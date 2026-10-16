"""Bandwidth limiting at the chip's port: an exact token bucket and a two-class port.

RFD 033 caps traffic to anything outside the pod at a low rate (its example: 1 GB/s against 900 GB/s for NVLink).
This module is the smallest faithful model of that enforcement point: every byte leaving the chip is charged to
either the *pod* bucket (only for peers whose link is currently HIGH) or the single *external* bucket shared by
every other destination.

The bucket uses integer arithmetic only, so its guarantee is exact rather than approximate: over any window of
length ``dt`` it grants at most ``burst + rate * dt`` bytes. Two details matter for security and are covered by
tests: the clock high-water mark never moves backwards (a rolled-back clock must not mint credit) and the external
bucket is per chip, not per destination (splitting traffic over many destinations must not multiply the limit).
"""

from __future__ import annotations

from .pod import Chip, LinkMode

NS = 10**9
HIGH_BANDWIDTH_BPS = 900 * 10**9  # per the RFD (NVLink example)
LOW_BANDWIDTH_BPS = 1 * 10**9  # per the RFD ("e.g., 1 GB/s")


class TokenBucket:
    """Exact token bucket. Credit is held in byte-nanoseconds so no rounding can leak bytes."""

    def __init__(self, rate_bps: int, burst_bytes: int, *, now_ns: int = 0) -> None:
        if rate_bps <= 0 or burst_bytes <= 0:
            raise ValueError("rate and burst must be positive")
        self.rate_bps = rate_bps
        self.burst_bytes = burst_bytes
        self._cap = burst_bytes * NS
        self._credit = self._cap
        self._last = now_ns
        self.granted_total = 0

    def _refill(self, now_ns: int) -> None:
        if now_ns > self._last:  # a clock that goes backwards earns nothing and does not lower the mark
            self._credit = min(self._cap, self._credit + self.rate_bps * (now_ns - self._last))
            self._last = now_ns

    def take_up_to(self, nbytes: int, now_ns: int) -> int:
        """Grant as many of ``nbytes`` as the credit allows (possibly zero)."""
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        self._refill(now_ns)
        grant = min(nbytes, self._credit // NS)
        self._credit -= grant * NS
        self.granted_total += grant
        return grant

    def try_send(self, nbytes: int, now_ns: int) -> bool:
        """All-or-nothing variant of ``take_up_to``."""
        self._refill(now_ns)
        if nbytes * NS > self._credit:
            return False
        self._credit -= nbytes * NS
        self.granted_total += nbytes
        return True

    def wait_ns(self, nbytes: int, now_ns: int) -> int:
        """Nanoseconds from ``now_ns`` until ``nbytes`` (at most the burst) could be sent in one piece."""
        if nbytes > self.burst_bytes:
            raise ValueError("a single send larger than the burst can never be granted")
        self._refill(now_ns)
        deficit = nbytes * NS - self._credit
        return 0 if deficit <= 0 else -(-deficit // self.rate_bps)

    def stream(self, nbytes: int, now_ns: int) -> int:
        """Push ``nbytes`` through the bucket as fast as it allows; returns the completion time in ns."""
        t = now_ns
        remaining = nbytes
        while remaining > 0:
            chunk = min(remaining, self.burst_bytes)
            t += self.wait_ns(chunk, t)
            if not self.try_send(chunk, t):  # pragma: no cover - wait_ns guarantees success
                raise AssertionError("bucket refused a send it had just promised")
            remaining -= chunk
        return t


class ChipPort:
    """A chip's egress: HIGH-link peers use the pod bucket, everything else shares one external bucket."""

    def __init__(
        self,
        chip: Chip,
        *,
        external_bps: int = LOW_BANDWIDTH_BPS,
        internal_bps: int = HIGH_BANDWIDTH_BPS,
        burst_ns: int = 1_000_000,
    ) -> None:
        self.chip = chip
        self.external = TokenBucket(external_bps, max(1, external_bps * burst_ns // NS))
        self.internal = TokenBucket(internal_bps, max(1, internal_bps * burst_ns // NS))

    def send(self, dest: bytes, nbytes: int, now_ns: int) -> int:
        """Bytes of ``nbytes`` to ``dest`` that the port lets through at ``now_ns``."""
        bucket = self.internal if self.chip.link_mode(dest) is LinkMode.HIGH else self.external
        return bucket.take_up_to(nbytes, now_ns)
