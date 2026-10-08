"""Signed, hash-chained counter readouts with authorised resets (the RFD's "secure readout" and "counter monotonicity").

Roles
-----
``ReadoutBlock``     small trusted hardware holding the device signing key. It samples the counter registers and signs
                     (device, epoch, sequence, tick, counters, hash of the previous readout). Firmware only receives
                     the signed blob; it cannot reach the key.
``ResetAuthority``   issues ``ResetTicket`` s: signed, bound to one device and one epoch, with an expiry tick.
``ReadoutVerifier``  checks signature, sequence, hash chain, epoch handling and that every counter delta is
                     plausible (``0 <= delta <= max_rate * elapsed ticks`` in modular arithmetic, so a wrap is handled
                     and a regression shows up as an implausibly large positive delta).

A legitimate reset runs through the block: it verifies the ticket, emits a *closing* readout that reports the final
counters of the old epoch, zeroes the registers and increments the epoch. The chain continues across epochs, so a
reset never erases already-reported work.

What this protects and what it cannot: the *channel* (forgery, replay, dropping, reordering, unauthorised reset
through the block). It does not protect the *sensor*: a physical fault that makes the registers count less (clock
gating, a cut wire, a glitch timed to regain the old value) produces readouts that are validly signed, monotone and
plausible. Only the cross-checks of the other modules can see that.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from aisecurity_rfds.common.hashing import ct_equal, domain_hash, u32, u64
from aisecurity_rfds.common.signing import verify as ed25519_verify

DOMAIN_READOUT = "rfd035/readout/v1"
DOMAIN_CHAIN = "rfd035/chain/v1"
DOMAIN_TICKET = "rfd035/reset-ticket/v1"
GENESIS = b"\x00" * 32

VerifyFn = Callable[[bytes, bytes, bytes], bool]


class Signer(Protocol):
    public_bytes: bytes

    def sign(self, message: bytes) -> bytes: ...


class IdealSigner:
    """Stand-in for an *ideal* signature scheme, used only by the exhaustive exploration (it is fast).

    Unforgeable because the explored adversary never reads ``_key``. It is not a real scheme: the "public" bytes are
    the key itself. Real Ed25519 is used everywhere else.
    """

    def __init__(self, key: bytes) -> None:
        self._key = key
        self.public_bytes = key

    def sign(self, message: bytes) -> bytes:
        return domain_hash("rfd035/ideal-signature", self._key, message)


def ideal_verify(public_bytes: bytes, message: bytes, signature: bytes) -> bool:
    return ct_equal(domain_hash("rfd035/ideal-signature", public_bytes, message), signature)


@dataclass(frozen=True)
class ResetTicket:
    device_id: bytes
    from_epoch: int
    ticket_id: int
    not_after_tick: int
    signature: bytes

    def digest(self) -> bytes:
        return domain_hash(DOMAIN_TICKET, self.device_id, u64(self.from_epoch), u64(self.ticket_id), u64(self.not_after_tick))

    def wire_size(self) -> int:
        return len(self.device_id) + 3 * 8 + len(self.signature)


class ResetAuthority:
    def __init__(self, signer: Signer) -> None:
        self._signer = signer
        self._next_id = 1

    @property
    def public_bytes(self) -> bytes:
        return self._signer.public_bytes

    def issue(self, device_id: bytes, from_epoch: int, not_after_tick: int) -> ResetTicket:
        unsigned = ResetTicket(device_id, from_epoch, self._next_id, not_after_tick, b"")
        self._next_id += 1
        return replace(unsigned, signature=self._signer.sign(unsigned.digest()))

    def snapshot(self) -> int:
        return self._next_id

    def restore(self, state: int) -> None:
        self._next_id = state


@dataclass(frozen=True)
class Readout:
    device_id: bytes
    epoch: int
    seq: int
    ticks: int
    closing: bool
    counters: tuple[int, ...]
    prev_hash: bytes
    ticket: ResetTicket | None
    signature: bytes

    def digest(self) -> bytes:
        ticket_blob = b"" if self.ticket is None else self.ticket.digest() + self.ticket.signature
        return domain_hash(
            DOMAIN_READOUT,
            self.device_id,
            u64(self.epoch),
            u64(self.seq),
            u64(self.ticks),
            b"\x01" if self.closing else b"\x00",
            u32(len(self.counters)),
            *[u64(c) for c in self.counters],
            self.prev_hash,
            ticket_blob,
        )

    def chain_hash(self) -> bytes:
        return domain_hash(DOMAIN_CHAIN, self.digest(), self.signature)

    def wire_size(self) -> int:
        fixed = len(self.device_id) + 3 * 8 + 1 + 8 * len(self.counters) + len(self.prev_hash) + len(self.signature)
        return fixed + (self.ticket.wire_size() if self.ticket else 0)


class TicketRefused(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ReadoutBlock:
    """The trusted block plus the physical counter registers it samples (the registers are the attack surface)."""

    def __init__(
        self,
        device_id: bytes,
        signer: Signer,
        authority_public: bytes,
        *,
        n_counters: int = 2,
        width_bits: int = 64,
        verify_fn: VerifyFn = ed25519_verify,
    ) -> None:
        self.device_id = device_id
        self._signer = signer
        self._authority_public = authority_public
        self._verify = verify_fn
        self.n_counters = n_counters
        self.width_bits = width_bits
        self._mask = (1 << width_bits) - 1
        self.registers = [0] * n_counters
        self.epoch = 0
        self.seq = 0
        self.ticks = 0
        self.head = GENESIS
        self.stalled = False  # physical fault: increments are suppressed
        self.scale: float | None = None  # physical fault: increments are scaled down

    # --- the physical process and its faults ---
    def advance(self, dticks: int, increments: Sequence[int]) -> None:
        self.ticks += dticks
        if self.stalled:
            return
        for i, inc in enumerate(increments):
            inc = int(inc * self.scale) if self.scale is not None else int(inc)
            self.registers[i] = (self.registers[i] + inc) & self._mask

    def glitch_clear_registers(self) -> None:
        self.registers = [0] * self.n_counters

    def set_stalled(self, on: bool) -> None:
        self.stalled = on

    def set_scale(self, factor: float | None) -> None:
        self.scale = factor

    # --- the protocol ---
    def _emit(self, closing: bool, ticket: ResetTicket | None) -> Readout:
        self.seq += 1
        unsigned = Readout(self.device_id, self.epoch, self.seq, self.ticks, closing, tuple(self.registers), self.head, ticket, b"")
        signed = replace(unsigned, signature=self._signer.sign(unsigned.digest()))
        self.head = signed.chain_hash()
        return signed

    def read(self) -> Readout:
        return self._emit(False, None)

    def _ticket_problem(self, ticket: ResetTicket) -> str | None:
        if not self._verify(self._authority_public, ticket.digest(), ticket.signature):
            return "bad-ticket-signature"
        if ticket.device_id != self.device_id:
            return "ticket-for-another-device"
        if ticket.from_epoch != self.epoch:
            return "ticket-epoch-mismatch"
        if self.ticks > ticket.not_after_tick:
            return "ticket-expired"
        return None

    def reset(self, ticket: ResetTicket) -> Readout:
        """Authorised reset: returns the closing readout (final counters of the old epoch) or raises ``TicketRefused``."""
        problem = self._ticket_problem(ticket)
        if problem is not None:
            raise TicketRefused(problem)
        closing = self._emit(True, ticket)
        self.registers = [0] * self.n_counters
        self.epoch += 1
        return closing

    def snapshot(self) -> tuple:
        return (tuple(self.registers), self.epoch, self.seq, self.ticks, self.head, self.stalled, self.scale)

    def restore(self, state: tuple) -> None:
        regs, self.epoch, self.seq, self.ticks, self.head, self.stalled, self.scale = state
        self.registers = list(regs)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = "ok"


class ReadoutVerifier:
    """Accepts a readout only if it extends the verified history in an authorised, plausible way."""

    def __init__(
        self,
        device_id: bytes,
        device_public: bytes,
        authority_public: bytes,
        *,
        max_rate_per_tick: Sequence[int],
        n_counters: int = 2,
        width_bits: int = 64,
        max_gap_ticks: int | None = None,
        verify_fn: VerifyFn = ed25519_verify,
    ) -> None:
        self.device_id = device_id
        self._device_public = device_public
        self._authority_public = authority_public
        self._verify = verify_fn
        self.n_counters = n_counters
        self._mask = (1 << width_bits) - 1
        self.max_rate = tuple(max_rate_per_tick)
        self.max_gap_ticks = max_gap_ticks
        self.epoch = 0
        self.seq = 0
        self.head = GENESIS
        self.ticks = 0
        self.last_counters = [0] * n_counters
        self.credited = [0] * n_counters  # sum of accepted (modular) deltas: work the verifier has accounted for
        self.closed = False  # the last accepted readout was a closing one, so the next must open a new epoch

    def _reject(self, reason: str) -> Verdict:
        return Verdict(False, reason)

    def _ticket_problem(self, ro: Readout) -> str | None:
        t = ro.ticket
        if t is None:
            return "closing-without-ticket"
        if not self._verify(self._authority_public, t.digest(), t.signature):
            return "bad-ticket-signature"
        if t.device_id != self.device_id or t.from_epoch != ro.epoch:
            return "ticket-not-for-this-epoch"
        if ro.ticks > t.not_after_tick:
            return "ticket-expired"
        return None

    def accept(self, ro: Readout) -> Verdict:
        if len(ro.counters) != self.n_counters:
            return self._reject("malformed")
        if ro.device_id != self.device_id:
            return self._reject("wrong-device")
        if not self._verify(self._device_public, ro.digest(), ro.signature):
            return self._reject("bad-signature")
        if ro.seq <= self.seq:
            return self._reject("replay")
        if ro.seq > self.seq + 1:
            return self._reject("gap")
        if not ct_equal(ro.prev_hash, self.head):
            return self._reject("chain-break")
        if ro.ticks <= self.ticks:
            return self._reject("time-regression")
        if self.max_gap_ticks is not None and ro.ticks - self.ticks > self.max_gap_ticks:
            return self._reject("stale")
        if self.closed:
            if ro.epoch != self.epoch + 1:
                return self._reject("epoch-jump")
            baseline = [0] * self.n_counters
        else:
            if ro.epoch != self.epoch:
                return self._reject("epoch-jump")
            baseline = self.last_counters
        elapsed = ro.ticks - self.ticks
        deltas = [(c - b) & self._mask for c, b in zip(ro.counters, baseline, strict=True)]
        if any(d > rate * elapsed for d, rate in zip(deltas, self.max_rate, strict=True)):
            return self._reject("implausible-delta")  # a regression within an epoch lands here
        if ro.closing:
            problem = self._ticket_problem(ro)
            if problem is not None:
                return self._reject(problem)
        elif ro.ticket is not None:
            return self._reject("ticket-on-open-readout")
        self.epoch = ro.epoch
        self.closed = ro.closing
        self.seq = ro.seq
        self.head = ro.chain_hash()
        self.ticks = ro.ticks
        self.last_counters = list(ro.counters)
        self.credited = [c + d for c, d in zip(self.credited, deltas, strict=True)]
        return Verdict(True)

    def snapshot(self) -> tuple:
        return (self.epoch, self.seq, self.head, self.ticks, tuple(self.last_counters), tuple(self.credited), self.closed)

    def restore(self, state: tuple) -> None:
        self.epoch, self.seq, self.head, self.ticks, last, credited, self.closed = state
        self.last_counters = list(last)
        self.credited = list(credited)


def protocol_overhead(n_devices: int, readouts_per_second: float, n_counters: int = 2, device_id_bytes: int = 8) -> dict[str, float]:
    """Deterministic arithmetic for the *communication and verification* overhead of the protocol.

    It says nothing about silicon area or power, which this package cannot estimate.
    """
    sample = Readout(b"\x00" * device_id_bytes, 0, 0, 0, False, (0,) * n_counters, GENESIS, None, b"\x00" * 64)
    closing = replace(sample, closing=True, ticket=ResetTicket(b"\x00" * device_id_bytes, 0, 0, 0, b"\x00" * 64))
    return {
        "readout_bytes": float(sample.wire_size()),
        "closing_readout_bytes": float(closing.wire_size()),
        "bytes_per_second_total": sample.wire_size() * readouts_per_second * n_devices,
        "signature_verifications_per_second": readouts_per_second * n_devices,
    }
