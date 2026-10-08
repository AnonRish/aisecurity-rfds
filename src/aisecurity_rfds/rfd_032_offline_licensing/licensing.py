"""Signed compute licenses: device serial + per-meter budgets, verified on-chip against a fused public key.

Security-relevant choices:

* The signed digest covers the serial (so a license cannot be moved to another chip), the sequence number,
  the mode and every (meter, amount) pair. Each field is length-prefixed, so the encoding is injective.
* ``mode="add"`` tops up a balance; ``mode="set"`` overwrites it, which lets an authority lower or zero a
  balance (a revocation) but only on a chip that actually applies the license.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass

from aisecurity_rfds.common.hashing import domain_hash, u64
from aisecurity_rfds.common.signing import KeyPair

from .meters import U64_MAX, Meter

_DOMAIN = "rfd032/license/v1"
MODES = ("add", "set")


@dataclass(frozen=True)
class ComputeLicense:
    device_serial: bytes
    sequence: int
    mode: str
    budgets: tuple[tuple[str, int], ...]  # canonical form: strictly increasing meter names
    signature: bytes = b""

    def digest(self) -> bytes:
        parts = [self.device_serial, u64(self.sequence), self.mode.encode(), u64(len(self.budgets))]
        for name, amount in self.budgets:
            parts += [name.encode(), u64(amount)]
        return domain_hash(_DOMAIN, *parts)


class LicenseAuthority:
    """Signs licenses. Its key pair stands in for the private half of the key fused into every chip."""

    def __init__(self, key: KeyPair) -> None:
        self._key = key

    @property
    def public_key(self) -> bytes:
        return self._key.public_bytes

    def sign(self, unsigned: ComputeLicense) -> ComputeLicense:
        """Sign an arbitrary (possibly non-canonical) license; used by tests to build malformed inputs."""
        return dataclasses.replace(unsigned, signature=self._key.sign(unsigned.digest()))

    def issue(
        self, serial: bytes, *, sequence: int, budgets: Mapping[Meter, int], mode: str = "add"
    ) -> ComputeLicense:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not 1 <= sequence <= U64_MAX:
            raise ValueError("sequence must be in 1..2^64-1")
        if not budgets:
            raise ValueError("a license must carry at least one budget")
        items = []
        for meter, amount in budgets.items():
            if not 0 <= amount <= U64_MAX:
                raise ValueError("budget amounts must be in 0..2^64-1")
            items.append((Meter(meter).value, int(amount)))
        return self.sign(ComputeLicense(serial, sequence, mode, tuple(sorted(items))))
