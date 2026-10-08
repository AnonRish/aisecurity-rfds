"""On-chip license verifier and meter block.

State kept across power cycles lives in one journaled flash record (``last_sequence`` plus a *persisted*
balance per enforced meter). Volatile registers hold the true remaining balance.

Fail-safe checkpointing: ``persisted <= true remaining`` at every instant, because a new checkpoint of
``remaining - interval`` is written *before* any consumption that would take the balance below the old one.
A power cycle therefore restores a balance that is never larger than the true one (no gain for an attacker
who cuts power) and at most ``interval`` smaller (the legitimate user's loss).

Replay protection here is the plain strictly-increasing sequence number; RFD 034 analyses it in depth.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from aisecurity_rfds.common.hashing import u64
from aisecurity_rfds.common.nvm import FlashStore, JournaledRecord
from aisecurity_rfds.common.signing import verify

from .licensing import MODES, ComputeLicense
from .meters import U64_MAX, Meter


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str


class OfflineLicenseChip:
    def __init__(
        self,
        serial: bytes,
        authority_public: bytes,
        *,
        enforced: Iterable[Meter],
        max_balance: Mapping[Meter, int] | None = None,
        checkpoint_interval: Mapping[Meter, int] | None = None,
        flash: FlashStore | None = None,
    ) -> None:
        self.serial = serial
        self._public = authority_public
        self._enforced = tuple(sorted({Meter(m) for m in enforced}, key=lambda m: m.value))
        if not self._enforced:
            raise ValueError("at least one meter must be enforced")
        self._cap = {m: U64_MAX for m in self._enforced}
        self._cap.update({Meter(m): v for m, v in (max_balance or {}).items() if Meter(m) in self._cap})
        self._interval = {m: 1 for m in self._enforced}
        self._interval.update({Meter(m): v for m, v in (checkpoint_interval or {}).items() if Meter(m) in self._interval})
        if any(v < 1 for v in self._interval.values()):
            raise ValueError("checkpoint intervals must be >= 1")
        self._flash = flash if flash is not None else FlashStore()
        self._state = JournaledRecord(self._flash, "license-state")
        self.checkpoint_writes = 0
        self.power_cycle()

    # ---------------------------------------------------------------- persistence

    def _encode(self, last_sequence: int, persisted: Mapping[Meter, int]) -> bytes:
        return u64(last_sequence) + b"".join(u64(persisted[m]) for m in self._enforced)

    def _persist(self, last_sequence: int, persisted: Mapping[Meter, int]) -> None:
        self._state.write(self._encode(last_sequence, persisted))

    def power_cycle(self) -> None:
        """Lose all volatile state and reload the conservative persisted balances."""
        blob = self._state.read()
        if blob is None:
            self._last_sequence = 0
            self._persisted = {m: 0 for m in self._enforced}
        else:
            self._last_sequence = int.from_bytes(blob[:8], "big")
            self._persisted = {
                m: int.from_bytes(blob[8 + 8 * i : 16 + 8 * i], "big") for i, m in enumerate(self._enforced)
            }
        self._remaining = dict(self._persisted)

    # ---------------------------------------------------------------- licenses

    def apply_license(self, lic: ComputeLicense) -> Decision:
        if lic.device_serial != self.serial:
            return Decision(False, "wrong-device")
        if not verify(self._public, lic.digest(), lic.signature):
            return Decision(False, "bad-signature")
        names = [name for name, _ in lic.budgets]
        if lic.mode not in MODES or not names or names != sorted(set(names)):
            return Decision(False, "malformed")
        if lic.sequence <= self._last_sequence:
            return Decision(False, "replay-or-stale")
        enforced_names = {m.value for m in self._enforced}
        if any(name not in enforced_names for name in names):
            return Decision(False, "unknown-meter")
        updated = dict(self._remaining)
        for name, amount in lic.budgets:
            meter = Meter(name)
            value = amount if lic.mode == "set" else self._remaining[meter] + amount
            if value > U64_MAX:
                return Decision(False, "overflow")
            if value > self._cap[meter]:
                return Decision(False, "balance-cap")
            updated[meter] = value
        self._persist(lic.sequence, updated)  # atomic: either the whole license is applied or none of it
        self._last_sequence = lic.sequence
        self._remaining = updated
        self._persisted = dict(updated)
        return Decision(True, "accepted")

    # ---------------------------------------------------------------- metering

    @property
    def enforced(self) -> tuple[Meter, ...]:
        return self._enforced

    @property
    def last_sequence(self) -> int:
        return self._last_sequence

    def remaining(self, meter: Meter) -> int:
        return self._remaining[Meter(meter)]

    def is_throttled(self) -> bool:
        """Throttled as soon as any enforced meter is empty (an unlicensed chip starts empty)."""
        return any(v == 0 for v in self._remaining.values())

    def units_available(self, cost: Mapping[Meter, int]) -> int | None:
        """Units of work (each costing ``cost``) that fit in the remaining balances; None if unmetered."""
        limits = [self._remaining[Meter(m)] // c for m, c in cost.items() if c > 0 and Meter(m) in self._remaining]
        return min(limits) if limits else None

    def consume(self, usage: Mapping[Meter, int]) -> None:
        """Debit the meters (clamped at zero). Checkpoints are written before the debit takes effect."""
        remaining, persisted, dirty = dict(self._remaining), dict(self._persisted), False
        for meter, amount in usage.items():
            m = Meter(meter)
            if amount < 0:
                raise ValueError("usage must be non-negative")
            if m not in remaining:
                continue  # unenforced meters are not counted
            remaining[m] = max(0, remaining[m] - amount)
            if persisted[m] > remaining[m]:
                persisted[m] = max(0, remaining[m] - self._interval[m])
                dirty = True
        if dirty:
            self._persist(self._last_sequence, persisted)
            self.checkpoint_writes += 1
        self._remaining, self._persisted = remaining, persisted
