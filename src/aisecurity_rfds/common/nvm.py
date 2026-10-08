"""Simulated non-volatile memories with the failure modes that matter for anti-replay and metering.

* ``OtpFuseBank``: bits only go 0 -> 1. ``hv_disrupted`` models an attacker starving the high-voltage
  programming supply, so burns silently do nothing unless the caller reads back (RFD 034).
* ``UnaryCounter``: a monotonic counter stored as a contiguous prefix of burned fuses.
* ``FlashStore``: key/value flash with torn writes (power loss) and an attacker-side
  snapshot/restore primitive for rollback attacks.
* ``JournaledRecord``: double-buffered, CRC-protected record that survives torn writes.
* ``BatteryBackedSram``: contents vanish when the battery is removed.
"""

from __future__ import annotations

import zlib

import numpy as np


class NvmError(Exception):
    pass


class WriteVerifyError(NvmError):
    """A write did not take effect (detected by read-back)."""


class CounterExhaustedError(NvmError):
    pass


class CounterTamperedError(NvmError):
    """The unary counter has a hole: a later fuse is burned while an earlier one is not."""


class OtpFuseBank:
    def __init__(self, n_bits: int):
        self._bits = np.zeros(n_bits, dtype=np.uint8)
        self.hv_disrupted = False
        self.burn_count = 0

    @property
    def n_bits(self) -> int:
        return len(self._bits)

    def read(self, i: int) -> int:
        return int(self._bits[i])

    def burn(self, i: int, *, verify: bool = False) -> None:
        if not self.hv_disrupted:
            self._bits[i] = 1
            self.burn_count += 1
        if verify and self._bits[i] != 1:
            raise WriteVerifyError(f"fuse {i} did not burn (high-voltage supply disrupted?)")

    def burned(self) -> int:
        return int(self._bits.sum())


class UnaryCounter:
    """Monotonic counter: value = number of burned fuses in a contiguous prefix."""

    def __init__(self, bank: OtpFuseBank, offset: int = 0, capacity: int | None = None):
        self._bank = bank
        self._offset = offset
        self._capacity = capacity if capacity is not None else bank.n_bits - offset
        if offset + self._capacity > bank.n_bits:
            raise ValueError("counter does not fit in the fuse bank")

    @property
    def capacity(self) -> int:
        return self._capacity

    def value(self) -> int:
        window = self._bank._bits[self._offset : self._offset + self._capacity]
        zeros = np.flatnonzero(window == 0)
        first_zero = int(zeros[0]) if len(zeros) else self._capacity
        if window[first_zero:].any():
            raise CounterTamperedError("burned fuse found after an unburned one")
        return first_zero

    def increment(self, *, verify: bool = True) -> int:
        v = self.value()
        if v >= self._capacity:
            raise CounterExhaustedError("all fuses burned")
        self._bank.burn(self._offset + v, verify=verify)
        return v + 1


class FlashStore:
    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self.write_count = 0

    def read(self, key: str) -> bytes | None:
        return self._data.get(key)

    def write(self, key: str, value: bytes, *, torn_at: int | None = None) -> None:
        """Write ``value``. ``torn_at`` simulates power loss after that many bytes were programmed."""
        old = self._data.get(key, b"")
        if torn_at is None:
            new = value
        else:
            cut = max(0, min(torn_at, len(value)))
            new = value[:cut] + old[cut:]
        self._data[key] = new
        self.write_count += 1

    def snapshot(self) -> dict[str, bytes]:
        return dict(self._data)

    def physical_restore(self, snapshot: dict[str, bytes]) -> None:
        """Attacker primitive: put back an earlier flash image (rollback attack)."""
        self._data = dict(snapshot)


class JournaledRecord:
    """Atomic record in two flash slots; a torn write can never destroy the last committed value."""

    def __init__(self, store: FlashStore, name: str):
        self._store = store
        self._slots = (f"{name}.A", f"{name}.B")

    @staticmethod
    def _encode(seq: int, payload: bytes) -> bytes:
        body = seq.to_bytes(8, "big") + len(payload).to_bytes(4, "big") + payload
        return body + zlib.crc32(body).to_bytes(4, "big")

    @staticmethod
    def _decode(blob: bytes | None) -> tuple[int, bytes] | None:
        if blob is None or len(blob) < 16:
            return None
        body, crc = blob[:-4], blob[-4:]
        if zlib.crc32(body).to_bytes(4, "big") != crc:
            return None
        seq = int.from_bytes(body[:8], "big")
        length = int.from_bytes(body[8:12], "big")
        payload = body[12:]
        if len(payload) != length:
            return None
        return seq, payload

    def _latest(self) -> tuple[int, int, bytes] | None:
        best: tuple[int, int, bytes] | None = None
        for slot_index, name in enumerate(self._slots):
            decoded = self._decode(self._store.read(name))
            if decoded is not None and (best is None or decoded[0] > best[0]):
                best = (decoded[0], slot_index, decoded[1])
        return best

    def read(self) -> bytes | None:
        latest = self._latest()
        return latest[2] if latest else None

    def write(self, payload: bytes, *, torn_at: int | None = None) -> None:
        latest = self._latest()
        seq = (latest[0] + 1) if latest else 1
        target = 1 - latest[1] if latest else 0
        self._store.write(self._slots[target], self._encode(seq, payload), torn_at=torn_at)


class BatteryBackedSram:
    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self.battery_ok = True

    def write(self, key: str, value: bytes) -> None:
        if not self.battery_ok:
            raise NvmError("SRAM unpowered")
        self._data[key] = value

    def read(self, key: str) -> bytes | None:
        return self._data.get(key) if self.battery_ok else None

    def remove_battery(self) -> None:
        self._data.clear()
        self.battery_ok = False

    def replace_battery(self) -> None:
        self.battery_ok = True
