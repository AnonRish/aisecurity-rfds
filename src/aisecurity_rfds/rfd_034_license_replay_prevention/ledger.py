"""External commitment option of RFD 034: the chip signs ``(device_id, sequence, timestamp)`` every N licenses.

The ledger can flag a *regression* (a later timestamp with a lower sequence). What it can and cannot catch
is exercised in ``scenarios.ledger_detection``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from aisecurity_rfds.common.hashing import domain_hash, u64
from aisecurity_rfds.common.signing import KeyPair, verify


@dataclass(frozen=True)
class Commitment:
    device_id: bytes
    sequence: int
    timestamp: int
    signature: bytes

    def message(self) -> bytes:
        return domain_hash("rfd034/commitment/v1", self.device_id, u64(self.sequence), u64(self.timestamp))


class CommitmentEmitter:
    """Chip side: emits a signed commitment after every ``every``-th accepted license (virtual clock)."""

    def __init__(self, device_key: KeyPair, device_id: bytes, every: int) -> None:
        if every < 1:
            raise ValueError("every must be >= 1")
        self._key, self._device_id, self._every = device_key, device_id, every
        self._applied = 0
        self._clock = 0

    def on_accepted(self, high_water_sequence: int) -> Commitment | None:
        self._applied += 1
        self._clock += 1
        if self._applied % self._every:
            return None
        unsigned = Commitment(self._device_id, high_water_sequence, self._clock, b"")
        return dataclasses.replace(unsigned, signature=self._key.sign(unsigned.message()))


class ExternalLedger:
    """Verifier side: stores commitments and reports anomalies; it never learns about licenses directly."""

    def __init__(self, device_id: bytes, device_public: bytes) -> None:
        self._device_id, self._public = device_id, device_public
        self._seen: list[Commitment] = []

    def submit(self, commitment: Commitment) -> list[str]:
        if commitment.device_id != self._device_id or not verify(self._public, commitment.message(), commitment.signature):
            return ["rejected: bad signature or wrong device"]
        anomalies: list[str] = []
        for prev in self._seen:
            later_but_lower = commitment.timestamp > prev.timestamp and commitment.sequence < prev.sequence
            earlier_but_higher = commitment.timestamp < prev.timestamp and commitment.sequence > prev.sequence
            if later_but_lower or earlier_but_higher:
                anomalies.append("regression")
                break
        self._seen.append(commitment)
        return anomalies
