"""A small world to run the registry in: a manufacturer, three organisations, an operator and an accredited recycler.

Used by the entry point, the attack catalogue and the tests so that they all act on the same, deterministic parties.
Keys come from ``party_key`` and anyone can recompute them: this is a demonstration, not a key-management example.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aisecurity_rfds.common.signing import KeyPair

from .identity import FRESH, Attestation, CounterTracker, DeviceKeyChip, party_key
from .registry import (
    COMPLETE_TRANSFER,
    ENROLL,
    RELOCATE,
    RETIRE,
    START_TRANSFER,
    LogEntry,
    Policy,
    Record,
    Registry,
    SignedRequest,
    make_request,
    sign_request,
    transfer_challenge,
)

EMBARGOED = "XX"


class ForgetfulTracker(CounterTracker):
    """A tracker that remembers nothing: stands in for an operator that skips its attestation-counter check."""

    def observe(self, attestation: Attestation) -> str:
        return FRESH


@dataclass
class World:
    registry: Registry
    policy: Policy
    fab: KeyPair
    alice: KeyPair
    bob: KeyPair
    mallory: KeyPair
    operator: KeyPair
    recycler: KeyPair
    clock: int = 1_000_000
    chips: dict[str, DeviceKeyChip] = field(default_factory=dict)

    @classmethod
    def new(cls, require_destruction_witness: bool = True, skip_counter_check: bool = False) -> World:
        fab, alice, bob, mallory, operator, recycler = (
            party_key(name) for name in ("fab", "alice", "bob", "mallory", "operator", "recycler")
        )
        policy = Policy(
            manufacturers=frozenset({fab.public_bytes}),
            embargoed=frozenset({EMBARGOED}),
            destruction_witnesses=frozenset({recycler.public_bytes}) if require_destruction_witness else frozenset(),
        )
        registry = Registry(operator, policy)
        if skip_counter_check:
            registry.tracker = ForgetfulTracker()
        return cls(registry, policy, fab, alice, bob, mallory, operator, recycler)

    def tick(self, seconds: int = 10) -> int:
        self.clock += seconds
        return self.clock

    def submit(self, signed: SignedRequest) -> LogEntry:
        return self.registry.submit(signed, self.tick())

    def record(self, chip: DeviceKeyChip) -> Record:
        record = self.registry.record(chip.chip_id)
        assert record is not None, "chip is not registered"
        return record

    def enroll(self, label: str, jurisdiction: str = "CA", location: bytes = b"") -> DeviceKeyChip:
        chip = DeviceKeyChip(label.encode())
        request = make_request(ENROLL, chip.chip_id, 0, public_key=chip.public_key, jurisdiction=jurisdiction, location=location)
        self.submit(sign_request(request, self.fab))
        self.chips[label] = chip
        return chip

    def offer(self, chip: DeviceKeyChip, seller: KeyPair, buyer: KeyPair, window: int = 3600) -> LogEntry:
        record = self.record(chip)
        request = make_request(
            START_TRANSFER, chip.chip_id, record.version, buyer=buyer.public_bytes, expires=self.clock + window
        )
        return self.submit(sign_request(request, seller))

    def completion(self, chip: DeviceKeyChip, buyer: KeyPair, jurisdiction: str = "US", attested: DeviceKeyChip | None = None):
        """The signed completion request for the pending transfer, with an attestation from ``attested`` (default: the chip)."""
        record = self.record(chip)
        attestation = (attested or chip).attest(transfer_challenge(record))
        request = make_request(
            COMPLETE_TRANSFER,
            chip.chip_id,
            record.version,
            jurisdiction=jurisdiction,
            att_counter=attestation.counter,
            att_signature=attestation.signature,
        )
        return sign_request(request, buyer)

    def sell(self, chip: DeviceKeyChip, seller: KeyPair, buyer: KeyPair, jurisdiction: str = "US") -> LogEntry:
        self.offer(chip, seller, buyer)
        return self.submit(self.completion(chip, buyer, jurisdiction))

    def relocate(self, chip: DeviceKeyChip, owner: KeyPair, jurisdiction: str, location: bytes = b"") -> LogEntry:
        request = make_request(RELOCATE, chip.chip_id, self.record(chip).version, jurisdiction=jurisdiction, location=location)
        return self.submit(sign_request(request, owner))

    def retire(self, chip: DeviceKeyChip, owner: KeyPair, reason: str, witnessed: bool = False) -> LogEntry:
        request = make_request(RETIRE, chip.chip_id, self.record(chip).version, reason=reason)
        return self.submit(sign_request(request, owner, *([self.recycler] if witnessed else [])))
