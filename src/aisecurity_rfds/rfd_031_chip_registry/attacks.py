"""The attack catalogue: every row is executed against the real registry, and the outcome is what the code did.

``protocol_attacks`` are requests that the registry's rules must refuse. ``operator_attacks`` are things a dishonest
operator can do to its own log, map and heads, and which of the clients, auditors and witnesses notices.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from aisecurity_rfds.common.merkle import MerkleLog

from .identity import DeviceKeyChip, party_key
from .registry import (
    CANCEL_TRANSFER,
    ENROLL,
    RELOCATE,
    START_TRANSFER,
    HistoryResponse,
    LogEntry,
    QueryResponse,
    Registry,
    Rejected,
    Request,
    SignedTreeHead,
    Verdict,
    audit_log,
    make_request,
    sign_request,
    sth_message,
    verify_history,
    verify_query,
)
from .scenario import EMBARGOED, World
from .smt import SparseMerkleMap
from .witness import Witness, WitnessRefused, fork_by_enumeration


@dataclass(frozen=True)
class AttackResult:
    attack: str
    defence: str
    outcome: str


def _refusal(action: Callable[[], object]) -> str:
    try:
        action()
    except Rejected as exc:
        return f"rejected: {exc.code}"
    return "accepted"


def protocol_attacks() -> list[AttackResult]:
    results: list[AttackResult] = []

    def add(attack: str, defence: str, action: Callable[[], object]) -> None:
        results.append(AttackResult(attack, defence, _refusal(action)))

    def forged_enrolment() -> None:
        w, chip = World.new(), DeviceKeyChip(b"forged")
        w.submit(sign_request(make_request(ENROLL, chip.chip_id, 0, public_key=chip.public_key, jurisdiction="CA"), w.mallory))

    def wrong_chip_id() -> None:
        w, chip = World.new(), DeviceKeyChip(b"wrong-id")
        w.submit(sign_request(make_request(ENROLL, b"\x11" * 32, 0, public_key=chip.public_key, jurisdiction="CA"), w.fab))

    def duplicate_enrolment() -> None:
        w = World.new()
        chip = w.enroll("dup")
        w.submit(sign_request(make_request(ENROLL, chip.chip_id, 0, public_key=chip.public_key, jurisdiction="CA"), w.fab))

    def stranger_sells() -> None:
        w = World.new()
        w.offer(w.enroll("c"), w.mallory, w.bob)

    def replayed_offer() -> None:
        w = World.new()
        chip = w.enroll("c")
        version = w.record(chip).version
        offer = make_request(START_TRANSFER, chip.chip_id, version, buyer=w.alice.public_bytes, expires=w.clock + 3600)
        signed = sign_request(offer, w.fab)
        w.submit(signed)
        w.submit(sign_request(make_request(CANCEL_TRANSFER, chip.chip_id, version + 1), w.alice))
        w.submit(signed)

    def double_sale() -> None:
        w = World.new()
        chip = w.enroll("c")
        w.offer(chip, w.fab, w.alice)
        w.offer(chip, w.fab, w.bob)

    def wrong_buyer() -> None:
        w = World.new()
        chip = w.enroll("c")
        w.offer(chip, w.fab, w.alice)
        w.submit(w.completion(chip, w.mallory))

    def late_completion() -> None:
        w = World.new()
        chip = w.enroll("c")
        w.offer(chip, w.fab, w.alice, window=100)
        w.tick(1000)
        w.submit(w.completion(chip, w.alice))

    def reused_attestation() -> None:
        w = World.new()
        chip = w.enroll("c")
        w.offer(chip, w.fab, w.alice)
        old = w.completion(chip, w.alice).request
        w.submit(sign_request(make_request(CANCEL_TRANSFER, chip.chip_id, w.record(chip).version), w.alice))
        w.offer(chip, w.fab, w.alice)
        request = make_request(
            "complete-transfer",
            chip.chip_id,
            w.record(chip).version,
            jurisdiction="US",
            att_counter=old.get("att_counter"),
            att_signature=old.get("att_signature"),
        )
        w.submit(sign_request(request, w.alice))

    def copy_seen_first() -> None:
        w = World.new()
        chip = w.enroll("c")
        w.sell(chip, w.fab, w.alice)
        copy = chip.clone()
        w.registry.observe_attestation(copy.attest(b"inspection"), b"inspection", w.tick())
        w.sell(chip, w.alice, w.bob)

    def early_withdrawal() -> None:
        w = World.new()
        chip = w.enroll("c")
        w.offer(chip, w.fab, w.alice)
        w.submit(sign_request(make_request(CANCEL_TRANSFER, chip.chip_id, w.record(chip).version), w.fab))

    def embargoed_sale() -> None:
        w = World.new()
        w.sell(w.enroll("c"), w.fab, w.alice, jurisdiction=EMBARGOED)

    def after_retirement() -> None:
        w = World.new()
        chip = w.enroll("c")
        w.retire(chip, w.fab, "recycled")
        w.relocate(chip, w.fab, "US")

    def unwitnessed_destruction() -> None:
        w = World.new()
        w.retire(w.enroll("c"), w.fab, "destroyed")

    def relocate_in_transit() -> None:
        w = World.new()
        chip = w.enroll("c")
        w.offer(chip, w.fab, w.alice)
        w.relocate(chip, w.fab, "US")

    def duplicated_field() -> None:
        w = World.new()
        chip = w.enroll("c")
        request = Request(RELOCATE, chip.chip_id, w.record(chip).version, (("jurisdiction", b"US"), ("jurisdiction", b"GB")))
        w.submit(sign_request(request, w.fab))

    add("Enrolment signed by a key that is not an approved manufacturer", "manufacturer allow-list", forged_enrolment)
    add("Enrolment under an id that is not the hash of the chip's key", "chip id binding", wrong_chip_id)
    add("A second enrolment of a registered chip", "unique chip id", duplicate_enrolment)
    add("A transfer offered by someone who does not own the chip", "owner's signature", stranger_sells)
    add("An old offer replayed after the record has moved on", "expected version", replayed_offer)
    add("A second offer while the first is pending", "transfer state machine", double_sale)
    add("Completion by a party other than the named buyer", "buyer's signature", wrong_buyer)
    add("Completion after the offer has lapsed", "offer expiry", late_completion)
    add("Completion that reuses the chip's attestation from an earlier offer", "challenge bound to the offer", reused_attestation)
    add("The genuine chip completes a sale after a copy of it attested", "attestation counter", copy_seen_first)
    add("The seller withdraws an offer before it lapses", "offer window", early_withdrawal)
    add("A sale into an embargoed jurisdiction", "policy list", embargoed_sale)
    add("Any change to a retired chip", "retirement is final", after_retirement)
    add("Destruction recorded without an accredited witness", "witness signature", unwitnessed_destruction)
    add("A relocation while a transfer is pending", "transfer state machine", relocate_in_transit)
    add("A request that names the same field twice", "canonical fields", duplicated_field)

    w = World.new()
    chip = w.enroll("c")
    w.retire(chip, w.fab, "lost")
    alerts = w.registry.observe_attestation(chip.attest(b"inspection"), b"inspection", w.tick())
    results.append(
        AttackResult(
            "A chip recorded as lost attests to an inspector",
            "attestation monitoring",
            "alert: " + ", ".join(a.kind for a in alerts),
        )
    )
    return results


def forked_heads() -> tuple[bytes, SignedTreeHead, SignedTreeHead, int]:
    """(operator key, head A, head B, time): two registries agree on three entries, then log different relocations."""
    w = World.new()
    chip = w.enroll("chip-0")
    w.sell(chip, w.fab, w.alice)
    entries, now = w.registry.entries(), w.clock
    reg_a = Registry.replicate(w.operator, w.policy, entries)
    reg_b = Registry.replicate(w.operator, w.policy, entries)
    reg_a.submit(sign_request(make_request(RELOCATE, chip.chip_id, 3, jurisdiction="FR"), w.alice), now + 10)
    reg_b.submit(sign_request(make_request(RELOCATE, chip.chip_id, 3, jurisdiction="GB"), w.alice), now + 10)
    return w.operator.public_bytes, reg_a.sth(now + 20), reg_b.sth(now + 20), now + 20


def _forge(entries: tuple[LogEntry, ...], now: int, operator) -> tuple[MerkleLog, SignedTreeHead]:
    """A log and a signed head over ``entries`` as claimed: the operator can sign whatever it likes."""
    log, smap = MerkleLog(), SparseMerkleMap()
    for entry in entries:
        log.append(entry.encode())
        smap.set(entry.record_after.chip_id, entry.record_after.encode())
    size, log_root, map_root = len(entries), log.root(), smap.root()
    return log, SignedTreeHead(size, log_root, map_root, now, operator.sign(sth_message(size, log_root, map_root, now)))


def _verdict(v: Verdict) -> str:
    return "accepted" if v.ok else f"rejected: {v.reason}"


def operator_attacks() -> list[AttackResult]:
    results: list[AttackResult] = []

    def add(attack: str, defence: str, outcome: str) -> None:
        results.append(AttackResult(attack, defence, outcome))

    w = World.new()
    chip = w.enroll("chip-0")
    w.sell(chip, w.fab, w.alice)
    early_head = w.registry.sth(w.clock + 5)  # three entries: the sale is complete; the head is signed 5 s later
    other = w.enroll("chip-1")
    entries = w.registry.entries()
    now = w.clock
    head = w.registry.sth(now)
    op, policy = w.operator, w.policy

    def detected(report) -> str:
        return "accepted: audit passes" if report.ok else f"detected: {report.problems[0]}"

    # 1. rewrite an old entry and re-sign a consistent head
    altered = replace(entries[2], record_after=replace(entries[2].record_after, owner=w.mallory.public_bytes))
    rewritten = entries[:2] + (altered,) + entries[3:]
    rewritten_log, rewritten_head = _forge(rewritten, now, op)
    add("Rewrite an old entry and re-sign the head", "audit replay", detected(audit_log(rewritten, policy, rewritten_head, op.public_bytes)))
    witness = Witness(party_key("witness-0"), op.public_bytes)
    witness.cosign(early_head, [], now)
    try:
        witness.cosign(rewritten_head, rewritten_log.consistency_proof(early_head.size, rewritten_head.size), now)
        outcome = "accepted"
    except WitnessRefused as exc:
        outcome = f"refused: {exc.code}"
    add("The same rewrite shown to a witness that cosigned the earlier head", "consistency proof", outcome)

    # 2. append a transfer that the owner never signed
    current = w.record(chip)
    bogus = sign_request(make_request(START_TRANSFER, chip.chip_id, current.version, buyer=w.mallory.public_bytes, expires=now + 100), w.mallory)
    claimed = replace(current, owner=w.mallory.public_bytes, version=current.version + 1)
    forged_entries = entries + (LogEntry(len(entries), now, bogus, claimed),)
    _, forged_head = _forge(forged_entries, now, op)
    add("Append a transfer the owner never signed", "audit replay", detected(audit_log(forged_entries, policy, forged_head, op.public_bytes)))

    # 2b. an operator that lets a copied chip complete a sale: the copy's counter is not above the genuine chip's
    lax = World.new(skip_counter_check=True)
    original = lax.enroll("chip-2")
    copy = original.clone()  # the key leaks before the first sale, while both counters stand at zero
    lax.sell(original, lax.fab, lax.alice)  # the genuine chip attests once, so its counter is now 1
    lax.offer(original, lax.alice, lax.bob)
    lax.submit(lax.completion(original, lax.bob, attested=copy))  # the copy's first attestation is also 1
    lax_now = lax.clock
    lax_head = lax.registry.sth(lax_now)
    lax_key = lax.operator.public_bytes
    add(
        "Operator lets a copy complete a sale: the whole log is replayed",
        "audit replay",
        detected(audit_log(lax.registry.entries(), lax.policy, lax_head, lax_key)),
    )
    lax_history = lax.registry.history(original.chip_id, lax_now)
    lax_current = lax.registry.query(original.chip_id, lax_now)
    add(
        "The same: the chip's history is replayed",
        "history replay",
        _verdict(verify_history(lax_key, lax.policy, lax_history, lax_current)),
    )

    # 3. change a record in the map without a log entry
    records = {e.record_after.chip_id: e.record_after for e in entries}
    doctored = replace(records[chip.chip_id], owner=w.mallory.public_bytes)
    doctored_map = SparseMerkleMap()
    for record in records.values():
        doctored_map.set(record.chip_id, (doctored if record.chip_id == chip.chip_id else record).encode())
    honest_log, _ = _forge(entries, now, op)
    size, log_root, map_root = len(entries), honest_log.root(), doctored_map.root()
    doctored_head = SignedTreeHead(size, log_root, map_root, now, op.sign(sth_message(size, log_root, map_root, now)))
    response = QueryResponse(chip.chip_id, doctored, doctored_map.prove(chip.chip_id), doctored_head)
    add("Change a record in the map but not in the log: a single query", "map proof", _verdict(verify_query(op.public_bytes, response, now=now, max_age=60)))
    history = HistoryResponse(chip.chip_id, w.registry.history(chip.chip_id, now).items, doctored_head)
    add("The same change: the chip's history is replayed", "history replay", _verdict(verify_history(op.public_bytes, policy, history, response)))
    add("The same change: the whole log is replayed", "audit replay", detected(audit_log(entries, policy, doctored_head, op.public_bytes)))

    # reordering, and backdating
    first, second = entries[2], entries[3]
    swapped = entries[:2] + (
        replace(second, index=2, timestamp=first.timestamp),
        replace(first, index=3, timestamp=second.timestamp),
    )
    _, swapped_head = _forge(swapped, now, op)
    add("Swap two unrelated entries and re-sign the head", "audit replay", detected(audit_log(swapped, policy, swapped_head, op.public_bytes)))
    add(
        "The same, to an auditor that kept the earlier head",
        "audit replay with earlier heads",
        detected(audit_log(swapped, policy, swapped_head, op.public_bytes, prior_heads=(early_head,))),
    )
    backdated = entries[:3] + (replace(entries[3], timestamp=early_head.timestamp - 3),)
    _, backdated_head = _forge(backdated, now, op)
    add(
        "Date an entry before a head that was published without it",
        "audit replay with earlier heads",
        detected(audit_log(backdated, policy, backdated_head, op.public_bytes, prior_heads=(early_head,))),
    )
    inside = entries[:3] + (replace(entries[3], timestamp=early_head.timestamp + 1),)
    _, inside_head = _forge(inside, now, op)
    add(
        "Date an entry earlier, but after the head published without it",
        "audit replay with earlier heads",
        detected(audit_log(inside, policy, inside_head, op.public_bytes, prior_heads=(early_head,))),
    )

    # 4-5. freshness
    old = w.registry.query(chip.chip_id, now - 10_000)
    add("Serve a head from before a recent transfer", "head age limit", _verdict(verify_query(op.public_bytes, old, now=now, max_age=3600)))
    future = w.registry.query(chip.chip_id, now + 10_000_000)
    add("Serve a head dated far in the future so it stays fresh", "clock skew limit", _verdict(verify_query(op.public_bytes, future, now=now, max_age=3600)))

    # 6. forks
    operator_key, head_a, head_b, fork_time = forked_heads()
    for n, t, colluding in ((3, 2, 0), (3, 2, 1), (5, 4, 1), (5, 4, 3)):
        forked = fork_by_enumeration(n, t, colluding, operator_key, head_a, head_b, fork_time)
        add(
            f"Show two clients different histories: {n} witnesses, {t} must cosign, {colluding} colluding",
            "witness quorum",
            "both heads accepted" if forked else "no way found to get both accepted",
        )

    # 7. truncation
    _, short_head = _forge(entries[:3], now, op)
    witness = Witness(party_key("witness-1"), op.public_bytes)
    witness.cosign(head, [], now)
    try:
        witness.cosign(short_head, [], now)
        outcome = "accepted"
    except WitnessRefused as exc:
        outcome = f"refused: {exc.code}"
    add("Drop the newest entries and sign a shorter head", "witness", outcome)
    add("The same, to an auditor holding the shorter log and the longer head", "audit replay", detected(audit_log(entries[:3], policy, head, op.public_bytes)))

    # 8. censorship
    add("Never log a request that was submitted", "none", "accepted: audit passes" if audit_log(entries, policy, head, op.public_bytes).ok else "detected")

    # 9-10. dishonest query answers
    honest = w.registry.query(chip.chip_id, now)
    add("Claim that a registered chip is not registered", "non-membership proof", _verdict(verify_query(op.public_bytes, QueryResponse(chip.chip_id, None, honest.proof, honest.sth), now=now, max_age=60)))
    substituted = QueryResponse(chip.chip_id, w.record(other), honest.proof, honest.sth)
    add("Answer a query with a different chip's record", "record and chip id check", _verdict(verify_query(op.public_bytes, substituted, now=now, max_age=60)))
    return results
