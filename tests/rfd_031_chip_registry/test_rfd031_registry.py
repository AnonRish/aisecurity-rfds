"""The operator's registry and what clients and auditors can check about it."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aisecurity_rfds.common.hashing import sha256
from aisecurity_rfds.common.merkle import MerkleLog
from aisecurity_rfds.rfd_031_chip_registry import identity as I
from aisecurity_rfds.rfd_031_chip_registry import registry as R
from aisecurity_rfds.rfd_031_chip_registry.scenario import World
from aisecurity_rfds.rfd_031_chip_registry.smt import SparseMerkleMap


def code(action) -> str:
    with pytest.raises(R.Rejected) as info:
        action()
    return info.value.code


def lifecycle(w: World):
    """enrol, sell to alice, relocate, sell to bob: returns (chip, number of entries)."""
    chip = w.enroll("life")
    w.sell(chip, w.fab, w.alice)
    w.relocate(chip, w.alice, "GB", location=sha256(b"site"))
    w.sell(chip, w.alice, w.bob, jurisdiction="FR")
    return chip


def forged(entries, operator, now):
    """A log, a map and a head over ``entries`` exactly as given: an operator can sign anything."""
    log, smap = MerkleLog(), SparseMerkleMap()
    for e in entries:
        log.append(e.encode())
        smap.set(e.record_after.chip_id, e.record_after.encode())
    size = len(entries)
    head = R.SignedTreeHead(size, log.root(), smap.root(), now, operator.sign(R.sth_message(size, log.root(), smap.root(), now)))
    return log, smap, head


def forged_answers(entries, operator, now, chip_id):
    log, smap, head = forged(entries, operator, now)
    items = tuple(R.HistoryItem(e, tuple(log.inclusion_proof(i, len(entries)))) for i, e in enumerate(entries) if e.signed.request.chip_id == chip_id)
    latest = [e.record_after for e in entries if e.record_after.chip_id == chip_id]
    query = R.QueryResponse(chip_id, latest[-1] if latest else None, smap.prove(chip_id), head)
    return R.HistoryResponse(chip_id, items, head), query


# ---- submitting ----------------------------------------------------------------------------------------------------


def test_accepted_requests_are_logged_in_order_with_the_record_they_produced():
    w = World.new()
    chip = lifecycle(w)
    entries = w.registry.entries()
    assert len(w.registry) == len(entries) == 6
    assert [e.index for e in entries] == list(range(6))
    assert [e.timestamp for e in entries] == sorted(e.timestamp for e in entries)
    assert entries[-1].record_after == w.registry.record(chip.chip_id)
    assert [e.record_after.version for e in entries] == [1, 2, 3, 4, 5, 6]
    assert isinstance(entries, tuple)


def test_a_rejected_request_changes_nothing():
    w = World.new()
    chip = w.enroll("a")
    before = (len(w.registry), w.registry.sth(5).log_root, w.registry.sth(5).map_root, w.registry.record(chip.chip_id))
    assert code(lambda: w.relocate(chip, w.mallory, "US")) == "unauthorised"
    assert code(lambda: w.registry.submit(R.sign_request(R.make_request(R.RELOCATE, chip.chip_id, 1, jurisdiction="US"), w.fab), 1)) == "clock"
    after = (len(w.registry), w.registry.sth(5).log_root, w.registry.sth(5).map_root, w.registry.record(chip.chip_id))
    assert before == after


def test_time_may_stand_still_but_not_go_backwards():
    w = World.new()
    chip = w.enroll("a")
    signed = R.sign_request(R.make_request(R.RELOCATE, chip.chip_id, 1, jurisdiction="US"), w.fab)
    assert code(lambda: w.registry.submit(signed, w.clock - 1)) == "clock"
    entry = w.registry.submit(signed, w.clock)
    assert entry.timestamp == w.clock


def test_the_signed_head_commits_to_size_both_roots_and_time():
    w = World.new()
    empty = w.registry.sth(7)
    assert empty.size == 0 and empty.timestamp == 7 and empty.verify(w.operator.public_bytes)
    assert empty.log_root == MerkleLog().root() and empty.map_root == SparseMerkleMap().root()
    lifecycle(w)
    head = w.registry.sth(99)
    assert head.size == 6 and head.verify(w.operator.public_bytes) and not head.verify(w.alice.public_bytes)
    for edited in (
        replace(head, size=5),
        replace(head, log_root=sha256(b"x")),
        replace(head, map_root=sha256(b"x")),
        replace(head, timestamp=100),
    ):
        assert not edited.verify(w.operator.public_bytes)
    assert head.log_root != empty.log_root and head.map_root != empty.map_root


def test_the_head_roots_are_those_of_the_log_and_the_map():
    w = World.new()
    lifecycle(w)
    log, smap, head = forged(w.registry.entries(), w.operator, 5)
    real = w.registry.sth(5)
    assert (real.log_root, real.map_root) == (head.log_root, head.map_root)
    assert real == head  # Ed25519 signatures are deterministic


def test_the_operators_public_key_is_exposed():
    w = World.new()
    assert w.registry.operator_public_key == w.operator.public_bytes
    assert w.registry.record(b"\x00" * 32) is None


# ---- queries -------------------------------------------------------------------------------------------------------


def test_a_query_for_a_registered_chip_verifies_and_returns_its_record():
    w = World.new()
    chip = lifecycle(w)
    answer = w.registry.query(chip.chip_id, w.clock)
    verdict = R.verify_query(w.operator.public_bytes, answer, now=w.clock, max_age=60)
    assert verdict == R.Verdict(True, "ok", w.record(chip))
    assert answer.record.owner == w.bob.public_bytes and answer.record.jurisdiction == "FR"


def test_a_query_for_an_unregistered_chip_verifies_as_absent():
    w = World.new()
    lifecycle(w)
    answer = w.registry.query(b"\x09" * 32, w.clock)
    assert answer.record is None
    assert R.verify_query(w.operator.public_bytes, answer, now=w.clock, max_age=60) == R.Verdict(True, "ok", None)


def test_a_query_on_an_empty_registry_verifies():
    w = World.new()
    answer = w.registry.query(b"\x09" * 32, w.clock)
    assert R.verify_query(w.operator.public_bytes, answer, now=w.clock, max_age=60).ok


def test_a_query_is_refused_when_signed_by_someone_else():
    w = World.new()
    chip = lifecycle(w)
    answer = w.registry.query(chip.chip_id, w.clock)
    assert R.verify_query(w.mallory.public_bytes, answer, now=w.clock, max_age=60).reason == "head-signature"


def test_a_query_is_refused_when_the_record_or_proof_is_changed():
    w = World.new()
    chip = lifecycle(w)
    answer = w.registry.query(chip.chip_id, w.clock)
    forged_record = replace(answer, record=replace(answer.record, owner=w.mallory.public_bytes))
    assert R.verify_query(w.operator.public_bytes, forged_record, now=w.clock, max_age=60).reason == "bad-proof"
    assert R.verify_query(w.operator.public_bytes, replace(answer, record=None), now=w.clock, max_age=60).reason == "bad-proof"
    other = w.enroll("other")
    swapped = replace(w.registry.query(chip.chip_id, w.clock), record=w.record(other))
    assert R.verify_query(w.operator.public_bytes, swapped, now=w.clock, max_age=60).reason == "wrong-record"
    stale_proof = w.registry.query(chip.chip_id, w.clock)
    other_proof = w.registry.query(other.chip_id, w.clock).proof
    assert R.verify_query(w.operator.public_bytes, replace(stale_proof, proof=other_proof), now=w.clock, max_age=60).reason == "bad-proof"


def test_a_query_head_must_be_neither_too_old_nor_from_the_future():
    w = World.new()
    chip = lifecycle(w)
    answer = w.registry.query(chip.chip_id, 10_000)
    ok = lambda now, **kw: R.verify_query(w.operator.public_bytes, answer, now=now, max_age=100, **kw)  # noqa: E731
    assert ok(10_000).ok and ok(10_100).ok
    assert ok(10_101).reason == "stale-head"
    assert ok(9_940).ok  # up to 60 s ahead of the client's clock is tolerated
    assert ok(9_939).reason == "future-head"
    assert ok(9_900, max_skew=100).ok and ok(9_899, max_skew=100).reason == "future-head"


# ---- histories -----------------------------------------------------------------------------------------------------


def test_a_chips_history_replays_to_the_record_the_map_proves():
    w = World.new()
    chip = lifecycle(w)
    other = w.enroll("other")
    history = w.registry.history(chip.chip_id, w.clock)
    current = w.registry.query(chip.chip_id, w.clock)
    assert [i.entry.record_after.version for i in history.items] == [1, 2, 3, 4, 5, 6]
    assert all(i.entry.signed.request.chip_id == chip.chip_id for i in history.items)
    assert R.verify_history(w.operator.public_bytes, w.policy, history, current) == R.Verdict(True, "ok", w.record(chip))
    solo = w.registry.history(other.chip_id, w.clock)
    assert len(solo.items) == 1 and R.verify_history(w.operator.public_bytes, w.policy, solo, w.registry.query(other.chip_id, w.clock)).ok


def test_the_history_of_an_unregistered_chip_is_empty_and_agrees_with_absence():
    w = World.new()
    lifecycle(w)
    history = w.registry.history(b"\x08" * 32, w.clock)
    current = w.registry.query(b"\x08" * 32, w.clock)
    assert history.items == () and R.verify_history(w.operator.public_bytes, w.policy, history, current) == R.Verdict(True, "ok", None)


def test_a_history_with_an_entry_removed_or_reordered_does_not_verify():
    w = World.new()
    chip = lifecycle(w)
    history = w.registry.history(chip.chip_id, w.clock)
    current = w.registry.query(chip.chip_id, w.clock)
    items = list(history.items)
    for mutated in (items[1:], items[:-1], items[:2] + items[3:], items[::-1], items[:3] + [items[3], items[3]] + items[4:]):
        verdict = R.verify_history(w.operator.public_bytes, w.policy, replace(history, items=tuple(mutated)), current)
        assert not verdict.ok, [i.entry.index for i in mutated]


def test_dropping_the_newest_entries_is_caught_by_the_map():
    w = World.new()
    chip = lifecycle(w)
    history = w.registry.history(chip.chip_id, w.clock)
    current = w.registry.query(chip.chip_id, w.clock)
    shorter = replace(history, items=history.items[:-2])
    assert R.verify_history(w.operator.public_bytes, w.policy, shorter, current).reason == "history-disagrees-with-map"


def test_a_history_entry_must_be_in_the_log_the_head_commits_to():
    w = World.new()
    chip = lifecycle(w)
    history = w.registry.history(chip.chip_id, w.clock)
    current = w.registry.query(chip.chip_id, w.clock)
    first = history.items[0]
    bad_proof = replace(history, items=(R.HistoryItem(first.entry, first.proof[:-1]), *history.items[1:]))
    assert R.verify_history(w.operator.public_bytes, w.policy, bad_proof, current).reason == "inclusion"
    edited = replace(first.entry, timestamp=first.entry.timestamp + 1)
    assert R.verify_history(w.operator.public_bytes, w.policy, replace(history, items=(R.HistoryItem(edited, first.proof), *history.items[1:])), current).reason == "inclusion"
    beyond = replace(first.entry, index=history.sth.size + 5)
    assert R.verify_history(w.operator.public_bytes, w.policy, replace(history, items=(R.HistoryItem(beyond, ()), *history.items[1:])), current).reason == "inclusion"


def test_a_history_and_a_query_must_come_from_the_same_head():
    w = World.new()
    chip = lifecycle(w)
    history = w.registry.history(chip.chip_id, w.clock)
    w.enroll("another")
    current = w.registry.query(chip.chip_id, w.clock)
    assert R.verify_history(w.operator.public_bytes, w.policy, history, current).reason == "heads-differ"
    assert R.verify_history(w.mallory.public_bytes, w.policy, w.registry.history(chip.chip_id, w.clock), current).reason == "head-signature"


def test_a_history_entry_for_another_chip_is_refused():
    w = World.new()
    chip = lifecycle(w)
    other = w.enroll("other")
    mixed = w.registry.history(chip.chip_id, w.clock)
    foreign = w.registry.history(other.chip_id, w.clock).items[0]
    verdict = R.verify_history(w.operator.public_bytes, w.policy, replace(mixed, items=(foreign, *mixed.items[1:])), w.registry.query(chip.chip_id, w.clock))
    assert verdict.reason == "wrong-chip"


def test_a_history_that_is_in_the_log_but_not_a_valid_chain_is_refused():
    w = World.new()
    chip = lifecycle(w)
    entries = list(w.registry.entries())
    # an entry that claims a state its request does not produce, committed to by a consistent head
    altered = replace(entries[2], record_after=replace(entries[2].record_after, owner=w.mallory.public_bytes))
    history, query = forged_answers(entries[:2] + [altered] + entries[3:], w.operator, w.clock, chip.chip_id)
    assert R.verify_history(w.operator.public_bytes, w.policy, history, query).reason == "state-mismatch"
    # an entry whose request nobody authorised
    bogus_request = R.sign_request(R.make_request(R.RELOCATE, chip.chip_id, 6, jurisdiction="US"), w.mallory)
    bogus = R.LogEntry(len(entries), w.clock, bogus_request, replace(w.record(chip), jurisdiction="US", version=7))
    history, query = forged_answers(entries + [bogus], w.operator, w.clock, chip.chip_id)
    assert R.verify_history(w.operator.public_bytes, w.policy, history, query).reason == "invalid-entry:unauthorised"


def test_a_doctored_map_is_caught_by_the_history_but_not_by_a_single_query():
    w = World.new()
    chip = lifecycle(w)
    entries = w.registry.entries()
    records = {e.record_after.chip_id: e.record_after for e in entries}
    doctored = replace(records[chip.chip_id], owner=w.mallory.public_bytes)
    smap = SparseMerkleMap()
    smap.set(chip.chip_id, doctored.encode())
    _, _, honest = forged(entries, w.operator, w.clock)
    head = R.SignedTreeHead(honest.size, honest.log_root, smap.root(), w.clock, w.operator.sign(R.sth_message(honest.size, honest.log_root, smap.root(), w.clock)))
    query = R.QueryResponse(chip.chip_id, doctored, smap.prove(chip.chip_id), head)
    history = R.HistoryResponse(chip.chip_id, w.registry.history(chip.chip_id, w.clock).items, head)
    assert R.verify_query(w.operator.public_bytes, query, now=w.clock, max_age=60).ok
    assert R.verify_history(w.operator.public_bytes, w.policy, history, query).reason == "history-disagrees-with-map"


# ---- audits --------------------------------------------------------------------------------------------------------


def test_an_honest_registry_passes_an_audit_at_every_stage():
    w = World.new()
    heads = [w.registry.sth(w.clock)]
    chip = w.enroll("a")
    heads.append(w.registry.sth(w.clock))
    w.sell(chip, w.fab, w.alice)
    heads.append(w.registry.sth(w.clock))
    w.relocate(chip, w.alice, "GB")
    report = R.audit_log(w.registry.entries(), w.policy, w.registry.sth(w.clock), w.operator.public_bytes, prior_heads=heads)
    assert report == R.AuditReport(True, 4, ())
    assert R.audit_log((), w.policy, R.Registry(w.operator, w.policy).sth(1), w.operator.public_bytes).ok


def test_an_audit_of_an_earlier_head_ignores_later_entries():
    w = World.new()
    chip = w.enroll("a")
    early = w.registry.sth(w.clock)
    w.sell(chip, w.fab, w.alice)
    report = R.audit_log(w.registry.entries(), w.policy, early, w.operator.public_bytes)
    assert report.ok and report.entries == 1


def test_an_audit_reports_a_head_signed_by_someone_else():
    w = World.new()
    lifecycle(w)
    report = R.audit_log(w.registry.entries(), w.policy, w.registry.sth(w.clock), w.mallory.public_bytes)
    assert report.problems == ("head signature is invalid",)


def test_an_audit_reports_a_log_shorter_than_its_head():
    w = World.new()
    lifecycle(w)
    entries = w.registry.entries()
    report = R.audit_log(entries[:5], w.policy, w.registry.sth(w.clock), w.operator.public_bytes)
    assert report.problems == ("log has 5 entries but the head covers 6",)
    assert report.entries == 5


def test_an_audit_reports_entries_that_are_edited_forged_or_misnumbered():
    w = World.new()
    chip = lifecycle(w)
    entries = list(w.registry.entries())
    op = w.operator.public_bytes

    def audit(mutated):
        _, _, head = forged(mutated, w.operator, w.clock)
        return list(R.audit_log(mutated, w.policy, head, op).problems)

    altered = replace(entries[2], record_after=replace(entries[2].record_after, jurisdiction="GB"))
    assert audit(entries[:2] + [altered] + entries[3:])[0] == "entry 2 records a state its request does not produce"
    bogus_request = R.sign_request(R.make_request(R.RELOCATE, chip.chip_id, 6, jurisdiction="US"), w.mallory)
    bogus = R.LogEntry(6, w.clock, bogus_request, replace(w.record(chip), jurisdiction="US", version=7))
    assert audit(entries + [bogus])[0] == "entry 6 is not a valid request (unauthorised)"
    renumbered = replace(entries[3], index=9)
    assert audit(entries[:3] + [renumbered] + entries[4:]) == ["entry 3 claims index 9"]
    early = replace(entries[4], timestamp=entries[3].timestamp - 1)
    assert audit(entries[:4] + [early] + entries[5:])[0] == "entry 4 is dated before the entry preceding it"


def copied_chip_completes_a_sale(w: World):
    """With an operator that skips its counter check: the key leaks, the genuine chip is sold, then the copy sells."""
    chip = w.enroll("a")
    copy = chip.clone()
    w.sell(chip, w.fab, w.alice)  # the genuine chip attests: its counter is now 1
    w.offer(chip, w.alice, w.bob)
    w.submit(w.completion(chip, w.bob, attested=copy))  # the copy's first attestation is 1 as well
    return chip


def test_an_honest_operator_refuses_the_sale_that_an_operator_without_the_counter_check_logs():
    honest = World.new()
    chip = honest.enroll("a")
    copy = chip.clone()
    honest.sell(chip, honest.fab, honest.alice)
    honest.offer(chip, honest.alice, honest.bob)
    before = len(honest.registry)
    assert code(lambda: honest.submit(honest.completion(chip, honest.bob, attested=copy))) == "counter-regression"
    assert len(honest.registry) == before
    lax = World.new(skip_counter_check=True)
    copied_chip_completes_a_sale(lax)
    assert len(lax.registry) == before + 1
    assert lax.record(lax.chips["a"]).owner == lax.bob.public_bytes


def test_an_audit_reports_a_transfer_whose_attestation_counter_did_not_rise():
    w = World.new(skip_counter_check=True)
    copied_chip_completes_a_sale(w)
    report = R.audit_log(w.registry.entries(), w.policy, w.registry.sth(w.clock), w.operator.public_bytes)
    assert report.problems == ("entry 4 completes a transfer with a counter that did not rise",)
    assert report.entries == 5


def test_a_history_whose_transfer_counter_did_not_rise_is_refused():
    w = World.new(skip_counter_check=True)
    chip = copied_chip_completes_a_sale(w)
    verdict = R.verify_history(
        w.operator.public_bytes, w.policy, w.registry.history(chip.chip_id, w.clock), w.registry.query(chip.chip_id, w.clock)
    )
    assert (verdict.ok, verdict.reason) == (False, "counter-regression")


def test_a_copy_that_started_ahead_passes_until_the_genuine_chip_completes_a_sale():
    w = World.new(skip_counter_check=True)
    chip = w.enroll("a")
    w.sell(chip, w.fab, w.alice)  # counter 1
    copy = chip.clone(advance=5)
    w.offer(chip, w.alice, w.bob)
    w.submit(w.completion(chip, w.bob, attested=copy))  # counter 7: above the genuine chip's
    op = w.operator.public_bytes
    head = w.registry.sth(w.clock)
    assert R.audit_log(w.registry.entries(), w.policy, head, op).ok
    w.offer(chip, w.bob, w.alice)
    w.submit(w.completion(chip, w.alice))  # the genuine chip's next attestation is 2: below the copy's
    report = R.audit_log(w.registry.entries(), w.policy, w.registry.sth(w.clock), op)
    assert report.problems == ("entry 6 completes a transfer with a counter that did not rise",)


def test_attestation_counters_of_different_chips_are_judged_apart():
    w = World.new()
    first, second = w.enroll("a"), w.enroll("b")
    w.sell(first, w.fab, w.alice)
    w.sell(second, w.fab, w.bob)  # both chips attest for the first time: counter 1 each
    assert first.counter == second.counter == 1
    assert R.audit_log(w.registry.entries(), w.policy, w.registry.sth(w.clock), w.operator.public_bytes).ok
    for chip in (first, second):
        history, query = w.registry.history(chip.chip_id, w.clock), w.registry.query(chip.chip_id, w.clock)
        assert R.verify_history(w.operator.public_bytes, w.policy, history, query).ok


def test_an_audit_reports_roots_that_do_not_follow_from_the_entries():
    w = World.new()
    lifecycle(w)
    entries = w.registry.entries()
    honest = w.registry.sth(w.clock)
    wrong_log = R.SignedTreeHead(honest.size, sha256(b"x"), honest.map_root, w.clock, w.operator.sign(R.sth_message(honest.size, sha256(b"x"), honest.map_root, w.clock)))
    wrong_map = R.SignedTreeHead(honest.size, honest.log_root, sha256(b"y"), w.clock, w.operator.sign(R.sth_message(honest.size, honest.log_root, sha256(b"y"), w.clock)))
    assert R.audit_log(entries, w.policy, wrong_log, w.operator.public_bytes).problems == ("log root differs from the signed head",)
    assert R.audit_log(entries, w.policy, wrong_map, w.operator.public_bytes).problems == ("map root differs from the signed head",)


def test_earlier_heads_expose_a_history_that_was_rewritten_into_another_valid_one():
    w = World.new()
    chip = w.enroll("a")
    w.sell(chip, w.fab, w.alice)
    early = w.registry.sth(w.clock + 5)
    other = w.enroll("b")
    entries = w.registry.entries()
    first, second = entries[2], entries[3]
    swapped = list(entries[:2]) + [replace(second, index=2, timestamp=first.timestamp), replace(first, index=3, timestamp=second.timestamp)]
    _, _, head = forged(swapped, w.operator, w.clock)
    op = w.operator.public_bytes
    assert R.audit_log(swapped, w.policy, head, op).ok  # internally consistent
    problems = R.audit_log(swapped, w.policy, head, op, prior_heads=(early,)).problems
    assert problems == (
        "log root at size 3 differs from an earlier head",
        "map root at size 3 differs from an earlier head",
    )
    assert other.chip_id != chip.chip_id


def test_earlier_heads_catch_a_changed_map_at_an_earlier_size():
    w = World.new()
    chip = w.enroll("a")
    w.sell(chip, w.fab, w.alice)
    honest = w.registry.sth(w.clock)
    liar = R.SignedTreeHead(honest.size, honest.log_root, sha256(b"other map"), honest.timestamp, w.operator.sign(R.sth_message(honest.size, honest.log_root, sha256(b"other map"), honest.timestamp)))
    w.relocate(chip, w.alice, "GB")
    problems = R.audit_log(w.registry.entries(), w.policy, w.registry.sth(w.clock), w.operator.public_bytes, prior_heads=(liar,)).problems
    assert problems == ("map root at size 3 differs from an earlier head",)


def test_earlier_heads_bound_how_far_an_entry_can_be_backdated():
    w = World.new()
    chip = w.enroll("a")
    w.sell(chip, w.fab, w.alice)
    head_time = w.clock + 5
    early = w.registry.sth(head_time)
    w.relocate(chip, w.alice, "GB")
    entries = list(w.registry.entries())
    op = w.operator.public_bytes

    def audit(timestamp):
        mutated = entries[:-1] + [replace(entries[-1], timestamp=timestamp)]
        _, _, head = forged(mutated, w.operator, head_time + 100)
        return list(R.audit_log(mutated, w.policy, head, op, prior_heads=(early,)).problems)

    assert audit(head_time) == []
    assert audit(head_time + 50) == []
    assert audit(head_time - 1) == ["entry 3 is dated before a head that was published without it"]
    assert audit(head_time - 4) == ["entry 3 is dated before a head that was published without it"]


def test_an_entry_dated_after_the_head_that_covers_it_is_reported():
    w = World.new()
    chip = w.enroll("a")
    w.relocate(chip, w.fab, "GB")
    entries = list(w.registry.entries())
    stamp = entries[-1].timestamp
    late = entries[:-1] + [replace(entries[-1], timestamp=stamp + 1)]
    op = w.operator.public_bytes
    _, _, too_early = forged(late, w.operator, stamp)  # a head that is older than an entry it covers
    _, _, head = forged(late, w.operator, stamp + 10)
    problems = R.audit_log(late, w.policy, head, op, prior_heads=(too_early,)).problems
    assert problems == ("entry 1 is dated after a head that covers it",)
    _, _, on_time = forged(late, w.operator, stamp + 1)
    assert R.audit_log(late, w.policy, head, op, prior_heads=(on_time,)).ok


def test_an_earlier_head_that_is_forged_or_longer_than_the_log_is_reported():
    w = World.new()
    chip = w.enroll("a")
    w.relocate(chip, w.fab, "GB")
    entries = w.registry.entries()
    head = w.registry.sth(w.clock)
    unsigned = replace(head, size=1, signature=bytes(64))
    problems = R.audit_log(entries, w.policy, head, w.operator.public_bytes, prior_heads=(unsigned,)).problems
    assert "an earlier head (size 1) has an invalid signature" in problems
    stamp = w.clock + 100
    longer = R.SignedTreeHead(5, head.log_root, head.map_root, stamp, w.operator.sign(R.sth_message(5, head.log_root, head.map_root, stamp)))
    problems = R.audit_log(entries, w.policy, head, w.operator.public_bytes, prior_heads=(longer,)).problems
    assert problems == ("an earlier head covers 5 entries but the log has 2",)


# ---- replication ---------------------------------------------------------------------------------------------------


def test_a_replica_built_from_the_log_has_the_same_roots_and_state():
    w = World.new()
    chip = lifecycle(w)
    copy = R.Registry.replicate(w.operator, w.policy, w.registry.entries())
    assert copy.sth(5) == w.registry.sth(5)
    assert copy.record(chip.chip_id) == w.record(chip)
    assert copy.tracker.last(chip.chip_id) == w.registry.tracker.last(chip.chip_id) > 0


def test_replication_revalidates_every_request():
    w = World.new()
    chip = lifecycle(w)
    entries = list(w.registry.entries())
    bogus_request = R.sign_request(R.make_request(R.RELOCATE, chip.chip_id, 6, jurisdiction="US"), w.mallory)
    bogus = R.LogEntry(6, w.clock, bogus_request, replace(w.record(chip), jurisdiction="US", version=7))
    assert code(lambda: R.Registry.replicate(w.operator, w.policy, entries + [bogus])) == "unauthorised"


def test_replicas_that_apply_different_requests_diverge():
    w = World.new()
    chip = lifecycle(w)
    a = R.Registry.replicate(w.operator, w.policy, w.registry.entries())
    b = R.Registry.replicate(w.operator, w.policy, w.registry.entries())
    a.submit(R.sign_request(R.make_request(R.RELOCATE, chip.chip_id, 6, jurisdiction="US"), w.bob), w.clock + 1)
    b.submit(R.sign_request(R.make_request(R.RELOCATE, chip.chip_id, 6, jurisdiction="GB"), w.bob), w.clock + 1)
    ha, hb = a.sth(w.clock + 2), b.sth(w.clock + 2)
    assert ha.size == hb.size == 7 and ha.log_root != hb.log_root and ha.map_root != hb.map_root


def test_consistency_proofs_link_an_earlier_head_to_a_later_one():
    from aisecurity_rfds.common.merkle import verify_consistency

    w = World.new()
    w.enroll("a")
    early = w.registry.sth(w.clock)
    lifecycle(w)
    late = w.registry.sth(w.clock)
    proof = w.registry.consistency_proof(early.size, late.size)
    assert verify_consistency(early.size, late.size, early.log_root, late.log_root, proof)
    assert not verify_consistency(early.size, late.size, early.log_root, sha256(b"x"), proof)


# ---- monitoring ----------------------------------------------------------------------------------------------------


def test_a_fresh_valid_attestation_raises_no_alert():
    w = World.new()
    chip = w.enroll("a")
    assert w.registry.observe_attestation(chip.attest(b"c"), b"c", w.tick()) == []
    assert w.registry.alerts == []
    assert w.registry.tracker.last(chip.chip_id) == 1


def test_an_attestation_by_an_unregistered_chip_is_an_alert():
    w = World.new()
    stranger = I.DeviceKeyChip(b"stranger")
    (alert,) = w.registry.observe_attestation(stranger.attest(b"c"), b"c", 55)
    assert (alert.kind, alert.chip_id, alert.time) == ("unknown-chip", stranger.chip_id, 55)
    assert w.registry.alerts == [alert]


def test_an_attestation_that_does_not_verify_is_an_alert_and_does_not_move_the_counter():
    w = World.new()
    chip = w.enroll("a")
    (alert,) = w.registry.observe_attestation(chip.attest(b"c"), b"other challenge", 5)
    assert alert.kind == "invalid-attestation" and alert.chip_id == chip.chip_id
    assert w.registry.tracker.last(chip.chip_id) == 0
    impostor = I.DeviceKeyChip(b"impostor")
    forged_att = I.Attestation(chip.chip_id, b"c", 1, impostor.attest(b"c").signature)
    assert [a.kind for a in w.registry.observe_attestation(forged_att, b"c", 6)] == ["invalid-attestation"]


def test_a_repeated_or_lower_counter_is_a_clone_alert_with_the_evidence():
    w = World.new()
    chip = w.enroll("a")
    copy = chip.clone()
    assert w.registry.observe_attestation(chip.attest(b"1"), b"1", 1) == []
    (alert,) = w.registry.observe_attestation(copy.attest(b"2"), b"2", 2)
    assert alert == R.Alert("clone-suspected", chip.chip_id, 2, "counter 1 <= 1")
    assert w.registry.alerts == [alert]


def test_a_retired_chip_that_attests_is_a_zombie_and_the_reason_is_reported():
    w = World.new()
    chip = w.enroll("a")
    w.retire(chip, w.fab, "lost")
    (alert,) = w.registry.observe_attestation(chip.attest(b"c"), b"c", w.tick())
    assert (alert.kind, alert.detail) == ("zombie", "retired as lost")


def test_a_retired_chip_whose_counter_also_regressed_raises_both_alerts():
    w = World.new()
    chip = w.enroll("a")
    w.registry.observe_attestation(chip.attest(b"1"), b"1", w.tick())
    w.registry.observe_attestation(chip.attest(b"2"), b"2", w.tick())
    w.retire(chip, w.fab, "recycled")
    copy = I.DeviceKeyChip(b"a")  # the same key, with a counter that starts from zero
    alerts = w.registry.observe_attestation(copy.attest(b"3"), b"3", w.tick())
    assert [a.kind for a in alerts] == ["zombie", "clone-suspected"]


def test_the_transfer_counter_check_rejects_a_chip_that_went_backwards_and_logs_nothing():
    w = World.new()
    chip = w.enroll("a")
    w.sell(chip, w.fab, w.alice)
    copy = chip.clone()
    w.registry.observe_attestation(copy.attest(b"inspection"), b"inspection", w.tick())
    entries_before = len(w.registry)
    w.offer(chip, w.alice, w.bob)
    entries_before += 1
    assert code(lambda: w.submit(w.completion(chip, w.bob))) == "counter-regression"
    assert len(w.registry) == entries_before and w.record(chip).status == R.IN_TRANSIT
    (alert,) = w.registry.alerts
    assert alert.kind == "clone-suspected" and "transfer attestation counter" in alert.detail and alert.chip_id == chip.chip_id


def test_a_completed_transfer_moves_the_trackers_mark():
    w = World.new()
    chip = w.enroll("a")
    w.sell(chip, w.fab, w.alice)
    assert w.registry.tracker.last(chip.chip_id) == 1
    w.sell(chip, w.alice, w.bob)
    assert w.registry.tracker.last(chip.chip_id) == 2


# ---- a whole life --------------------------------------------------------------------------------------------------


def test_every_stage_of_a_chips_life_verifies_for_a_client_and_an_auditor():
    w = World.new()
    chip = w.enroll("life")
    stages = [
        lambda: w.sell(chip, w.fab, w.alice),
        lambda: w.relocate(chip, w.alice, "GB", location=sha256(b"site")),
        lambda: w.offer(chip, w.alice, w.bob),
        lambda: w.submit(R.sign_request(R.make_request(R.CANCEL_TRANSFER, chip.chip_id, w.record(chip).version), w.bob)),
        lambda: w.sell(chip, w.alice, w.bob, jurisdiction="FR"),
        lambda: w.retire(chip, w.bob, "destroyed", witnessed=True),
    ]
    heads = []
    for stage in stages:
        stage()
        history = w.registry.history(chip.chip_id, w.clock)
        current = w.registry.query(chip.chip_id, w.clock)
        assert R.verify_query(w.operator.public_bytes, current, now=w.clock, max_age=60).ok
        assert R.verify_history(w.operator.public_bytes, w.policy, history, current).ok
        versions = [i.entry.record_after.version for i in history.items]
        assert versions == list(range(1, len(versions) + 1))
        heads.append(w.registry.sth(w.clock))
        assert R.audit_log(w.registry.entries(), w.policy, heads[-1], w.operator.public_bytes, prior_heads=heads[:-1]).ok
    assert w.record(chip).status == R.RETIRED and w.record(chip).owner == w.bob.public_bytes
