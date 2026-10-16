"""Witnesses and quorums: what they refuse, who they convince, and how many must collude to fork the registry."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aisecurity_rfds.common.hashing import sha256
from aisecurity_rfds.common.signing import verify
from aisecurity_rfds.rfd_031_chip_registry import identity as I
from aisecurity_rfds.rfd_031_chip_registry import registry as R
from aisecurity_rfds.rfd_031_chip_registry import witness as W
from aisecurity_rfds.rfd_031_chip_registry.attacks import forked_heads
from aisecurity_rfds.rfd_031_chip_registry.scenario import World


def refusal(action) -> str:
    with pytest.raises(W.WitnessRefused) as info:
        action()
    return info.value.code


def world_with_heads():
    w = World.new()
    chip = w.enroll("a")
    h1 = w.registry.sth(w.clock)
    w.sell(chip, w.fab, w.alice)
    h3 = w.registry.sth(w.clock)
    w.relocate(chip, w.alice, "GB")
    h4 = w.registry.sth(w.clock)
    return w, chip, h1, h3, h4


def witness_for(w: World, name: str = "w0", **kw) -> W.Witness:
    return W.Witness(I.party_key(name), w.operator.public_bytes, **kw)


def test_a_witness_cosigns_a_genuine_head_and_remembers_it():
    w, _, h1, *_ = world_with_heads()
    witness = witness_for(w)
    assert witness.head is None
    signature = witness.cosign(h1, [], w.clock)
    assert witness.head == h1
    assert verify(witness.public_key, W.cosign_message(h1), signature)
    assert not verify(witness.public_key, W.cosign_message(replace(h1, size=9)), signature)


def test_the_cosigned_message_binds_the_whole_head():
    w, _, h1, h3, _ = world_with_heads()
    messages = {W.cosign_message(h) for h in (h1, h3, replace(h1, timestamp=h1.timestamp + 1), replace(h1, map_root=sha256(b"x")))}
    assert len(messages) == 4
    assert W.cosign_message(h1) != h1.message()


def test_a_head_the_operator_did_not_sign_is_refused():
    w, _, h1, *_ = world_with_heads()
    other_operator = W.Witness(I.party_key("w0"), I.party_key("someone else").public_bytes)
    assert refusal(lambda: other_operator.cosign(h1, [], w.clock)) == "head-signature"
    assert refusal(lambda: witness_for(w).cosign(replace(h1, size=h1.size + 1), [], w.clock)) == "head-signature"
    assert witness_for(w).head is None


def test_a_head_from_the_future_is_refused_beyond_the_skew():
    w, _, h1, *_ = world_with_heads()
    witness = witness_for(w, max_skew=60)
    assert witness.cosign(h1, [], h1.timestamp - 60)  # exactly at the limit
    assert refusal(lambda: witness_for(w, max_skew=60).cosign(h1, [], h1.timestamp - 61)) == "future-head"
    assert refusal(lambda: witness_for(w, max_skew=5).cosign(h1, [], h1.timestamp - 6)) == "future-head"
    assert witness_for(w, max_skew=5).cosign(h1, [], h1.timestamp - 5)


def test_a_witness_extends_its_head_only_with_a_consistency_proof():
    w, _, h1, h3, h4 = world_with_heads()
    witness = witness_for(w)
    witness.cosign(h1, [], w.clock)
    assert refusal(lambda: witness.cosign(h3, [], w.clock)) == "inconsistent"
    assert witness.head == h1  # a refusal changes nothing
    assert refusal(lambda: witness.cosign(h3, [sha256(b"junk")], w.clock)) == "inconsistent"
    witness.cosign(h3, w.registry.consistency_proof(1, 3), w.clock)
    assert witness.head == h3
    witness.cosign(h4, w.registry.consistency_proof(3, 4), w.clock)
    assert witness.head == h4


def test_a_consistency_proof_for_the_wrong_pair_of_sizes_is_refused():
    w, _, h1, h3, h4 = world_with_heads()
    witness = witness_for(w)
    witness.cosign(h3, [], w.clock)
    assert refusal(lambda: witness.cosign(h4, w.registry.consistency_proof(1, 4), w.clock)) == "inconsistent"
    assert witness.cosign(h4, w.registry.consistency_proof(3, 4), w.clock)


def test_a_witness_that_has_cosigned_nothing_accepts_any_genuine_head():
    w, _, _, h3, _ = world_with_heads()
    assert witness_for(w).cosign(h3, [], w.clock)


def test_a_witness_will_not_go_back_to_a_shorter_log():
    w, _, h1, h3, _ = world_with_heads()
    witness = witness_for(w)
    witness.cosign(h3, [], w.clock)
    stamp = h3.timestamp + 5  # a freshly signed head over the first entry only
    shorter = R.SignedTreeHead(h1.size, h1.log_root, h1.map_root, stamp, w.operator.sign(R.sth_message(h1.size, h1.log_root, h1.map_root, stamp)))
    assert refusal(lambda: witness.cosign(shorter, [], w.clock + 10)) == "shrinking"
    assert refusal(lambda: witness.cosign(h1, [], w.clock)) == "timestamp-regression"  # an old head fails on its age
    assert witness.head == h3


def test_a_witness_will_not_cosign_two_different_heads_of_the_same_size():
    operator, head_a, head_b, now = forked_heads()
    witness = W.Witness(I.party_key("w0"), operator)
    witness.cosign(head_a, [], now)
    assert refusal(lambda: witness.cosign(head_b, [], now)) == "fork"
    assert witness.head == head_a
    other = W.Witness(I.party_key("w1"), operator)
    other.cosign(head_b, [], now)
    assert refusal(lambda: other.cosign(head_a, [], now)) == "fork"


def test_a_witness_notices_a_changed_map_root_under_an_unchanged_log():
    w, _, h1, *_ = world_with_heads()
    witness = witness_for(w)
    witness.cosign(h1, [], w.clock)
    other_map = R.SignedTreeHead(h1.size, h1.log_root, sha256(b"m"), h1.timestamp, w.operator.sign(R.sth_message(h1.size, h1.log_root, sha256(b"m"), h1.timestamp)))
    assert refusal(lambda: witness.cosign(other_map, [], w.clock)) == "fork"


def test_the_same_head_may_be_cosigned_again_but_time_may_not_run_backwards():
    w, _, h1, *_ = world_with_heads()
    witness = witness_for(w)
    first = witness.cosign(h1, [], w.clock)
    assert witness.cosign(h1, [], w.clock) == first
    later = R.SignedTreeHead(h1.size, h1.log_root, h1.map_root, h1.timestamp + 30, w.operator.sign(R.sth_message(h1.size, h1.log_root, h1.map_root, h1.timestamp + 30)))
    witness.cosign(later, [], w.clock + 100)
    assert witness.head == later
    assert refusal(lambda: witness.cosign(h1, [], w.clock + 100)) == "timestamp-regression"


def test_a_compromised_witness_cosigns_anything_and_remembers_nothing_it_should_refuse():
    operator, head_a, head_b, now = forked_heads()
    witness = W.CompromisedWitness(I.party_key("w0"), operator)
    sig_a, sig_b = witness.cosign(head_a, [], now), witness.cosign(head_b, [], now)
    assert verify(witness.public_key, W.cosign_message(head_a), sig_a)
    assert verify(witness.public_key, W.cosign_message(head_b), sig_b)
    stranger = replace(head_a, signature=bytes(64))
    assert witness.cosign(stranger, [], now)


# ---- quorums -------------------------------------------------------------------------------------------------------


def quorum_world(n: int, t: int):
    w, _, h1, *_ = world_with_heads()
    witnesses = [witness_for(w, f"w{i}") for i in range(n)]
    quorum = W.Quorum(frozenset(x.public_key for x in witnesses), t)
    return w, h1, witnesses, quorum


def test_a_head_is_accepted_with_exactly_the_threshold_of_distinct_witnesses():
    w, head, witnesses, quorum = quorum_world(5, 3)
    sigs = [(x.public_key, x.cosign(head, [], w.clock)) for x in witnesses]
    op = w.operator.public_bytes
    assert quorum.accepts(W.CosignedHead(head, tuple(sigs[:3])), op)
    assert quorum.accepts(W.CosignedHead(head, tuple(sigs)), op)
    assert not quorum.accepts(W.CosignedHead(head, tuple(sigs[:2])), op)
    assert not quorum.accepts(W.CosignedHead(head, ()), op)


def test_the_same_witness_counts_once():
    w, head, witnesses, quorum = quorum_world(5, 3)
    sig = (witnesses[0].public_key, witnesses[0].cosign(head, [], w.clock))
    assert not quorum.accepts(W.CosignedHead(head, (sig, sig, sig)), w.operator.public_bytes)


def test_unknown_witnesses_and_bad_signatures_do_not_count():
    w, head, witnesses, quorum = quorum_world(4, 3)
    good = [(x.public_key, x.cosign(head, [], w.clock)) for x in witnesses[:2]]
    outsider = witness_for(w, "outsider")
    unknown = (outsider.public_key, outsider.cosign(head, [], w.clock))
    bad = (witnesses[2].public_key, bytes(64))
    wrong_message = (witnesses[3].public_key, I.party_key("w3").sign(b"another message"))
    op = w.operator.public_bytes
    assert not quorum.accepts(W.CosignedHead(head, tuple(good) + (unknown, bad, wrong_message)), op)
    third = (witnesses[2].public_key, witnesses[2].cosign(head, [], w.clock))
    assert quorum.accepts(W.CosignedHead(head, tuple(good) + (unknown, bad, third)), op)


def test_a_quorum_needs_the_operators_own_signature_on_the_head_too():
    w, head, witnesses, quorum = quorum_world(3, 2)
    sigs = tuple((x.public_key, x.cosign(head, [], w.clock)) for x in witnesses)
    assert quorum.accepts(W.CosignedHead(head, sigs), w.operator.public_bytes)
    assert not quorum.accepts(W.CosignedHead(head, sigs), I.party_key("not the operator").public_bytes)
    assert not quorum.accepts(W.CosignedHead(replace(head, signature=bytes(64)), sigs), w.operator.public_bytes)


def test_cosignatures_on_one_head_do_not_carry_over_to_another():
    w, _, h1, h3, _ = world_with_heads()
    witnesses = [witness_for(w, f"w{i}") for i in range(3)]
    quorum = W.Quorum(frozenset(x.public_key for x in witnesses), 2)
    sigs = tuple((x.public_key, x.cosign(h1, [], w.clock)) for x in witnesses)
    assert not quorum.accepts(W.CosignedHead(h3, sigs), w.operator.public_bytes)


def test_gathering_cosignatures_asks_each_witness_with_the_right_proof():
    w, chip, h1, h3, h4 = world_with_heads()
    witnesses = [witness_for(w, f"w{i}") for i in range(4)]
    quorum = W.Quorum(frozenset(x.public_key for x in witnesses), 4)
    op = w.operator.public_bytes
    for head in (h1, h3, h4):  # each step needs a consistency proof from the witness's previous head
        gathered = W.gather_cosignatures(w.registry, head, witnesses, w.clock)
        assert len(gathered.cosignatures) == 4 and quorum.accepts(gathered, op)
        assert all(x.head == head for x in witnesses)
    again = W.gather_cosignatures(w.registry, h4, witnesses, w.clock)
    assert len(again.cosignatures) == 4  # re-cosigning the same head needs no proof


def test_gathering_skips_witnesses_that_refuse():
    w, chip, h1, h3, h4 = world_with_heads()
    behind = witness_for(w, "behind")
    ahead = witness_for(w, "ahead")
    ahead.cosign(h4, [], w.clock)
    gathered = W.gather_cosignatures(w.registry, h3, [behind, ahead], w.clock)
    assert [pk for pk, _ in gathered.cosignatures] == [behind.public_key]  # `ahead` will not go back to size 3
    assert behind.head == h3 and ahead.head == h4


# ---- how many must collude -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,t,needed,margin",
    [(1, 1, 1, 0), (3, 1, 0, 2), (3, 2, 1, 1), (3, 3, 3, 0), (4, 2, 0, 2), (4, 3, 2, 1), (5, 3, 1, 2), (5, 4, 3, 1), (5, 5, 5, 0), (7, 4, 1, 3), (7, 6, 5, 1), (9, 7, 5, 2)],
)
def test_the_quorum_formulas_at_known_points(n, t, needed, margin):
    assert W.min_colluders_to_fork(n, t) == needed
    assert W.liveness_margin(n, t) == margin


@pytest.mark.parametrize("n,t", [(0, 0), (3, 0), (3, 4), (3, -1)])
def test_the_quorum_formulas_reject_impossible_thresholds(n, t):
    with pytest.raises(ValueError):
        W.min_colluders_to_fork(n, t)
    with pytest.raises(ValueError):
        W.liveness_margin(n, t)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_enumerating_every_way_to_show_two_heads_to_real_witnesses_matches_the_formula(n):
    operator, head_a, head_b, now = forked_heads()
    for t in range(1, n + 1):
        needed = W.min_colluders_to_fork(n, t)
        for colluders in range(n + 1):
            forked = W.fork_by_enumeration(n, t, colluders, operator, head_a, head_b, now)
            assert forked == (colluders >= needed), (n, t, colluders)


def test_enumeration_rejects_an_impossible_number_of_colluders():
    operator, head_a, head_b, now = forked_heads()
    with pytest.raises(ValueError):
        W.fork_by_enumeration(3, 2, 4, operator, head_a, head_b, now)
    with pytest.raises(ValueError):
        W.fork_by_enumeration(3, 2, -1, operator, head_a, head_b, now)


def test_the_enumerated_heads_really_conflict():
    operator, head_a, head_b, now = forked_heads()
    assert head_a.size == head_b.size
    assert head_a.log_root != head_b.log_root and head_a.map_root != head_b.map_root
    assert head_a.verify(operator) and head_b.verify(operator) and now > head_a.timestamp - 1
