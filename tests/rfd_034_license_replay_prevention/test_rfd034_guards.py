"""Behaviour of each replay guard on its own: acceptance, rejection, known limitations and parameter limits."""

from __future__ import annotations

import pytest

from aisecurity_rfds.rfd_034_license_replay_prevention.guards import HashLogGuard
from aisecurity_rfds.rfd_034_license_replay_prevention.licensing import License
from aisecurity_rfds.rfd_034_license_replay_prevention.scenarios import (
    BUDGET,
    CONFIG_BY_LABEL,
    CONFIGS,
    DEVICE_ID,
    N_FUSES,
    Rig,
    attack_ring_eviction,
    hybrid_config,
    lenient_verifier,
    padded_signature_replay,
    sequence_config,
    weak_rng_replay,
)
from aisecurity_rfds.rfd_034_license_replay_prevention.scenarios import ed25519_verify

IDS = [c.label for c in CONFIGS]


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_fresh_license_is_accepted_and_credits_its_budget(cfg):
    rig = Rig(cfg)
    decision = rig.apply(rig.issue())
    assert decision.accepted and decision.budget_credited == BUDGET and rig.total_credited == BUDGET


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_license_for_another_device_is_rejected_without_touching_state(cfg):
    rig = Rig(cfg)
    lic = rig.issue()
    other = rig.authority.issue(b"OTHER-CHIP", sequence=lic.sequence, budget_flops=BUDGET, challenge=lic.challenge)
    before = rig.nvm.flash.snapshot()
    assert rig.apply(other).reason == "wrong-device"
    assert rig.nvm.flash.snapshot() == before


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_forged_license_never_changes_persistent_state(cfg):
    rig = Rig(cfg)
    lic = rig.issue()
    flash_before, fuses_before = rig.nvm.flash.snapshot(), rig.nvm.fuses.burned()
    forged = License(DEVICE_ID, 2**40, bytes(16), BUDGET, lic.challenge, bytes(64))
    decision = rig.apply(forged)
    assert (decision.accepted, decision.reason) == (False, "bad-signature")
    assert rig.nvm.flash.snapshot() == flash_before and rig.nvm.fuses.burned() == fuses_before
    assert rig.apply(lic).accepted


def test_hash_log_keyed_on_signed_content_ignores_signature_encoding():
    assert padded_signature_replay("content", lenient_verifier) is False
    assert padded_signature_replay("content", ed25519_verify) is False


def test_hash_log_keyed_on_full_license_is_bypassed_by_a_lenient_verifier():
    """Known limitation of 'hash the license' if the verifier accepts a second encoding of one signature."""
    assert padded_signature_replay("full", lenient_verifier) is True
    assert padded_signature_replay("full", ed25519_verify) is False


def test_bounded_hash_log_forgets_its_oldest_entries():
    assert attack_ring_eviction(CONFIG_BY_LABEL["hash-log/ring4"]).succeeded
    assert not attack_ring_eviction(CONFIG_BY_LABEL["hash-log"]).succeeded


def test_hash_log_rejects_invalid_parameters():
    rig = Rig(CONFIG_BY_LABEL["hash-log"])
    args = (rig.nvm, DEVICE_ID, rig.authority.public_key)
    with pytest.raises(ValueError):
        HashLogGuard(*args, capacity=0)
    with pytest.raises(ValueError):
        HashLogGuard(*args, hash_scope="signature")


def test_sequence_guard_strands_a_license_that_arrives_after_a_later_one():
    rig = Rig(sequence_config())
    first, second = rig.issue(), rig.issue()
    assert rig.apply(second).accepted
    decision = rig.apply(first)
    assert (decision.accepted, decision.reason) == (False, "replay-or-stale")


def test_sequence_guard_rejects_an_equal_sequence_number_even_with_a_new_id():
    rig = Rig(sequence_config())
    first = rig.issue()
    rig.apply(first)
    twin = rig.authority.issue(DEVICE_ID, sequence=first.sequence, budget_flops=BUDGET)
    assert not rig.apply(twin).accepted


def test_one_signed_far_future_sequence_number_permanently_strands_lower_licenses():
    """Inherent to monotonic counters: the authority's own mistake bricks every lower license."""
    rig = Rig(sequence_config())
    pending = rig.issue()
    far = rig.authority.issue(DEVICE_ID, sequence=10**6, budget_flops=BUDGET)
    assert rig.apply(far).accepted
    assert not rig.apply(pending).accepted


def test_sequence_gaps_are_safe_for_replay_but_cost_fuses_under_an_otp_floor():
    rig = Rig(sequence_config(4))
    first = rig.issue()
    rig.apply(first)
    skipped_ahead = rig.authority.issue(DEVICE_ID, sequence=41, budget_flops=BUDGET)  # authority skipped 39 numbers
    assert rig.apply(skipped_ahead).accepted
    assert rig.nvm.fuses.burned() == 41 // 4  # one fuse per stride crossed, however large the gap
    assert not rig.apply(first).accepted and not rig.apply(skipped_ahead).accepted


def test_hybrid_accepts_late_licenses_inside_the_window_and_rejects_older_ones():
    rig = Rig(hybrid_config(2))
    licenses = [rig.issue() for _ in range(6)]  # sequences 1..6
    assert rig.apply(licenses[5]).accepted  # high-water mark 6
    assert rig.apply(licenses[3]).accepted  # sequence 4: 4 + 2 >= 6
    decision = rig.apply(licenses[2])  # sequence 3: 3 + 2 < 6
    assert (decision.accepted, decision.reason) == (False, "too-old")


def test_hybrid_log_is_pruned_by_sequence_and_stays_bounded():
    rig = Rig(hybrid_config(3))
    for _ in range(50):
        rig.apply(rig.issue())
    entries = rig.guard.log_entries()
    assert len(entries) <= 4
    assert min(sequence for sequence, _ in entries) >= 50 - 3


def test_hybrid_rejects_a_reused_license_id_inside_the_window():
    rig = Rig(hybrid_config(4))
    first = rig.issue()
    rig.apply(first)
    twin = rig.authority.issue(DEVICE_ID, sequence=first.sequence + 1, budget_flops=BUDGET, license_id=first.license_id)
    decision = rig.apply(twin)
    assert (decision.accepted, decision.reason) == (False, "replay")


def test_challenge_license_is_single_use_and_bound_to_the_pending_challenge():
    rig = Rig(CONFIG_BY_LABEL["challenge"])
    lic = rig.issue()
    wrong = rig.authority.issue(DEVICE_ID, sequence=0, budget_flops=BUDGET, challenge=bytes(32))
    assert rig.apply(wrong).reason == "challenge-mismatch"
    assert rig.apply(lic).accepted
    assert rig.apply(lic).reason == "no-pending-challenge"


def test_requesting_a_new_challenge_invalidates_the_previous_outstanding_license():
    """Liveness cost of one pending challenge: an attacker with driver access can void licenses in flight."""
    rig = Rig(CONFIG_BY_LABEL["challenge"])
    first, second = rig.issue(), rig.issue()
    assert rig.apply(first).reason == "challenge-mismatch"
    assert rig.apply(second).accepted


def test_repeating_rng_enables_replay_unless_the_challenge_is_bound_to_the_otp_epoch():
    assert weak_rng_replay(use_otp_epoch=False) is True
    assert weak_rng_replay(use_otp_epoch=True) is False


def test_challenge_epoch_exhausts_after_one_fuse_per_license():
    rig = Rig(CONFIG_BY_LABEL["challenge+otp-epoch"])
    for _ in range(N_FUSES):
        assert rig.apply(rig.issue()).accepted
    assert rig.apply(rig.issue()).reason == "otp-exhausted"


def test_otp_floor_rejects_unreachable_sequence_numbers_without_burning_fuses():
    rig = Rig(sequence_config(1))
    far = rig.authority.issue(DEVICE_ID, sequence=N_FUSES + 10, budget_flops=BUDGET)
    assert rig.apply(far).reason == "otp-exhausted"
    assert rig.nvm.fuses.burned() == 0
