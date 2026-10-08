"""The attack matrix: expectations were derived from each design before the numbers were printed."""

from __future__ import annotations

import pytest

from aisecurity_rfds.rfd_034_license_replay_prevention.scenarios import (
    ATTACKS,
    BUDGET,
    CONFIG_BY_LABEL,
    CONFIGS,
    Rig,
    attack_flash_rollback,
    attack_hv_disruption,
    attack_stockpile,
    attack_matrix,
    liveness_out_of_order,
    sequence_config,
    worst_rollback_replay,
)

ALL = {c.label for c in CONFIGS}
EXPECTED_BYPASS = {
    "plain replay": set(),
    "replay after power cycle": set(),
    "flash rollback, replay 10": ALL - {"challenge+otp-epoch"},
    "HV disruption, no read-back": ALL,
    "HV disruption, read-back": set(),
    "forged huge sequence": set(),
    "oldest of 6 after eviction": {"hash-log/ring4"},
    "stockpile 5 licenses": ALL - {"challenge+otp-epoch"},
}


@pytest.fixture(scope="module")
def matrix():
    return attack_matrix()


def test_matrix_has_one_row_per_attack_and_one_cell_per_guard(matrix):
    assert [name for name, _ in matrix] == [name for name, _ in ATTACKS] == list(EXPECTED_BYPASS)
    assert all(set(row) == ALL for _, row in matrix)


def test_matrix_matches_the_outcomes_derived_from_each_design(matrix):
    for name, row in matrix:
        bypassed = {label for label, outcome in row.items() if outcome.succeeded}
        assert bypassed == EXPECTED_BYPASS[name], name


@pytest.mark.parametrize(
    ("label", "detail"),
    [
        ("hash-log", "10/10 replayed"),  # no memory survives a rollback
        ("hash-log/ring4", "10/10 replayed"),
        ("sequence", "10/10 replayed"),
        ("sequence+otp(S=4)", "2/10 replayed"),  # floor 8 after 10 licenses: sequences 9 and 10 are replayable
        ("hybrid(W=4)", "10/10 replayed"),
        ("hybrid(W=4)+otp(S=4)", "7/10 replayed"),  # floor 8, window 4: sequences 4..10
        ("challenge", "1/10 replayed"),  # only the challenge that was pending in the image
    ],
)
def test_flash_rollback_replay_counts_follow_the_derivation(label, detail):
    outcome = attack_flash_rollback(CONFIG_BY_LABEL[label])
    assert outcome.succeeded and outcome.detail == detail


def test_flash_rollback_is_fully_blocked_only_by_the_otp_epoch_challenge():
    assert not attack_flash_rollback(CONFIG_BY_LABEL["challenge+otp-epoch"]).succeeded


@pytest.mark.parametrize("cfg", CONFIGS, ids=[c.label for c in CONFIGS])
def test_hv_disruption_is_defeated_only_by_reading_back_before_crediting(cfg):
    assert attack_hv_disruption(cfg, verify_after_write=False).succeeded
    assert not attack_hv_disruption(cfg, verify_after_write=True).succeeded


def test_fuse_only_disruption_without_read_back_leaves_the_floor_behind_the_flash():
    rig = Rig(sequence_config(4), verify_after_write=False)
    licenses = [rig.issue() for _ in range(8)]
    rig.nvm.fuses.hv_disrupted = True
    assert all(rig.apply(lic).accepted for lic in licenses)
    assert rig.guard.flash_last() == 8
    assert rig.guard.floor.last_sequence_floor() == 0  # the two burns that should have happened did not


def test_fuse_only_disruption_with_read_back_turns_the_attack_into_a_lost_license():
    rig = Rig(sequence_config(4))
    licenses = [rig.issue() for _ in range(4)]
    for lic in licenses[:3]:
        assert rig.apply(lic).accepted
    rig.nvm.fuses.hv_disrupted = True
    decision = rig.apply(licenses[3])  # sequence 4 needs a fuse burn
    assert (decision.accepted, decision.reason) == (False, "commit-failed")
    assert rig.total_credited == 3 * BUDGET
    rig.nvm.fuses.hv_disrupted = False
    assert rig.apply(licenses[3]).reason == "replay-or-stale"  # flash already moved: safe but the license is gone
    rig.reboot()
    assert rig.guard.floor.last_sequence_floor() == 4  # boot repairs the floor from the flash value


def test_stockpiling_is_limited_to_one_license_only_with_epoch_binding():
    assert attack_stockpile(CONFIG_BY_LABEL["challenge"]).detail.startswith("5/5")
    assert not attack_stockpile(CONFIG_BY_LABEL["challenge+otp-epoch"]).succeeded


def test_out_of_order_delivery_strands_licenses_only_in_strictly_ordered_guards():
    expected = {
        "hash-log": 3,
        "hash-log/ring4": 3,
        "sequence": 1,
        "sequence+otp(S=4)": 1,
        "hybrid(W=4)": 3,
        "hybrid(W=4)+otp(S=4)": 3,
        "challenge": None,
        "challenge+otp-epoch": None,
    }
    assert {c.label: liveness_out_of_order(c) for c in CONFIGS} == expected


def test_sequence_floor_trades_one_fuse_per_stride_for_a_replay_window_of_stride_minus_one():
    for stride in (1, 2, 4, 8):
        worst, fuses = worst_rollback_replay(sequence_config(stride), n=16)
        assert (worst, fuses) == (stride - 1, 16 // stride)


def test_hybrid_floor_replay_window_is_window_plus_stride():
    from aisecurity_rfds.rfd_034_license_replay_prevention.scenarios import hybrid_config

    for stride in (1, 2, 4, 8):
        worst, _ = worst_rollback_replay(hybrid_config(4, stride), n=32)
        assert worst == 4 + stride
