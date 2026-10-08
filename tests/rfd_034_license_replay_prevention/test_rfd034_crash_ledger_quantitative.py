"""Exhaustive power-loss enumeration, the external commitment ledger and the quantitative trade-offs."""

from __future__ import annotations

import pytest

from aisecurity_rfds.common.signing import KeyPair
from aisecurity_rfds.rfd_034_license_replay_prevention.hardware import PowerLoss
from aisecurity_rfds.rfd_034_license_replay_prevention.ledger import Commitment, CommitmentEmitter, ExternalLedger
from aisecurity_rfds.rfd_034_license_replay_prevention.scenarios import (
    BUDGET,
    CONFIG_BY_LABEL,
    CONFIGS,
    DEVICE_ID,
    NAIVE_SEQUENCE,
    Rig,
    crash_consistency,
    crash_points,
    ledger_detection,
    mean_stranded_fraction,
    run_crash_point,
    sequence_config,
    storage_profile,
)

IDS = [c.label for c in CONFIGS]


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_no_power_loss_point_lets_any_license_be_credited_twice(cfg):
    report = crash_consistency(cfg)
    assert report.points >= 17
    assert report.violations == 0


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_a_license_is_lost_only_when_the_commit_was_already_durable(cfg):
    """Non-OTP guards end with one flash write (1 lost point: right after it). OTP guards end with a flash
    write then a fuse burn: lost after the flash write, before the burn, and after the burn."""
    report = crash_consistency(cfg)
    assert report.lost == (3 if "otp" in cfg.label else 1)
    assert report.recoverable == report.points - report.lost


def test_non_atomic_counter_regresses_under_exactly_one_torn_write_offset():
    """Negative control: the harness can report a failure. A raw little-endian counter 0x01FF -> 0x0200 torn
    after one byte reads 0x0100, so a credited license with a higher sequence number is replayable."""
    points = crash_points(NAIVE_SEQUENCE)
    violating = [p for p in points if run_crash_point(NAIVE_SEQUENCE, p)[0]]
    assert violating == [(0, "torn", 1)]


def test_power_loss_after_the_final_write_credits_nothing_and_loses_the_license():
    rig = Rig(CONFIG_BY_LABEL["sequence"])
    lic = rig.issue()
    rig.nvm.plan.arm(0, "after")
    with pytest.raises(PowerLoss):
        rig.apply(lic)
    assert rig.total_credited == 0
    rig.reboot()
    assert rig.apply(lic).reason == "replay-or-stale"


def test_torn_journal_write_leaves_the_previous_value_authoritative():
    rig = Rig(CONFIG_BY_LABEL["sequence"])
    first, second = rig.issue(), rig.issue()
    rig.apply(first)
    rig.nvm.plan.arm(0, "torn", 5)
    with pytest.raises(PowerLoss):
        rig.apply(second)
    rig.reboot()
    assert rig.guard.last_sequence() == first.sequence
    assert rig.apply(second).accepted


def test_boot_repairs_the_otp_floor_after_a_crash_between_flash_and_fuse():
    rig = Rig(sequence_config(4))
    for _ in range(3):  # sequences 1..3 need no fuse burn (stride 4)
        rig.apply(rig.issue())
    fourth = rig.issue()
    rig.nvm.plan.arm(1, "before")  # op 0 = flash write of sequence 4, op 1 = the fuse burn
    with pytest.raises(PowerLoss):
        rig.apply(fourth)
    assert rig.guard.flash_last() == 4 and rig.guard.floor.last_sequence_floor() == 0
    rig.reboot()
    assert rig.guard.floor.last_sequence_floor() == 4


def _chip_with_ledger(every: int):
    rig = Rig(sequence_config())
    key = KeyPair.from_seed(bytes(range(32)))
    return rig, CommitmentEmitter(key, DEVICE_ID, every), ExternalLedger(DEVICE_ID, key.public_bytes), key


def test_honest_chip_is_never_flagged_by_the_ledger():
    rig, emitter, ledger, _ = _chip_with_ledger(3)
    anomalies = []
    for _ in range(30):
        assert rig.apply(rig.issue()).accepted
        commitment = emitter.on_accepted(rig.guard.last_sequence())
        if commitment is not None:
            anomalies += ledger.submit(commitment)
    assert anomalies == []


def test_ledger_flags_a_careless_rollback_attacker_as_a_regression():
    anomalies, credited = ledger_detection(careful=False)
    assert "regression" in anomalies
    assert credited == 25  # the attacker still got 10 extra licenses; the ledger only notices afterwards


def test_ledger_does_not_flag_a_careful_attacker_who_withholds_replay_commitments():
    """Known limitation of the external-commitment option: detection needs the attacker's cooperation."""
    anomalies, credited = ledger_detection(careful=True)
    assert anomalies == [] and credited == 25


def test_ledger_rejects_forged_commitments_and_other_devices():
    rig, emitter, ledger, key = _chip_with_ledger(1)
    rig.apply(rig.issue())
    genuine = emitter.on_accepted(1)
    forged = KeyPair.from_seed(bytes(32)).sign(genuine.message())
    assert ledger.submit(Commitment(genuine.device_id, 1, genuine.timestamp, forged))[0].startswith("rejected")
    other = Commitment(b"OTHER", genuine.sequence, genuine.timestamp, genuine.signature)
    assert ledger.submit(other)[0].startswith("rejected")
    assert ledger.submit(genuine) == []


def test_commitment_emitter_validates_its_cadence():
    with pytest.raises(ValueError):
        CommitmentEmitter(KeyPair.from_seed(bytes(32)), DEVICE_ID, 0)


def test_storage_follows_the_rfd_formulas_for_each_approach():
    assert storage_profile(CONFIG_BY_LABEL["hash-log"], 10)[0] == 8 + 32 * 10  # 32 bytes per license (+ 8 count)
    assert storage_profile(CONFIG_BY_LABEL["hash-log/ring4"], 100)[0] == 8 + 32 * 4  # bounded by the ring
    assert storage_profile(CONFIG_BY_LABEL["sequence"], 100)[0] == 8  # a single 8-byte counter
    assert storage_profile(CONFIG_BY_LABEL["hybrid(W=4)"], 100)[0] == 8 + 24 * 5  # window + 1 entries of 24 bytes
    assert storage_profile(CONFIG_BY_LABEL["challenge+otp-epoch"], 100)[2] == 100  # one fuse per license


def test_double_slot_journaling_makes_physical_flash_larger_than_the_logical_payload():
    payload, physical, fuses = storage_profile(CONFIG_BY_LABEL["sequence"], 10)
    assert fuses == 0 and physical == 2 * (payload + 16)


def test_strictly_ordered_delivery_strands_more_licenses_as_displacement_grows():
    fractions = [mean_stranded_fraction(None, d) for d in (0, 2, 4, 8)]
    assert fractions[0] == 0.0
    assert fractions == sorted(fractions) and fractions[-1] > 0.3


def test_hybrid_window_at_least_the_displacement_strands_nothing_and_smaller_windows_do():
    for displacement in (2, 4, 8):
        assert mean_stranded_fraction(displacement, displacement) == 0.0
    assert mean_stranded_fraction(2, 8) > 0.0
