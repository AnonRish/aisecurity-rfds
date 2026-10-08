"""License format and the on-chip verifier: binding, tamper, replay, atomicity, caps and power-cycle safety."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aisecurity_rfds.common.hashing import sha256
from aisecurity_rfds.common.nvm import FlashStore
from aisecurity_rfds.common.signing import KeyPair, verify
from aisecurity_rfds.rfd_032_offline_licensing.chip import OfflineLicenseChip
from aisecurity_rfds.rfd_032_offline_licensing.licensing import ComputeLicense, LicenseAuthority
from aisecurity_rfds.rfd_032_offline_licensing.meters import U64_MAX, Meter
from aisecurity_rfds.rfd_032_offline_licensing.policy_sim import (
    checkpoint_interval_for_endurance,
    power_cycle_experiment,
)

SERIAL = b"CHIP-A"
F, M = Meter.FLOPS, Meter.MEM_BYTES


@pytest.fixture
def authority():
    return LicenseAuthority(KeyPair.from_seed(sha256(b"test authority")))


def make_chip(authority, **kwargs):
    kwargs.setdefault("enforced", [F, M])
    return OfflineLicenseChip(SERIAL, authority.public_key, **kwargs)


class TearingFlash(FlashStore):
    """Tears one chosen write after `torn_at` bytes and then raises, as a power loss would."""

    def __init__(self):
        super().__init__()
        self._countdown = None
        self._torn_at = 0

    def arm(self, writes_from_now: int, torn_at: int) -> None:
        self._countdown, self._torn_at = writes_from_now, torn_at

    def write(self, key, value, *, torn_at=None):
        if self._countdown is not None:
            if self._countdown == 0:
                self._countdown = None
                super().write(key, value, torn_at=self._torn_at)
                raise RuntimeError("power loss")
            self._countdown -= 1
        super().write(key, value, torn_at=torn_at)


# ------------------------------------------------------------------ license format


@pytest.mark.parametrize(
    "mutation",
    [
        {"device_serial": b"CHIP-B"},
        {"sequence": 2},
        {"mode": "set"},
        {"budgets": (("flops", 101), ("mem_bytes", 50))},
        {"budgets": (("flops", 100), ("int_ops", 50))},
    ],
)
def test_signature_covers_serial_sequence_mode_and_every_budget(authority, mutation):
    lic = authority.issue(SERIAL, sequence=1, budgets={F: 100, M: 50})
    assert verify(authority.public_key, lic.digest(), lic.signature)
    forged = dataclasses.replace(lic, **mutation)
    assert not verify(authority.public_key, forged.digest(), forged.signature)


def test_digest_encoding_is_injective_across_field_boundaries():
    a = ComputeLicense(b"ab", 1, "add", (("c", 1),))
    b = ComputeLicense(b"a", 1, "add", (("bc", 1),))
    assert a.digest() != b.digest()


def test_authority_rejects_invalid_licenses(authority):
    with pytest.raises(ValueError):
        authority.issue(SERIAL, sequence=1, budgets={F: 1}, mode="subtract")
    with pytest.raises(ValueError):
        authority.issue(SERIAL, sequence=0, budgets={F: 1})
    with pytest.raises(ValueError):
        authority.issue(SERIAL, sequence=1, budgets={})
    with pytest.raises(ValueError):
        authority.issue(SERIAL, sequence=1, budgets={F: -1})
    with pytest.raises(ValueError):
        authority.issue(SERIAL, sequence=1, budgets={F: U64_MAX + 1})


def test_issued_budgets_are_in_canonical_sorted_order(authority):
    lic = authority.issue(SERIAL, sequence=1, budgets={Meter.MEM_BYTES: 1, Meter.CYCLES: 2, Meter.FLOPS: 3})
    assert [name for name, _ in lic.budgets] == ["cycles", "flops", "mem_bytes"]


# ------------------------------------------------------------------ on-chip verification


def test_unlicensed_chip_is_throttled_until_every_enforced_meter_is_funded(authority):
    chip = make_chip(authority)
    assert chip.is_throttled()
    assert chip.apply_license(authority.issue(SERIAL, sequence=1, budgets={F: 100})).accepted
    assert chip.is_throttled()  # the memory meter is still empty
    assert chip.apply_license(authority.issue(SERIAL, sequence=2, budgets={M: 50})).accepted
    assert not chip.is_throttled()


def test_license_for_another_serial_is_rejected_and_changes_nothing(authority):
    chip = make_chip(authority)
    other = authority.issue(b"CHIP-B", sequence=1, budgets={F: 100, M: 50})
    assert chip.apply_license(other).reason == "wrong-device"
    assert chip.remaining(F) == 0 and chip.last_sequence == 0


def test_edited_budget_is_rejected_as_a_bad_signature(authority):
    chip = make_chip(authority)
    lic = authority.issue(SERIAL, sequence=1, budgets={F: 100, M: 50})
    forged = dataclasses.replace(lic, budgets=(("flops", 10**9), ("mem_bytes", 50)))
    assert chip.apply_license(forged).reason == "bad-signature"
    assert chip.remaining(F) == 0


def test_replayed_and_stale_licenses_are_rejected(authority):
    chip = make_chip(authority)
    first = authority.issue(SERIAL, sequence=2, budgets={F: 100, M: 50})
    assert chip.apply_license(first).accepted
    assert chip.apply_license(first).reason == "replay-or-stale"
    assert chip.apply_license(authority.issue(SERIAL, sequence=1, budgets={F: 100, M: 50})).reason == "replay-or-stale"
    assert chip.remaining(F) == 100


def test_license_with_an_unenforced_meter_is_rejected_atomically(authority):
    chip = make_chip(authority, enforced=[F])
    lic = authority.issue(SERIAL, sequence=1, budgets={F: 5, Meter.CYCLES: 5})
    assert chip.apply_license(lic).reason == "unknown-meter"
    assert chip.remaining(F) == 0 and chip.last_sequence == 0


def test_duplicate_meter_entries_are_malformed_not_double_counted(authority):
    chip = make_chip(authority)
    dup = authority.sign(ComputeLicense(SERIAL, 1, "add", (("flops", 5), ("flops", 5))))
    assert verify(authority.public_key, dup.digest(), dup.signature)  # genuinely signed
    assert chip.apply_license(dup).reason == "malformed"
    assert chip.remaining(F) == 0


@pytest.mark.parametrize(
    "bad",
    [
        ComputeLicense(SERIAL, 1, "add", (("mem_bytes", 1), ("flops", 1))),  # not in canonical order
        ComputeLicense(SERIAL, 1, "add", ()),  # empty
        ComputeLicense(SERIAL, 1, "multiply", (("flops", 1),)),  # unknown mode
    ],
)
def test_signed_but_non_canonical_licenses_are_rejected(authority, bad):
    assert make_chip(authority).apply_license(authority.sign(bad)).reason == "malformed"


def test_top_up_that_would_overflow_u64_is_rejected_and_state_is_unchanged(authority):
    chip = make_chip(authority, enforced=[F])
    chip.apply_license(authority.issue(SERIAL, sequence=1, budgets={F: 5}))
    big = authority.issue(SERIAL, sequence=2, budgets={F: U64_MAX})
    assert chip.apply_license(big).reason == "overflow"
    assert chip.remaining(F) == 5 and chip.last_sequence == 1


def test_balance_cap_rejects_the_whole_license_not_just_one_meter(authority):
    chip = make_chip(authority, max_balance={F: 150})
    assert chip.apply_license(authority.issue(SERIAL, sequence=1, budgets={F: 100, M: 10})).accepted
    assert chip.apply_license(authority.issue(SERIAL, sequence=2, budgets={F: 100, M: 10})).reason == "balance-cap"
    assert (chip.remaining(F), chip.remaining(M), chip.last_sequence) == (100, 10, 1)


def test_set_mode_overwrites_the_balance_and_can_revoke_it(authority):
    chip = make_chip(authority, enforced=[F])
    chip.apply_license(authority.issue(SERIAL, sequence=1, budgets={F: 1000}))
    chip.apply_license(authority.issue(SERIAL, sequence=2, budgets={F: 0}, mode="set"))
    assert chip.remaining(F) == 0 and chip.is_throttled()


def test_exhausting_one_meter_throttles_and_a_new_license_restores_full_speed(authority):
    chip = make_chip(authority)
    chip.apply_license(authority.issue(SERIAL, sequence=1, budgets={F: 100, M: 100}))
    chip.consume({F: 100})
    assert chip.is_throttled() and chip.remaining(M) == 100
    chip.apply_license(authority.issue(SERIAL, sequence=2, budgets={F: 100}))
    assert not chip.is_throttled()


def test_units_available_is_set_by_the_scarcest_meter(authority):
    chip = make_chip(authority)
    chip.apply_license(authority.issue(SERIAL, sequence=1, budgets={F: 100, M: 30}))
    assert chip.units_available({F: 1, M: 2}) == 15
    assert chip.units_available({}) is None
    assert chip.units_available({Meter.CYCLES: 5}) is None


def test_consume_clamps_at_zero_ignores_unenforced_meters_and_rejects_negative_usage(authority):
    chip = make_chip(authority, enforced=[F])
    chip.apply_license(authority.issue(SERIAL, sequence=1, budgets={F: 10}))
    chip.consume({F: 25, Meter.CYCLES: 99})
    assert chip.remaining(F) == 0
    with pytest.raises(ValueError):
        chip.consume({F: -1})


def test_chip_constructor_validates_its_parameters(authority):
    with pytest.raises(ValueError):
        OfflineLicenseChip(SERIAL, authority.public_key, enforced=[])
    with pytest.raises(ValueError):
        make_chip(authority, checkpoint_interval={F: 0})


# ------------------------------------------------------------------ power cycling


def test_sequence_number_survives_a_power_cycle_so_replay_stays_rejected(authority):
    chip = make_chip(authority, enforced=[F])
    lic = authority.issue(SERIAL, sequence=1, budgets={F: 100})
    chip.apply_license(lic)
    chip.power_cycle()
    assert chip.apply_license(lic).reason == "replay-or-stale"


def test_power_cycling_never_increases_the_balance_and_loses_at_most_one_interval(authority):
    rng = np.random.default_rng(3)
    for interval in (1, 7, 100):
        report = power_cycle_experiment(10**5, interval, rng, steps=300, cycle_prob=0.2)
        assert report.cycles > 20
        assert report.max_gain <= 0
        assert report.max_loss <= interval


def test_checkpoint_writes_scale_with_budget_over_interval_plus_one(authority):
    """One write per interval + 1 units consumed (the balance may sit exactly on the checkpoint without a
    write), plus the first write that moves the checkpoint below the exact post-license value."""
    for interval in (10, 50):
        chip = make_chip(authority, enforced=[F], checkpoint_interval={F: interval})
        chip.apply_license(authority.issue(SERIAL, sequence=1, budgets={F: 1000}))
        for _ in range(1000):
            chip.consume({F: 1})
        assert abs(chip.checkpoint_writes - (1000 / (interval + 1) + 1)) <= 1.5


def test_torn_checkpoint_write_never_restores_more_than_the_true_balance(authority):
    for torn_at in range(0, 60):
        flash = TearingFlash()
        chip = make_chip(authority, enforced=[F], checkpoint_interval={F: 100}, flash=flash)
        chip.apply_license(authority.issue(SERIAL, sequence=1, budgets={F: 1000}))
        flash.arm(0, torn_at)
        with pytest.raises(RuntimeError):
            chip.consume({F: 600})  # the checkpoint is written before the work is allowed to happen
        true_before = chip.remaining(F)
        chip.power_cycle()
        assert chip.remaining(F) in (1000, 300)
        assert chip.remaining(F) <= true_before


def test_torn_license_write_leaves_the_chip_either_unlicensed_or_fully_licensed(authority):
    outcomes = set()
    for torn_at in range(0, 60):
        flash = TearingFlash()
        chip = make_chip(authority, enforced=[F], flash=flash)
        flash.arm(0, torn_at)
        with pytest.raises(RuntimeError):
            chip.apply_license(authority.issue(SERIAL, sequence=1, budgets={F: 100}))
        chip.power_cycle()
        outcomes.add((chip.last_sequence, chip.remaining(F)))
    assert outcomes == {(0, 0), (1, 100)}


def test_flash_rollback_restores_an_older_larger_balance_documented_limitation(authority):
    """Known gap: balance and sequence both live in flash, so restoring an old image un-spends the budget.
    RFD 034 analyses the same attack on replay state; a fuse floor or a rollback-proof counter is needed."""
    flash = FlashStore()
    chip = make_chip(authority, enforced=[F], flash=flash)
    chip.apply_license(authority.issue(SERIAL, sequence=1, budgets={F: 1000}))
    image = flash.snapshot()
    chip.consume({F: 1000})
    assert chip.is_throttled()
    flash.physical_restore(image)
    chip.power_cycle()
    assert chip.remaining(F) == 1000 and chip.last_sequence == 1


def test_endurance_arithmetic_and_validation():
    assert checkpoint_interval_for_endurance(10**18, 10**5) == 10**13
    assert checkpoint_interval_for_endurance(10, 3) == 4
    with pytest.raises(ValueError):
        checkpoint_interval_for_endurance(10, 0)
