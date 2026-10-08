"""Throttle-action arithmetic, job execution, mid-job policies and stockpiling."""

from __future__ import annotations

import numpy as np
import pytest

from aisecurity_rfds.common.hashing import sha256
from aisecurity_rfds.common.signing import KeyPair
from aisecurity_rfds.rfd_032_offline_licensing.chip import OfflineLicenseChip
from aisecurity_rfds.rfd_032_offline_licensing.licensing import LicenseAuthority
from aisecurity_rfds.rfd_032_offline_licensing.meters import Meter
from aisecurity_rfds.rfd_032_offline_licensing.policy_sim import (
    SimParams,
    drawer_stockpile,
    revocation_outcome,
    simulate,
)
from aisecurity_rfds.rfd_032_offline_licensing.throttle import (
    HardwareProfile,
    Job,
    ThrottleAction,
    preset_jobs,
    run_job,
    throttle_rate,
)

PROFILE = HardwareProfile()


def funded_chip(flops: int) -> OfflineLicenseChip:
    authority = LicenseAuthority(KeyPair.from_seed(sha256(b"throttle tests")))
    chip = OfflineLicenseChip(b"S", authority.public_key, enforced=[Meter.FLOPS])
    chip.apply_license(authority.issue(b"S", sequence=1, budgets={Meter.FLOPS: flops}))
    return chip


# ------------------------------------------------------------------ throttle actions


def test_clock_reduction_slows_every_workload_equally_including_gaming():
    rates = {n: throttle_rate(ThrottleAction.CLOCK_REDUCTION, j, PROFILE) for n, j in preset_jobs().items()}
    assert set(rates.values()) == {PROFILE.baseline_clock}


def test_functional_unit_disable_spares_non_tensor_work_and_slows_tensor_work_by_the_assumed_ratio():
    jobs = preset_jobs()
    assert throttle_rate(ThrottleAction.FUNCTIONAL_UNIT_DISABLE, jobs["gaming"], PROFILE) == 1.0
    for ratio in (1 / 4, 1 / 16, 1 / 64):
        profile = HardwareProfile(cuda_over_tensor=ratio)
        assert throttle_rate(ThrottleAction.FUNCTIONAL_UNIT_DISABLE, jobs["ai-training"], profile) == ratio


def test_memory_bandwidth_limit_hits_the_more_memory_bound_job_harder():
    jobs = preset_jobs()
    rate = {n: throttle_rate(ThrottleAction.MEMORY_BW_LIMIT, j, PROFILE) for n, j in jobs.items()}
    assert rate["ai-inference"] < rate["ai-training"]
    assert rate["gaming"] == pytest.approx(rate["ai-training"])  # identical memory-bound fraction: not separable
    assert rate["gaming"] == pytest.approx(1 / (0.7 + 0.3 / 0.05))


def test_complete_disable_stops_everything():
    assert all(throttle_rate(ThrottleAction.COMPLETE_DISABLE, j, PROFILE) == 0 for j in preset_jobs().values())


def test_hardware_profile_and_job_validate_their_parameters():
    with pytest.raises(ValueError):
        HardwareProfile(baseline_clock=0)
    with pytest.raises(ValueError):
        HardwareProfile(cuda_over_tensor=1.5)
    with pytest.raises(ValueError):
        Job("bad", 10, {}, 0.0, False, 0.1)
    with pytest.raises(ValueError):
        Job("bad", 10, {}, 1.0, False, 1.5)


# ------------------------------------------------------------------ job execution


def test_run_job_splits_full_speed_and_throttled_work_and_charges_the_meter():
    chip = funded_chip(400)
    job = Job("j", 1000, {Meter.FLOPS: 1}, 100.0, True, 0.3)
    result = run_job(chip, job, ThrottleAction.CLOCK_REDUCTION, PROFILE)
    assert (result.full_speed_units, result.throttled_units, result.completed) == (400, 600, True)
    assert result.seconds == pytest.approx(400 / 100 + 600 / (100 * 0.01))
    assert chip.remaining(Meter.FLOPS) == 0 and chip.is_throttled()


def test_functional_unit_disable_finishes_an_ai_job_far_sooner_than_clock_reduction():
    job = Job("j", 1000, {Meter.FLOPS: 1}, 100.0, True, 0.3)
    slow = run_job(funded_chip(400), job, ThrottleAction.CLOCK_REDUCTION, PROFILE).seconds
    faster = run_job(funded_chip(400), job, ThrottleAction.FUNCTIONAL_UNIT_DISABLE, PROFILE).seconds
    assert faster == pytest.approx(4 + 600 / (100 * 0.0625)) and faster < slow


def test_complete_disable_leaves_the_job_unfinished_after_the_budget_runs_out():
    result = run_job(funded_chip(400), Job("j", 1000, {Meter.FLOPS: 1}, 100.0, True, 0.3), ThrottleAction.COMPLETE_DISABLE)
    assert (result.completed, result.units_done, result.seconds) == (False, 400, 4.0)


def test_a_job_within_budget_never_touches_the_throttle():
    chip = funded_chip(10_000)
    result = run_job(chip, Job("j", 1000, {Meter.FLOPS: 1}, 100.0, True, 0.3), ThrottleAction.COMPLETE_DISABLE)
    assert result.completed and result.throttled_units == 0 and chip.remaining(Meter.FLOPS) == 9000


# ------------------------------------------------------------------ mid-job policies (ASSUMED job-size model)


def results(budget: float, policy: str, **kwargs):
    return simulate(policy, SimParams(budget=budget, **kwargs), np.random.default_rng(1))


@pytest.mark.parametrize("budget", [5, 20, 100])
def test_checkpoint_policy_slowdown_is_one_plus_delay_over_budget(budget):
    expected = 1 + 10.0 / budget
    assert results(budget, "checkpoint").mean_slowdown == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("budget", [5, 20, 100])
def test_each_license_interrupts_about_one_job_so_reissues_match_work_over_budget(budget):
    sizes = np.random.default_rng(1).lognormal(0.0, 0.5, 5000)
    result = results(budget, "checkpoint")
    assert abs(result.reissues - sizes.sum() / budget) <= 1


def test_hard_stop_never_finishes_jobs_larger_than_one_license_and_wastes_work():
    tiny = results(1, "hard-stop")
    assert 0.45 <= tiny.never_completed_fraction <= 0.55  # median job size is 1, so about half exceed a budget of 1
    assert results(20, "hard-stop").wasted_fraction > 0.0
    for policy in ("checkpoint", "throttle"):
        assert results(1, policy).never_completed_fraction == 0.0
        assert results(20, policy).wasted_fraction == 0.0


@pytest.mark.parametrize("budget", [1, 5, 20, 100])
def test_throttle_policy_slowdown_follows_the_closed_form_and_never_beats_pausing_by_much(budget):
    """Each license cycle does `budget` units at full speed plus baseline_rate * delay units while waiting, so
    slowdown = (B + D) / (B + r * D); pausing gives (B + D) / B."""
    delay, rate = 10.0, 0.01
    throttle, pause = results(budget, "throttle"), results(budget, "checkpoint")
    assert throttle.mean_slowdown == pytest.approx((budget + delay) / (budget + rate * delay), rel=0.02)
    assert throttle.mean_slowdown <= pause.mean_slowdown
    if budget >= 20:  # the baseline's contribution is negligible once a license dwarfs rate * delay
        assert pause.mean_slowdown - throttle.mean_slowdown < 0.01


def test_simulation_validates_its_parameters():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate("overdraft", SimParams(budget=10), rng)
    with pytest.raises(ValueError):
        simulate("throttle", SimParams(budget=0), rng)
    with pytest.raises(ValueError):
        simulate("throttle", SimParams(budget=10, delay=-1), rng)


# ------------------------------------------------------------------ stockpiling


@pytest.mark.parametrize("cap", [100, 300, 500])
def test_balance_cap_bounds_the_burst_but_not_the_total_a_stockpile_provides(cap):
    total, peak = drawer_stockpile(5, 100, cap)
    assert total == 500 and peak == cap


def test_a_cap_below_one_license_makes_every_license_unusable():
    with pytest.raises(RuntimeError):
        drawer_stockpile(2, 100, 50)


def test_revocation_helps_only_if_the_chip_applies_it_before_the_stockpile():
    assert revocation_outcome(5, 100, deliver_revocation_first=True) == 0
    assert revocation_outcome(5, 100, deliver_revocation_first=False) == 500
