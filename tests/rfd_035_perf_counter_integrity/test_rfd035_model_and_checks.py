"""Telemetry model and verifier checks: consistency, calibration, one test per check, and the power statistic."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.rfd_035_perf_counter_integrity import checks, model
from aisecurity_rfds.rfd_035_perf_counter_integrity.attacks import (
    analytic_power_detection_probability,
    dynamic_power,
    honest_readings,
)
from aisecurity_rfds.rfd_035_perf_counter_integrity.model import CLASSES, PEAK_ISSUE, generate_reading


@pytest.fixture(scope="module")
def th():
    return checks.calibrate(honest_readings(CLASSES, 100, make_rng(1)))


@pytest.fixture(scope="module")
def fresh():
    return honest_readings(CLASSES, 80, make_rng(2))


@pytest.fixture
def base():
    return generate_reading(CLASSES["dense-matmul"], make_rng(4))


def flagged(reading, th):
    return {name for name, hit in checks.inspect(reading, th).items() if hit}


def test_generated_readings_are_internally_consistent(fresh):
    for r in fresh:
        assert np.all(r.issued >= r.retired)
        assert np.all(r.mem >= r.miss)
        assert np.all(r.issued <= PEAK_ISSUE)
        assert np.all(r.retired >= 0)
        assert np.isclose(sum(r.agg_flops), r.flops.sum())


def test_generation_is_deterministic_for_a_seed():
    a = generate_reading(CLASSES["elementwise"], make_rng(5))
    b = generate_reading(CLASSES["elementwise"], make_rng(5))
    assert np.array_equal(a.flops, b.flops) and np.array_equal(a.mem, b.mem) and a.power_w == b.power_w
    c = generate_reading(CLASSES["elementwise"], make_rng(6))
    assert not np.array_equal(a.flops, c.flops)


def test_measured_power_is_model_power_plus_two_percent_noise():
    rng = make_rng(3)
    rel = []
    for _ in range(2000):
        r = generate_reading(CLASSES["dense-matmul"], rng)
        rel.append(r.power_w / model.true_power(float(r.flops.sum()), float(r.mem.sum())) - 1.0)
    assert abs(np.mean(rel)) < 0.003
    assert 0.0185 < np.std(rel) < 0.0215


def test_honest_readings_of_calibration_classes_trip_no_structural_check(th, fresh):
    structural = [(i, flagged(r, th) - {"power-consistency"}) for i, r in enumerate(fresh)]
    assert [x for x in structural if x[1]] == []
    # The power check is statistical (two-sided rate about 2e-4 per reading), so allow the rare noise tail.
    assert sum("power-consistency" in flagged(r, th) for r in fresh) <= 2


def test_calibrated_bands_cover_every_calibration_ratio():
    readings = honest_readings(CLASSES, 60, make_rng(8))
    th = checks.calibrate(readings, margin=0.0)
    for r in readings:
        on = r.retired > 0
        rho, mu = r.flops[on] / r.retired[on], r.mem[on] / r.retired[on]
        assert th.rho_lo <= rho.min() and rho.max() <= th.rho_hi
        assert th.mu_lo <= mu.min() and mu.max() <= th.mu_hi


def test_margin_widens_the_bands_exactly():
    readings = honest_readings(CLASSES, 30, make_rng(9))
    tight, wide = checks.calibrate(readings, margin=0.0), checks.calibrate(readings, margin=0.10)
    assert wide.rho_lo == pytest.approx(0.9 * tight.rho_lo) and wide.rho_hi == pytest.approx(1.1 * tight.rho_hi)
    assert wide.mu_lo == pytest.approx(0.9 * tight.mu_lo) and wide.mu_hi == pytest.approx(1.1 * tight.mu_hi)


def test_issued_below_retired_is_flagged_alone(th, base):
    issued = base.issued.copy()
    issued[0] = base.retired[0] - 1
    assert flagged(base.with_counters(issued=issued), th) == {"issued>=retired"}


def test_more_misses_than_accesses_is_flagged_alone(th, base):
    miss = base.miss.copy()
    miss[3] = base.mem[3] + 1
    assert flagged(base.with_counters(miss=miss), th) == {"accesses>=misses"}


def test_issue_rate_above_hardware_peak_is_flagged_alone(th, base):
    issued = base.issued.copy()
    issued[5] = int(PEAK_ISSUE) + 1
    assert flagged(base.with_counters(issued=issued), th) == {"throughput-bound"}


def test_flop_ratio_outside_band_is_flagged(th, base):
    flops = base.flops.copy()
    flops[0] = base.retired[0] * th.rho_hi * 1.2
    got = flagged(base.with_counters(flops=flops), th)
    assert "flop-per-instruction-band" in got
    assert not got & {"issued>=retired", "accesses>=misses", "throughput-bound", "aggregator-cross-check"}


def test_memory_ratio_outside_band_is_flagged(th, base):
    mem = base.mem.copy()
    mem[0] = int(base.retired[0] * th.mu_hi * 1.2)
    assert "memory-per-instruction-band" in flagged(base.with_counters(mem=mem), th)


def test_one_sm_halving_its_flops_is_caught_by_peers_not_by_bands(th, base):
    flops = base.flops.copy()
    flops[0] *= 0.5  # ratio 2.8 stays inside the class-agnostic band [0.73, 6.06]
    assert flagged(base.with_counters(flops=flops), th) == {"peer-outlier"}


def test_aggregator_that_disagrees_with_the_sm_counters_is_flagged_alone(th, base):
    bad = dataclasses.replace(base, agg_flops=(base.agg_flops[0] * 0.9, base.agg_flops[1]))
    assert flagged(bad, th) == {"aggregator-cross-check"}


def test_power_far_from_the_counters_prediction_is_flagged_alone(th, base):
    assert flagged(dataclasses.replace(base, power_w=base.power_w * 1.3), th) == {"power-consistency"}
    assert flagged(dataclasses.replace(base, power_w=base.power_w * 0.7), th) == {"power-consistency"}


def test_idle_sm_invariant_flags_counters_on_sms_that_retired_nothing(th):
    r = generate_reading(CLASSES["dense-matmul"], make_rng(10), active_fraction=0.5)
    assert flagged(r, th) == set()
    idle = int(np.flatnonzero(r.retired == 0)[0])
    mem, flops = r.mem.copy(), r.flops.copy()
    mem[idle] = 5
    assert "memory-per-instruction-band" in flagged(r.with_counters(mem=mem), th)
    flops[idle] = 100.0
    assert "flop-per-instruction-band" in flagged(r.with_counters(flops=flops), th)


def test_power_statistic_is_close_to_standard_normal_on_honest_data():
    rng = make_rng(6)
    z = np.array([checks.power_z([generate_reading(CLASSES["decode-memory-bound"], rng)], 0.02) for _ in range(3000)])
    assert abs(z.mean()) < 0.06  # normalised by predicted power, the honest statistic is exactly noise / sigma_rel
    assert 0.93 < z.std() < 1.07


def test_power_false_positive_rate_matches_the_exact_tail_probability():
    rng = make_rng(7)
    n, z_crit = 6000, 2.0
    hits = sum(abs(checks.power_z([generate_reading(CLASSES["elementwise"], rng)], 0.02)) > z_crit for _ in range(n))
    expected = analytic_power_detection_probability(0.0, 100.0, 300.0, 0.02, z_crit)
    assert abs(hits / n - expected) < 4 * np.sqrt(expected * (1 - expected) / n)


def test_default_threshold_gives_a_two_sided_false_positive_rate_near_two_in_ten_thousand():
    p = analytic_power_detection_probability(0.0, 100.0, 300.0, 0.02, 3.72)
    assert 1.5e-4 < p < 3.0e-4
    assert checks.power_z_threshold(1e-4) == pytest.approx(3.719, abs=0.001)


def test_minimum_detectable_fraction_is_the_fifty_percent_point():
    r = generate_reading(CLASSES["dense-matmul"], make_rng(8), power_noise=0.0)
    pdyn, p = dynamic_power(r), r.power_w
    for n in (1, 4, 25):
        h = checks.min_detectable_fraction(pdyn, p, 0.02, 3.72, n)
        assert 0.4 < analytic_power_detection_probability(h, pdyn, p, 0.02, 3.72, n=n) < 0.6


def test_minimum_detectable_fraction_follows_its_closed_form_and_scales_roughly_as_root_n_and_sigma():
    d, p, z = 200.0, 300.0, 3.72
    for n, sigma in ((1, 0.02), (4, 0.02), (25, 0.02), (1, 0.10)):
        expected = z * sigma * p / (d * (np.sqrt(n) + z * sigma))
        assert checks.min_detectable_fraction(d, p, sigma, z, n) == pytest.approx(expected)
    h1 = checks.min_detectable_fraction(d, p, 0.02, z, 1)
    assert checks.min_detectable_fraction(d, p, 0.02, z, 4) == pytest.approx(h1 / 2, rel=0.05)
    assert checks.min_detectable_fraction(d, p, 0.10, z, 1) > 3.5 * h1  # grows with the assumed noise
    assert checks.min_detectable_hidden_fraction(1e5, 300.0, 0.02, 3.72, 1) == pytest.approx(
        checks.min_detectable_fraction(model.K_FLOP * 1e5, 300.0, 0.02, 3.72, 1)
    )


def test_exact_detection_formula_agrees_with_direct_simulation_of_the_statistic_for_several_windows():
    rng = make_rng(12)
    power, p_dyn, sigma = 388.0, 288.0, 0.02
    for n, h in ((1, 0.05), (4, 0.05), (25, 0.02)):
        eps = rng.standard_normal((40_000, n))
        measured = power * (1 + sigma * eps)
        predicted = power - h * p_dyn
        z = ((measured - predicted) / (sigma * predicted)).sum(axis=1) / np.sqrt(n)
        empirical = float(np.mean(np.abs(z) > 3.72))
        formula = analytic_power_detection_probability(h, p_dyn, power, sigma, 3.72, n=n)
        assert abs(empirical - formula) < 4 * np.sqrt(formula * (1 - formula) / 40_000) + 1e-4


def test_detection_formula_is_monotone_in_hidden_fraction_and_antitone_in_assumed_noise():
    probs = [analytic_power_detection_probability(h, 288.0, 388.0, 0.02, 3.72) for h in np.linspace(0, 0.3, 13)]
    assert all(b >= a for a, b in zip(probs, probs[1:]))
    assert probs[0] < 3e-4 and probs[-1] > 0.999
    noisy = [analytic_power_detection_probability(0.1, 288.0, 388.0, s, 3.72, noise_rel=0.02) for s in (0.02, 0.04, 0.08)]
    assert noisy[0] > noisy[1] > noisy[2]


def test_delta_based_power_statistic_agrees_with_the_per_reading_one(th):
    rs = honest_readings(CLASSES, 5, make_rng(11))
    direct = checks.power_z(rs, th.sigma_rel)
    via_deltas = checks.power_z_from_deltas(
        [float(r.flops.sum()) for r in rs], [float(r.mem.sum()) for r in rs], [r.power_w for r in rs], th.sigma_rel
    )
    assert direct == pytest.approx(via_deltas)
