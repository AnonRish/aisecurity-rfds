"""The RFD's six attack vectors against the verifier checks: who catches what, and what slips through."""

from __future__ import annotations

import numpy as np
import pytest

from aisecurity_rfds.common.rng import derive_seed, make_rng
from aisecurity_rfds.rfd_035_perf_counter_integrity import attacks, checks
from aisecurity_rfds.rfd_035_perf_counter_integrity.model import CLASSES, N_SM, NOVEL_CLASS, generate_reading


@pytest.fixture(scope="module")
def th():
    return checks.calibrate(attacks.honest_readings(CLASSES, 100, make_rng(21)))


@pytest.fixture(scope="module")
def fresh():
    return attacks.honest_readings(CLASSES, 80, make_rng(22))


def stats(name, readings, th, fn=None):
    return attacks.evaluate_attack(fn or attacks.ATTACKS[name], readings, th, make_rng(derive_seed("t035", name)))


@pytest.mark.parametrize("name", list(attacks.ATTACKS))
def test_every_attack_is_flagged_by_some_check_almost_always(name, th, fresh):
    assert stats(name, fresh, th).any_flag >= 0.99


@pytest.mark.parametrize("name", list(attacks.ATTACKS))
def test_attacks_never_edit_the_measured_power(name, fresh):
    rng = make_rng(derive_seed("power", name))
    for r in fresh[:40]:
        assert attacks.ATTACKS[name](r, rng).power_w == r.power_w


def test_sm_level_attacks_leave_aggregators_consistent_with_the_edited_counters(fresh):
    rng = make_rng(23)
    for fn in (attacks.emp_bit_flips, attacks.voltage_glitch_reset, attacks.laser_flop_scale, attacks.fib_wire_cut, attacks.firmware_consistent_scale):
        attacked = fn(fresh[0], rng)
        assert sum(attacked.agg_flops) == pytest.approx(float(attacked.flops.sum()))


def test_aggregator_attack_changes_only_the_aggregators(fresh):
    rng = make_rng(24)
    r = fresh[0]
    attacked = attacks.aggregator_tamper(r, rng, h=0.3)
    assert np.array_equal(attacked.flops, r.flops) and np.array_equal(attacked.retired, r.retired)
    assert sum(attacked.agg_flops) == pytest.approx(0.7 * sum(r.agg_flops))
    assert attacks.hidden_fraction(r, attacked) == pytest.approx(0.3)


def test_firmware_scaling_every_counter_is_caught_only_by_the_power_check(th, fresh):
    st = stats("firmware: all counters -30%", fresh, th)
    assert st.exclusive["power-consistency"] >= 0.99


def test_aggregator_tampering_is_caught_only_by_the_cross_check(th, fresh):
    st = stats("aggregators -30%", fresh, th)
    assert st.exclusive["aggregator-cross-check"] == 1.0


def test_clearing_whole_sms_looks_idle_to_every_check_except_power(th, fresh):
    st = stats("voltage glitch clears half the SMs", fresh, th)
    assert st.exclusive["power-consistency"] >= 0.99


def test_clearing_only_the_flop_registers_is_caught_by_the_ratio_band(th, fresh):
    st = stats("flop-only", fresh, th, fn=lambda r, g: attacks.voltage_glitch_reset(r, g, fraction=0.25, scope="flops"))
    assert st.flagged_by["flop-per-instruction-band"] >= 0.99
    assert st.flagged_by["peer-outlier"] >= 0.99


def test_laser_scaling_on_a_few_sms_is_caught_by_peer_comparison_not_by_power(th, fresh):
    st = stats("laser: FLOPs -50% on 4 SMs", fresh, th)
    assert st.flagged_by["peer-outlier"] >= 0.99
    assert st.flagged_by["power-consistency"] <= 0.05


def test_wire_cut_is_caught_by_band_and_peer_checks(th, fresh):
    st = stats("FIB: FLOP wire cut on 4 SMs", fresh, th)
    assert st.flagged_by["flop-per-instruction-band"] >= 0.99
    assert st.flagged_by["peer-outlier"] >= 0.99


def test_peer_check_bounds_per_sm_hiding_to_about_a_dozen_percent(th):
    """Below the peer tolerance (about 12% of one SM's FLOPs) an SM can under-report undetected, so little is hidden."""
    rs = attacks.honest_readings(CLASSES, 100, make_rng(25))
    quiet = stats("laser-5%", rs, th, fn=lambda r, g: attacks.laser_flop_scale(r, g, h=0.05, n_sms=1))
    loud = stats("laser-30%", rs, th, fn=lambda r, g: attacks.laser_flop_scale(r, g, h=0.30, n_sms=1))
    assert quiet.any_flag < 0.1  # 5% of one SM is inside the peer tolerance
    assert loud.any_flag > 0.9
    assert quiet.undetected_hidden_max < 0.05 * 1.25 / N_SM  # 5% of one SM, whose share is at most 1.25x the average


def test_emp_flips_that_evade_every_check_hide_almost_nothing(th):
    rs = attacks.honest_readings(CLASSES, 500, make_rng(37))
    st = stats("emp-1", rs, th, fn=lambda r, g: attacks.emp_bit_flips(r, g, n_flips=1))
    assert 0.70 < st.any_flag < 0.92  # a single flip is often too small to notice...
    assert st.undetected_hidden_max < 0.01  # ...but then it cannot have hidden much (bounded by peer tolerance / N_SM)


def test_more_emp_flips_raise_the_detection_rate(th):
    rs = attacks.honest_readings(CLASSES, 300, make_rng(38))
    rates = [stats(f"emp-{n}", rs, th, fn=lambda r, g, n=n: attacks.emp_bit_flips(r, g, n_flips=n)).any_flag for n in (1, 2, 4)]
    assert rates[0] < rates[1] < rates[2]


@pytest.mark.parametrize("cname", list(CLASSES))
def test_consistent_scaling_detection_rate_matches_the_exact_formula(cname, th):
    n = 1500
    measured, formula = attacks.scale_detection(CLASSES[cname], th, 0.10, n, make_rng(derive_seed("sd", cname)))
    assert abs(measured - formula) < 4 * np.sqrt(formula * (1 - formula) / n) + 0.01


def test_consistent_scaling_below_the_power_floor_goes_unnoticed(th):
    measured, formula = attacks.scale_detection(CLASSES["dense-matmul"], th, 0.02, 600, make_rng(26))
    assert measured < 0.02 and formula < 0.01


def test_scaling_hides_exactly_the_fraction_requested(fresh):
    for h in (0.05, 0.3, 0.6):
        r = fresh[7]
        attacked = attacks.firmware_consistent_scale(r, make_rng(27), h=h)
        assert attacks.hidden_fraction(r, attacked) == pytest.approx(h, abs=0.005)


def test_hidden_fraction_is_zero_for_over_reporting_and_for_no_attack(fresh):
    r = fresh[0]
    assert attacks.hidden_fraction(r, r) == 0.0
    bigger = r.with_counters(flops=r.flops * 2)
    assert attacks.hidden_fraction(r, bigger) == 0.0


def test_more_noise_than_the_verifier_assumes_weakens_the_power_check():
    formula = attacks.analytic_power_detection_probability
    assumed = formula(0.10, 288.0, 388.0, 0.02, 3.72)
    underestimated = formula(0.10, 288.0, 388.0, 0.02, 3.72, noise_rel=0.05)
    assert underestimated < assumed
    honest_alarms = formula(0.0, 288.0, 388.0, 0.02, 3.72, noise_rel=0.05)
    assert honest_alarms > 100 * formula(0.0, 288.0, 388.0, 0.02, 3.72)  # far more false alarms when noise is underestimated


def test_attack_evaluation_is_reproducible(th, fresh):
    a = stats("firmware: all counters -30%", fresh, th)
    b = stats("firmware: all counters -30%", fresh, th)
    assert a == b


def test_attack_functions_do_not_mutate_their_input(fresh):
    r = fresh[3]
    before = (r.flops.copy(), r.mem.copy(), r.retired.copy(), r.agg_flops)
    for fn in attacks.ATTACKS.values():
        fn(r, make_rng(28))
    assert np.array_equal(r.flops, before[0]) and np.array_equal(r.mem, before[1]) and np.array_equal(r.retired, before[2])
    assert r.agg_flops == before[3]


def test_unattacked_novel_workload_is_flagged_so_flags_alone_do_not_mean_attack(th):
    """A flag means 'outside what was calibrated', not 'attack': an honest unseen workload trips the bands."""
    novel = [generate_reading(NOVEL_CLASS, make_rng(derive_seed("novel", i))) for i in range(30)]
    assert all(any(checks.inspect(r, th).values()) for r in novel)
