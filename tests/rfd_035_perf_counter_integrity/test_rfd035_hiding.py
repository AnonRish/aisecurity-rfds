"""What a threshold-aware attacker can hide: closed form versus brute-force search, band width, and idle SMs."""

from __future__ import annotations

import numpy as np
import pytest

from aisecurity_rfds.common.rng import derive_seed, make_rng
from aisecurity_rfds.rfd_035_perf_counter_integrity import checks, hiding
from aisecurity_rfds.rfd_035_perf_counter_integrity.attacks import metered_flops
from aisecurity_rfds.rfd_035_perf_counter_integrity.model import CLASSES, K_FLOP, K_MEM, N_SM, NOVEL_CLASS, generate_reading


@pytest.fixture(scope="module")
def per_class():
    return hiding.calibration_readings(CLASSES, 100, make_rng(31))


@pytest.fixture(scope="module")
def agnostic(per_class):
    return checks.calibrate([r for rs in per_class.values() for r in rs])


def flagged(reading, th):
    return {name for name, hit in checks.inspect(reading, th).items() if hit}


def test_ideal_readings_are_noise_free_and_pass_every_check(per_class, agnostic):
    for name, cls in CLASSES.items():
        r = hiding.ideal_reading(cls)
        assert flagged(r, agnostic) == set()
        assert np.all(r.flops[r.retired > 0] / r.retired[r.retired > 0] == pytest.approx(cls.flops_per_retired))


@pytest.mark.parametrize("cname", list(CLASSES))
@pytest.mark.parametrize("mode", ["agnostic", "class-aware"])
def test_closed_form_matches_brute_force_search_on_ideal_readings(cname, mode, per_class, agnostic):
    th = agnostic if mode == "agnostic" else checks.calibrate(per_class[cname])
    cls = CLASSES[cname]
    searched = hiding.max_hidden_fraction(hiding.ideal_reading(cls), th)
    closed = hiding.closed_form_max_hidden_fraction(cls.flops_per_retired, cls.mem_fraction, th, 500)
    assert abs(searched - closed) < 0.01


@pytest.mark.parametrize("cname", list(CLASSES))
def test_noisy_readings_allow_slightly_less_hiding_than_the_closed_form(cname, agnostic):
    cls = CLASSES[cname]
    closed = hiding.closed_form_max_hidden_fraction(cls.flops_per_retired, cls.mem_fraction, agnostic, 500)
    found = [hiding.max_hidden_fraction(generate_reading(cls, make_rng(derive_seed("noisy", cname, i))), agnostic) for i in range(6)]
    assert max(found) < closed + 0.01  # per-SM jitter only ever tightens the feasible set
    assert min(found) > closed - 0.12


def test_the_issue_cap_binds_for_busy_memory_bound_sms(agnostic):
    cls = CLASSES["decode-memory-bound"]
    busy = hiding.ideal_reading(cls, retired=1000)
    closed = hiding.closed_form_max_hidden_fraction(cls.flops_per_retired, cls.mem_fraction, agnostic, 1000)
    band_only = hiding.closed_form_max_hidden_fraction(cls.flops_per_retired, cls.mem_fraction, agnostic, 1)
    assert closed < band_only - 0.2  # the throughput cap, not the ratio band, is the binding constraint here
    assert abs(hiding.max_hidden_fraction(busy, agnostic) - closed) < 0.01


def test_feasible_attack_passes_every_check_and_hides_what_it_claims(agnostic):
    ideal = hiding.ideal_reading(CLASSES["dense-matmul"])
    h_max = hiding.max_hidden_fraction(ideal, agnostic)
    h = h_max - 0.02
    report = hiding.find_hiding_report(ideal, agnostic, h)
    assert report is not None
    assert flagged(report, agnostic) == set()
    assert metered_flops(report) == pytest.approx((1 - h) * metered_flops(ideal), rel=1e-9)


def test_beyond_the_maximum_every_retired_scale_trips_some_check(agnostic):
    ideal = hiding.ideal_reading(CLASSES["dense-matmul"])
    h = hiding.max_hidden_fraction(ideal, agnostic) + 0.02
    assert hiding.find_hiding_report(ideal, agnostic, h) is None
    for s in np.geomspace(0.05, 20.0, 400):
        assert flagged(hiding.power_matched_report(ideal, h, float(s)), agnostic), s


def test_power_matching_leaves_no_power_residual():
    ideal = hiding.ideal_reading(CLASSES["decode-memory-bound"])
    # The only residual is rounding of the fake memory counters to integers: at most half an access per SM.
    rounding_bound = 0.5 * K_MEM * N_SM / (0.02 * ideal.power_w)
    for h in (0.1, 0.3, 0.5):
        attacked = hiding.power_matched_report(ideal, h, 1.0)
        assert abs(checks.power_z([attacked], 0.02)) <= 1.01 * rounding_bound
        assert metered_flops(attacked) == pytest.approx((1 - h) * metered_flops(ideal))
    assert hiding.POWER_MATCH_RATIO == pytest.approx(K_FLOP / K_MEM)


def test_without_power_matching_the_same_hiding_is_caught_by_power(agnostic):
    ideal = hiding.ideal_reading(CLASSES["dense-matmul"])
    sloppy = ideal.with_counters(flops=ideal.flops * 0.7)  # hides 30% but adds no fake traffic
    assert "power-consistency" in flagged(sloppy, agnostic)


def test_class_aware_bands_shrink_what_can_be_hidden(per_class):
    for row in hiding.hiding_table(per_class):
        assert row.closed_aware < row.closed_agnostic
        assert row.searched_aware < row.searched_agnostic


def test_class_aware_hiding_is_a_few_percent_for_dense_matmul(per_class):
    row = next(r for r in hiding.hiding_table(per_class) if r.cls == "dense-matmul")
    assert row.closed_aware < 0.05 and row.closed_agnostic > 0.8


def test_hiding_scales_with_how_memory_light_the_workload_is(agnostic):
    """With class-agnostic bands, a workload with a small memory/FLOP ratio leaves the most room for fake traffic."""
    closed = {n: hiding.closed_form_max_hidden_fraction(c.flops_per_retired, c.mem_fraction, agnostic, 500) for n, c in CLASSES.items()}
    assert closed["dense-matmul"] > closed["decode-memory-bound"] > closed["elementwise"]


def test_wider_bands_flag_the_unseen_workload_less_but_let_the_attacker_hide_more(per_class):
    rows = hiding.margin_tradeoff(per_class, (0.0, 0.25, 0.5, 0.7, 0.9), make_rng(32), n_test=300)
    assert [r.margin for r in rows] == [0.0, 0.25, 0.5, 0.7, 0.9]
    hideable = [r.dense_matmul_hidden for r in rows]
    assert all(b > a for a, b in zip(hideable, hideable[1:]))
    novel = [r.novel_false_positive_rate for r in rows]
    assert all(b <= a for a, b in zip(novel, novel[1:]))
    assert novel[0] == 1.0 and novel[-2] == 0.0
    assert rows[-2].dense_matmul_hidden > 0.95  # the margin that stops false alarms also lets >95% be hidden
    # Bands equal to the observed sample extremes flag some fresh honest readings of the *calibration* classes too;
    # any real margin removes that.
    assert 0.0 < rows[0].seen_false_positive_rate < 0.2
    assert all(r.seen_false_positive_rate == 0.0 for r in rows[1:])


def test_the_unseen_class_is_rejected_by_default_bands_on_the_flop_ratio(agnostic):
    novel = [generate_reading(NOVEL_CLASS, make_rng(derive_seed("novel-hiding", i))) for i in range(40)]
    assert all("flop-per-instruction-band" in flagged(r, agnostic) for r in novel)


def test_fake_traffic_on_idle_sms_is_caught_only_by_the_idle_sm_invariant(agnostic):
    r = generate_reading(CLASSES["dense-matmul"], make_rng(3), active_fraction=0.5)
    assert flagged(r, agnostic) == set()
    attacked = hiding.fake_traffic_on_idle_sms(r, 0.5)
    assert metered_flops(attacked) == pytest.approx(0.5 * metered_flops(r))
    assert flagged(attacked, agnostic) == {"memory-per-instruction-band"}


def test_the_power_match_ratio_is_the_watts_per_flop_over_the_watts_per_access():
    assert hiding.POWER_MATCH_RATIO == pytest.approx(0.075)
    ideal = hiding.ideal_reading(CLASSES["dense-matmul"])
    attacked = hiding.power_matched_report(ideal, 0.2, 1.0)
    extra_mem = float(attacked.mem.sum() - ideal.mem.sum())
    hidden_flops = float(ideal.flops.sum() - attacked.flops.sum())
    assert K_MEM * extra_mem == pytest.approx(K_FLOP * hidden_flops, rel=0.01)


def test_search_returns_zero_when_the_honest_reading_is_itself_rejected(per_class):
    tight = checks.calibrate(per_class["dense-matmul"], margin=0.0)
    elementwise = hiding.ideal_reading(CLASSES["elementwise"])
    assert hiding.max_hidden_fraction(elementwise, tight) == 0.0
