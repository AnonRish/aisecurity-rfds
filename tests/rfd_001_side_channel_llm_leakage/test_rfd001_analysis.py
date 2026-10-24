import math

import numpy as np
import pytest
from scipy import stats as sps

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.rfd_001_side_channel_llm_leakage import analysis as an
from aisecurity_rfds.rfd_001_side_channel_llm_leakage import leakage as lk


def test_welch_t_matches_scipy(rng):
    a, b = rng.normal(0, 1, (7, 3)), rng.normal(0.5, 2, (9, 3))
    assert np.allclose(an.welch_t(a, b), sps.ttest_ind(a, b, equal_var=False).statistic)


def test_welch_t_is_signed_and_antisymmetric(rng):
    a, b = rng.normal(0, 1, (30, 4)), rng.normal(1, 1, (30, 4))
    assert np.allclose(an.welch_t(a, b), -an.welch_t(b, a)) and (an.welch_t(a, b) < 0).all()


def test_tvla_traces_needed_at_known_points():
    assert an.tvla_traces_needed(1.0, power=0.5) == 41  # 2 * 4.5^2 = 40.5
    assert an.tvla_traces_needed(1.0, power=0.9) == 67  # 2 * (4.5 + 1.2816)^2 = 66.85
    assert an.tvla_traces_needed(0.5, power=0.9) == 268


def test_tvla_traces_scale_with_the_inverse_square_of_the_effect_and_grow_with_power():
    assert an.tvla_traces_needed(0.1) / an.tvla_traces_needed(0.2) == pytest.approx(4.0, rel=0.01)
    assert an.tvla_traces_needed(0.3, 0.5) < an.tvla_traces_needed(0.3, 0.9) < an.tvla_traces_needed(0.3, 0.99)
    assert an.tvla_traces_needed(0.3, 0.9, threshold=3.0) < an.tvla_traces_needed(0.3, 0.9, threshold=4.5)


@pytest.mark.parametrize("effect, power", [(0.0, 0.9), (-1.0, 0.9), (1.0, 0.0), (1.0, 1.0)])
def test_tvla_traces_needed_rejects_nonsense(effect, power):
    with pytest.raises(ValueError):
        an.tvla_traces_needed(effect, power)


def test_the_t_test_flags_a_planted_leak_at_the_rate_the_formula_promises():
    d = 0.3
    n = an.tvla_traces_needed(d, 0.9)
    r = make_rng(11)
    a, b = r.normal(0, 1, (n, 1500)), r.normal(d, 1, (n, 1500))
    assert 0.86 <= (an.welch_t(b, a) > an.TVLA_THRESHOLD).mean() <= 0.94


def test_the_t_test_flags_almost_nothing_when_there_is_no_leak():
    r = make_rng(12)
    t = an.welch_t(r.normal(0, 1, (200, 20000)), r.normal(0, 1, (200, 20000)))
    assert (np.abs(t) > an.TVLA_THRESHOLD).sum() <= 2  # expected 20000 * 6.8e-6 = 0.14


def test_theoretical_rho_at_known_points():
    assert an.theoretical_rho(0.0) == 0.0
    assert an.theoretical_rho(1.0) == pytest.approx(1 / math.sqrt(2))
    assert an.theoretical_rho(3.0) == pytest.approx(3 / math.sqrt(10))
    assert an.theoretical_rho(1e6) == pytest.approx(1.0)


def test_fisher_traces_at_a_known_point_and_in_both_directions():
    assert an.fisher_traces(0.5, 3.0) == pytest.approx(32.83, abs=0.01)  # 3 + (3 / atanh(0.5))^2
    assert an.fisher_traces(0.9, 3.0) < an.fisher_traces(0.5, 3.0) < an.fisher_traces(0.1, 3.0)
    assert an.fisher_traces(0.5, 5.0) > an.fisher_traces(0.5, 3.0)


def test_the_leak_spread_and_the_per_secret_correlation():
    assert an.LEAK_SD == pytest.approx(2.55, abs=0.005)
    assert an.secret_rho(77, 0.0) == pytest.approx(1.0)
    assert an.secret_rho(0, 5.0) == 0.0  # a zero activation has nothing to leak
    assert an.secret_rho(77, 1.0) > an.secret_rho(77, 2.0) > an.secret_rho(77, 8.0) > 0
    rho = an.secret_rho(77, 2.0)
    assert 2.0 / math.sqrt(rho**-2 - 1) == pytest.approx(2.336, abs=0.001)  # rho = sd / sqrt(sd^2 + 4), solved for sd


def test_correlation_detection_traces_matches_a_known_formula_value():
    rho = an.secret_rho(77, 2.0)
    assert an.correlation_detection_traces(rho, power=0.9) == math.ceil(3 + ((4.5 + 1.2816) / math.atanh(rho)) ** 2)
    assert an.correlation_detection_traces(rho, power=0.99) > an.correlation_detection_traces(rho, power=0.5)


def test_the_detection_formula_delivers_about_the_power_it_promises():
    sigma = 2.0
    n = an.correlation_detection_traces(an.secret_rho(77, sigma))
    rate = an.correlation_flag_rate(77, sigma, n, 300, make_rng(13))
    assert 0.85 <= rate.p <= 0.99
    # with far fewer samples the leak is mostly missed
    assert an.correlation_flag_rate(77, sigma, n // 4, 300, make_rng(14)).p < 0.5


def test_correlations_match_numpy(rng):
    observed, predictions = rng.normal(5, 2, 50), rng.normal(-1, 3, (5, 50))
    expected = [np.corrcoef(observed, p)[0, 1] for p in predictions]
    assert np.allclose(an.correlations(observed, predictions), expected)


def test_a_constant_prediction_has_correlation_zero_not_nan(rng):
    out = an.correlations(rng.normal(0, 1, 20), np.vstack([np.full(20, 3.0), rng.normal(0, 1, 20)]))
    assert out[0] == 0.0 and np.isfinite(out).all()


def test_an_activation_is_recovered_exactly_from_noise_free_leakage():
    weights = np.arange(-128, 128)
    for secret in (-128, -93, -1, 1, 5, 77, 127):
        assert an.cpa_recover(lk.hw16(weights * secret).astype(float), weights) == secret


def test_a_zero_activation_cannot_be_recovered_because_its_prediction_is_constant():
    weights = np.arange(-128, 128)
    predictions = an.cpa_predictions(weights)
    assert not predictions[128].any()  # hypothesis 0 predicts zero leakage for every weight
    assert an.correlations(lk.hw16(weights * 0).astype(float) + make_rng(15).normal(0, 1, 256), predictions)[128] == 0.0


def test_the_attack_succeeds_more_often_with_more_samples_and_less_noise():
    few = an.cpa_success_rate(2.0, 8, 150, make_rng(16)).p
    many = an.cpa_success_rate(2.0, 200, 150, make_rng(17)).p
    noisy = an.cpa_success_rate(8.0, 100, 150, make_rng(18)).p
    assert few < 0.5 < many and an.cpa_success_rate(2.0, 100, 150, make_rng(19)).p > noisy


def test_recovering_a_value_takes_more_samples_than_detecting_that_it_leaks():
    detect = an.correlation_detection_traces(an.secret_rho(77, 4.0))
    recover = an.samples_to_recover(4.0, 0.9, 100, 1, secret=77)
    assert recover is not None and recover > detect


def test_quadrupling_the_noise_variance_costs_several_times_the_samples():
    low = an.samples_to_recover(2.0, 0.9, 100, 1, secret=77)
    high = an.samples_to_recover(4.0, 0.9, 100, 1, secret=77)
    assert low is not None and high is not None and 2.5 <= high / low <= 8.0


def test_samples_to_recover_gives_up_past_the_ladder():
    assert an.samples_to_recover(1000.0, 0.9, 20, 1) is None


def test_the_ladder_grows_by_half_each_step():
    assert an.LADDER[0] == 8 and an.LADDER[1] == 12 and an.LADDER[-1] > 5000
    assert all(b > a for a, b in zip(an.LADDER, an.LADDER[1:]))
    assert all(1.3 < b / a < 1.7 for a, b in zip(an.LADDER, an.LADDER[1:]))


def test_the_closest_confusable_activation_of_a_doubling_is_the_double():
    # x and 2x give products that differ by one shift, so their Hamming weights track each other
    assert an.closest_confusable(5)[0] == 10 and an.closest_confusable(5)[1] > 0.99
    assert an.closest_confusable(100)[0] == 50 and an.closest_confusable(100)[1] > 0.98
    other, r = an.closest_confusable(77)
    assert other != 77 and 0.5 < r < 0.9


def test_pair_error_at_known_values():
    assert an.pair_error(0.0, 1.0) == 0.5 and an.pair_error(0.0, 0.0) == 0.5
    assert an.pair_error(2.0, 1.0) == pytest.approx(0.158655, abs=1e-6)  # Q(1)
    assert an.pair_error(10.0, 0.0) == 0.0
    assert an.pair_error(4.0, 1.0) == pytest.approx(0.0227501, abs=1e-6)  # Q(2)


def test_pair_error_falls_with_distance_and_rises_with_noise():
    assert an.pair_error(1.0, 1.0) > an.pair_error(2.0, 1.0) > an.pair_error(3.0, 1.0)
    assert an.pair_error(2.0, 0.5) < an.pair_error(2.0, 1.0) < an.pair_error(2.0, 2.0)
    assert np.allclose(an.pair_error(np.array([0.0, 2.0]), 1.0), [0.5, 0.158655], atol=1e-6)


def test_distances_between_rows():
    d = an.distances(np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]]))
    assert d[0, 1] == 5.0 and d[1, 0] == 5.0 and d[0, 2] == 0.0 and np.allclose(np.diag(d), 0)


def test_the_error_bounds_on_three_collinear_hypotheses():
    q1, q2 = 0.158655, 0.0227501  # Q(1), Q(2)
    lower, upper = an.identification_error_bounds(np.array([[0.0], [2.0], [4.0]]), 1.0)
    assert lower == pytest.approx(q1, abs=1e-5)  # the largest pairwise error, for every hypothesis
    assert upper == pytest.approx((2 * (q1 + q2) + 2 * q1) / 3, abs=1e-5)  # the end points sum two, the middle sums two


def test_two_hypotheses_have_equal_bounds_because_the_error_is_exact():
    lower, upper = an.identification_error_bounds(np.array([[0.0], [3.0]]), 1.5)
    assert lower == pytest.approx(upper) == pytest.approx(an.pair_error(3.0, 1.5))


def test_the_union_bound_is_capped_at_one():
    expected = np.zeros((50, 1)) + np.arange(50)[:, None] * 1e-3  # fifty nearly identical hypotheses
    assert an.identification_error_bounds(expected, 1.0)[1] == 1.0


def test_the_bounds_sandwich_the_simulated_error_at_several_noise_levels():
    expected = lk.SCOPE.view(lk.ToyLayer.make().templates())
    for sigma in (6.0, 12.0, 24.0, 48.0):
        lower, upper = an.identification_error_bounds(expected, sigma)
        est = an.identification_accuracy(expected, sigma, 3000, make_rng(int(sigma)))
        error, slack = 1 - est.p, 3 * math.sqrt(est.p * (1 - est.p) / 3000) + 1e-9
        assert lower - slack <= error <= upper + slack, sigma


def test_identical_templates_cap_the_accuracy_because_one_of_two_copies_is_always_wrong():
    rows = np.vstack([np.eye(7) * 5.0, np.eye(7)[0] * 5.0])  # eight hypotheses, the last a copy of the first
    est = an.identification_accuracy(rows, 1e-6, 4000, make_rng(20))
    assert est.p == pytest.approx(7 / 8, abs=0.02)
    lower, upper = an.identification_error_bounds(rows, 1e-6)
    assert lower == pytest.approx(2 * 0.5 / 8) and upper >= lower


def test_the_noise_free_ceiling_is_the_share_of_rows_that_are_distinct():
    assert an.noise_free_identification_accuracy(np.array([[0.0], [1.0], [2.0]])) == 1.0
    assert an.noise_free_identification_accuracy(np.array([[0.0], [0.0], [1.0]])) == pytest.approx(2 / 3)
    assert an.noise_free_identification_accuracy(np.array([[1.0, 2.0]] * 3 + [[3.0, 4.0]])) == 0.5
    assert an.noise_free_identification_accuracy(np.zeros((5, 2))) == pytest.approx(1 / 5)


def test_the_noise_free_ceiling_compares_whole_rows_not_columns():
    assert an.noise_free_identification_accuracy(np.array([[1.0, 1.0], [2.0, 2.0]])) == 1.0  # equal columns, distinct rows
    assert an.noise_free_identification_accuracy(np.array([[1.0, 2.0], [1.0, 3.0]])) == 1.0  # equal first column only


def test_a_noiseless_attacker_reaches_exactly_the_ceiling_it_is_given():
    layer = lk.ToyLayer.make()
    slow = lk.SOFTWARE.view(layer.templates())  # one number per token; some tokens share it
    ceiling = an.noise_free_identification_accuracy(slow)
    assert ceiling < 1.0
    est = an.identification_accuracy(slow, 0.0, 6400, make_rng(21))
    assert est.p == pytest.approx(ceiling, abs=0.02)
    assert an.noise_free_identification_accuracy(lk.SCOPE.view(layer.templates())) == 1.0


def test_identify_returns_the_nearest_row():
    expected = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    assert an.identify(np.array([[9.0, 1.0], [1.0, 8.0], [-3.0, -2.0]]), expected).tolist() == [1, 2, 0]


def test_averaging_runs_is_the_same_as_dividing_the_noise_by_the_square_root():
    expected = lk.SCOPE.view(lk.ToyLayer.make().templates())
    averaged = an.identification_accuracy(expected, 24.0, 3000, make_rng(21), repeats=4)
    halved = an.identification_accuracy(expected, 12.0, 3000, make_rng(22))
    assert abs(averaged.p - halved.p) < 0.05
    assert averaged.p > an.identification_accuracy(expected, 24.0, 3000, make_rng(23)).p + 0.1


def test_repeats_needed_is_the_smallest_count_that_guarantees_the_target():
    expected = lk.SCOPE.view(lk.ToyLayer.make().templates())
    sigma = lk.SCOPE.noise
    n = an.repeats_needed(expected, sigma, 0.99)
    assert n is not None and n > 1
    assert 1 - an.identification_error_bounds(expected, sigma / math.sqrt(n))[1] >= 0.99
    assert 1 - an.identification_error_bounds(expected, sigma / math.sqrt(n - 1))[1] < 0.99
    assert an.repeats_needed(expected, sigma, 0.9) <= n <= an.repeats_needed(expected, sigma, 0.999)


def test_repeats_needed_is_none_when_no_amount_of_averaging_can_separate_the_hypotheses():
    twin = np.vstack([np.eye(5) * 3.0, np.eye(5)[:1] * 3.0])
    assert an.repeats_needed(twin, 1.0, 0.99) is None
    assert an.repeats_needed(lk.ACOUSTIC.view(lk.ToyLayer.make().templates()), lk.ACOUSTIC.noise, 0.99) is None


@pytest.mark.parametrize("target", [0.0, 1.0, -0.5, 1.5])
def test_repeats_needed_rejects_a_target_outside_zero_one(target):
    with pytest.raises(ValueError):
        an.repeats_needed(np.eye(3), 1.0, target)
