import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.rfd_002_moe_expert_fingerprinting import signatures as sg


def q(x):
    """Gaussian upper tail, written with erfc so that the tests do not lean on the code's own scipy call."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


# --- the space of expert sets ------------------------------------------------------------------------------------


def test_the_sets_are_listed_in_lexicographic_order_and_numbered_by_bitmask():
    space = sg.SetSpace.make(4, 2)
    assert space.sets.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    assert space.count == 6 and space.experts == 4 and space.k == 2
    assert len(space.index) == 16
    assert {m: int(space.index[m]) for m in (3, 5, 9, 6, 10, 12)} == {3: 0, 5: 1, 9: 2, 6: 3, 10: 4, 12: 5}
    assert (space.index[[0, 1, 2, 4, 7, 8, 15]] == -1).all()


@pytest.mark.parametrize("experts,k", [(1, 1), (5, 1), (5, 5), (8, 3), (16, 2), (12, 4)])
def test_the_number_of_sets_is_the_binomial_coefficient(experts, k):
    space = sg.SetSpace.make(experts, k)
    assert space.count == math.comb(experts, k)
    assert len({tuple(s) for s in space.sets.tolist()}) == space.count
    assert (np.diff(space.sets, axis=1) > 0).all()


def test_encoding_inverts_the_listing_and_ignores_the_order_inside_a_route():
    space = sg.SetSpace.make(6, 3)
    assert space.encode(space.sets).tolist() == list(range(space.count))
    assert space.encode(space.sets[:, ::-1]).tolist() == list(range(space.count))


def test_encoding_keeps_the_shape_of_what_it_is_given():
    space = sg.SetSpace.make(4, 2)
    routes = np.array([[[0, 1], [2, 3]], [[3, 1], [0, 2]], [[2, 1], [0, 3]]])
    assert space.encode(routes).tolist() == [[0, 5], [4, 1], [3, 2]]
    assert space.encode(np.array([0, 3])).item() == 2


@pytest.mark.parametrize("bad", [[[1, 1]], [[0, 1, 2]], [[0]]])
def test_encoding_refuses_a_route_that_is_not_k_distinct_experts(bad):
    with pytest.raises(ValueError):
        sg.SetSpace.make(4, 2).encode(np.array(bad))


@pytest.mark.parametrize("experts,k", [(3, 4), (4, 0), (0, 1), (21, 2), (4, -1)])
def test_a_set_space_with_impossible_sizes_is_refused(experts, k):
    with pytest.raises(ValueError):
        sg.SetSpace.make(experts, k)


def test_twenty_experts_is_the_largest_space_and_twenty_one_is_refused():
    space = sg.SetSpace.make(20, 1)
    assert space.count == 20 and space.encode(np.array([[7]])).item() == 7
    with pytest.raises(ValueError):
        sg.SetSpace.make(21, 1)


# --- signatures ---------------------------------------------------------------------------------------------------


def test_signatures_are_dealt_to_experts_in_turn():
    sigs = sg.Signatures.make(16, 4, 5, seed=3)
    assert sigs.vectors.shape == (16, 4) and sigs.classes == 5
    for e in range(16):
        assert np.array_equal(sigs.vectors[e], sigs.vectors[e % 5])
    assert not np.array_equal(sigs.vectors[0], sigs.vectors[1])


def test_the_first_experts_get_the_rows_of_the_seeded_draw_in_order():
    base = make_rng(3).normal(0.0, 1.0, (5, 4))
    sigs = sg.Signatures.make(16, 4, 5, seed=3)
    assert np.array_equal(sigs.vectors[:5], base) and np.array_equal(sigs.vectors[5:10], base)


def test_signatures_are_reproducible_and_depend_on_the_seed():
    a = sg.Signatures.make(8, 3, 8, seed=1)
    assert np.array_equal(a.vectors, sg.Signatures.make(8, 3, 8, seed=1).vectors)
    assert not np.array_equal(a.vectors, sg.Signatures.make(8, 3, 8, seed=2).vectors)


def test_the_spread_scales_every_signature():
    one = sg.Signatures.make(8, 3, 8, seed=1).vectors
    assert np.allclose(sg.Signatures.make(8, 3, 8, seed=1, spread=3.0).vectors, 3.0 * one)
    assert abs(sg.Signatures.make(300, 50, 300, seed=1, spread=2.0).vectors.std() - 2.0) < 0.05


@pytest.mark.parametrize(
    "kwargs",
    [{"classes": 0}, {"classes": 9}, {"features": 0}, {"spread": 0.0}, {"spread": -1.0}],
)
def test_impossible_signature_parameters_are_refused(kwargs):
    args = {"experts": 8, "features": 3, "classes": 4, "seed": 1, **kwargs}
    with pytest.raises(ValueError):
        sg.Signatures.make(**args)


def test_classes_counts_distinct_rows_and_treats_rounding_error_as_equal():
    assert sg.Signatures(np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])).classes == 2
    assert sg.Signatures(np.array([[1.0], [1.0 + 1e-12]])).classes == 1
    assert sg.Signatures(np.array([[1.0], [1.0 + 1e-6]])).classes == 2
    assert sg.Signatures(np.array([[0.3], [0.4]])).classes == 2


def test_the_signature_of_a_set_is_the_sum_of_its_experts():
    space = sg.SetSpace.make(4, 2)
    powers = sg.Signatures(np.array([[1.0], [2.0], [4.0], [8.0]]))
    assert powers.of_sets(space)[:, 0].tolist() == [3, 5, 9, 6, 10, 12]  # each sum is the set's own bitmask
    two = sg.Signatures(np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [0.5, -1.0]]))
    assert np.allclose(two.of_sets(space)[3], [2.0, 3.0]) and two.of_sets(space).shape == (6, 2)


def test_signatures_with_equal_experts_give_sets_with_equal_signatures():
    sigs = sg.Signatures(np.array([[1.0], [1.0], [5.0], [5.0]]))
    assert sg.groups_of(sigs.of_sets(sg.SetSpace.make(4, 2))).tolist() == [0, 1, 1, 1, 1, 2]


@pytest.mark.parametrize("classes,expected", [(1, 1), (2, 3), (4, 10), (8, 36)])
def test_the_number_of_distinct_set_signatures_is_c_times_c_plus_one_over_two(classes, expected):
    # two experts of one class can run together, so the "same class twice" signatures count as well
    space = sg.SetSpace.make(16, 2)
    sigs = sg.Signatures.make(16, 4, classes, seed=5)
    assert sg.distinct_signatures(sigs.of_sets(space)) == expected == classes * (classes + 1) // 2


def test_with_one_expert_per_class_no_class_can_run_twice_and_every_set_has_its_own_signature():
    space = sg.SetSpace.make(16, 2)
    assert sg.distinct_signatures(sg.Signatures.make(16, 4, 16, seed=5).of_sets(space)) == 120 == space.count


def test_drift_of_zero_changes_nothing_and_drift_has_the_stated_scale():
    sigs = sg.Signatures(np.full((60, 200), 3.0))  # rms entry 3
    assert np.array_equal(sigs.drifted(0.0, make_rng(1)).vectors, sigs.vectors)
    moved = sigs.drifted(0.1, make_rng(1)).vectors - sigs.vectors
    assert moved.std() == pytest.approx(0.3, rel=0.03) and abs(moved.mean()) < 0.01


def test_drift_is_measured_against_the_rms_and_not_the_spread():
    flat = sg.Signatures(np.full((60, 200), 3.0))
    spread = sg.Signatures(np.tile([-3.0, 3.0], (60, 100)))
    a = (flat.drifted(0.2, make_rng(2)).vectors - flat.vectors).std()
    b = (spread.drifted(0.2, make_rng(2)).vectors - spread.vectors).std()
    assert a == pytest.approx(0.6, rel=0.03) and b == pytest.approx(0.6, rel=0.03)


def test_drift_is_reproducible():
    sigs = sg.Signatures.make(8, 3, 8, seed=1)
    assert np.array_equal(sigs.drifted(0.3, make_rng(4)).vectors, sigs.drifted(0.3, make_rng(4)).vectors)


# --- grouping, measuring, decoding --------------------------------------------------------------------------------


def test_groups_number_equal_signatures_alike_in_sorted_order():
    sigs = np.array([[1.0], [2.0], [1.0], [3.0], [2.0]])
    assert sg.groups_of(sigs).tolist() == [0, 1, 0, 2, 1]
    near = np.array([[1.0], [1.0 + 1e-12], [1.0 + 1e-6]])
    assert sg.groups_of(near).tolist() == [0, 0, 1]


def test_groups_of_two_dimensional_signatures_compare_whole_rows():
    sigs = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 2.0]])
    assert sg.groups_of(sigs).tolist() == [0, 1, 0]


def test_a_noise_free_reading_is_the_signature_and_uses_no_randomness():
    sigs = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    rng = make_rng(5)
    reading = sg.measure(sigs, np.array([2, 0, 0, 1]), 0.0, rng)
    assert np.array_equal(reading, sigs[[2, 0, 0, 1]])
    assert rng.random() == make_rng(5).random()


def test_a_noisy_reading_has_the_signature_as_its_mean_and_sigma_as_its_spread():
    sigs = np.array([[1.0, -2.0], [5.0, 0.0]])
    reading = sg.measure(sigs, np.zeros(40000, dtype=int), 0.7, make_rng(6))
    assert reading.shape == (40000, 2)
    assert np.allclose(reading.mean(axis=0), sigs[0], atol=0.02) and np.allclose(reading.std(axis=0), 0.7, rtol=0.02)
    assert np.array_equal(reading, sg.measure(sigs, np.zeros(40000, dtype=int), 0.7, make_rng(6)))


def test_decoding_names_the_set_with_the_nearest_signature():
    sigs = np.array([[1.0], [2.0], [3.0]])
    assert sg.decode(np.array([[0.9], [3.2], [2.4], [1.6]]), sigs).tolist() == [0, 2, 1, 1]
    two = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    assert sg.decode(np.array([[1.0, 2.0], [3.0, 0.5], [0.4, 0.4]]), two).tolist() == [2, 1, 0]


def test_decoding_a_tie_picks_the_lowest_index():
    sigs = np.array([[1.0], [1.0], [2.0]])
    assert sg.decode(np.array([[1.0]]), sigs).tolist() == [0]


def test_a_noise_free_reading_decodes_to_its_own_set_when_signatures_differ():
    space = sg.SetSpace.make(8, 2)
    set_sigs = sg.Signatures.make(8, 4, 8, seed=2).of_sets(space)
    assert sg.decode(set_sigs, set_sigs).tolist() == list(range(space.count))


# --- pairwise error probabilities -------------------------------------------------------------------------------


def test_the_pair_error_of_two_signatures_is_the_gaussian_tail_at_half_their_distance():
    sigs = np.array([[0.0, 0.0], [3.0, 4.0]])
    pair = sg.pair_errors(sigs, 2.5)
    assert pair[0, 1] == pytest.approx(q(1.0), rel=1e-9) and pair[0, 1] == pytest.approx(0.158655, rel=1e-5)
    assert np.array_equal(pair, pair.T) and pair[0, 0] == pair[1, 1] == 0.0


def test_pair_errors_fall_with_distance_and_rise_with_noise():
    sigs = np.array([[0.0], [1.0], [3.0]])
    pair = sg.pair_errors(sigs, 1.0)
    assert pair[0, 1] > pair[0, 2] > 0 and pair[0, 1] == pytest.approx(q(0.5))
    assert (sg.pair_errors(sigs, 2.0) >= pair).all() and (sg.pair_errors(sigs, 2.0) > pair)[0, 1]


def test_equal_signatures_are_confused_half_the_time_whatever_the_noise():
    sigs = np.array([[1.0], [1.0], [4.0]])
    for sigma in (0.0, 0.3, 5.0):
        pair = sg.pair_errors(sigs, sigma)
        assert pair[0, 1] == pair[1, 0] == 0.5 and pair[0, 0] == pair[1, 1] == 0.0


def test_with_no_noise_only_equal_signatures_are_confused():
    sigs = np.array([[1.0], [1.0], [4.0], [4.0 + 1e-6]])
    pair = sg.pair_errors(sigs, 0.0)
    assert pair[0, 1] == 0.5 and pair[2, 3] == 0.0 and pair[0, 2] == 0.0
    assert np.array_equal(sg.pair_errors(sigs, -1.0), pair)


def test_pair_errors_with_no_noise_raise_no_division_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pair = sg.pair_errors(np.array([[0.0], [1.0], [2.0]]), 0.0)
    assert not pair.any()


def test_pair_errors_treat_signatures_closer_than_rounding_as_equal():
    assert sg.pair_errors(np.array([[1.0], [1.0 + 1e-12]]), 0.5)[0, 1] == 0.5
    assert sg.pair_errors(np.array([[1.0], [1.0 + 1e-6]]), 0.5)[0, 1] < 0.5


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10_000), st.floats(0.05, 3.0))
def test_pair_errors_are_symmetric_probabilities_at_most_one_half(seed, sigma):
    sigs = make_rng(seed).normal(size=(7, 3))
    pair = sg.pair_errors(sigs, sigma)
    assert np.allclose(pair, pair.T) and (pair >= 0).all() and (pair <= 0.5).all() and (np.diag(pair) == 0).all()


# --- bounds on set decoding ---------------------------------------------------------------------------------------


def test_bounds_for_two_sets_are_the_pair_error_on_both_sides():
    low, high = sg.set_error_bounds(np.array([[0.0], [2.0]]), 1.0)
    assert low == pytest.approx(q(1.0)) and high == pytest.approx(q(1.0))


def test_bounds_for_three_equally_spaced_sets_by_hand():
    low, high = sg.set_error_bounds(np.array([[0.0], [2.0], [4.0]]), 1.0)
    outer = q(1.0) + q(2.0)
    assert low == pytest.approx(q(1.0))
    assert high == pytest.approx((outer + (2 * q(1.0)) + outer) / 3)


def test_the_upper_bound_is_capped_at_one():
    low, high = sg.set_error_bounds(np.zeros((10, 2)), 1.0)
    assert low == 0.5 and high == 1.0


def test_with_no_noise_and_distinct_signatures_both_bounds_are_zero():
    low, high = sg.set_error_bounds(np.array([[0.0], [1.0], [5.0]]), 0.0)
    assert low == 0.0 and high == 0.0


def test_two_equal_signatures_among_distinct_ones_cost_a_third_of_the_accuracy_at_least():
    low, high = sg.set_error_bounds(np.array([[0.0], [0.0], [100.0]]), 0.0)
    assert low == pytest.approx(2 / 3 * 0.5) and high == pytest.approx(2 / 3 * 0.5)


@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.6, 1.0, 2.0])
def test_simulated_set_accuracy_sits_between_the_two_bounds(sigma):
    set_sigs = sg.Signatures.make(8, 4, 8, seed=11).of_sets(sg.SetSpace.make(8, 2))
    low, high = sg.set_error_bounds(set_sigs, sigma)
    est = sg.set_accuracy(set_sigs, sigma, 6000, make_rng(int(sigma * 100)))
    assert est.hi >= 1 - high - 1e-9 and est.lo <= 1 - low + 1e-9
    assert low <= high


def test_both_bounds_get_worse_as_the_noise_grows():
    set_sigs = sg.Signatures.make(8, 4, 8, seed=11).of_sets(sg.SetSpace.make(8, 2))
    bounds = [sg.set_error_bounds(set_sigs, s) for s in (0.1, 0.3, 1.0, 3.0)]
    assert [b[0] for b in bounds] == sorted(b[0] for b in bounds) and [b[1] for b in bounds] == sorted(b[1] for b in bounds)


def test_the_noise_free_ceiling_is_distinct_signatures_over_sets():
    assert sg.noise_free_ceiling(np.array([[1.0], [2.0], [1.0], [3.0]])) == 0.75
    assert sg.noise_free_ceiling(np.array([[1.0], [2.0]])) == 1.0
    assert sg.distinct_signatures(np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 1.0]])) == 2
    assert sg.distinct_signatures(np.array([[1.0], [1.0 + 1e-12], [1.0 + 1e-6]])) == 2


# --- simulated set accuracy -----------------------------------------------------------------------------------------


def test_set_accuracy_without_noise_is_one_when_signatures_differ():
    set_sigs = sg.Signatures.make(8, 4, 8, seed=2).of_sets(sg.SetSpace.make(8, 2))
    est = sg.set_accuracy(set_sigs, 0.0, 500, make_rng(1))
    assert est.p == 1.0 and est.trials == 500 and est.hi == pytest.approx(1.0) and est.lo < 1.0


def test_set_accuracy_without_noise_is_the_ceiling_when_signatures_repeat():
    set_sigs = sg.Signatures.make(8, 4, 3, seed=2).of_sets(sg.SetSpace.make(8, 2))
    ceiling = sg.noise_free_ceiling(set_sigs)
    est = sg.set_accuracy(set_sigs, 0.0, 20000, make_rng(1))
    assert ceiling == 6 / 28 and abs(est.p - ceiling) < 0.01


def test_set_accuracy_of_two_sets_matches_the_gaussian_tail():
    est = sg.set_accuracy(np.array([[0.0], [2.0]]), 1.0, 40000, make_rng(2))
    assert est.p == pytest.approx(1 - q(1.0), abs=0.006) and est.lo <= est.p <= est.hi


def test_a_hopelessly_noisy_sensor_gives_chance_accuracy():
    set_sigs = sg.Signatures.make(8, 4, 8, seed=2).of_sets(sg.SetSpace.make(8, 2))
    assert abs(sg.set_accuracy(set_sigs, 1000.0, 20000, make_rng(3)).p - 1 / 28) < 0.01


def test_set_accuracy_is_reproducible_and_rises_with_the_number_of_features():
    space = sg.SetSpace.make(8, 2)
    a = sg.set_accuracy(sg.Signatures.make(8, 4, 8, seed=2).of_sets(space), 0.8, 3000, make_rng(7))
    b = sg.set_accuracy(sg.Signatures.make(8, 4, 8, seed=2).of_sets(space), 0.8, 3000, make_rng(7))
    assert a == b
    accs = [sg.set_accuracy(sg.Signatures.make(8, f, 8, seed=2).of_sets(space), 0.8, 4000, make_rng(f)).p for f in (1, 4, 16)]
    assert accs[0] < accs[1] < accs[2]


def test_set_accuracy_rises_with_the_number_of_distinguishable_experts_at_a_fixed_noise():
    space = sg.SetSpace.make(8, 2)
    accs = [sg.set_accuracy(sg.Signatures.make(8, 4, c, seed=2).of_sets(space), 0.0, 4000, make_rng(c)).p for c in (2, 4, 8)]
    assert accs[0] < accs[1] < accs[2] == 1.0


@pytest.mark.parametrize("sets,bits", [(1, 0.0), (2, 1.0), (8, 3.0), (120, math.log2(120))])
def test_bits_per_slot_is_the_log_of_the_number_of_sets(sets, bits):
    assert sg.bits_per_slot(sets) == pytest.approx(bits)
