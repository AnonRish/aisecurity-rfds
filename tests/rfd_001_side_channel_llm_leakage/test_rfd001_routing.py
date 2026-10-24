import itertools
import math

import numpy as np
import pytest

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.rfd_001_side_channel_llm_leakage import leakage as lk
from aisecurity_rfds.rfd_001_side_channel_llm_leakage import routing as rt
from aisecurity_rfds.rfd_001_side_channel_llm_leakage.leakage import Channel
from aisecurity_rfds.rfd_001_side_channel_llm_leakage.routing import MoeRouter


def hand_router(probs, levels) -> MoeRouter:
    probs = np.asarray(probs, dtype=float)
    return MoeRouter(sets=tuple((i,) for i in range(probs.shape[1])), probs=probs, levels=np.asarray(levels, dtype=float))


def sensor(noise: float, rate_hz: float = 50.0, gain: float = 1.0) -> Channel:
    return Channel("test sensor", gain=gain, noise_op=0.0, noise_out=noise, rate_hz=rate_hz)


def test_a_random_router_has_every_pair_and_normalised_rows():
    router = MoeRouter.make(seed=4)
    assert router.sets == tuple(itertools.combinations(range(8), 2)) and len(router.sets) == math.comb(8, 2) == 28
    assert router.probs.shape == (4, 28) and np.allclose(router.probs.sum(axis=1), 1.0)
    assert router.categories == 4 and router.levels.shape == (28,)


def test_an_expert_sets_level_is_the_sum_of_its_experts_shifts():
    router = MoeRouter.make(seed=5, n_experts=5, k=3, expert_spread=0.2)
    shifts = make_rng(5).normal(0.0, 0.2, size=5)
    for level, members in zip(router.levels, router.sets):
        assert level == pytest.approx(sum(shifts[e] for e in members))
    assert len(router.sets[0]) == 3 and not MoeRouter.make(seed=5, expert_spread=0.0).levels.any()


def test_a_router_is_reproducible_from_its_seed():
    a, b, c = MoeRouter.make(seed=1), MoeRouter.make(seed=1), MoeRouter.make(seed=2)
    assert np.array_equal(a.probs, b.probs) and np.array_equal(a.levels, b.levels)
    assert not np.array_equal(a.probs, c.probs)


def test_draw_follows_the_routers_probabilities():
    router = hand_router([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]], [0.0, 1.0, 2.0])
    drawn = router.draw(np.array([0] * 4000 + [1] * 4000), 5, make_rng(30))
    assert drawn.shape == (8000, 5) and drawn.min() >= 0 and drawn.max() <= 2
    for category, expected in ((0, [0.7, 0.2, 0.1]), (1, [0.1, 0.1, 0.8])):
        rows = drawn[4000 * category : 4000 * (category + 1)]
        assert np.allclose(np.bincount(rows.ravel(), minlength=3) / rows.size, expected, atol=0.02)


def test_mutual_information_is_zero_when_every_category_uses_experts_the_same_way():
    assert rt.mutual_information(hand_router([[0.5, 0.3, 0.2]] * 4, [0, 1, 2])) == pytest.approx(0.0, abs=1e-12)


def test_mutual_information_is_log2_of_the_categories_when_their_experts_are_disjoint():
    router = hand_router(np.eye(4), [0.0, 1.0, 2.0, 3.0])
    assert rt.mutual_information(router) == pytest.approx(2.0)
    assert rt.noise_free_category_accuracy(router) == pytest.approx(1.0)


def test_mutual_information_and_bayes_accuracy_at_a_hand_computed_point():
    router = hand_router([[0.75, 0.25], [0.25, 0.75]], [0.0, 1.0])
    h = -(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75))  # entropy of a 0.25 coin: 0.8113 bits
    assert rt.mutual_information(router) == pytest.approx(1 - h)
    assert rt.noise_free_category_accuracy(router) == pytest.approx(0.75)


def test_mutual_information_is_between_zero_and_log2_of_the_categories_for_random_routers():
    for seed in range(5):
        mi = rt.mutual_information(MoeRouter.make(seed=seed))
        assert 0.0 < mi < math.log2(4)
    assert rt.mutual_information(MoeRouter.make(concentration=0.05)) > rt.mutual_information(MoeRouter.make(concentration=5.0))


def test_a_dense_model_is_a_router_with_one_choice_and_leaks_nothing():
    dense = hand_router([[1.0]] * 4, [0.0])
    assert rt.mutual_information(dense) == pytest.approx(0.0, abs=1e-12)
    est = rt.category_accuracy(dense, sensor(1e-9), 16, 2000, make_rng(31))
    assert 0.20 <= est.p <= 0.30  # chance for four topics


def test_a_slot_that_the_sensor_cannot_resolve_is_refused():
    with pytest.raises(ValueError, match="cannot resolve"):
        rt.slot_noise(sensor(0.01, rate_hz=10.0))
    with pytest.raises(ValueError):
        rt.category_accuracy(hand_router([[1.0]], [0.0]), sensor(0.01, rate_hz=10.0), 4, 10, make_rng(1))
    with pytest.raises(ValueError):
        rt.slot_noise(sensor(0.01, rate_hz=49.0))  # 0.98 samples per token
    assert rt.slot_noise(sensor(0.01, rate_hz=50.0)) == pytest.approx(0.01)  # exactly one sample per token is enough


def test_slot_noise_averages_the_samples_taken_during_one_token():
    assert rt.slot_noise(sensor(0.02, rate_hz=50.0)) == pytest.approx(0.02)  # one sample per token
    assert rt.slot_noise(sensor(0.02, rate_hz=1000.0)) == pytest.approx(0.02 / math.sqrt(20))
    assert rt.slot_noise(lk.SCOPE) < rt.slot_noise(lk.SOFTWARE) == pytest.approx(0.01)


def test_with_no_noise_the_attacker_reaches_the_noise_free_bayes_accuracy():
    router = MoeRouter.make(seed=7)
    est = rt.category_accuracy(router, sensor(1e-9), 1, 6000, make_rng(32))
    assert abs(est.p - rt.noise_free_category_accuracy(router)) < 0.03


def test_more_tokens_give_more_accuracy():
    router = MoeRouter.make()
    few = rt.category_accuracy(router, lk.SOFTWARE, 1, 800, make_rng(33)).p
    many = rt.category_accuracy(router, lk.SOFTWARE, 64, 800, make_rng(34)).p
    assert many > few + 0.2


def test_accuracy_is_chance_when_every_expert_set_gives_the_same_level():
    flat = hand_router(MoeRouter.make().probs, np.zeros(28))
    assert 0.20 <= rt.category_accuracy(flat, sensor(0.01), 8, 2000, make_rng(35)).p <= 0.30


def test_a_noisier_sensor_is_worse():
    router = MoeRouter.make()
    clean = rt.category_accuracy(router, sensor(0.003), 8, 800, make_rng(36)).p
    noisy = rt.category_accuracy(router, sensor(0.1), 8, 800, make_rng(37)).p
    assert clean > noisy + 0.2


def test_a_channel_gain_scales_the_levels_and_therefore_the_accuracy():
    router = MoeRouter.make()
    weak = rt.category_accuracy(router, sensor(0.01, gain=0.1), 8, 800, make_rng(38)).p
    strong = rt.category_accuracy(router, sensor(0.01, gain=1.0), 8, 800, make_rng(39)).p
    assert strong > weak + 0.15


def test_the_set_error_bounds_sandwich_the_simulated_set_accuracy():
    router = MoeRouter.make()
    for ch in (lk.ACOUSTIC, lk.SOFTWARE):
        lower, upper = rt.set_identification_error(router, ch)
        est = rt.set_accuracy(router, ch, 4000, make_rng(40))
        error, slack = 1 - est.p, 3 * math.sqrt(est.p * (1 - est.p) / 4000)
        assert lower - slack <= error <= upper + slack, ch.name
    assert rt.set_identification_error(router, lk.SCOPE)[0] < rt.set_identification_error(router, lk.SOFTWARE)[0]
