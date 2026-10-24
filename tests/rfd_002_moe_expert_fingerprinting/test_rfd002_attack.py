import math
from dataclasses import replace

import numpy as np
import pytest

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.common.stats import wilson_interval
from aisecurity_rfds.rfd_002_moe_expert_fingerprinting import attack as at
from aisecurity_rfds.rfd_002_moe_expert_fingerprinting import moe
from aisecurity_rfds.rfd_002_moe_expert_fingerprinting import signatures as sg

CFG = moe.Config(vocab=10, dim=6, hidden=6, layers=2, experts=6, k=2, ctx=0.0)
SPACE = sg.SetSpace.make(6, 2)
V = CFG.vocab


def model(seed=5):
    """A random model in which tokens 0 and 1 are the same input, so they always take the same route."""
    p = moe.init_params(CFG, seed)
    p["emb"][1] = p["emb"][0]
    return p


def exact_sigs(seed=3):
    return sg.Signatures.make(6, 4, 6, seed).of_sets(SPACE)


class Zeros:
    """A stand-in generator whose sensor noise is exactly zero."""

    def normal(self, loc, scale, size):
        return np.zeros(size)


class Shift:
    """A stand-in generator whose sensor noise is always +0.9."""

    def normal(self, loc, scale, size):
        return np.full(size, 0.9)


# --- the profile -------------------------------------------------------------------------------------------------


def test_the_profile_of_a_context_free_model_puts_nearly_all_mass_on_each_tokens_route():
    p = model()
    profile = at.build_profile(CFG, p, SPACE, make_rng(1), per_token=20, alpha=0.5)
    assert profile.table.shape == (CFG.layers, SPACE.count, V)
    assert np.allclose(profile.table.sum(axis=1), 1.0)
    routes = moe.routes_of(CFG, p, np.arange(V), np.zeros((V, CFG.dim)))
    which = SPACE.encode(routes)
    hit, miss = 20.5 / 27.5, 0.5 / 27.5
    for layer in range(CFG.layers):
        for t in range(V):
            column = profile.table[layer, :, t]
            assert column[which[t, layer]] == pytest.approx(hit)
            assert np.delete(column, which[t, layer]) == pytest.approx(np.full(SPACE.count - 1, miss))


def test_without_context_or_noise_the_profile_does_not_depend_on_the_random_draws():
    a = at.build_profile(CFG, model(), SPACE, make_rng(1), per_token=10)
    b = at.build_profile(CFG, model(), SPACE, make_rng(2), per_token=10)
    assert np.array_equal(a.table, b.table)


def test_with_context_the_profile_spreads_over_routes_and_depends_on_the_draws():
    cfg = replace(CFG, ctx=1.5)
    a = at.build_profile(cfg, model(), SPACE, make_rng(1), per_token=60)
    b = at.build_profile(cfg, model(), SPACE, make_rng(2), per_token=60)
    assert np.allclose(a.table.sum(axis=1), 1.0) and not np.array_equal(a.table, b.table)
    assert a.table.max(axis=1).mean() < at.build_profile(CFG, model(), SPACE, make_rng(1), per_token=60).table.max(axis=1).mean()


def test_routing_noise_flattens_the_profile():
    sharp = at.build_profile(CFG, model(), SPACE, make_rng(1), per_token=100, tau=0.0)
    flat = at.build_profile(CFG, model(), SPACE, make_rng(1), per_token=100, tau=2.0)
    assert np.allclose(flat.table.sum(axis=1), 1.0)
    assert flat.table.max(axis=1).mean() < sharp.table.max(axis=1).mean() - 0.3


def test_more_smoothing_flattens_the_profile_towards_uniform():
    soft = at.build_profile(CFG, model(), SPACE, make_rng(1), per_token=10, alpha=1e9)
    assert np.allclose(soft.table, 1 / SPACE.count, atol=1e-6)
    low = at.build_profile(CFG, model(), SPACE, make_rng(1), per_token=10, alpha=0.5)
    mid = at.build_profile(CFG, model(), SPACE, make_rng(1), per_token=10, alpha=5.0)
    assert low.table.max() > mid.table.max() > soft.table.max()


@pytest.mark.parametrize("kwargs", [{"per_token": 0}, {"per_token": -3}, {"alpha": 0.0}, {"alpha": -1.0}])
def test_a_profile_needs_runs_and_positive_smoothing(kwargs):
    with pytest.raises(ValueError):
        at.build_profile(CFG, model(), SPACE, make_rng(1), **kwargs)


def test_a_profile_with_one_run_per_token_is_still_a_distribution():
    profile = at.build_profile(CFG, model(), SPACE, make_rng(1), per_token=1, alpha=0.1)
    assert np.allclose(profile.table.sum(axis=1), 1.0) and profile.table.min() == pytest.approx(0.1 / (1 + 0.1 * SPACE.count))


# --- the likelihood of a reading ------------------------------------------------------------------------------------


def test_with_an_exact_sensor_every_set_with_the_same_signature_is_equally_likely():
    sigs = np.array([[1.0], [1.0], [5.0], [7.0]])
    lik = at.set_likelihood(sigs, np.array([[0, 3], [2, 1]]), 0.0, Zeros())
    assert lik.shape == (2, 2, 4)
    assert lik[0, 0].tolist() == [1, 1, 0, 0] and lik[0, 1].tolist() == [0, 0, 0, 1]
    assert lik[1, 0].tolist() == [0, 0, 1, 0] and lik[1, 1].tolist() == [1, 1, 0, 0]


def test_an_exact_sensor_uses_no_randomness():
    rng = make_rng(8)
    at.set_likelihood(exact_sigs(), np.array([[0, 1, 2]]), 0.0, rng)
    assert rng.random() == make_rng(8).random()


def test_with_a_noisy_sensor_the_likelihood_is_a_gaussian_kernel_around_the_reading():
    sigs = np.array([[0.0], [2.0], [4.0]])
    lik = at.set_likelihood(sigs, np.array([[0, 2]]), 1.0, Zeros())
    assert np.allclose(lik[0, 0], [1.0, math.exp(-2.0), math.exp(-8.0)])
    assert np.allclose(lik[0, 1], [math.exp(-8.0), math.exp(-2.0), 1.0])


def test_noise_is_added_to_the_true_signature_and_the_likelihood_is_scaled_by_sigma():
    sigs = np.array([[0.0], [2.0], [4.0]])
    lik = at.set_likelihood(sigs, np.array([[0]]), 1.0, Shift())  # the reading is 0.9
    assert np.allclose(lik[0, 0], [1.0, math.exp(-0.2), math.exp(-4.4)])
    wide = at.set_likelihood(sigs, np.array([[0]]), 2.0, Shift())  # the reading is still 0.9; sigma now 2
    assert np.allclose(wide[0, 0], [1.0, math.exp(-0.4 / 8), math.exp(-8.8 / 8)])


def test_the_noisy_likelihood_never_exceeds_one_and_peaks_at_the_nearest_signature():
    sigs = sg.Signatures.make(6, 3, 6, 1).of_sets(SPACE)
    lik = at.set_likelihood(sigs, SPACE.encode(SPACE.sets)[None, :].repeat(3, axis=0), 0.4, make_rng(2))
    assert lik.max() == 1.0 and (lik.max(axis=-1) == 1.0).all() and lik.min() >= 0.0


def test_the_noisy_likelihood_is_reproducible():
    args = (exact_sigs(), np.array([[0, 5, 9]]), 0.3)
    assert np.array_equal(at.set_likelihood(*args, make_rng(4)), at.set_likelihood(*args, make_rng(4)))
    assert not np.array_equal(at.set_likelihood(*args, make_rng(4)), at.set_likelihood(*args, make_rng(5)))


def test_a_target_with_other_signatures_is_read_with_its_own_signatures_but_decoded_with_the_attackers():
    mine = np.array([[0.0], [2.0], [4.0]])
    theirs = mine + 0.5
    lik = at.set_likelihood(mine, np.array([[1]]), 1.0, Zeros(), true_sigs=theirs)  # the reading is 2.5
    assert np.allclose(lik[0, 0], [math.exp(-(6.25 - 0.25) / 2), 1.0, math.exp(-(2.25 - 0.25) / 2)])


def test_signature_drift_needs_a_noisy_sensor():
    sigs = np.array([[0.0], [2.0]])
    with pytest.raises(ValueError):
        at.set_likelihood(sigs, np.array([[0]]), 0.0, Zeros(), true_sigs=sigs + 0.1)


# --- scores ------------------------------------------------------------------------------------------------------------


TABLE = at.Profile(
    np.array(
        [
            [[0.5, 0.1], [0.3, 0.2], [0.2, 0.7]],
            [[0.1, 0.6], [0.8, 0.3], [0.1, 0.1]],
        ]
    )
)


def test_scores_add_the_log_likelihood_of_each_block():
    lik = np.array([[[1, 0, 0], [0, 1, 0]], [[1, 1, 0], [0, 0, 1]]], dtype=float)
    scores = at.token_scores(TABLE, lik)
    assert scores.shape == (2, 2)
    assert np.allclose(scores[0], [math.log(0.5) + math.log(0.8), math.log(0.1) + math.log(0.3)])
    assert np.allclose(scores[1], [math.log(0.8) + math.log(0.1), math.log(0.3) + math.log(0.1)])


def test_only_the_blocks_the_sensor_saw_count():
    lik = np.array([[[0, 0, 1], [0, 1, 0]]], dtype=float)
    assert np.allclose(at.token_scores(TABLE, lik[:, :1])[0], np.log(TABLE.table[0, 2]))
    assert np.allclose(at.token_scores(TABLE, lik)[0], np.log(TABLE.table[0, 2]) + np.log(TABLE.table[1, 1]))


def test_an_impossible_reading_scores_a_large_negative_number_and_not_minus_infinity():
    scores = at.token_scores(TABLE, np.zeros((1, 2, 3)))
    assert np.isfinite(scores).all() and scores[0, 0] == pytest.approx(2 * math.log(1e-300))


def test_a_flat_likelihood_scores_every_token_alike():
    scores = at.token_scores(TABLE, np.ones((1, 2, 3)))
    assert np.allclose(scores, 0.0, atol=1e-12)


# --- the attack -----------------------------------------------------------------------------------------------------


def run(p, sigma, sigs=None, per_token=10, **kw):
    sigs = exact_sigs() if sigs is None else sigs
    prof = at.build_profile(CFG, p, SPACE, make_rng(1), per_token=per_token)
    kw.setdefault("tokens", np.arange(V))
    return at.attack(CFG, p, prof, SPACE, sigs, sigma, 1, make_rng(2), **kw)


@pytest.mark.parametrize("layers", [1, 2])
def test_with_no_context_and_an_exact_sensor_the_attack_reaches_the_deterministic_ceiling_exactly(layers):
    p = model()
    est = run(p, 0.0, layers=layers)
    assert est.p == at.deterministic_ceiling(CFG, p, layers) and est.trials == V


def test_the_ceiling_is_below_one_here_and_a_second_block_raises_it():
    p = model()
    first, both = at.deterministic_ceiling(CFG, p, 1), at.deterministic_ceiling(CFG, p, 2)
    assert first < both <= 0.9  # tokens 0 and 1 are the same input, so they can never be told apart


def test_the_ceiling_ignores_the_context_scale_of_the_config():
    p = model()
    assert at.deterministic_ceiling(replace(CFG, ctx=2.0), p) == at.deterministic_ceiling(CFG, p)


def test_the_ceiling_at_hand_computed_models():
    cfg = moe.Config(vocab=3, dim=3, hidden=2, layers=1, experts=3, k=1, ctx=0.0)
    p = moe.init_params(cfg, 1)
    p["Wr"][0] = np.eye(3)
    p["emb"] = np.eye(3)
    assert at.deterministic_ceiling(cfg, p) == 1.0  # every token picks its own expert
    p["emb"] = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert at.deterministic_ceiling(cfg, p) == pytest.approx(2 / 3)
    p["emb"] = np.ones((3, 3))
    assert at.deterministic_ceiling(cfg, p) == pytest.approx(1 / 3)


def test_the_ceiling_never_falls_when_the_sensor_sees_more_blocks():
    cfg = moe.Config(vocab=30, dim=8, hidden=6, layers=4, experts=6, k=2, ctx=0.0)
    p = moe.init_params(cfg, 4)
    ceilings = [at.deterministic_ceiling(cfg, p, n) for n in (1, 2, 3, 4)]
    assert ceilings == sorted(ceilings) and ceilings[0] < ceilings[-1]


def test_a_sensor_that_cannot_tell_any_experts_apart_leaves_only_a_constant_guess():
    flat = sg.Signatures.make(6, 4, 1, 3).of_sets(SPACE)
    assert run(model(), 0.0, sigs=flat).p <= 0.2


def test_repeating_an_observation_changes_nothing_when_nothing_is_random():
    p = model()
    assert run(p, 0.0, repeats=3).p == run(p, 0.0, repeats=1).p


def test_a_barely_noisy_sensor_does_as_well_as_an_exact_one():
    p = model()
    assert run(p, 1e-3).p == run(p, 0.0).p == at.deterministic_ceiling(CFG, p)


def test_a_hopelessly_noisy_sensor_leaves_close_to_chance():
    p = model()
    prof = at.build_profile(CFG, p, SPACE, make_rng(1), per_token=10)
    est = at.attack(CFG, p, prof, SPACE, exact_sigs(), 1000.0, 3000, make_rng(2))
    assert est.p < 0.2 and est.trials == 3000


def test_extra_noise_never_helps_the_attacker_on_a_context_free_model():
    p = model()
    prof = at.build_profile(CFG, p, SPACE, make_rng(1), per_token=10)
    accs = [at.attack(CFG, p, prof, SPACE, exact_sigs(), s, 3000, make_rng(2)).p for s in (0.0, 0.5, 2.0, 8.0)]
    assert accs == sorted(accs, reverse=True) and accs[0] - accs[-1] > 0.4


def test_matching_signatures_in_the_drift_argument_change_nothing():
    p, sigs = model(), exact_sigs()
    assert run(p, 0.3, sigs, true_sigs=sigs).p == run(p, 0.3, sigs).p


def test_a_target_whose_signatures_are_scrambled_defeats_the_attacker():
    p, sigs = model(), exact_sigs()
    honest = run(p, 0.05, sigs).p
    scrambled = run(p, 0.05, sigs, true_sigs=sigs[::-1].copy()).p
    assert honest > 0.7 and scrambled < 0.3


def test_the_token_list_replaces_the_number_of_trials():
    p = model()
    prof = at.build_profile(CFG, p, SPACE, make_rng(1), per_token=10)
    est = at.attack(CFG, p, prof, SPACE, exact_sigs(), 0.0, 99999, make_rng(2), tokens=np.array([3, 3, 7, 9, 2]))
    assert est.trials == 5 and est.p in {0, 0.2, 0.4, 0.6, 0.8, 1.0}


def test_without_a_token_list_tokens_are_drawn_uniformly():
    p = model()
    prof = at.build_profile(CFG, p, SPACE, make_rng(1), per_token=10)
    est = at.attack(CFG, p, prof, SPACE, exact_sigs(), 0.0, 4000, make_rng(2))
    assert est.trials == 4000 and abs(est.p - at.deterministic_ceiling(CFG, p)) < 0.03


def test_the_estimate_carries_a_wilson_interval_of_the_hit_count():
    est = run(model(), 0.0)
    lo, hi = wilson_interval(round(est.p * est.trials), est.trials)
    assert (est.lo, est.hi) == (lo, hi) and est.lo <= est.p <= est.hi


def test_the_attack_is_reproducible_given_the_generator():
    cfg = replace(CFG, ctx=1.0)
    p = model()
    prof = at.build_profile(cfg, p, SPACE, make_rng(1), per_token=30)
    runs = [at.attack(cfg, p, prof, SPACE, exact_sigs(), 0.3, 600, make_rng(s), tau=0.5, repeats=2) for s in (7, 7, 8)]
    assert runs[0] == runs[1] and runs[0] != runs[2]


@pytest.mark.parametrize("kwargs", [{"layers": 0}, {"layers": CFG.layers + 1}, {"layers": -1}, {"repeats": 0}])
def test_the_attack_refuses_impossible_views(kwargs):
    with pytest.raises(ValueError):
        run(model(), 0.0, **kwargs)


def test_the_attack_may_see_every_block_and_the_default_is_every_block():
    p = model()
    assert run(p, 0.0, layers=CFG.layers).p == run(p, 0.0).p


# --- how much two models differ --------------------------------------------------------------------------------


def test_a_model_never_differs_from_itself():
    p = model()
    assert at.route_change(CFG, p, p, make_rng(1)) == 0.0
    assert at.route_change(CFG, p, moe.copy_params(p), make_rng(1)) == 0.0


def test_a_change_to_weights_that_do_not_steer_the_first_block_leaves_its_routes_alone():
    cfg = replace(CFG, layers=1)
    p, q = moe.init_params(cfg, 5), moe.init_params(cfg, 5)
    q["Wo"] = q["Wo"] + 1.0
    q["W1"] = q["W1"] * 2.0
    assert at.route_change(cfg, p, q, make_rng(1)) == 0.0


def test_two_models_that_choose_different_experts_for_every_token_differ_everywhere():
    cfg = moe.Config(vocab=5, dim=3, hidden=2, layers=1, experts=4, k=2, ctx=0.0)
    a, b = moe.init_params(cfg, 1), moe.init_params(cfg, 1)
    a["emb"][:] = b["emb"][:] = 1.0
    a["Wr"][0] = np.zeros((3, 4))
    b["Wr"][0] = np.zeros((3, 4))
    a["Wr"][0][0, 0], a["Wr"][0][0, 1] = 5.0, 4.0  # experts 0 and 1
    b["Wr"][0][0, 0], b["Wr"][0][0, 2] = 5.0, 4.0  # experts 0 and 2: one in common is not enough
    assert at.route_change(cfg, a, b, make_rng(1)) == 1.0
    assert at.route_change(cfg, b, a, make_rng(1)) == 1.0


def test_route_change_is_a_share_between_zero_and_one_and_is_large_for_independent_models():
    a, b = moe.init_params(CFG, 1), moe.init_params(CFG, 2)
    changed = at.route_change(CFG, a, b, make_rng(1))
    assert 0.7 < changed <= 1.0 and at.route_change(CFG, a, b, make_rng(1), n=50) <= 1.0


def test_route_change_counts_a_block_once_however_many_of_its_experts_moved():
    cfg = moe.Config(vocab=5, dim=3, hidden=2, layers=1, experts=4, k=2, ctx=0.0)
    a, b = moe.init_params(cfg, 1), moe.init_params(cfg, 1)
    a["emb"][:] = b["emb"][:] = 1.0
    for model_, hot in ((a, (0, 1)), (b, (2, 3))):
        model_["Wr"][0] = np.zeros((3, 4))
        model_["Wr"][0][0, hot[0]], model_["Wr"][0][0, hot[1]] = 5.0, 4.0
    assert at.route_change(cfg, a, b, make_rng(1)) == 1.0  # both experts differ, and it still counts as one block
