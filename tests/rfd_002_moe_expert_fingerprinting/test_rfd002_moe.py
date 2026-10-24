import math
from dataclasses import replace
from functools import cache

import numpy as np
import pytest

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.rfd_002_moe_expert_fingerprinting import moe

TINY = moe.Config(vocab=8, dim=5, hidden=4, layers=2, experts=4, k=2, ctx=0.5)


def tiny_params(seed=3, cfg=TINY):
    p = moe.init_params(cfg, seed)
    p["b1"] = p["b1"] + 0.1  # a non-zero bias, so that its gradient is tested
    return p


def batch(cfg, n, seed):
    rng = make_rng(seed)
    return rng.integers(0, cfg.vocab, n), rng.integers(0, cfg.vocab, n), rng.normal(size=(n, cfg.dim))


def reference_forward(cfg, p, tokens, c):
    """The same network written with plain loops, one token and one expert at a time."""
    out = []
    for token, ctx in zip(tokens, c):
        x = p["emb"][token] + cfg.ctx * ctx
        for layer in range(cfg.layers):
            z = x @ p["Wr"][layer]
            chosen = sorted(np.argsort(-z)[: cfg.k])
            weights = np.exp(z[chosen] - z[chosen].max())
            weights /= weights.sum()
            y = np.zeros(cfg.dim)
            for w, e in zip(weights, chosen):
                h = np.maximum(x @ p["W1"][layer, e] + p["b1"][layer, e], 0.0)
                y += w * (h @ p["W2"][layer, e])
            x = x + y
        out.append(x @ p["Wo"])
    return np.array(out)


# --- helpers ------------------------------------------------------------------------------------------------------


def test_softmax_rows_sum_to_one_and_match_the_formula():
    z = np.array([[0.0, 1.0, 2.0], [5.0, 5.0, 5.0]])
    out = moe.softmax(z)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out[0], np.exp(z[0]) / np.exp(z[0]).sum())
    assert np.allclose(out[1], 1 / 3)


def test_softmax_is_stable_for_huge_inputs_and_shift_invariant():
    z = np.array([1000.0, 1001.0, 999.0])
    assert np.allclose(moe.softmax(z), moe.softmax(z - 1000.0))
    assert np.isfinite(moe.softmax(z)).all()


def test_top_k_returns_the_largest_entries_in_index_order(rng):
    scores = rng.normal(size=(50, 9))
    got = moe.top_k(scores, 3)
    assert got.shape == (50, 3)
    for row, picked in zip(scores, got):
        assert list(picked) == sorted(np.argsort(-row)[:3])


def test_top_k_of_one_is_the_argmax(rng):
    scores = rng.normal(size=(20, 6))
    assert moe.top_k(scores, 1)[:, 0].tolist() == scores.argmax(axis=1).tolist()


@pytest.mark.parametrize(
    "kwargs",
    [{"k": 0}, {"k": 17}, {"vocab": 0}, {"dim": 0}, {"hidden": 0}, {"layers": 0}, {"ctx": -0.1}],
)
def test_a_nonsensical_config_is_refused(kwargs):
    with pytest.raises(ValueError):
        moe.Config(**kwargs)


def test_init_params_has_the_documented_shapes_and_is_reproducible():
    cfg = moe.Config()
    p = moe.init_params(cfg, 7)
    assert p["emb"].shape == (64, 24) and p["Wr"].shape == (3, 24, 16) and p["Wo"].shape == (24, 64)
    assert p["W1"].shape == (3, 16, 24, 24) and p["W2"].shape == (3, 16, 24, 24) and p["b1"].shape == (3, 16, 24)
    assert not p["b1"].any()
    again = moe.init_params(cfg, 7)
    assert all(np.array_equal(p[name], again[name]) for name in p)
    assert not np.array_equal(p["emb"], moe.init_params(cfg, 8)["emb"])


def test_init_scales_are_the_documented_ones():
    cfg = moe.Config()
    p = moe.init_params(cfg, 1)
    assert abs(p["emb"].std() - 1.0) < 0.05
    assert abs(p["Wr"].std() - 1 / math.sqrt(cfg.dim)) < 0.01
    assert abs(p["W2"].std() - 0.5 / math.sqrt(cfg.hidden)) < 0.005


def test_copy_params_is_independent_of_the_original():
    p = tiny_params()
    q = moe.copy_params(p)
    q["emb"][0, 0] += 1.0
    assert q["emb"][0, 0] != p["emb"][0, 0]


# --- the synthetic language ------------------------------------------------------------------------------------


def test_chain_rows_are_distributions_and_reproducible():
    chain = moe.Chain.make(4, vocab=20)
    assert chain.trans.shape == (20, 20) and (chain.trans > 0).all()
    assert np.allclose(chain.trans.sum(axis=1), 1.0)
    assert np.array_equal(chain.trans, moe.Chain.make(4, vocab=20).trans)
    assert not np.array_equal(chain.trans, moe.Chain.make(5, vocab=20).trans)


def test_chain_entropy_at_hand_computed_points():
    uniform = moe.Chain(np.full((3, 3), 1 / 3))
    assert uniform.entropy == pytest.approx(math.log(3))
    chain = moe.Chain(np.array([[0.5, 0.5], [0.9, 0.1]]))
    h2 = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
    assert chain.entropy == pytest.approx((math.log(2) + h2) / 2)
    assert moe.Chain(np.eye(3)).entropy == 0.0  # zero entries contribute nothing


def test_next_tokens_follow_the_rows(rng):
    chain = moe.Chain(np.array([[0.7, 0.2, 0.1], [0.0, 1.0, 0.0], [0.1, 0.1, 0.8]]))
    tokens = np.repeat([0, 1, 2], 20000)
    nxt = chain.next_tokens(tokens, rng)
    for t in range(3):
        freq = np.bincount(nxt[tokens == t], minlength=3) / 20000
        assert np.allclose(freq, chain.trans[t], atol=0.01)
    assert nxt.min() >= 0 and nxt.max() <= 2


def test_next_tokens_of_a_deterministic_chain_are_its_successors(rng):
    successor = np.array([2, 0, 1])
    chain = moe.Chain(np.eye(3)[successor])
    assert chain.next_tokens(np.array([0, 1, 2, 0]), rng).tolist() == [2, 0, 1, 2]


def test_mixing_two_chains_is_linear_in_the_share():
    a, b = moe.Chain.make(1, vocab=6), moe.Chain.make(2, vocab=6)
    assert np.allclose(a.mixed_with(b, 0.0).trans, a.trans)
    assert np.allclose(a.mixed_with(b, 1.0).trans, b.trans)
    mix = a.mixed_with(b, 0.25)
    assert np.allclose(mix.trans, 0.75 * a.trans + 0.25 * b.trans) and np.allclose(mix.trans.sum(axis=1), 1.0)


# --- forward ---------------------------------------------------------------------------------------------------


def test_forward_matches_a_loop_reference_on_random_models():
    for seed in range(3):
        p = tiny_params(seed)
        tokens, _, c = batch(TINY, 10, seed)
        logits, routes, _ = moe.forward(TINY, p, tokens, c)
        assert np.allclose(logits, reference_forward(TINY, p, tokens, c), atol=1e-10)
        assert routes.shape == (10, TINY.layers, TINY.k)


def test_forward_on_a_hand_worked_model():
    cfg = moe.Config(vocab=3, dim=2, hidden=2, layers=1, experts=2, k=1, ctx=0.0)
    p = {
        "emb": np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]]),
        "Wr": np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        "W1": np.array([[np.eye(2), np.eye(2)]]),
        "b1": np.zeros((1, 2, 2)),
        "W2": np.array([[2 * np.eye(2), -np.eye(2)]]),
        "Wo": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
    }
    logits, routes, _ = moe.forward(cfg, p, np.array([0, 1, 2]), np.zeros((3, 2)))
    assert routes[:, 0, 0].tolist() == [0, 1, 0]
    assert np.allclose(logits, [[3, 0, 3], [0, 0, 0], [6, 3, 9]])


def test_routes_are_the_top_k_of_the_clean_logits_and_routes_of_matches_forward():
    p = tiny_params()
    tokens, _, c = batch(TINY, 30, 9)
    routes = moe.routes_of(TINY, p, tokens, c)
    x = p["emb"][tokens] + TINY.ctx * c
    assert np.array_equal(routes[:, 0], moe.top_k(x @ p["Wr"][0], TINY.k))
    assert np.array_equal(routes, moe.forward(TINY, p, tokens, c)[1])


def test_routing_noise_changes_the_choice_but_not_the_mixing_weights():
    cfg = moe.Config(vocab=3, dim=2, hidden=2, layers=1, experts=3, k=1, ctx=0.0)
    p = tiny_params(5, cfg)
    tokens, c = np.array([0, 1]), np.zeros((2, 2))
    clean = moe.forward(cfg, p, tokens, c)
    noise = np.zeros((2, 1, 3))
    noise[:, 0, :] = 100.0 * np.eye(3)[(clean[1][:, 0, 0] + 1) % 3]  # push every token to a different expert
    noisy = moe.forward(cfg, p, tokens, c, noise)
    assert (noisy[1] != clean[1]).all()
    cache_clean, cache_noisy = clean[2][0][0], noisy[2][0][0]
    assert np.allclose(cache_clean[3], 1.0) and np.allclose(cache_noisy[3], 1.0)  # k = 1: the gate is exactly 1
    cfg2 = replace(cfg, k=2)
    forced = np.array([[[0, 1]], [[0, 2]]])
    g = moe.forward(cfg2, p, tokens, c, forced=forced)[2][0][0][3]
    z = (p["emb"][tokens] @ p["Wr"][0])
    assert np.allclose(g[0], moe.softmax(z[0, [0, 1]])) and np.allclose(g[1], moe.softmax(z[1, [0, 2]]))


def test_with_two_active_experts_noise_forces_the_worst_expert_in_and_the_gates_still_use_the_clean_logits():
    cfg = moe.Config(vocab=3, dim=2, hidden=2, layers=1, experts=3, k=2, ctx=0.0)
    p = tiny_params(5, cfg)
    tokens, c = np.array([0, 1, 2]), np.zeros((3, 2))
    z = p["emb"][tokens] @ p["Wr"][0]
    worst = z.argmin(axis=1)
    noise = np.zeros((3, 1, 3))
    noise[np.arange(3), 0, worst] = 100.0
    _, routes, (cache, _) = moe.forward(cfg, p, tokens, c, noise)
    gates = cache[0][3]
    for i in range(3):
        assert worst[i] in routes[i, 0]
        assert np.allclose(gates[i], moe.softmax(z[i, routes[i, 0]]))
    assert not np.array_equal(routes, moe.routes_of(cfg, p, tokens, c))


def test_forced_routes_are_honoured():
    p = tiny_params()
    tokens, _, c = batch(TINY, 6, 2)
    forced = np.tile(np.array([[0, 1], [2, 3]]), (6, 1, 1))
    assert np.array_equal(moe.forward(TINY, p, tokens, c, forced=forced)[1], forced)


def test_without_experts_every_block_is_the_identity():
    p = tiny_params()
    tokens, _, c = batch(TINY, 7, 4)
    logits = moe.forward(TINY, p, tokens, c, experts_on=False)[0]
    assert np.allclose(logits, (p["emb"][tokens] + TINY.ctx * c) @ p["Wo"])


# --- gradients -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 0.05])
def test_gradients_match_finite_differences_for_every_tensor(alpha):
    p = tiny_params()
    tokens, targets, c = batch(TINY, 12, 5)
    routes = moe.routes_of(TINY, p, tokens, c)
    _, _, grads = moe.loss_and_grads(TINY, p, tokens, targets, c, None, routes, alpha)
    rng = make_rng(17)
    for name in p:
        for i in rng.choice(p[name].size, size=min(8, p[name].size), replace=False):
            up, down = moe.copy_params(p), moe.copy_params(p)
            up[name].ravel()[i] += 1e-6
            down[name].ravel()[i] -= 1e-6
            numeric = (
                moe.loss_and_grads(TINY, up, tokens, targets, c, None, routes, alpha)[0]
                - moe.loss_and_grads(TINY, down, tokens, targets, c, None, routes, alpha)[0]
            ) / 2e-6
            analytic = grads[name].ravel()[i]
            assert abs(numeric - analytic) <= 1e-6 + 1e-5 * max(abs(numeric), abs(analytic)), (name, i, numeric, analytic)


def test_cross_entropy_is_the_negative_log_probability_of_the_target():
    p = tiny_params()
    tokens, targets, c = batch(TINY, 9, 6)
    logits = moe.forward(TINY, p, tokens, c)[0]
    expected = -np.log(moe.softmax(logits)[np.arange(9), targets]).mean()
    total, ce, _ = moe.loss_and_grads(TINY, p, tokens, targets, c)
    assert ce == pytest.approx(expected) and total == pytest.approx(expected)


def test_the_load_balancing_loss_at_a_hand_computed_point():
    cfg = moe.Config(vocab=4, dim=2, hidden=2, layers=1, experts=2, k=1, ctx=0.0)
    p = tiny_params(2, cfg)
    tokens, targets, c = np.arange(4), np.arange(4), np.zeros((4, 2))
    forced = np.array([[[0]], [[0]], [[0]], [[1]]])  # three tokens to expert 0, one to expert 1
    alpha = 0.3
    total, ce, _ = moe.loss_and_grads(cfg, p, tokens, targets, c, None, forced, alpha)
    pz = moe.softmax(p["emb"][tokens] @ p["Wr"][0])
    share = np.array([0.75, 0.25])
    assert total - ce == pytest.approx(alpha * 2 * (share * pz.mean(axis=0)).sum())


def test_the_load_balancing_loss_divides_dispatches_by_tokens_times_k_when_k_is_two():
    cfg = moe.Config(vocab=4, dim=2, hidden=2, layers=1, experts=3, k=2, ctx=0.0)
    p = tiny_params(2, cfg)
    tokens, targets, c = np.arange(4), np.arange(4), np.zeros((4, 2))
    forced = np.array([[[0, 1]], [[0, 1]], [[0, 2]], [[1, 2]]])  # dispatches: expert 0 three times, 1 three times, 2 twice
    alpha = 0.4
    total, ce, _ = moe.loss_and_grads(cfg, p, tokens, targets, c, None, forced, alpha)
    pz = moe.softmax(p["emb"][tokens] @ p["Wr"][0])
    share = np.array([3, 3, 2]) / 8  # four tokens times two experts each
    assert share.sum() == pytest.approx(1.0)
    assert total - ce == pytest.approx(alpha * 3 * (share * pz.mean(axis=0)).sum())


def test_the_load_balancing_loss_is_smallest_when_the_experts_share_the_work_evenly():
    cfg = moe.Config(vocab=4, dim=2, hidden=2, layers=1, experts=2, k=1, ctx=0.0)
    p = tiny_params(2, cfg)
    p["Wr"][:] = 0.0  # every router probability is exactly one half
    tokens, targets, c = np.arange(4), np.arange(4), np.zeros((4, 2))
    even = moe.loss_and_grads(cfg, p, tokens, targets, c, None, np.array([[[0]], [[0]], [[1]], [[1]]]), 1.0)
    lopsided = moe.loss_and_grads(cfg, p, tokens, targets, c, None, np.array([[[0]], [[0]], [[0]], [[0]]]), 1.0)
    assert even[0] - even[1] == pytest.approx(1.0)  # alpha * E * sum(share * 0.5) = 1 for any shares
    assert lopsided[0] - lopsided[1] == pytest.approx(1.0)


def test_unchosen_experts_get_no_gradient_and_removed_experts_get_none_at_all():
    p = tiny_params()
    tokens, targets, c = batch(TINY, 5, 7)
    forced = np.tile(np.array([[0, 1], [0, 1]]), (5, 1, 1))
    grads = moe.loss_and_grads(TINY, p, tokens, targets, c, None, forced)[2]
    assert not grads["W1"][:, 2:].any() and not grads["W2"][:, 2:].any() and grads["W1"][:, :2].any()
    off = moe.loss_and_grads(TINY, p, tokens, targets, c, experts_on=False)[2]
    assert not off["W1"].any() and not off["W2"].any() and not off["b1"].any() and off["Wo"].any()


# --- routing noise ---------------------------------------------------------------------------------------------


def test_no_routing_noise_at_tau_zero_and_a_negative_tau_is_refused(rng):
    assert moe.routing_noise(TINY, 5, 0.0, rng) is None
    with pytest.raises(ValueError):
        moe.routing_noise(TINY, 5, -0.1, rng)


def test_routing_noise_has_the_gumbel_scale(rng):
    noise = moe.routing_noise(TINY, 4000, 2.0, rng)
    assert noise.shape == (4000, TINY.layers, TINY.experts)
    assert noise.std() == pytest.approx(2.0 * math.pi / math.sqrt(6), rel=0.03)


@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0])
def test_noisy_top_one_samples_from_the_softmax_at_temperature_tau(tau):
    logits = np.array([0.5, 0.0, -0.5])
    rng = make_rng(31)
    draws = (logits[None, :] + tau * rng.gumbel(size=(60000, 3))).argmax(axis=1)
    freq = np.bincount(draws, minlength=3) / 60000
    assert np.allclose(freq, moe.softmax(logits / tau), atol=0.01)


# --- training and evaluation ------------------------------------------------------------------------------------


@cache
def trained_tiny():
    cfg = moe.Config(vocab=12, dim=8, hidden=8, layers=2, experts=4, k=2, ctx=0.3)
    chain = moe.Chain.make(1, vocab=12, latent=3, width=8)
    p = moe.train(cfg, moe.init_params(cfg, 2), chain, 300, make_rng(3), batch=128)
    return cfg, chain, p


def test_training_lowers_the_loss():
    cfg, chain, p = trained_tiny()
    before = moe.evaluate(cfg, moe.init_params(cfg, 2), chain, 6000, make_rng(4))
    after = moe.evaluate(cfg, p, chain, 6000, make_rng(4))
    assert after < before - 0.5 and after > chain.entropy - 0.05


def test_training_is_reproducible_and_leaves_its_input_alone():
    cfg = moe.Config(vocab=8, dim=5, hidden=4, layers=2, experts=4, k=2, ctx=0.3)
    chain = moe.Chain.make(1, vocab=8, latent=3, width=6)
    start = moe.init_params(cfg, 1)
    frozen = moe.copy_params(start)
    a = moe.train(cfg, start, chain, 20, make_rng(8), batch=32)
    b = moe.train(cfg, start, chain, 20, make_rng(8), batch=32)
    assert all(np.array_equal(a[n], b[n]) for n in a)
    assert all(np.array_equal(start[n], frozen[n]) for n in start)
    assert not np.array_equal(a["emb"], start["emb"])


def test_zero_learning_rate_changes_nothing_and_training_without_experts_leaves_them_alone():
    cfg = moe.Config(vocab=8, dim=5, hidden=4, layers=2, experts=4, k=2, ctx=0.3)
    chain = moe.Chain.make(1, vocab=8, latent=3, width=6)
    start = moe.init_params(cfg, 1)
    still = moe.train(cfg, start, chain, 10, make_rng(8), lr=0.0, batch=32)
    assert all(np.array_equal(still[n], start[n]) for n in start)
    off = moe.train(cfg, start, chain, 10, make_rng(8), batch=32, experts_on=False)
    assert np.array_equal(off["W1"], start["W1"]) and np.array_equal(off["W2"], start["W2"])
    assert not np.array_equal(off["Wo"], start["Wo"])


def test_training_with_routing_noise_runs_and_differs_from_training_without():
    cfg = moe.Config(vocab=8, dim=5, hidden=4, layers=2, experts=4, k=2, ctx=0.3)
    chain = moe.Chain.make(1, vocab=8, latent=3, width=6)
    start = moe.init_params(cfg, 1)
    quiet = moe.train(cfg, start, chain, 15, make_rng(8), batch=32)
    noisy = moe.train(cfg, start, chain, 15, make_rng(8), batch=32, tau=1.0)
    assert not np.array_equal(quiet["Wr"], noisy["Wr"])


def test_a_model_with_no_output_weights_guesses_uniformly():
    cfg = moe.Config(vocab=8, dim=5, hidden=4, layers=2, experts=4, k=2, ctx=0.3)
    chain = moe.Chain.make(1, vocab=8, latent=3, width=6)
    p = moe.init_params(cfg, 1)
    p["Wo"][:] = 0.0
    assert moe.evaluate(cfg, p, chain, 500, make_rng(1)) == pytest.approx(math.log(8))


def test_evaluate_is_reproducible_and_noise_raises_the_loss_of_a_trained_model():
    cfg, chain, p = trained_tiny()
    assert moe.evaluate(cfg, p, chain, 2000, make_rng(5)) == moe.evaluate(cfg, p, chain, 2000, make_rng(5))
    clean = moe.evaluate(cfg, p, chain, 6000, make_rng(6))
    assert moe.evaluate(cfg, p, chain, 6000, make_rng(6), tau=3.0) > clean + 0.05


def test_removing_the_experts_of_a_trained_model_raises_the_loss():
    cfg, chain, p = trained_tiny()
    assert moe.evaluate(cfg, p, chain, 6000, make_rng(7), experts_on=False) > moe.evaluate(cfg, p, chain, 6000, make_rng(7)) + 0.3


def test_expert_load_counts_dispatches_per_block():
    cfg = moe.Config(vocab=4, dim=2, hidden=2, layers=2, experts=3, k=2)
    routes = np.array([[[0, 1], [1, 2]], [[0, 1], [0, 2]], [[0, 2], [0, 1]]])
    load = moe.expert_load(cfg, routes)
    assert np.allclose(load[0], [3 / 6, 2 / 6, 1 / 6])
    assert np.allclose(load[1], [2 / 6, 2 / 6, 2 / 6])
    assert np.allclose(load.sum(axis=1), 1.0)


def test_a_trained_router_spreads_the_work_over_every_expert():
    cfg, _, p = trained_tiny()
    tokens = np.repeat(np.arange(cfg.vocab), 40)
    load = moe.expert_load(cfg, moe.routes_of(cfg, p, tokens, make_rng(9).normal(size=(len(tokens), cfg.dim))))
    assert load.min() > 0.05  # an even share would be 0.25


# --- quantization ----------------------------------------------------------------------------------------------


def test_quantization_rounds_to_a_grid_with_at_most_two_to_the_bits_levels():
    p = tiny_params()
    q = moe.quantize(p, 4)
    for name in moe.WEIGHT_NAMES:
        scale = np.abs(p[name]).max() / 7
        assert len(np.unique(q[name])) <= 15
        assert np.abs(q[name] - p[name]).max() <= scale / 2 + 1e-12
        assert np.allclose(q[name] / scale, np.round(q[name] / scale))
    assert np.array_equal(q["b1"], p["b1"])


def test_quantization_is_idempotent_and_finer_with_more_bits():
    p = tiny_params()
    assert all(np.allclose(moe.quantize(moe.quantize(p, 5), 5)[n], moe.quantize(p, 5)[n]) for n in p)
    errors = [np.abs(moe.quantize(p, bits)["W1"] - p["W1"]).max() for bits in (3, 4, 6, 8)]
    assert errors == sorted(errors, reverse=True)


def test_quantization_only_touches_the_tensors_it_is_given():
    p = tiny_params()
    q = moe.quantize(p, 3, ("W1", "W2"))
    assert np.array_equal(q["Wr"], p["Wr"]) and np.array_equal(q["emb"], p["emb"]) and np.array_equal(q["Wo"], p["Wo"])
    assert not np.array_equal(q["W1"], p["W1"])


def test_quantization_needs_two_bits_and_leaves_an_all_zero_tensor_alone():
    p = tiny_params()
    with pytest.raises(ValueError):
        moe.quantize(p, 1)
    p["W2"][:] = 0.0
    assert not moe.quantize(p, 4)["W2"].any()


def test_quantizing_the_router_moves_routes_and_leaving_it_alone_keeps_the_first_block():
    p = tiny_params()
    tokens, _, c = batch(TINY, 200, 12)
    base = moe.routes_of(TINY, p, tokens, c)
    kept = moe.routes_of(TINY, moe.quantize(p, 2, ("W1", "W2", "Wo")), tokens, c)
    assert np.array_equal(base[:, 0], kept[:, 0])  # the first block sees the unquantized embedding and router
    rough = moe.routes_of(TINY, moe.quantize(p, 2), tokens, c)
    assert (base != rough).any()


def test_quantization_covers_every_weight_tensor_and_no_bias_by_default():
    p = tiny_params()
    q = moe.quantize(p, 3)
    assert set(moe.WEIGHT_NAMES) == {"emb", "Wr", "W1", "W2", "Wo"}
    for name in ("emb", "Wr", "W1", "W2", "Wo"):
        assert not np.array_equal(q[name], p[name]), name
    assert np.array_equal(q["b1"], p["b1"])


def test_quantizing_an_all_zero_tensor_returns_a_copy_and_not_the_tensor_itself():
    p = tiny_params()
    p["W2"][:] = 0.0
    q = moe.quantize(p, 4)
    q["W2"][0, 0, 0, 0] = 1.0
    assert not p["W2"].any()


def test_quantizing_does_not_change_the_model_it_is_given():
    p = tiny_params()
    frozen = moe.copy_params(p)
    moe.quantize(p, 3)
    assert all(np.array_equal(p[n], frozen[n]) for n in p)


# --- more about the model and the trainer -----------------------------------------------------------------------


def test_k_may_equal_the_number_of_experts_and_then_every_expert_runs():
    cfg = moe.Config(vocab=5, dim=3, hidden=2, layers=1, experts=3, k=3)
    p = moe.init_params(cfg, 1)
    routes = moe.routes_of(cfg, p, np.arange(5), np.zeros((5, 3)))
    assert (routes == [0, 1, 2]).all()


def test_the_language_is_a_scaled_tanh_readout_of_random_token_features():
    vocab, latent, width, scale = 9, 3, 5, 2.0
    rng = make_rng(7)
    u = rng.normal(size=(vocab, latent))
    hidden = np.tanh(u @ rng.normal(size=(latent, width)))
    expected = moe.softmax(scale * hidden @ rng.normal(size=(width, vocab)))
    got = moe.Chain.make(7, vocab=vocab, latent=latent, width=width, scale=scale)
    assert np.allclose(got.trans, expected)


def test_the_scale_sharpens_the_language_and_a_zero_scale_makes_it_uniform():
    flat = moe.Chain.make(1, vocab=16, scale=0.0)
    assert np.allclose(flat.trans, 1 / 16) and flat.entropy == pytest.approx(math.log(16))
    assert moe.Chain.make(1, vocab=16, scale=4.0).entropy < moe.Chain.make(1, vocab=16, scale=1.6).entropy < flat.entropy


class Top:
    """A stand-in generator whose draw is just below one."""

    def random(self, n):
        return np.full(n, 1.0 - 1e-12)


def test_a_draw_beyond_a_rounded_cumulative_sum_still_names_a_valid_token():
    chain = moe.Chain(np.array([[0.5, 0.5 - 1e-9], [0.5, 0.5 - 1e-9]]))  # rows sum to just under one
    assert chain.next_tokens(np.array([0, 1]), Top()).tolist() == [1, 1]


def test_the_initial_scales_of_the_other_tensors_are_the_documented_ones():
    cfg = moe.Config()
    p = moe.init_params(cfg, 1)
    assert abs(p["W1"].std() - math.sqrt(2 / cfg.dim)) < 0.01
    assert abs(p["Wo"].std() - 1 / math.sqrt(cfg.dim)) < 0.01


def test_training_is_adam_with_the_stated_constants_on_the_stated_gradients():
    cfg = moe.Config(vocab=8, dim=5, hidden=4, layers=2, experts=4, k=2, ctx=0.3)
    chain = moe.Chain.make(1, vocab=8, latent=3, width=6)
    start = moe.init_params(cfg, 1)
    got = moe.train(cfg, start, chain, 3, make_rng(8))  # the defaults: learning rate 3e-3, batch 256, balance weight 0.01
    rng = make_rng(8)
    p = moe.copy_params(start)
    m = {name: np.zeros_like(value) for name, value in p.items()}
    v = {name: np.zeros_like(value) for name, value in p.items()}
    for t in (1, 2, 3):
        tokens = rng.integers(0, 8, 256)
        targets = chain.next_tokens(tokens, rng)
        c = rng.normal(size=(256, 5))
        grads = moe.loss_and_grads(cfg, p, tokens, targets, c, None, None, 0.01)[2]
        for name in p:
            m[name] = 0.9 * m[name] + 0.1 * grads[name]
            v[name] = 0.999 * v[name] + 0.001 * grads[name] ** 2
            p[name] = p[name] - 3e-3 * (m[name] / (1 - 0.9**t)) / (np.sqrt(v[name] / (1 - 0.999**t)) + 1e-8)
    for name in p:
        assert np.allclose(got[name], p[name], rtol=1e-9, atol=1e-12), name


def test_the_first_adam_step_moves_every_weight_with_a_gradient_by_the_learning_rate():
    cfg = moe.Config(vocab=8, dim=5, hidden=4, layers=2, experts=4, k=2, ctx=0.3)
    chain = moe.Chain.make(1, vocab=8, latent=3, width=6)
    start = moe.init_params(cfg, 1)
    after = moe.train(cfg, start, chain, 1, make_rng(8), lr=1e-2)
    moved = np.abs(after["Wo"] - start["Wo"])
    assert moved.max() <= 1e-2 * (1 + 1e-6) and (moved > 9e-3).mean() > 0.9


def test_the_load_balancing_weight_makes_the_busiest_expert_less_busy():
    cfg = moe.Config(vocab=16, dim=8, hidden=6, layers=1, experts=4, k=1, ctx=0.3)
    chain = moe.Chain.make(1, vocab=16, latent=3, width=6)
    start = moe.init_params(cfg, 3)
    tokens = np.repeat(np.arange(16), 60)
    c = make_rng(9).normal(size=(len(tokens), 8))

    def busiest(alpha):
        p = moe.train(cfg, start, chain, 200, make_rng(4), alpha=alpha, batch=128)
        return moe.expert_load(cfg, moe.routes_of(cfg, p, tokens, c))[0].max()

    free, balanced = busiest(0.0), busiest(2.0)
    assert balanced < free
