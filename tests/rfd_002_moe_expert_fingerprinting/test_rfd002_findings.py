"""The qualitative claims of the README, checked on a smaller model than the one the program prints (so that they
do not just restate its numbers): 24 tokens, 3 blocks of 8 experts, 2 active, trained for 400 Adam steps."""

from dataclasses import replace
from functools import cache
from itertools import pairwise

import numpy as np

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.rfd_002_moe_expert_fingerprinting import attack as at
from aisecurity_rfds.rfd_002_moe_expert_fingerprinting import moe
from aisecurity_rfds.rfd_002_moe_expert_fingerprinting import signatures as sg

CFG = moe.Config(vocab=24, dim=12, hidden=12, layers=3, experts=8, k=2, ctx=0.3)
SPACE = sg.SetSpace.make(8, 2)
KEPT = ("W1", "W2", "Wo")


@cache
def chain():
    return moe.Chain.make(1, vocab=24, latent=4, width=12)


def fit(cfg, seed, steps=400):
    return moe.train(cfg, moe.init_params(cfg, seed), chain(), steps, make_rng(3), batch=128)


@cache
def model(name):
    """The model called ``name``, as ``(config, parameters)``. Callers must not change the parameters."""
    base = fit(CFG, 2)
    other = moe.Chain.make(2, 24)
    if name == "base":
        return CFG, base
    if name == "ctx1":
        return replace(CFG, ctx=1.0), fit(replace(CFG, ctx=1.0), 2)
    if name == "ctx2":
        return replace(CFG, ctx=2.0), fit(replace(CFG, ctx=2.0), 2)
    if name == "int8":
        return CFG, moe.quantize(base, 8, KEPT)
    if name == "int4":
        return CFG, moe.quantize(base, 4)
    if name == "int3":
        return CFG, moe.quantize(base, 3)
    if name == "ft10":
        return CFG, moe.train(CFG, base, chain().mixed_with(other, 0.1), 100, make_rng(60), lr=1e-3, batch=128)
    if name == "ft50":
        return CFG, moe.train(CFG, base, chain().mixed_with(other, 0.5), 100, make_rng(61), lr=1e-3, batch=128)
    if name == "indep":
        return CFG, fit(CFG, 99)
    if name == "control":
        return CFG, moe.train(CFG, base, chain(), 200, make_rng(110), lr=1e-3, batch=128)
    if name == "aware":
        return CFG, moe.train(CFG, base, chain(), 200, make_rng(111), lr=1e-3, batch=128, tau=0.5)
    raise KeyError(name)


@cache
def table(name, tau=0.0):
    cfg, p = model(name)
    return at.build_profile(cfg, p, SPACE, make_rng(5), per_token=150, tau=tau)


def sigs(classes=8, features=4, seed=6):
    return sg.Signatures.make(8, features, classes, seed)


def acc(name, profile, signatures, sigma, label, trials=1500, **kw):
    cfg, p = model(name)
    return at.attack(cfg, p, profile, SPACE, signatures.of_sets(SPACE), sigma, trials, make_rng(label), **kw).p


def excess(name, tau=0.0):
    cfg, p = model(name)
    return moe.evaluate(cfg, p, chain(), 8000, make_rng(4), tau=tau) - chain().entropy


def flipped(tau):
    cfg, p = model("base")
    tokens = np.repeat(np.arange(24), 100)
    c = make_rng(90).normal(size=(len(tokens), cfg.dim))
    clean = moe.routes_of(cfg, p, tokens, c)
    noisy = moe.routes_of(cfg, p, tokens, c, moe.routing_noise(cfg, len(tokens), tau, make_rng(91)))
    return float((clean != noisy).any(axis=2).mean())


# --- the toy model is a real model -------------------------------------------------------------------------------


def test_the_toy_model_has_learned_its_language_and_depends_on_its_experts():
    cfg, p = model("base")
    uniform = np.log(24) - chain().entropy
    assert excess("base") < 0.15 < uniform / 5
    assert moe.evaluate(cfg, p, chain(), 8000, make_rng(4), experts_on=False) - chain().entropy > 0.5


# --- §1: how many experts must be distinguishable ---------------------------------------------------------------------


def test_token_recovery_rises_with_the_number_of_distinguishable_experts():
    accs =[acc("base", table("base"), sigs(c), 0.0, 10 + c) for c in (2, 4, 8)]
    distinct = [sg.distinct_signatures(sigs(c).of_sets(SPACE)) for c in (2, 4, 8)]
    assert distinct == [3, 10, 28]
    assert accs[0] < 0.6 and accs[0] + 0.2 < accs[1] < accs[2] and accs[2] > 0.9


def test_token_recovery_rises_with_the_number_of_blocks_observed():
    a1, a2, a3 = (acc("base", table("base"), sigs(8), 0.0, 20 + n, layers=n) for n in (1, 2, 3))
    assert a1 < 0.75 and a1 + 0.2 < a2 <= a3 + 0.02 and a3 > 0.9


def test_one_block_of_two_distinguishable_experts_is_close_to_chance():
    assert acc("base", table("base"), sigs(2), 0.0, 30, layers=1) < 0.2  # chance is 1/24


def test_with_no_context_vector_the_exact_attack_reaches_exactly_the_share_of_tokens_with_their_own_route():
    cfg, p = model("base")
    cfg0 = replace(cfg, ctx=0.0)
    prof = at.build_profile(cfg0, p, SPACE, make_rng(5), per_token=20)
    ceilings = [at.deterministic_ceiling(cfg, p, n) for n in (1, 2, 3)]
    for n, ceiling in zip((1, 2, 3), ceilings):
        got = at.attack(cfg0, p, prof, SPACE, sigs(8).of_sets(SPACE), 0.0, 1, make_rng(1), layers=n, tokens=np.arange(24))
        assert got.p == ceiling
    assert ceilings == sorted(ceilings) and ceilings[0] < 0.7 and ceilings[1] > 0.9


# --- §2: what the sensor must resolve ---------------------------------------------------------------------------------


def test_set_decoding_collapses_before_token_recovery_does():
    set_sigs = sigs(8).of_sets(SPACE)
    set_acc = sg.set_accuracy(set_sigs, 1.0, 3000, make_rng(20)).p
    token_acc = acc("base", table("base"), sigs(8), 1.0, 30)
    assert set_acc < 0.6 and token_acc > 0.6 and token_acc > set_acc + 0.2


def test_a_sensor_with_little_noise_does_both_jobs_and_a_very_noisy_one_neither():
    set_sigs = sigs(8).of_sets(SPACE)
    assert sg.set_accuracy(set_sigs, 0.1, 3000, make_rng(20)).p > 0.95 and acc("base", table("base"), sigs(8), 0.1, 31) > 0.9
    assert sg.set_accuracy(set_sigs, 4.0, 3000, make_rng(20)).p < 0.15 and acc("base", table("base"), sigs(8), 4.0, 32) < 0.25


def test_token_recovery_falls_as_the_sensor_gets_noisier():
    accs = [acc("base", table("base"), sigs(8), s, 33) for s in (0.1, 0.5, 1.0, 2.0)]
    assert accs == sorted(accs, reverse=True) and accs[0] - accs[-1] > 0.4


def test_measuring_more_features_raises_token_recovery():
    accs = [acc("base", table("base"), sigs(8, f, 7), 0.5, 40 + f) for f in (1, 2, 4, 8)]
    assert accs[0] < accs[1] < accs[2] <= accs[3] + 0.02 and accs[0] < 0.6 and accs[2] > 0.85


# --- §3: routes that depend on the context -----------------------------------------------------------------------------


def test_a_larger_context_vector_lowers_recovery_and_costs_quality():
    accs = []
    for name in ("base", "ctx1", "ctx2"):
        cfg, p = model(name)
        prof = at.build_profile(cfg, p, SPACE, make_rng(5), per_token=150)
        accs.append(at.attack(cfg, p, prof, SPACE, sigs(8).of_sets(SPACE), 0.0, 1500, make_rng(50)).p)
    assert accs[0] > accs[1] + 0.1 and accs[1] > accs[2] + 0.1 and accs[0] > 0.9 and accs[2] < 0.5
    assert excess("base") < excess("ctx1") < excess("ctx2") and excess("ctx2") > 0.5


# --- §4: does a table transfer? ----------------------------------------------------------------------------------------


def test_a_table_built_on_the_original_works_on_an_int8_copy():
    cfg, p = model("int8")
    assert at.route_change(cfg, model("base")[1], p, make_rng(70)) < 0.05
    assert abs(acc("int8", table("base"), sigs(8), 0.0, 72) - acc("base", table("base"), sigs(8), 0.0, 72)) < 0.03


def test_coarser_quantization_changes_more_routes_and_a_rebuilt_table_repairs_the_damage():
    changes = [at.route_change(CFG, model("base")[1], model(n)[1], make_rng(70)) for n in ("int8", "int4", "int3")]
    assert changes == sorted(changes) and changes[2] > 0.3
    old = [acc(n, table("base"), sigs(8), 0.0, 72) for n in ("int4", "int3")]
    new = [acc(n, table(n), sigs(8), 0.0, 72) for n in ("int4", "int3")]
    assert old[0] > 0.85 and old[1] < new[1] - 0.1 and min(new) > 0.9


def test_a_table_survives_fine_tuning_with_a_loss_of_some_accuracy_and_a_rebuilt_table_loses_none():
    base_acc = acc("base", table("base"), sigs(8), 0.0, 72)
    for name in ("ft10", "ft50"):
        assert 0.0 < at.route_change(CFG, model("base")[1], model(name)[1], make_rng(70)) < 0.5
        assert 0.85 < acc(name, table("base"), sigs(8), 0.0, 72) <= base_acc + 0.02
        assert acc(name, table(name), sigs(8), 0.0, 72) > 0.93


def test_a_table_is_useless_on_an_independently_trained_model_until_it_is_rebuilt_on_it():
    assert at.route_change(CFG, model("base")[1], model("indep")[1], make_rng(70)) > 0.8
    assert acc("indep", table("base"), sigs(8), 0.0, 72) < 0.15  # chance is 1/24
    assert acc("indep", table("indep"), sigs(8), 0.0, 72) > 0.9
    assert excess("indep") < 0.15  # it is as good a model as the original


def test_signature_drift_lowers_recovery_once_it_is_a_sizeable_part_of_the_signature():
    base = sigs(8)
    accs = {}
    for eps in (0.0, 0.1, 0.3, 0.6):
        drifted = base.drifted(eps, make_rng(80)).of_sets(SPACE)
        cfg, p = model("base")
        accs[eps] = at.attack(cfg, p, table("base"), SPACE, base.of_sets(SPACE), 0.25, 1500, make_rng(81), true_sigs=drifted).p
    assert accs[0.0] > 0.9 and abs(accs[0.1] - accs[0.0]) < 0.05
    assert accs[0.3] < accs[0.1] and accs[0.6] < accs[0.0] - 0.3


# --- §5: routing noise ------------------------------------------------------------------------------------------------


def test_routing_noise_changes_the_expert_choice_more_and_more_often_as_tau_grows():
    shares = [flipped(t) for t in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)]
    assert shares[0] == 0.0 and shares == sorted(shares) and len(set(shares)) == 6 and shares[-1] > 0.8


def test_small_routing_noise_costs_almost_no_quality_and_large_noise_costs_a_lot():
    base = excess("base")
    assert excess("base", 0.1) - base < 0.02 and excess("base", 0.25) - base < 0.03
    assert excess("base", 1.0) - base > 0.2 and excess("base", 2.0) - base > 0.6
    costs = [excess("base", t) for t in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)]
    assert all(b >= a - 0.002 for a, b in pairwise(costs))


def test_an_attacker_who_ignores_the_noise_loses_little_at_small_tau_and_most_of_the_accuracy_at_large_tau():
    a0 = acc("base", table("base"), sigs(8), 0.0, 94)
    accs = {t: acc("base", table("base"), sigs(8), 0.0, 94, tau=t) for t in (0.25, 1.0, 2.0)}
    assert a0 > 0.9 and accs[0.25] > a0 - 0.08 and accs[1.0] < 0.6 and accs[2.0] < 0.3


def test_a_table_that_knows_the_noise_recovers_part_of_the_loss():
    for tau in (0.5, 1.0):
        ignorant = acc("base", table("base"), sigs(8), 0.0, 95, tau=tau)
        aware = acc("base", table("base", tau), sigs(8), 0.0, 95, tau=tau)
        assert aware > ignorant + 0.05


def test_seeing_the_same_token_several_times_defeats_moderate_noise():
    knows = table("base", 1.0)
    once = acc("base", knows, sigs(8), 0.0, 96, tau=1.0)
    four = acc("base", knows, sigs(8), 0.0, 96, tau=1.0, repeats=4)
    assert four > once + 0.2 and four > 0.8
    half = acc("base", table("base", 0.5), sigs(8), 0.0, 96, tau=0.5, repeats=4)
    assert half > 0.95


def test_more_repeats_never_hurt_the_attacker_much_and_help_most_when_the_noise_is_large():
    knows = table("base", 2.0)
    accs = [acc("base", knows, sigs(8), 0.0, 97, tau=2.0, repeats=r) for r in (1, 4, 16)]
    assert accs[0] < accs[1] < accs[2] and accs[2] - accs[0] > 0.3


def test_noise_aware_training_lowers_the_loss_under_noise_compared_with_equally_long_plain_training():
    assert abs(excess("aware") - excess("control")) < 0.02
    for tau in (0.5, 1.0, 2.0):
        assert excess("aware", tau) < excess("control", tau) - 0.01
    assert excess("aware", 1.0) > excess("aware") + 0.1  # noise still costs the noise-trained model something
