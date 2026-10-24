"""Print the figures quoted in README.md. Deterministic; about half a minute of CPU."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from aisecurity_rfds.common.report import table
from aisecurity_rfds.common.rng import derive_seed, make_rng

from . import attack as at
from . import moe
from . import signatures as sg

SIGMA_READING = 0.25  # sensor noise (sd, in units of one feature's signature spread) for the physical-drift rows
TRIALS = 800  # tokens drawn per accuracy figure
STEPS = 600  # Adam steps to train a model
FINETUNE_STEPS = 150
AWARE_TAU = 0.5  # routing noise used while training the noise-trained model
TAUS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)


def rng(*label: object) -> np.random.Generator:
    return make_rng(derive_seed("rfd002", *label))


def train(cfg: moe.Config, chain: moe.Chain, seed: int = 0, steps: int = STEPS) -> dict[str, np.ndarray]:
    return moe.train(cfg, moe.init_params(cfg, derive_seed("init", seed)), chain, steps, rng("train", seed, cfg.ctx))


def excess(cfg: moe.Config, p: dict[str, np.ndarray], chain: moe.Chain, tau: float = 0.0, experts_on: bool = True) -> float:
    """Held-out cross-entropy minus the language's entropy, in nats: the loss the model has beyond the best possible."""
    return moe.evaluate(cfg, p, chain, 20000, rng("eval", tau, experts_on), tau, experts_on) - chain.entropy


def acc(cfg, target, profile, space, set_sigs, sigma, label, **kw) -> float:
    return at.attack(cfg, target, profile, space, set_sigs, sigma, TRIALS, rng("attack", *label), **kw).p


def assumptions(cfg: moe.Config, chain: moe.Chain, p: dict[str, np.ndarray], space: sg.SetSpace) -> None:
    print("RFD 002 - MoE expert fingerprinting. A simulation: every number is a property of a small model trained here")
    print("and of an assumed sensor, not a measurement of any GPU or any real mixture-of-experts model.\n")
    print(
        f"Toy model: {cfg.vocab} tokens, {cfg.layers} blocks of {cfg.experts} experts with {cfg.k} active per token, "
        f"width {cfg.dim}, expert hidden size {cfg.hidden}, trained for {STEPS} Adam steps on a synthetic first-order language."
    )
    print(f"A context vector of scale {cfg.ctx:g} is added to each token's embedding, so a token's route varies with context.")
    uniform = math.log(cfg.vocab) - chain.entropy
    print(
        f"Quality is the excess loss: held-out cross-entropy minus the language's entropy of {chain.entropy:.3f} nats. "
        f"Trained: {excess(cfg, p, chain):.3f}. Experts removed: {excess(cfg, p, chain, experts_on=False):.3f}. Uniform guess: {uniform:.3f}."
    )
    tokens = np.repeat(np.arange(cfg.vocab), 50)
    load = moe.expert_load(cfg, moe.routes_of(cfg, p, tokens, rng("load").normal(size=(len(tokens), cfg.dim))))
    print(f"Share of dispatches per expert, over all blocks: smallest {load.min():.3f}, largest {load.max():.3f} (even share {1 / cfg.experts:.3f}).")
    print(f"Sensor: each expert has a signature of 4 features drawn from N(0, 1); a block's reading is the sum of the signatures of its {cfg.k} experts plus noise.")
    print(f"A block's route has {space.count} possible values, {sg.bits_per_slot(space.count):.1f} bits; naming one of {cfg.vocab} tokens takes {math.log2(cfg.vocab):.1f}.")


def distinguishable(cfg, p, chain, space, profile) -> None:
    print("\n1. How many experts must be distinguishable? Token accuracy from exact routes (sensor noise 0)")
    rows = []
    for d in (2, 4, 8, 16):
        sigs = sg.Signatures.make(cfg.experts, 4, d, derive_seed("sigs", d))
        set_sigs = sigs.of_sets(space)
        cells = [f"{acc(cfg, p, profile, space, set_sigs, 0.0, ('q1', d, n), layers=n):.3f}" for n in (1, 2, 3)]
        rows.append([sigs.classes, sg.distinct_signatures(set_sigs), *cells])
    print(table(["distinguishable experts", "distinct set signatures", "1 block", "2 blocks", "3 blocks"], rows))
    ceilings = [at.deterministic_ceiling(cfg, p, n) for n in (1, 2, 3)]
    print(f"Chance {1 / cfg.vocab:.3f}. With no context vector the routes of the {cfg.vocab} tokens give a ceiling of {ceilings[0]:.3f} after 1 block and {ceilings[1]:.3f} after 2.")


def sensor(cfg, p, space, profile) -> None:
    print("\n2. What the sensor must resolve (16 distinguishable experts, 4 features of spread 1, 3 blocks)")
    sigs = sg.Signatures.make(cfg.experts, 4, cfg.experts, derive_seed("sigs", 16))
    set_sigs = sigs.of_sets(space)
    rows = []
    for sigma in (0.1, 0.25, 0.5, 1.0, 2.0):
        sim = sg.set_accuracy(set_sigs, sigma, 2000, rng("setacc", sigma)).p
        lo, hi = sg.set_error_bounds(set_sigs, sigma)
        rows.append([sigma, f"{sim:.3f}", f"{1 - hi:.3f}", f"{1 - lo:.3f}", f"{acc(cfg, p, profile, space, set_sigs, sigma, ('q2', sigma)):.3f}"])
    print(table(["noise sd", "set accuracy", "guaranteed at least", "cannot exceed", "token accuracy"], rows))
    print("Features measured at noise sd 0.5 (each feature has spread 1):")
    rows = []
    for f in (1, 2, 4, 8):
        fs = sg.Signatures.make(cfg.experts, f, cfg.experts, derive_seed("sigs", 16, f)).of_sets(space)
        rows.append([f, f"{sg.set_accuracy(fs, 0.5, 2000, rng('feat', f)).p:.3f}", f"{acc(cfg, p, profile, space, fs, 0.5, ('q2f', f)):.3f}"])
    print(table(["features", "set accuracy", "token accuracy"], rows))
    print("Spread over 20 other random draws of the signatures (16 distinguishable experts, 4 features, noise sd 0.5; the sd includes sampling error):")
    draws = np.array([
        [
            sg.set_accuracy(s, 0.5, 2000, rng("spread-set", i)).p,
            acc(cfg, p, profile, space, s, 0.5, ("spread-token", i)),
        ]
        for i, s in enumerate(
            sg.Signatures.make(cfg.experts, 4, cfg.experts, derive_seed("sigs-spread", i)).of_sets(space) for i in range(20)
        )
    ])
    rows = [[name, *(f"{v:.3f}" for v in (c.mean(), c.std(ddof=1), c.min(), c.max()))] for name, c in zip(("set accuracy", "token accuracy"), draws.T)]
    print(table(["quantity", "mean", "sd", "min", "max"], rows))


def context(cfg, chain, base, space) -> None:
    print("\n3. Routes that depend on context: models trained with a larger context vector (16 distinguishable experts, exact routes)")
    sigs = sg.Signatures.make(cfg.experts, 4, cfg.experts, derive_seed("sigs", 16))
    set_sigs = sigs.of_sets(space)
    rows = []
    for scale in (0.5, 1.0, 2.0):
        c = replace(cfg, ctx=scale)
        p = base if scale == cfg.ctx else train(c, chain)
        prof = at.build_profile(c, p, space, rng("profile", scale))
        rows.append([f"{scale:g}", f"{excess(c, p, chain):.3f}", *(f"{acc(c, p, prof, space, set_sigs, 0.0, ('q3', scale, n), layers=n):.3f}" for n in (1, 3))])
    print(table(["context scale", "excess loss", "1 block", "3 blocks"], rows))


def transfer(cfg, chain, p, space, profile) -> None:
    print("\n4. Does a table built on one model work on another? (16 distinguishable experts, exact routes, 3 blocks)")
    sigs = sg.Signatures.make(cfg.experts, 4, cfg.experts, derive_seed("sigs", 16))
    set_sigs = sigs.of_sets(space)
    other = moe.Chain.make(2, cfg.vocab)
    kept = ("W1", "W2", "Wo")
    targets: list[tuple[str, dict[str, np.ndarray]]] = [("the original", p)]
    for bits in (8, 4, 3):
        targets.append((f"int{bits}, router kept", moe.quantize(p, bits, kept)))
        targets.append((f"int{bits}, all weights", moe.quantize(p, bits)))
    for share in (0.1, 0.5):
        tuned = moe.train(cfg, p, chain.mixed_with(other, share), FINETUNE_STEPS, rng("finetune", share), lr=1e-3)
        targets.append((f"fine-tuned on {share:.0%} new language", tuned))
    targets.append(("trained again from another seed", train(cfg, chain, seed=1)))
    rows = []
    for name, target in targets:
        changed = at.route_change(cfg, p, target, rng("change", name))
        rebuilt = at.build_profile(cfg, target, space, rng("profile", name))
        rows.append([
            name,
            f"{changed:.3f}",
            f"{excess(cfg, target, chain):.3f}",
            f"{acc(cfg, target, profile, space, set_sigs, 0.0, ('q4', name)):.3f}",
            f"{acc(cfg, target, rebuilt, space, set_sigs, 0.0, ('q4', name)):.3f}",
        ])
    print(table(["target model", "blocks with other experts", "excess loss, original language", "table from the original", "table rebuilt on the target"], rows))
    print(f"Signatures that differ between builds (sensor noise sd {SIGMA_READING:g}; the attacker keeps the original's signatures):")
    rows = []
    for eps in (0.0, 0.1, 0.3):
        drifted = sigs.drifted(eps, rng("drift", eps)).of_sets(space)
        rows.append([eps, f"{acc(cfg, p, profile, space, set_sigs, SIGMA_READING, ('q4d', eps), true_sigs=drifted):.3f}"])
    print(table(["drift (share of rms signature)", "token accuracy"], rows))


def defence(cfg, chain, p, space, profile) -> None:
    print("\n5. Routing noise: Gumbel noise of scale tau on the router logits before the experts are chosen (exact routes, 3 blocks)")
    sigs = sg.Signatures.make(cfg.experts, 4, cfg.experts, derive_seed("sigs", 16))
    set_sigs = sigs.of_sets(space)
    steps = FINETUNE_STEPS * 2
    control = moe.train(cfg, p, chain, steps, rng("control"), lr=1e-3)
    aware = moe.train(cfg, p, chain, steps, rng("aware"), lr=1e-3, tau=AWARE_TAU)
    print(f"trained: the model of the sections above. control: {steps} more steps without noise. noise-trained: {steps} more steps with tau {AWARE_TAU:g} noise.")
    rows = [[tau, *(f"{excess(cfg, m, chain, tau):.3f}" for m in (p, control, aware))] for tau in TAUS]
    print(table(["tau", "excess loss: trained", "control", "noise-trained"], rows))
    print("Token accuracy of the attacker with exact routes:")
    rows = []
    for tau in TAUS:
        knows = at.build_profile(cfg, p, space, rng("profile-p", tau), tau=tau)
        knows_control = at.build_profile(cfg, control, space, rng("profile-c", tau), tau=tau)
        knows_aware = at.build_profile(cfg, aware, space, rng("profile-a", tau), tau=tau)
        rows.append([
            tau,
            f"{acc(cfg, p, profile, space, set_sigs, 0.0, ('q5a', tau), tau=tau):.3f}",
            f"{acc(cfg, p, knows, space, set_sigs, 0.0, ('q5b', tau), tau=tau):.3f}",
            f"{acc(cfg, p, knows, space, set_sigs, 0.0, ('q5c', tau), tau=tau, repeats=4):.3f}",
            f"{acc(cfg, control, knows_control, space, set_sigs, 0.0, ('q5d', tau), tau=tau):.3f}",
            f"{acc(cfg, aware, knows_aware, space, set_sigs, 0.0, ('q5e', tau), tau=tau):.3f}",
        ])
    headers = ["tau", "trained, table without noise", "trained, table knows the noise", "trained, same token seen 4 times", "control, table knows the noise", "noise-trained, table knows the noise"]
    print(table(headers, rows))


def main() -> None:
    cfg = moe.Config()
    chain = moe.Chain.make(1, cfg.vocab)
    p = train(cfg, chain)
    space = sg.SetSpace.make(cfg.experts, cfg.k)
    profile = at.build_profile(cfg, p, space, rng("profile", "base"))
    assumptions(cfg, chain, p, space)
    distinguishable(cfg, p, chain, space, profile)
    sensor(cfg, p, space, profile)
    context(cfg, chain, p, space)
    transfer(cfg, chain, p, space, profile)
    defence(cfg, chain, p, space, profile)


if __name__ == "__main__":
    main()
