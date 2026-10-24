"""Print the figures quoted in README.md. Deterministic; a few seconds of CPU."""

from __future__ import annotations

import math
from dataclasses import replace

from aisecurity_rfds.common.report import sci, table
from aisecurity_rfds.common.rng import derive_seed, make_rng

from . import analysis as an
from . import leakage as lk
from . import routing as rt
from . import scenarios as sc

PROMPT_TOKENS = 16  # tokens of prompt (or of observed decode steps) the topic attackers get to see


def rng(*label: object):
    return make_rng(derive_seed("rfd001", *label))


def count(x: float | None) -> str:
    return "-" if x is None else (f"{x:,.0f}" if x < 10**5 else sci(x))


def at_most(x: int | None, cap: int = 1024) -> str:
    return f">{cap}" if x is None else str(x)


def assumptions(layer: lk.ToyLayer) -> None:
    print("RFD 001 - Side-channel leakage from LLM inference. A simulation: every number is a property of the assumed")
    print("leakage model and the assumed channels, not a measurement of any GPU.\n")
    print(f"Leakage: one multiplication leaks the Hamming weight of its 16-bit product ({an.LEAK_SD:.2f} LU of spread over random weights; LU = leakage unit).")
    print(f"Toy layer: {layer.weights.shape[0]} x {layer.weights.shape[1]} int8 weights = {layer.ops} multiplications per token, vocabulary {layer.vocab}.")
    print(f"Operations run at {lk.OP_RATE_HZ:,.0f} per second and a decode step lasts {lk.TOKEN_SLOT_S * 1000:.0f} ms.")
    print("Channels: each sample is the mean leakage over its window, plus noise (assumed values).")
    rows = [
        [c.name, f"{c.gain:g}", f"{c.noise_op:g}", f"{c.noise_out:g}", f"{c.rate_hz:,.0f}", f"{c.window:,}", f"{c.samples_per_slot:,.10g}"]
        for c in lk.CHANNELS
    ]
    print(table(["channel", "gain", "noise per op", "noise per sample", "samples per second", "ops per sample", "samples per token"], rows))


def detection(layer: lk.ToyLayer) -> None:
    print("\n1. Is there any leakage? A fixed-versus-random Welch t test; |t| above 4.5 flags a sample")
    rows = []
    for d in (0.5, 0.2, 0.1, 0.05):
        n = an.tvla_traces_needed(d)
        r = rng("formula", d)
        a, b = r.normal(0.0, 1.0, (n, 2000)), r.normal(d, 1.0, (n, 2000))
        rows.append([d, f"{n:,}", f"{(an.welch_t(b, a) > an.TVLA_THRESHOLD).mean():.3f}"])
    print(table(["effect d (standard deviations)", "traces per group for 90% power", "flagged in 2000 simulated tests"], rows))
    rows = []
    for ch in lk.CHANNELS:
        d = sc.median_best_effect(layer, ch)
        samples = ch.view(layer.template(0)).shape[-1]
        flagged = " / ".join(str(sc.tvla_flagged(layer, ch, k, rng("tvla", ch.name, k))[0]) for k in (100, 1000, 10000))
        rows.append([ch.name, f"{ch.window:,}", sci(d), count(an.tvla_traces_needed(d)), f"{flagged} of {samples}"])
    print(table(["channel", "ops per sample", "typical best effect d", "traces per group to flag (90%)", "flagged after 100 / 1,000 / 10,000 traces"], rows))
    flagged, top = sc.tvla_flagged(layer, lk.SCOPE, 10000, rng("null"), fixed=None)
    print(f"No leak to find (both groups random tokens, 10,000 traces): {flagged} of {layer.ops} samples flagged, largest |t| {top:.1f}.")


def correlation_attack() -> None:
    print("\n2. Correlation attack on one int8 activation (known weights, one measurement per multiplication, gain 1)")
    rows = []
    for sigma in (1.0, 2.0, 4.0):
        rho = an.secret_rho(77, sigma)
        n = an.correlation_detection_traces(rho)
        flag = an.correlation_flag_rate(77, sigma, n, 400, rng("flag", sigma)).p
        one = an.samples_to_recover(sigma, 0.9, 100, 1, secret=77)
        anyone = an.samples_to_recover(sigma, 0.9, 100, 2)
        rows.append([sigma, f"{rho:.3f}", n, f"{flag:.3f}", at_most(one, an.LADDER[-1]), at_most(anyone, an.LADDER[-1])])
    headers = ["noise sd (LU)", "rho (secret 77)", "samples to flag the leak", "flagged in simulation", "samples to recover 77 (90%)", "samples to recover a random secret (90%)"]
    print(table(headers, rows))
    ghosts = "; ".join(f"{s} ~ {h} ({r:.3f})" for s in (77, 5, 100) for h, r in [an.closest_confusable(s)])
    print(f"Closest confusable activation, by correlation of the predicted leakage over random weights: {ghosts}.")


def naming_a_token(layer: lk.ToyLayer) -> None:
    print("\n3. Naming one token of the vocabulary from the first-layer product (power rail, oscilloscope)")
    expected = lk.SCOPE.view(layer.templates())
    rows = []
    for n in (1, 2, 4, 8, 16):
        est = an.identification_accuracy(expected, lk.SCOPE.noise, 2000, rng("id", n), repeats=n)
        lo, hi = an.identification_error_bounds(expected, lk.SCOPE.noise / math.sqrt(n))
        rows.append([n, f"{est.p:.3f}", f"{1 - hi:.3f}", f"{1 - lo:.3f}"])
    print(table(["averaged runs", "simulated accuracy", "guaranteed at least", "cannot exceed"], rows))
    rows = []
    for ch in lk.CHANNELS:
        view = ch.view(layer.templates())
        needed = an.repeats_needed(view, ch.noise, 0.99)
        rows.append([ch.name, "never (>10^7 runs)" if needed is None else needed, f"{an.noise_free_identification_accuracy(view):.3f}"])
    print(table(["channel", "averaged runs for a guaranteed 99%", "ceiling even with no noise"], rows))
    slow = lk.ACOUSTIC.view(layer.templates())
    print(f"A slow sensor reports one number per token (the sum of its {layer.ops} leakages, scaled), and the {layer.vocab} tokens have only {len(set(slow[:, 0]))} distinct sums: tokens that share a sum can never be told apart.")


def channel_matrix(layer: lk.ToyLayer, router: rt.MoeRouter) -> None:
    print(f"\n4. Which channel sees which target (one run; {PROMPT_TOKENS}-token prompts; {len(router.sets)} expert pairs; {router.categories} topics)")
    rows = []
    for ch in lk.CHANNELS:
        token = sc.token_accuracy(layer, ch, 2000, rng("tok", ch.name)).p
        resolved = ch.samples_per_slot >= 1
        pair = rt.set_accuracy(router, ch, 2000, rng("pair", ch.name)).p if resolved else None
        moe = rt.category_accuracy(router, ch, PROMPT_TOKENS, 600, rng("moe", ch.name)).p if resolved else None
        dense = sc.dense_category_accuracy(layer, ch, PROMPT_TOKENS, 600, rng("dense", ch.name)).p
        hidden = sc.hidden_recovery(layer, ch, "post-hoc filter", 20, 32, rng("hidden", ch.name))
        cells = ["-" if v is None else f"{v:.3f}" for v in (pair, moe)]
        rows.append([ch.name, f"{token:.3f}", cells[0], cells[1], f"{dense:.3f}", f"{hidden:.3f}"])
    headers = ["channel", "token", "expert pair", "topic (routing)", "topic (dense)", "hidden response"]
    print(table(headers, rows))
    print(f"Chance: token {1 / layer.vocab:.3f}, expert pair {1 / len(router.sets):.3f}, topic {1 / router.categories:.3f}. '-' means the channel takes fewer than one sample per token.")


def dense_versus_moe(layer: lk.ToyLayer, router: rt.MoeRouter) -> None:
    print("\n5. Dense versus mixture-of-experts: what the routing choice adds")
    mi = rt.mutual_information(router)
    print(f"The expert pair a token uses reveals {mi:.3f} bits about the topic (at most {math.log2(router.categories):.3f}); read exactly, one token gives {rt.noise_free_category_accuracy(router):.3f} topic accuracy. A dense model makes no such choice: 0 bits.")
    rows = []
    for ch in lk.CHANNELS:
        moe = sc.tokens_for_category(router, ch, 0.9, 600, 0) if ch.samples_per_slot >= 1 else None
        dense = sc.tokens_for_dense_category(layer, ch, 0.9, 600, 0)
        rows.append([ch.name, "-" if ch.samples_per_slot < 1 else at_most(moe), at_most(dense)])
    print(table(["channel", "tokens for 90% topic accuracy: routing", "tokens for 90%: dense, naming tokens"], rows))
    print("Software sensor at 50 Hz with the sensor noise varied (each expert shifts the mean level by 0.05 LU, sd):")
    rows = []
    for noise in (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08):
        ch = replace(lk.SOFTWARE, noise_out=noise)
        rows.append([noise, f"{noise / 0.05:g}", at_most(sc.tokens_for_category(router, ch, 0.9, 600, 0))])
    print(table(["sensor noise (LU)", "noise / expert shift", "tokens for 90% topic accuracy"], rows))


def shared_sensor(layer: lk.ToyLayer, router: rt.MoeRouter) -> None:
    print("\n6. A software sensor shared with other workloads: tokens for 90% topic accuracy by exposed sample rate")
    rows = []
    for rate in (50, 200, 1000):
        base = replace(lk.SOFTWARE, rate_hz=rate)
        cells = [at_most(sc.tokens_for_category(router, base.with_tenants(n, sc.OTHER_TENANT_NOISE), 0.9, 600, 0)) for n in (1, 4, 16)]
        rows.append([f"{rate:,}", f"{base.samples_per_slot:,.10g}", *cells])
    print(table(["exposed rate (Hz)", "samples per token", "1 tenant", "4 tenants", "16 tenants"], rows))
    fast = replace(lk.SOFTWARE, rate_hz=1000)
    acc = sc.token_accuracy(layer, fast, 2000, rng("cloud-token"))
    print(f"Naming a token through the 1000 Hz sensor: accuracy {acc.p:.3f} (chance {1 / layer.vocab:.3f}); each sample averages {fast.window:,} operations, of which {layer.ops} carry the token.")


def refusal(layer: lk.ToyLayer) -> None:
    print("\n7. A response hidden behind a refusal: fraction of its 32 tokens recovered (mean of 20 responses)")
    channels = (lk.SCOPE, lk.EM_PROBE, lk.ACOUSTIC, lk.SOFTWARE)
    rows = []
    for mode in ("no refusal", "post-hoc filter", "pre-check"):
        cells = [f"{sc.hidden_recovery(layer, ch, mode, 20, 32, rng('hidden', ch.name)):.3f}" for ch in channels]
        rows.append([mode, *cells])
    print(table(["refusal applied", *(ch.name for ch in channels)], rows))


def main() -> None:
    layer = lk.ToyLayer.make()
    router = rt.MoeRouter.make()
    assumptions(layer)
    detection(layer)
    correlation_attack()
    naming_a_token(layer)
    channel_matrix(layer, router)
    dense_versus_moe(layer, router)
    shared_sensor(layer, router)
    refusal(layer)


if __name__ == "__main__":
    main()
