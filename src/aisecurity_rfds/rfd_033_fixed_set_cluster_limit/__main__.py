"""Print every number quoted in README.md. Deterministic; about ten seconds of CPU, no network, ASCII only."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from aisecurity_rfds.common.rng import make_rng

from . import inference, podsize
from . import primitives as prim
from . import scenarios as sc
from .cluster import bridge_cluster_size, build_fleet
from .gate import Capability, Design, bypass_matrix
from .limiter import NS, ChipPort, TokenBucket
from .pod import GENERATION_FUSES
from .training import (
    YEAR_S,
    External,
    Hardware,
    Model,
    Run,
    breakeven_dp_bps,
    breakeven_pp_bps,
    cost_multiplier,
    defeat_bandwidth_bps,
    dp_ceiling_flops,
    ideal_chips,
    min_stages,
    plan,
    pp_ceiling_flops,
    ring_factor,
    simulate_ring_sync_seconds,
)

HW = Hardware()
POD = 64
N_REF = 1.0e12
REF = Run(Model.dense(N_REF), 20 * N_REF)  # 1T parameters, 20T tokens, 16M-token batches: an assumed frontier-scale run


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [[str(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in cells]
    return "\n".join([line, rule, *body])


def bps(x: float) -> str:
    """Decimal units, as in the RFD (1 GB/s = 1e9 bytes/s)."""
    if math.isinf(x):
        return "never"
    for unit, scale in (("GB/s", 1e9), ("MB/s", 1e6), ("kB/s", 1e3)):
        if x >= scale:
            v = x / scale
            return f"{v:.3g} {unit}"
    return f"{x:.3g} B/s"


def mult(x: float) -> str:
    if math.isinf(x):
        return "none"
    return f"{x:.0f}x" if x >= 100 else f"{x:.3g}x"


def sci(x: float) -> str:
    return f"{x:.3g}"


def section_membership() -> None:
    print("1. Pod membership: one row per attack, each executed against the protocol code")
    verdicts = sc.all_verdicts()
    print(_table(["attack", "what stops it", "outcome"], [[v.attack, v.defence, v.outcome] for v in verdicts]))
    counts = {o: sum(v.outcome == o for v in verdicts) for o in ("blocked", "partial", "succeeds")}
    print(f"   rows: {len(verdicts)}; blocked {counts['blocked']}, partial {counts['partial']}, attacker succeeds {counts['succeeds']}")
    print("   'succeeds' and 'partial' rows are limitations of the design as modelled, not bugs the tests hide.")


def section_update_window() -> None:
    print("\n2. Updatable membership (open question 3): removing a chip by updating ONE member, then peer gossip")
    print("   Rounds = every member starts one handshake with a random peer from its own manifest (10 seeds each).")
    rows = []
    for n in (8, 16, 32, 64):
        runs = [sc.stale_window(n, seed) for seed in range(10)]
        rounds = [r.rounds_to_full_adoption for r in runs]
        rows.append([
            n,
            f"{sum(rounds) / len(rounds):.1f}",
            max(rounds),
            sorted({r.accepted_before for r in runs})[0],
            sorted({r.accepted_after for r in runs})[0],
        ])
    print(_table(["pod size", "mean rounds", "max rounds", "members accepting removed chip, right after update", "after gossip"], rows))
    print("   Between the facility update and the end of gossip, a removed chip still links with every member that has not heard.")
    print(f"   The one-way generation counter has {GENERATION_FUSES} fuses; every accepted update and every tamper response burns at least one.")


def section_key_extraction() -> None:
    print("\n3. Key extraction: the pod-size bound only holds while device keys stay secret")
    fleet = build_fleet(8, 8, seed=b"extract")
    rows = []
    for label, extracted in (
        ("none", []),
        ("one key from pod 1", [fleet.pods[0][0]]),
        ("all 8 keys of pod 1", fleet.pods[0]),
        *[(f"one key from each of {m} pods", [fleet.pods[p][0] for p in range(m)]) for m in (2, 3, 4, 8)],
    ):
        rows.append([label, bridge_cluster_size(fleet, extracted)])
    print("   8 pods of 8 chips; one attacker device presents every extracted identity")
    print(_table(["attacker extracts", "largest high-bandwidth cluster"], rows))


def section_gate() -> None:
    print("\n4. Where the high-bandwidth decision lives: designs x attacker capabilities (BYPASSED = outside traffic accepted as pod traffic)")
    got = {(o.design, o.capability): o.bypassed for o in bypass_matrix()}
    caps = list(Capability)
    rows = [[d.value, *("BYPASSED" if got[(d, c)] else "holds" for c in caps)] for d in Design]
    print(_table(["design", *(c.value for c in caps)], rows))
    print("   The physical row of the RFD (disabling transceivers) cannot be modelled in software and is not in this table.")


def section_limiter() -> None:
    print("\n5. Egress limiter: exact token bucket (integer arithmetic), checked against granted <= burst + rate * time")
    rows = []
    for rate, burst, seconds in ((10**9, 10**6, 10), (10**6, 10**3, 20), (3, 1, 100)):
        bucket = TokenBucket(rate, burst)
        step = max(1, NS // 1000)
        total = sum(bucket.take_up_to(1 << 40, i * step) for i in range(seconds * 1000 + 1))
        bound = burst + rate * seconds
        rows.append([bps(rate), burst, f"{seconds} s", total, bound, "yes" if total <= bound else "NO"])
    print("   A greedy sender polling every millisecond:")
    print(_table(["rate", "burst (B)", "duration", "bytes granted", "bound", "within bound"], rows))

    rng = make_rng(5)
    worst = 0.0
    windows = 0
    for _ in range(40):
        rate = int(rng.integers(1_000, 5_000_000))
        burst = int(rng.integers(1, 20_000))
        bucket = TokenBucket(rate, burst)
        now = 0
        log: list[tuple[int, int]] = []
        for _ in range(120):
            now += int(rng.choice([0, 1, 10, 1_000, 100_000, 5_000_000, 200_000_000]))
            log.append((now, bucket.take_up_to(int(rng.integers(0, 3 * burst + 1)), now)))
        prefix = [0]
        for _, g in log:
            prefix.append(prefix[-1] + g)
        for i in range(len(log)):
            for j in range(i + 1, len(log)):
                carried = (prefix[j + 1] - prefix[i + 1]) * NS
                allowed = burst * NS + rate * (log[j][0] - log[i][0])
                worst = max(worst, carried / allowed)
                windows += 1
    print(f"   Random request patterns: {windows} windows over 40 buckets, highest carried/allowed ratio {worst:.4f} (must be <= 1)")

    fleet = build_fleet(1, 2)
    port = ChipPort(fleet.chips[0], external_bps=1_000_000, burst_ns=1_000_000)
    destinations = [bytes([i]) * 32 for i in range(1, 51)]
    total = sum(port.send(dest, 10**6, step * 1_000_000) for step in range(2001) for dest in destinations)
    bound = port.external.burst_bytes + 1_000_000 * 2
    print(f"   50 destinations share ONE external bucket: {total} bytes in 2 s against a bound of {bound}")

    rows = []
    for replicas, vector, rate in ((2, 2e10, 64_000_000_000), (10, 2e10, 64_000_000_000), (100, 2e12, 64_000_000_000), (7, 5e9, 1_000_000_000)):
        sim = simulate_ring_sync_seconds(vector, replicas, rate)
        closed = float(ring_factor(replicas)) * vector / rate
        rows.append([replicas, sci(vector), bps(rate), f"{sim:.4f}", f"{closed:.4f}", f"{abs(sim - closed) / closed:.2e}"])
    print("   Ring all-reduce through the bucket (closed form vs simulated drain):")
    print(_table(["replicas", "vector B", "pod egress", "simulated s", "closed form s", "rel. diff"], rows))


def section_constants() -> None:
    print("\n6. Constants the bandwidth model uses, by execution (not from memory)")
    rng = make_rng(6)
    rows = []
    for p in (2, 4, 8):
        n = 8 * p
        _, sent = prim.ring_allreduce([rng.standard_normal(n) for _ in range(p)])
        rows.append([f"ring all-reduce, {p} workers", f"{sent[0] / (8 * n):.4f}", f"{float(ring_factor(p)):.4f}", "bytes sent per worker / vector bytes"])
    for widths, tokens in (([32, 64, 48, 16], 7), ([100, 30, 30, 30, 5], 13)):
        flops, params = prim.train_step_flops(widths, tokens, rng)
        rows.append([f"train step, widths {widths}", f"{flops / (params * tokens):.4f}", "6.0000", "matmul FLOPs / (params * tokens)"])
    boundary = prim.pipeline_boundary_bytes(512, 10, rng)
    rows.append(["pipeline boundary, fp16", f"{boundary / (10 * 512 * 2):.4f}", "2.0000", "bytes / (tokens * d * 2 B)"])
    block = prim.transformer_block_params(1024)
    rows.append(["decoder block, d=1024", f"{block / 1024**2:.4f}", "12.0000", "parameters / d^2"])
    print(_table(["quantity", "counted", "assumed by model", "meaning"], rows))


def section_reference() -> dict[str, float]:
    print("\n7. Reference run (every input is an assumption, listed so it can be changed)")
    m = REF.model
    ideal = ideal_chips(REF, HW, YEAR_S)
    print(f"   chip: {sci(HW.flops)} FLOP/s peak, {HW.mfu:.0%} sustained, {HW.mem_bytes / 1e9:.0f} GB memory; pod: {POD} chips")
    print(f"   model: {sci(m.n_params)} parameters, d_model {m.d_model:.0f}, {m.n_layers:.0f} layers; tokens {sci(REF.tokens)}; batch {sci(REF.batch_tokens)} tokens")
    print(f"   state {REF.state_bytes_per_param:.0f} B/param, gradients {REF.grad_bytes:.0f} B, activations {REF.act_bytes:.0f} B; deadline one year")
    print(f"   training compute {sci(REF.flops)} FLOP; communication-free ideal {ideal:.0f} chips; memory forces at least {min_stages(REF, HW, POD)} pods per replica")
    return {"ideal": ideal}


def section_cost() -> None:
    print("\n8. Chips an attacker needs relative to the communication-free ideal, by cap and by traffic reduction")
    print("   'reduction' = gradient compression x local steps (applies to data-parallel traffic only); none = no layout with <=512 stages and <=4096 replicas finishes in a year")
    caps = [1e9, 3e8, 1e8, 3e7, 1e7, 3e6, 1e6]
    for scope, label in (("per_chip", "cap applied per chip (pod egress = 64 x cap)"), ("per_pod", "the same numbers applied per pod")):
        rows = []
        for reduction in (1, 100, 10_000):
            rows.append([reduction, *(mult(cost_multiplier(REF, HW, POD, External(c, scope), YEAR_S, compression=reduction)) for c in caps)])
        print(f"   {label}:")
        print(_table(["reduction", *(bps(c) for c in caps)], rows))
    p = plan(REF, HW, POD, External(1e9, "per_chip"), YEAR_S)
    assert p is not None
    print(
        f"   RFD example cap (1 GB/s per chip, no reduction): cheapest plan {p.layout.stages} stages x {p.layout.replicas} replicas "
        f"= {p.chips} chips, utilisation {p.utilisation:.2f}"
    )


def section_defeat() -> None:
    print("\n9. Cap below which the cheapest run needs more than 10x the ideal chips ('defeat bandwidth')")
    rows = []
    for reduction in (1, 10, 100, 1000, 10_000):
        chip = defeat_bandwidth_bps(REF, HW, POD, YEAR_S, multiplier=10, scope="per_chip", compression=reduction)
        pod = defeat_bandwidth_bps(REF, HW, POD, YEAR_S, multiplier=10, scope="per_pod", compression=reduction)
        rows.append([reduction, bps(chip), f"{1e9 / chip:.3g}", bps(pod)])
    print(_table(["reduction", "per-chip cap", "RFD's 1 GB/s divided by it", "per-pod cap"], rows))
    print("   The per-chip column plateaus because pipeline traffic (activations) is not reduced by this model's gradient tricks.")


def section_sensitivity() -> None:
    print("\n10. Sensitivity of the defeat bandwidth (10x, per-chip scope), one assumption changed at a time")

    def defeat(run: Run, hw: Hardware, pod: int, deadline: float, reduction: int) -> float:
        return defeat_bandwidth_bps(run, hw, pod, deadline, multiplier=10, scope="per_chip", compression=reduction)

    def run_of(n: float = N_REF, **kw: float) -> Run:
        return Run(Model.dense(n), 20 * n, **kw)

    cases: list[tuple[str, Run, Hardware, int, float]] = [
        ("baseline", REF, HW, POD, YEAR_S),
        ("batch 4M tokens", run_of(batch_tokens=4e6), HW, POD, YEAR_S),
        ("batch 64M tokens", run_of(batch_tokens=64e6), HW, POD, YEAR_S),
        ("chip 0.5x speed", REF, Hardware(flops=HW.flops * 0.5), POD, YEAR_S),
        ("chip 2x speed", REF, Hardware(flops=HW.flops * 2), POD, YEAR_S),
        ("chip 4x speed", REF, Hardware(flops=HW.flops * 4), POD, YEAR_S),
        ("deadline 6 months", REF, HW, POD, YEAR_S / 2),
        ("deadline 2 years", REF, HW, POD, YEAR_S * 2),
        ("pod of 8 chips", REF, HW, 8, YEAR_S),
        ("pod of 256 chips", REF, HW, 256, YEAR_S),
        ("activations 1 byte", run_of(act_bytes=1.0), HW, POD, YEAR_S),
        ("activations 0.5 byte", run_of(act_bytes=0.5), HW, POD, YEAR_S),
        ("model 1e11 parameters", run_of(1e11), HW, POD, YEAR_S),
        ("model 1e13 parameters", run_of(1e13), HW, POD, YEAR_S),
    ]
    rows = [[label, bps(defeat(run, hw, pod, dl, 1)), bps(defeat(run, hw, pod, dl, 100))] for label, run, hw, pod, dl in cases]
    print(_table(["change", "reduction 1", "reduction 100"], rows))


def section_closed_forms() -> None:
    print("\n11. Closed forms for the reference run (per-chip scope)")
    rows = []
    for replicas in (2, 10, 100, 1000):
        rows.append([
            f"data parallel, R={replicas}",
            bps(breakeven_dp_bps(REF, HW, replicas)),
            bps(breakeven_dp_bps(REF, HW, replicas, compression=100)),
        ])
    print("   Break-even cap: communication takes as long as compute (reduction 1, then 100)")
    print(_table(["layout", "reduction 1", "reduction 100"], rows))
    print(f"   Break-even for a pipeline of K stages (any batch size): K=4 {bps(breakeven_pp_bps(REF, HW, 4))}, K=64 {bps(breakeven_pp_bps(REF, HW, 64))}")
    print(
        f"   Ceilings at 1 GB/s per chip: data parallelism {sci(dp_ceiling_flops(REF, POD, 1e9))} FLOP/s, "
        f"one pipeline {sci(pp_ceiling_flops(REF, POD, 1e9))} FLOP/s; the run needs {sci(REF.flops / YEAR_S)} FLOP/s"
    )
    print(
        f"   Pure data parallelism at 1 GB/s per chip (reduction 1) tops out at {dp_ceiling_flops(REF, POD, 1e9) / HW.effective_flops:.0f} "
        f"chip-equivalents of useful compute however many pods are added"
    )


def section_pod_size() -> None:
    print("\n12. Pod size (open question 1): compute one pod sustains, and how algorithmic progress erodes a fixed limit")
    target = REF.flops
    rows = []
    for size in (1, 8, 64, 65, 256, 1024):
        c = podsize.pod_compute(size, HW)
        rows.append([
            size,
            sci(c),
            f"{podsize.compute_optimal_params(c) / 1e9:.3g}",
            *(f"{podsize.years_until_pod_reaches(target, size, HW, d):.1f}" for d in (0.5, 1.0, 2.0)),
        ])
    print(f"   One year of sustained compute; target = the reference run ({sci(target)} FLOP); doubling time = how fast algorithms halve the compute needed")
    print(_table(["pod size", "FLOP per year", "compute-optimal params (B)", "years to reach target, doubling 0.5 y", "1 y", "2 y"], rows))
    caps = (1e24, 1e25, 1e26)
    print("   Largest pod whose year of compute stays at or below a cap: " + "; ".join(f"{sci(c)} FLOP -> {podsize.max_pod_size(c, HW)} chips" for c in caps))


def section_multipod() -> None:
    print("\n13. Legitimate multi-pod workloads (open question 4): serving across pods vs training, 1 GB/s per chip")
    rows = []
    for n in (1e11, 1e12, 1e13, 1e14):
        model = Model.dense(n)
        run = Run(model, 20 * n)
        pods = inference.weight_pods(model, POD, HW, weight_bytes=1.0)
        train_pods = min_stages(run, HW, POD)
        serve_b = inference.pipeline_bytes_per_token(model)
        train_b = inference.pipeline_bytes_per_token(model, training=True)
        if pods == 1:
            pipeline_headroom = tensor_headroom = "single pod"
        else:
            compute = inference.compute_tokens_per_s(model, pods * POD, HW)
            pipeline_headroom = f"{inference.boundary_tokens_per_s(serve_b, POD, 1e9) / compute:.3g}x"
            tensor_headroom = f"{inference.boundary_tokens_per_s(inference.tensor_parallel_bytes_per_token(model, pods), POD, 1e9) / compute:.3g}x"
        rows.append([sci(n), pods, train_pods, f"{serve_b / 1e3:.3g}", f"{train_b / 1e3:.3g}", pipeline_headroom, tensor_headroom])
    print(_table(
        ["params", "pods for int8 weights", "pods for training state", "pipeline kB/token serving", "kB/token training", "pipeline: link capacity / compute", "tensor-parallel: link capacity / compute"],
        rows,
    ))
    print("   Ratio above 1 = the link carries more tokens/s than the pods can compute; below 1 = the cap, not the chips, sets the speed.")


def main() -> None:
    print("RFD 033 - Fixed set cluster size limiting. Executable model: every number is a property of the code and its stated assumptions.\n")
    section_membership()
    section_update_window()
    section_key_extraction()
    section_gate()
    section_limiter()
    section_constants()
    section_reference()
    section_cost()
    section_defeat()
    section_sensitivity()
    section_closed_forms()
    section_pod_size()
    section_multipod()
    print("\nNot modelled: manufacturing cost and retrofit (open questions 5 and 6), the physical transceiver row, latency, pipeline bubbles,")
    print("stragglers, mixture-of-experts, and the extra tokens that compression and local steps cost in practice (so the attacker is favoured).")


if __name__ == "__main__":
    main()
