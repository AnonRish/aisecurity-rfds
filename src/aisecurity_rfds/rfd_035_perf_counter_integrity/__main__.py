"""Print the headline findings quoted in README.md. Deterministic; about fifteen seconds of CPU."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from aisecurity_rfds.common.rng import derive_seed, make_rng

from . import (
    CLASSES,
    NOVEL_CLASS,
    calibrate,
    detection_matrix,
    explore,
    generate_reading,
    hiding_table,
    inspect,
    margin_tradeoff,
)
from .attacks import dynamic_power, emp_bit_flips, evaluate_attack, honest_readings, scale_detection
from .checks import min_detectable_fraction
from .device_sim import SCENARIOS, power_miss_probability, simulate_device
from .hiding import calibration_readings, fake_traffic_on_idle_sms
from .monotonic import ALPHABET_HIDING, ALPHABET_PROTOCOL, aliased_work, bits_needed, safe_gap, seconds_to_wrap
from .readout import protocol_overhead
from .redundancy import emp_outcomes, exhaustive_detection, layouts


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [[str(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in cells]
    return "\n".join([line, rule, *body])


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    print("RFD 035 - Performance Counter Integrity Protection (synthetic telemetry; every constant is an assumption)")
    per_class = calibration_readings(CLASSES, 100, make_rng(35))
    th = calibrate([r for rs in per_class.values() for r in rs])
    fresh = honest_readings(CLASSES, 100, make_rng(36))
    fp = sum(any(inspect(r, th).values()) for r in fresh)
    print(f"\n[1] Calibration: bands rho in [{th.rho_lo:.3f}, {th.rho_hi:.3f}] FLOPs/retired instr, "
          f"mu in [{th.mu_lo:.3f}, {th.mu_hi:.3f}] accesses/retired instr (observed honest range +/-5%).")
    print(f"    Honest false positives on {len(fresh)} fresh readings of the calibration classes: {fp}")
    novel = [generate_reading(NOVEL_CLASS, make_rng(derive_seed("novel", i))) for i in range(100)]
    print(f"    Honest readings of an unseen workload class ({NOVEL_CLASS.name}) flagged: {sum(any(inspect(r, th).values()) for r in novel)} of 100")

    print("\n[2] Attack vs check, 300 attacked readings each (share flagged by each check that fires on >= 5%):")
    matrix = detection_matrix(fresh, th, lambda name: make_rng(derive_seed("attack", name)))
    rows = []
    for name, st in matrix.items():
        fired = sorted(((v, k) for k, v in st.flagged_by.items() if v >= 0.05), reverse=True)
        only = sorted(((v, k) for k, v in st.exclusive.items() if v >= 0.05), reverse=True)
        rows.append((name, _pct(st.any_flag), "; ".join(f"{k} {_pct(v)}" for v, k in fired), "; ".join(f"{k} {_pct(v)}" for v, k in only) or "-",
                     "- (none unflagged)" if st.any_flag == 1.0 else f"{st.undetected_hidden_max:.4f}"))
    print(_table(["attack", "flagged", "checks that fire", "flagged by that check alone", "max hidden if unflagged"], rows))

    big = honest_readings(CLASSES, 500, make_rng(37))
    for n_flips in (1, 2):
        st = evaluate_attack(lambda r, g, n=n_flips: emp_bit_flips(r, g, n_flips=n), big, th, make_rng(derive_seed("emp-large", n_flips)))
        print(f"    EMP, {n_flips} random bit flip(s) per reading, {st.n} readings: flagged {_pct(st.any_flag)}; "
              f"largest FLOP fraction hidden by a flip nothing flagged: {_pct(st.undetected_hidden_max)}")

    print("\n[3] Consistent under-reporting (every counter x (1-h)) is visible only through power. Detection rate, "
          "Monte Carlo (600 readings) vs exact formula:")
    rows = []
    for cname, cls in CLASSES.items():
        for h in (0.05, 0.10, 0.15):
            mc, an = scale_detection(cls, th, h, 600, make_rng(derive_seed("scale", cname, h)))
            rows.append((cname, f"{h:.2f}", _pct(mc), _pct(an)))
    print(_table(["class", "hidden h", "measured", "formula"], rows))
    rows = []
    for cname, cls in CLASSES.items():
        r0 = generate_reading(cls, make_rng(0), power_noise=0.0)
        pdyn = dynamic_power(r0)
        rows.append((cname, f"{r0.power_w:.0f} W", f"{pdyn:.0f} W",
                     _pct(min_detectable_fraction(pdyn, r0.power_w, th.sigma_rel, th.z_crit, 1)),
                     _pct(min_detectable_fraction(pdyn, r0.power_w, th.sigma_rel, th.z_crit, 40)),
                     _pct(min_detectable_fraction(pdyn, r0.power_w, 0.10, th.z_crit, 40))))
    print(_table(["class", "chip power", "dynamic", "50% point, 1 interval", "50% point, 40 intervals", "40 intervals, sigma 10%"], rows))

    print("\n[4] Power-matched hiding (attacker knows thresholds, adds fake memory traffic to hold predicted power):")
    rows = [(r.cls, _pct(r.closed_agnostic), _pct(r.searched_agnostic), _pct(r.closed_aware), _pct(r.searched_aware)) for r in hiding_table(per_class)]
    print(_table(["workload", "agnostic bands: formula", "agnostic: search", "class-aware bands: formula", "class-aware: search"], rows))

    print("\n[5] Widening the bands trades false positives on an unseen honest workload for room to hide:")
    rows = [(f"{r.margin:.2f}", _pct(r.novel_false_positive_rate), _pct(r.seen_false_positive_rate), _pct(r.dense_matmul_hidden))
            for r in margin_tradeoff(per_class, (0.0, 0.02, 0.05, 0.25, 0.5, 0.6, 0.7, 0.9), make_rng(7), n_test=600)]
    print(_table(["band margin", "unseen workload flagged", "seen workloads flagged", "dense-matmul hideable"], rows))
    idle_reading = generate_reading(CLASSES["dense-matmul"], make_rng(3), active_fraction=0.5)
    flags = [k for k, v in inspect(fake_traffic_on_idle_sms(idle_reading, 0.5), th).items() if v]
    print(f"    Fake traffic parked on idle SMs while hiding 50% of FLOPs is flagged by: {', '.join(flags)}")

    print("\n[6] Signed-readout protocol, 40 intervals, attack at interval 20 (seed 0):")
    rows = []
    for s in SCENARIOS:
        o = simulate_device(s, seed=0)
        rows.append((s, f"{o.intervals_accepted}/40", f"{o.power_z:.1f}", _pct(o.hidden_fraction), o.verdict))
    print(_table(["scenario", "accepted", "power z", "work hidden", "outcome"], rows))
    print(f"    Chance a constant 1% under-count escapes a 40-interval power test: {_pct(power_miss_probability(0.01, 40))}; "
          f"0.5%: {_pct(power_miss_probability(0.005, 40))}; 5%: {_pct(power_miss_probability(0.05, 40))}")

    print("\n[7] Exhaustive exploration of the protocol code (ideal signatures, tiny counters):")
    a = explore(ALPHABET_HIDING, 8)
    b = explore(ALPHABET_PROTOCOL, 6)
    rows = [
        ("hiding actions (work, read, glitch, stall)", 8, a.transitions, a.accepted_readouts, a.mature_windows, len(a.violations), dict(a.detected), _pct(a.max_window_hidden_fraction)),
        ("protocol actions (+replay, drop, tickets)", 6, b.transitions, b.accepted_readouts, b.mature_windows, len(b.violations), dict(b.detected), _pct(b.max_window_hidden_fraction)),
    ]
    print(_table(["alphabet", "depth", "transitions", "readouts accepted", "mature windows", "invariant violations", "traces ended by", "max window hidden"], rows))
    print(f"    Shortest accepted trace that hides a whole mature window: {' -> '.join(a.witness or ())}")

    print("\n[8] Redundant counters: exhaustive subsets (always detected / always recovered) and a random EMP disc of radius 0.15:")
    for k in (3, 5):
        print("    k=%d: " % k + "; ".join(f"t={t}: {d}/{r}" for t, (d, r) in exhaustive_detection(k).items()))
    rows = []
    for name, pos in layouts((1, 2, 3)).items():
        o = emp_outcomes(pos, 0.15, 100_000, make_rng(derive_seed("emp", name)))
        rows.append((name, _pct(o["no effect"]), _pct(o["detected"]), _pct(o["undetected"])))
    print(_table(["layout", "no effect", "detected", "undetected"], rows))

    print("\n[9] Overhead and width arithmetic:")
    ov = protocol_overhead(100_000, 1.0)
    print(f"    Readout {ov['readout_bytes']:.0f} B (closing {ov['closing_readout_bytes']:.0f} B); 100,000 devices at 1 readout/s: "
          f"{ov['bytes_per_second_total'] / 1e6:.1f} MB/s, {ov['signature_verifications_per_second']:,.0f} signature verifications/s")
    for exponent in (15, 16):
        print(f"    A 64-bit FLOP counter at 10^{exponent} FLOP/s wraps after {seconds_to_wrap(64, 10.0**exponent) / 3600:.2f} hours")
    print(f"    Counting 10^26 FLOPs needs {bits_needed(10**26)} bits. With an 8-bit counter growing at most 2 per tick, readouts must be at most "
          f"{safe_gap(8, 2)} ticks apart; if 600 units accumulate between two reads, {aliased_work(8, 600)} of them alias away")


if __name__ == "__main__":
    main()
