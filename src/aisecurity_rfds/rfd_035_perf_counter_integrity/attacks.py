"""The counter-tampering attacks named in the RFD, as transformations of an honest :class:`Reading`.

Each function takes an honest reading and returns what the verifier would receive after the attack. SM-level
attacks edit the per-SM counters and the (honest) aggregators recompute from them; the aggregator attack edits
only the aggregators. Physical power draw is never edited: the attacker cannot fake watts by editing registers.

"Hidden fraction" is ``(true FLOPs - metered FLOPs) / true FLOPs`` where metered FLOPs is what the aggregators
report, because that is the number a licence meter or compute-accounting scheme would use.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import norm

from .checks import CHECK_NAMES, Thresholds, inspect
from .model import K_FLOP, K_MEM, N_SM, P_IDLE, Reading, WorkloadClass, generate_reading

AttackFn = Callable[[Reading, np.random.Generator], Reading]

_COUNTER_FIELDS = ("issued", "retired", "flops", "mem", "miss")


def metered_flops(r: Reading) -> float:
    """The FLOP total a downstream meter would use: the sum of the aggregator outputs."""
    return float(sum(r.agg_flops))


def hidden_fraction(true: Reading, attacked: Reading) -> float:
    t = metered_flops(true)
    return max(0.0, (t - metered_flops(attacked)) / t) if t > 0 else 0.0


def _ints(r: Reading) -> dict[str, np.ndarray]:
    return {name: np.round(getattr(r, name)).astype(np.int64) for name in _COUNTER_FIELDS}


def _with_int_counters(r: Reading, arrays: dict[str, np.ndarray]) -> Reading:
    arrays = dict(arrays)
    arrays["flops"] = arrays["flops"].astype(np.float64)
    return r.with_counters(**arrays)


# --- the six attack vectors of the RFD's table -------------------------------------------------------------


def emp_bit_flips(r: Reading, rng: np.random.Generator, n_flips: int = 4, max_bit: int = 40) -> Reading:
    """EMP injection: uncontrolled single-bit upsets in random counters of random SMs."""
    arrays = _ints(r)
    for _ in range(n_flips):
        name = _COUNTER_FIELDS[int(rng.integers(len(_COUNTER_FIELDS)))]
        arrays[name][int(rng.integers(N_SM))] ^= 1 << int(rng.integers(max_bit))
    return _with_int_counters(r, arrays)


def voltage_glitch_reset(r: Reading, rng: np.random.Generator, fraction: float = 0.5, scope: str = "all") -> Reading:
    """Voltage glitch: counter registers of a random subset of SMs are cleared (``scope``: ``all`` or ``flops``)."""
    arrays = _ints(r)
    hit = rng.choice(N_SM, max(1, int(round(fraction * N_SM))), replace=False)
    for name in _COUNTER_FIELDS if scope == "all" else ("flops",):
        arrays[name][hit] = 0
    return _with_int_counters(r, arrays)


def laser_flop_scale(r: Reading, rng: np.random.Generator, h: float = 0.5, n_sms: int = 4) -> Reading:
    """Laser fault injection with precise control: FLOP counters of chosen SMs are scaled by ``1 - h``."""
    arrays = _ints(r)
    hit = rng.choice(N_SM, n_sms, replace=False)
    arrays["flops"][hit] = np.round(arrays["flops"][hit] * (1.0 - h)).astype(np.int64)
    return _with_int_counters(r, arrays)


def fib_wire_cut(r: Reading, rng: np.random.Generator, n_sms: int = 4) -> Reading:
    """FIB wire cutting: the FLOP-increment wire of chosen SMs is severed, so that counter stays at zero."""
    arrays = _ints(r)
    arrays["flops"][rng.choice(N_SM, n_sms, replace=False)] = 0
    return _with_int_counters(r, arrays)


def firmware_consistent_scale(r: Reading, rng: np.random.Generator, h: float = 0.3) -> Reading:
    """Unsigned firmware that scales *every* counter on *every* SM by ``1 - h`` (keeps all ratios intact)."""
    arrays = {k: np.round(v * (1.0 - h)).astype(np.int64) for k, v in _ints(r).items()}
    arrays["issued"] = np.maximum(arrays["issued"], arrays["retired"])
    arrays["mem"] = np.maximum(arrays["mem"], arrays["miss"])
    return _with_int_counters(r, arrays)


def aggregator_tamper(r: Reading, rng: np.random.Generator, h: float = 0.3) -> Reading:
    """Aggregation attack: the aggregators under-report; the per-SM counters they should sum are untouched."""
    return replace(r, agg_flops=tuple(a * (1.0 - h) for a in r.agg_flops))


ATTACKS: dict[str, AttackFn] = {
    "EMP bit flips (4 random)": emp_bit_flips,
    "voltage glitch clears half the SMs": voltage_glitch_reset,
    "laser: FLOPs -50% on 4 SMs": laser_flop_scale,
    "FIB: FLOP wire cut on 4 SMs": fib_wire_cut,
    "firmware: all counters -30%": firmware_consistent_scale,
    "aggregators -30%": aggregator_tamper,
}


# --- evaluation ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AttackStats:
    n: int
    flagged_by: dict[str, float]  # fraction of attacked readings each check flagged
    any_flag: float
    undetected_hidden_mean: float  # mean hidden fraction over readings nothing flagged (0 if none)
    undetected_hidden_max: float
    exclusive: dict[str, float]  # fraction flagged by this check and no other


def honest_readings(classes: Mapping[str, WorkloadClass], n_per_class: int, rng: np.random.Generator) -> list:
    return [generate_reading(c, rng) for c in classes.values() for _ in range(n_per_class)]


def evaluate_attack(attack: AttackFn, readings: Sequence, th: Thresholds, rng: np.random.Generator) -> AttackStats:
    counts = dict.fromkeys(CHECK_NAMES, 0)
    exclusive = dict.fromkeys(CHECK_NAMES, 0)
    any_flag = 0
    hidden_undetected: list[float] = []
    for honest in readings:
        attacked = attack(honest, rng)
        flags = inspect(attacked, th)
        raised = [name for name, hit in flags.items() if hit]
        for name in raised:
            counts[name] += 1
        if len(raised) == 1:
            exclusive[raised[0]] += 1
        if raised:
            any_flag += 1
        else:
            hidden_undetected.append(hidden_fraction(honest, attacked))
    n = len(readings)
    return AttackStats(
        n=n,
        flagged_by={k: v / n for k, v in counts.items()},
        any_flag=any_flag / n,
        undetected_hidden_mean=float(np.mean(hidden_undetected)) if hidden_undetected else 0.0,
        undetected_hidden_max=float(np.max(hidden_undetected)) if hidden_undetected else 0.0,
        exclusive={k: v / n for k, v in exclusive.items()},
    )


def detection_matrix(readings: Sequence, th: Thresholds, seed_rng: Callable[[str], np.random.Generator]) -> dict[str, AttackStats]:
    return {name: evaluate_attack(fn, readings, th, seed_rng(name)) for name, fn in ATTACKS.items()}


# --- consistent scaling: the power check alone ---------------------------------------------------------------


def analytic_power_detection_probability(
    h: float, p_dynamic: float, power: float, sigma_rel: float, z_crit: float, n: int = 1, noise_rel: float | None = None
) -> float:
    """P(|z| > z_crit) when counters behind ``p_dynamic`` watts are under-reported by ``h`` for ``n`` identical windows.

    ``sigma_rel`` is the noise the verifier assumes, ``noise_rel`` the noise actually present (default: equal),
    ``power`` the true chip power. The statistic normalises by the predicted power ``P' = power - h * p_dynamic``,
    so it is exactly Gaussian: ``z ~ N(sqrt(n) * h * p_dynamic / (sigma_rel * P'), (noise_rel * power / (sigma_rel * P'))**2)``.
    """
    nz = sigma_rel if noise_rel is None else noise_rel
    predicted = power - h * p_dynamic
    mean = np.sqrt(n) * h * p_dynamic / (sigma_rel * predicted)
    sd = nz * power / (sigma_rel * predicted)
    return float(norm.sf((z_crit - mean) / sd) + norm.cdf((-z_crit - mean) / sd))


def dynamic_power(r: Reading) -> float:
    return K_FLOP * float(r.flops.sum()) + K_MEM * float(r.mem.sum())


def scale_detection(cls: WorkloadClass, th: Thresholds, h: float, n_trials: int, rng: np.random.Generator) -> tuple[float, float]:
    """(Monte Carlo rate at which any check flags the consistent scaling attack, analytic expectation).

    Only the power check can flag it, so the two should agree. The analytic value is evaluated per honest
    reading (its own dynamic power and total power) and averaged over the same sample.
    """
    hits = 0
    analytic = 0.0
    for _ in range(n_trials):
        honest = generate_reading(cls, rng)
        hits += any(inspect(firmware_consistent_scale(honest, rng, h), th).values())
        analytic += analytic_power_detection_probability(h, dynamic_power(honest), total_baseline_power(honest), th.sigma_rel, th.z_crit)
    return hits / n_trials, analytic / n_trials


def total_baseline_power(r: Reading) -> float:
    return P_IDLE + dynamic_power(r)
