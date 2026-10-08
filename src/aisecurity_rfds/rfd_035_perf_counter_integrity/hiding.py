"""How much can an attacker hide while every sanity check stays quiet?

The attacker modelled here knows the verifier's thresholds and the power model. It under-reports a fraction ``h``
of FLOPs and, so that the power check is satisfied, adds fake memory traffic worth the same predicted watts
(``K_FLOP / K_MEM`` accesses per hidden FLOP). It then rescales the retired-instruction counter so that the
per-instruction ratios stay inside the verifier's bands. Whether such a scale exists is a feasibility question
with a closed form, which this module states, solves, and cross-checks by brute-force search against the real
checks.

With class-agnostic bands (calibrated on all workloads) the feasible ``h`` is large; with bands calibrated on the
workload actually declared it is small. Tight bands, however, reject honest workloads they were not calibrated on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .checks import Thresholds, calibrate, inspect
from .model import (
    CLASSES,
    K_FLOP,
    K_MEM,
    N_SM,
    NOVEL_CLASS,
    PEAK_ISSUE,
    Reading,
    WorkloadClass,
    aggregate,
    generate_reading,
    true_power,
)

POWER_MATCH_RATIO = K_FLOP / K_MEM  # fake memory accesses per hidden FLOP that keep predicted power unchanged


def ideal_reading(cls: WorkloadClass, *, retired: int = 500, n_active: int = N_SM) -> Reading:
    """A noise-free reading: every active SM identical and equal to the class means."""
    active = np.arange(N_SM) < n_active
    ret = np.where(active, retired, 0).astype(np.int64)
    flops = np.where(active, retired * cls.flops_per_retired, 0.0)
    mem = np.where(active, round(retired * cls.mem_fraction), 0).astype(np.int64)
    miss = np.where(active, round(retired * cls.mem_fraction * cls.miss_rate), 0).astype(np.int64)
    power = true_power(float(flops.sum()), float(mem.sum()))
    return Reading(ret.copy(), ret, flops, mem, miss, power, aggregate(flops, 2))


# --- the power-matched attack and its feasibility ------------------------------------------------------------


def power_matched_report(r: Reading, h: float, retired_scale: float) -> Reading:
    """What the attacker reports: FLOPs x (1-h), memory accesses raised to hold predicted power, retired x scale."""
    active = r.retired > 0
    flops = np.where(active, r.flops * (1.0 - h), r.flops)
    mem = np.where(active, np.round(r.mem + POWER_MATCH_RATIO * h * r.flops), r.mem).astype(np.int64)
    retired = np.where(active, np.maximum(1, np.round(r.retired * retired_scale)), r.retired).astype(np.int64)
    issued = np.where(active, np.maximum(r.issued, retired), r.issued).astype(np.int64)
    return r.with_counters(issued=issued, retired=retired, flops=flops, mem=mem)


def feasible_scale_interval(r: Reading, th: Thresholds, h: float) -> tuple[float, float]:
    """Interval of retired-instruction scales keeping every per-SM ratio in its band and issued <= peak."""
    active = r.retired > 0
    rho = r.flops[active] / r.retired[active]
    mu_new = (r.mem[active] + POWER_MATCH_RATIO * h * r.flops[active]) / r.retired[active]
    lo = max((1 - h) * rho.max() / th.rho_hi, mu_new.max() / th.mu_hi)
    hi = min((1 - h) * rho.min() / th.rho_lo, mu_new.min() / th.mu_lo, PEAK_ISSUE / r.retired[active].max())
    return float(lo), float(hi)


def find_hiding_report(r: Reading, th: Thresholds, h: float, tries: int = 41) -> Reading | None:
    """A power-matched report that no check flags, or ``None`` if the attacker cannot find one at this ``h``."""
    lo, hi = feasible_scale_interval(r, th, h)
    if lo > hi:
        return None
    for frac in np.linspace(0.0, 1.0, tries):
        for sign in (1, -1):
            s = (lo + hi) / 2 + sign * frac * (hi - lo) / 2
            candidate = power_matched_report(r, h, s)
            if not any(inspect(candidate, th).values()):
                return candidate
    return None


def max_hidden_fraction(r: Reading, th: Thresholds, iterations: int = 30) -> float:
    """Largest ``h`` for which :func:`find_hiding_report` succeeds (bisection; feasibility is monotone in ``h``)."""
    if find_hiding_report(r, th, 0.0) is None:
        return 0.0  # the honest reading is itself rejected by these thresholds
    lo, hi = 0.0, 0.999
    if find_hiding_report(r, th, hi) is not None:
        return hi
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if find_hiding_report(r, th, mid) is not None:
            lo = mid
        else:
            hi = mid
    return lo


def closed_form_max_hidden_fraction(rho: float, mu: float, th: Thresholds, retired_per_sm: float) -> float:
    """Closed form for identical SMs with ``rho`` FLOPs and ``mu`` memory accesses per retired instruction.

    Feasibility needs a retired scale ``s`` with ``(1-h) rho / s`` in ``[rho_lo, rho_hi]``, ``(mu + c h rho) / s``
    in ``[mu_lo, mu_hi]`` and ``s * retired <= PEAK`` where ``c = K_FLOP / K_MEM``. The binding constraints are

    * ``(mu + c h rho) / mu_hi <= (1-h) rho / rho_lo``  ->  ``h <= (kappa - mu/rho) / (kappa + c)``, kappa = mu_hi/rho_lo
    * ``(mu + c h rho) / mu_hi <= PEAK / retired``       ->  ``h <= (mu_hi PEAK / retired - mu) / (c rho)``
    """
    c = POWER_MATCH_RATIO
    kappa = th.mu_hi / th.rho_lo
    h_band = (kappa - mu / rho) / (kappa + c)
    h_cap = (th.mu_hi * PEAK_ISSUE / retired_per_sm - mu) / (c * rho)
    return float(max(0.0, min(1.0, h_band, h_cap)))


# --- class-agnostic versus class-aware bands -------------------------------------------------------------------


@dataclass(frozen=True)
class HidingRow:
    cls: str
    closed_agnostic: float
    searched_agnostic: float
    closed_aware: float
    searched_aware: float


def calibration_readings(classes: Mapping[str, WorkloadClass], n_per_class: int, rng: np.random.Generator) -> dict[str, list[Reading]]:
    return {name: [generate_reading(c, rng) for _ in range(n_per_class)] for name, c in classes.items()}


def hiding_table(per_class: Mapping[str, Sequence[Reading]], margin: float = 0.05) -> list[HidingRow]:
    """Maximum hidden FLOP fraction per workload under class-agnostic and class-aware bands (noise-free target)."""
    agnostic = calibrate([r for rs in per_class.values() for r in rs], margin=margin)
    rows = []
    for name, readings in per_class.items():
        cls = CLASSES[name]
        aware = calibrate(readings, margin=margin)
        target = ideal_reading(cls)
        rho, mu = cls.flops_per_retired, cls.mem_fraction
        rows.append(
            HidingRow(
                name,
                closed_form_max_hidden_fraction(rho, mu, agnostic, 500),
                max_hidden_fraction(target, agnostic),
                closed_form_max_hidden_fraction(rho, mu, aware, 500),
                max_hidden_fraction(target, aware),
            )
        )
    return rows


@dataclass(frozen=True)
class TradeoffRow:
    margin: float
    novel_false_positive_rate: float  # honest readings of the unseen class flagged by any structural check
    seen_false_positive_rate: float  # honest fresh readings of the calibration classes flagged by any structural check
    dense_matmul_hidden: float  # closed-form maximum hidden fraction for dense-matmul under these bands


def margin_tradeoff(
    per_class: Mapping[str, Sequence[Reading]], margins: Sequence[float], rng: np.random.Generator, n_test: int = 150
) -> list[TradeoffRow]:
    """Widening the bands to stop flagging an unseen honest workload also widens what an attacker can hide.

    False-positive rates count the structural checks only: the power check's own rate does not depend on the margin.
    """
    calib = [r for rs in per_class.values() for r in rs]

    def structural(r: Reading, th: Thresholds) -> bool:
        return any(hit for name, hit in inspect(r, th).items() if name != "power-consistency")

    novel = [generate_reading(NOVEL_CLASS, rng) for _ in range(n_test)]
    seen = [generate_reading(c, rng) for c in CLASSES.values() for _ in range(n_test // len(CLASSES))]
    dense = CLASSES["dense-matmul"]
    rows = []
    for m in margins:
        th = calibrate(calib, margin=m)
        rows.append(
            TradeoffRow(
                m,
                float(np.mean([structural(r, th) for r in novel])),
                float(np.mean([structural(r, th) for r in seen])),
                closed_form_max_hidden_fraction(dense.flops_per_retired, dense.mem_fraction, th, 500),
            )
        )
    return rows


def fake_traffic_on_idle_sms(r: Reading, h: float) -> Reading:
    """Hide a fraction ``h`` of FLOPs and park the power-matching fake memory traffic on SMs that report no work."""
    active = r.retired > 0
    extra_total = POWER_MATCH_RATIO * h * float(r.flops[active].sum())
    idle_idx = np.flatnonzero(~active)
    mem = r.mem.copy()
    mem[idle_idx] = np.round(extra_total / len(idle_idx)).astype(np.int64)
    flops = np.where(active, r.flops * (1.0 - h), r.flops)
    return r.with_counters(flops=flops, mem=mem)
