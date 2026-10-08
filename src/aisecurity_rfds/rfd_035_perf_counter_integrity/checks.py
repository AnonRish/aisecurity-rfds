"""Verifier-side sanity checks and their calibration.

Each check returns True when it *flags* a reading. Thresholds are calibrated on honest readings, so how tight
they are (and therefore how much an attacker can hide) is a modelling choice made visible here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .model import K_FLOP, K_MEM, N_SM, P_IDLE, PEAK_ISSUE, Reading, true_power

CHECK_NAMES = (
    "issued>=retired",
    "accesses>=misses",
    "throughput-bound",
    "flop-per-instruction-band",
    "memory-per-instruction-band",
    "peer-outlier",
    "aggregator-cross-check",
    "power-consistency",
)


@dataclass(frozen=True)
class Thresholds:
    rho_lo: float  # allowed FLOPs per retired instruction (per active SM)
    rho_hi: float
    mu_lo: float  # allowed memory accesses per retired instruction
    mu_hi: float
    peer_scale: float = 6.0  # robust z for the peer comparison
    agg_tol: float = 1e-9  # relative disagreement allowed between aggregators and the SM-level readout
    z_crit: float = 3.72  # two-sided false-positive rate 2 * Phi(-3.72), about 2e-4 per interval
    sigma_rel: float = 0.02  # relative power-measurement noise the verifier assumes


def calibrate(readings: Sequence[Reading], margin: float = 0.05, **overrides: float) -> Thresholds:
    """Bands = observed honest min/max per active SM, widened by ``margin``."""
    rho, mu = [], []
    for r in readings:
        ok = r.retired > 0
        rho.append(r.flops[ok] / r.retired[ok])
        mu.append(r.mem[ok] / r.retired[ok])
    rho_all, mu_all = np.concatenate(rho), np.concatenate(mu)
    return Thresholds(
        rho_lo=float(rho_all.min() * (1 - margin)),
        rho_hi=float(rho_all.max() * (1 + margin)),
        mu_lo=float(mu_all.min() * (1 - margin)),
        mu_hi=float(mu_all.max() * (1 + margin)),
        **overrides,  # type: ignore[arg-type]
    )


def predicted_power(r: Reading) -> float:
    return P_IDLE + K_FLOP * float(r.flops.sum()) + K_MEM * float(r.mem.sum())


def power_z(readings: Sequence[Reading], sigma_rel: float) -> float:
    """Standardised power residual over a window: positive when the chip drew more than the counters explain.

    Each residual is normalised by the *predicted* power (not the measured one), so for honest counters it is
    exactly ``noise / sigma_rel`` and the statistic is Gaussian with a symmetric two-sided false-positive rate.
    """
    resid = []
    for r in readings:
        pred = predicted_power(r)
        resid.append((r.power_w - pred) / (sigma_rel * pred))
    return float(np.sum(resid) / np.sqrt(len(resid)))


def power_z_from_deltas(flop_deltas: Sequence[float], mem_deltas: Sequence[float], powers: Sequence[float], sigma_rel: float) -> float:
    """Same statistic as :func:`power_z`, computed from per-interval counter increments (e.g. successive
    cumulative readouts) and the chip power measured in each interval."""
    resid = []
    for f, m, p in zip(flop_deltas, mem_deltas, powers, strict=True):
        pred = P_IDLE + K_FLOP * f + K_MEM * m
        resid.append((p - pred) / (sigma_rel * pred))
    return float(np.sum(resid) / np.sqrt(len(resid)))


def inspect(r: Reading, th: Thresholds) -> dict[str, bool]:
    active = r.retired > 0
    ratio = np.divide(r.flops, r.retired, out=np.zeros(N_SM), where=active)
    memr = np.divide(r.mem, r.retired, out=np.zeros(N_SM), where=active)
    if active.sum() >= 3:
        med = float(np.median(ratio[active]))
        scale = max(1.4826 * float(np.median(np.abs(ratio[active] - med))), 0.02 * med, 1e-12)
        peer = bool(np.any(np.abs(ratio[active] - med) > th.peer_scale * scale))
    else:
        peer = False
    group_sums = [float(g.sum()) for g in np.array_split(r.flops, len(r.agg_flops))]
    agg = any(abs(a - s) > th.agg_tol * max(1.0, s) for a, s in zip(r.agg_flops, group_sums))
    # Idle-SM invariant: an SM that retired nothing cannot have counted FLOPs or memory traffic. Without it the
    # per-instruction ratios are undefined there and fake traffic could be parked on "idle" SMs.
    idle = ~active
    orphan_flops = bool(np.any(idle & (r.flops > 0)))
    orphan_mem = bool(np.any(idle & ((r.mem > 0) | (r.miss > 0))))
    return {
        "issued>=retired": bool(np.any(r.retired > r.issued)),
        "accesses>=misses": bool(np.any(r.miss > r.mem)),
        "throughput-bound": bool(np.any(r.issued > PEAK_ISSUE)),
        "flop-per-instruction-band": orphan_flops
        or bool(np.any((ratio[active] < th.rho_lo) | (ratio[active] > th.rho_hi))),
        "memory-per-instruction-band": orphan_mem
        or bool(np.any((memr[active] < th.mu_lo) | (memr[active] > th.mu_hi))),
        "peer-outlier": peer,
        "aggregator-cross-check": agg,
        "power-consistency": abs(power_z([r], th.sigma_rel)) > th.z_crit,
    }


def power_z_threshold(false_positive_rate: float) -> float:
    return float(norm.isf(false_positive_rate))


def min_detectable_fraction(hidable_dynamic_w: float, power_w: float, sigma_rel: float, z_crit: float, n_windows: int = 1) -> float:
    """Fraction of the hidable dynamic power at which the power check has a 50% chance of firing.

    Under-reporting a fraction ``h`` of the counters behind ``hidable_dynamic_w`` watts of prediction shifts the
    standardised residual by ``sqrt(n) * h * D / (sigma_rel * (P - h * D))`` (``D`` = hidable watts, ``P`` = chip
    power); the 50% point is where that shift equals ``z_crit``, i.e. ``h = z sigma P / (D (sqrt(n) + z sigma))``.
    Values above 1 mean the attack cannot be seen at all.
    """
    return float(z_crit * sigma_rel * power_w / (hidable_dynamic_w * (np.sqrt(n_windows) + z_crit * sigma_rel)))


def min_detectable_hidden_fraction(flops_total: float, power_w: float, sigma_rel: float, z_crit: float, n_windows: int) -> float:
    """Same, when only FLOP counters are under-reported (memory counters stay honest)."""
    return min_detectable_fraction(K_FLOP * flops_total, power_w, sigma_rel, z_crit, n_windows)


def honest_dynamic_power(flops_total: float, mem_total: float) -> float:
    return true_power(flops_total, mem_total) - P_IDLE
