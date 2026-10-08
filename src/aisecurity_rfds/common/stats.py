"""Detection-probability and confidence-interval helpers shared by the sampling-based RFDs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats as _sps


def poisson_miss_probability(n_verified: float, fake_fraction: float) -> float:
    """P(no fake packet caught) ~ exp(-n_verified * fake_fraction) (Poisson approximation, RFD 018)."""
    return math.exp(-n_verified * fake_fraction)


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_miss_probability(total: int, fake: int, sampled: int) -> float:
    """Exact P(a uniform sample of ``sampled`` of ``total`` items, without replacement, contains no fake)."""
    if not 0 <= fake <= total or not 0 <= sampled <= total:
        raise ValueError("require 0 <= fake <= total and 0 <= sampled <= total")
    if sampled > total - fake:
        return 0.0
    return math.exp(_log_comb(total - fake, sampled) - _log_comb(total, sampled))


def samples_needed_poisson(fake_fraction: float, confidence: float) -> int:
    if not 0 < fake_fraction <= 1 or not 0 < confidence < 1:
        raise ValueError("fake_fraction in (0,1], confidence in (0,1)")
    return math.ceil(-math.log(1 - confidence) / fake_fraction)


def samples_needed_hypergeom(total: int, fake: int, confidence: float) -> int:
    """Smallest sample size whose exact miss probability is at most ``1 - confidence``."""
    if fake < 1 or fake > total or not 0 < confidence < 1:
        raise ValueError("need 1 <= fake <= total and 0 < confidence < 1")
    target = 1 - confidence
    lo, hi = 0, total
    while lo < hi:
        mid = (lo + hi) // 2
        if hypergeom_miss_probability(total, fake, mid) <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if successes == 0 else _sps.beta.ppf(alpha / 2, successes, trials - successes + 1)
    hi = 1.0 if successes == trials else _sps.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return float(lo), float(hi)


@dataclass(frozen=True)
class McEstimate:
    p: float
    lo: float
    hi: float
    trials: int

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi


def monte_carlo_probability(
    trial: Callable[[np.random.Generator], bool], n_trials: int, rng: np.random.Generator
) -> McEstimate:
    """Estimate P(trial is True) with a 95% Wilson interval. ``trial`` must use the rng it is given."""
    hits = sum(1 for _ in range(n_trials) if trial(rng))
    lo, hi = wilson_interval(hits, n_trials)
    return McEstimate(hits / n_trials, lo, hi, n_trials)
