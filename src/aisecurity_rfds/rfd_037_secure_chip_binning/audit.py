"""Sampling audits of whether restricted circuits are physically absent (RFD 037, open question 2).

The RFD lists five ways to inspect a chip (X-ray, decapping with optical imaging, full delayering, comparison with
design files, functional testing) and says each has a blind spot; only delayering is called complete, and it is
destructive and expensive. This module does not model those techniques. It takes one number as an input, the
probability ``detect`` that inspecting one chip that carries the restricted circuits reveals them, and answers the
statistical questions around it: how many chips must be inspected, what fraction of the lot that destroys, and how
much a cheating manufacturer can hide. Every ``detect`` value used by the callers is an assumption.

A comparison with design files reads what the manufacturer declares, so it passes whatever was fabricated; only
inspecting physical chips is independent of the manufacturer's honesty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from aisecurity_rfds.common.stats import McEstimate, monte_carlo_probability


_SLACK = 1e-8  # relative slack when comparing a miss probability with a target (the log-space error is ~4e-10)


def _log_comb(n: int | np.ndarray, k: int | np.ndarray) -> np.ndarray:
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def _check(lot: int, cheated: int, inspected: int, detect: float) -> None:
    if lot < 1 or not 0 <= cheated <= lot or not 0 <= inspected <= lot:
        raise ValueError("require lot >= 1, 0 <= cheated <= lot and 0 <= inspected <= lot")
    if not 0.0 <= detect <= 1.0:
        raise ValueError("detect must be in [0, 1]")


def miss_probability(lot: int, cheated: int, inspected: int, detect: float = 1.0) -> float:
    """P(an audit of ``inspected`` chips drawn without replacement from ``lot`` finds nothing).

    ``cheated`` chips of the lot carry the restricted circuits; inspecting one reveals them with probability
    ``detect``. The number of cheated chips drawn is hypergeometric, and each is missed independently.

    The terms are summed in log space; for a lot of 100,000 the relative error is about 4e-10 (checked against
    exact integer arithmetic), which is why the threshold searches below compare with a relative slack.
    """
    _check(lot, cheated, inspected, detect)
    hits = np.arange(max(0, inspected - (lot - cheated)), min(cheated, inspected) + 1)
    log_pmf = _log_comb(cheated, hits) + _log_comb(lot - cheated, inspected - hits) - _log_comb(lot, inspected)
    return float(np.sum(np.exp(log_pmf) * (1.0 - detect) ** hits))


def inspections_needed(lot: int, cheated: int, confidence: float, detect: float = 1.0) -> int | None:
    """Fewest chips to inspect so that the audit finds cheating with at least ``confidence``, or None if no number does.

    A miss probability within ``_SLACK`` (relative) above the target still counts as meeting it, so a lot where the
    answer is an exact tie (one cheated chip, 95% confidence, 95,000 chips) is not pushed up by one by rounding.
    """
    if cheated < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("require cheated >= 1 and 0 < confidence < 1")
    ceiling = (1.0 - confidence) * (1.0 + _SLACK)
    if miss_probability(lot, cheated, lot, detect) > ceiling:
        return None
    lo, hi = 0, lot
    while lo < hi:
        mid = (lo + hi) // 2
        if miss_probability(lot, cheated, mid, detect) <= ceiling:
            hi = mid
        else:
            lo = mid + 1
    return lo


def evasion_budget(lot: int, inspected: int, tolerated_miss: float, detect: float = 1.0) -> int:
    """Most cheated chips a manufacturer can ship while the audit still misses them with probability >= ``tolerated_miss``.

    As in ``inspections_needed``, a miss probability within ``_SLACK`` (relative) below the tolerance still counts.
    """
    if not 0.0 < tolerated_miss <= 1.0:
        raise ValueError("tolerated_miss must be in (0, 1]")
    floor = tolerated_miss * (1.0 - _SLACK)
    lo, hi = 0, lot
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if miss_probability(lot, mid, inspected, detect) >= floor:
            lo = mid
        else:
            hi = mid - 1
    return lo


def destroyed_fraction(lot: int, inspected: int) -> float:
    """Share of the lot consumed by destructive inspection."""
    _check(lot, 0, inspected, 1.0)
    return inspected / lot


def repeated_miss_probability(lot: int, cheated: int, inspected: int, detect: float, lots: int) -> float:
    """P(cheating the same way in each of ``lots`` independently audited lots is never found)."""
    if lots < 0:
        raise ValueError("lots cannot be negative")
    return miss_probability(lot, cheated, inspected, detect) ** lots


@dataclass(frozen=True)
class Lot:
    """A lot of chips. ``cheated`` are the indices that carry the restricted circuits; ``declared`` is the design
    the manufacturer says they were built to."""

    size: int
    cheated: frozenset[int]
    declared: str


def design_files_match(reference: str, lot: Lot) -> bool:
    """Comparison with design files: it can only compare what the manufacturer declares."""
    return lot.declared == reference


def sample_audit(lot: Lot, inspected: int, detect: float, rng: np.random.Generator) -> bool:
    """One physical audit: True if any inspected cheated chip is detected."""
    _check(lot.size, len(lot.cheated), inspected, detect)
    drawn = rng.choice(lot.size, size=inspected, replace=False)
    return any(int(i) in lot.cheated and rng.random() < detect for i in drawn)


def estimate_detection(
    lot: Lot, inspected: int, detect: float, trials: int, rng: np.random.Generator
) -> McEstimate:
    """Monte Carlo estimate of the probability that ``sample_audit`` finds the cheating."""
    return monte_carlo_probability(lambda r: sample_audit(lot, inspected, detect, r), trials, rng)
