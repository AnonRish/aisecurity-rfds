"""What a pod of a given size can train in a given time (RFD 033, open question 1).

The RFD's size table says qualitatively how capable each pod size is ("small models", "limited frontier
capability", "frontier-capable") and the RAND text it quotes tentatively puts the line at 65 chips pending work on
algorithmic efficiency. This module turns a pod size into training compute, and an algorithmic-efficiency doubling
time into the time until a fixed pod reaches a given capability. All hardware numbers and thresholds are inputs,
not findings: the point is the structure of the dependence (linear in chips and in per-chip throughput, exponential
in time through algorithmic progress).
"""

from __future__ import annotations

import math

from .training import YEAR_S, Hardware


def pod_compute(pod_size: int, hw: Hardware, seconds: float = YEAR_S) -> float:
    """FLOPs a pod of ``pod_size`` chips can sustain over ``seconds``."""
    if pod_size < 1 or seconds <= 0:
        raise ValueError("pod size and duration must be positive")
    return pod_size * hw.effective_flops * seconds


def max_pod_size(compute_cap: float, hw: Hardware, seconds: float = YEAR_S) -> int:
    """Largest pod whose sustained compute over ``seconds`` stays at or below ``compute_cap`` (0 if none does)."""
    return int(math.floor(compute_cap / (hw.effective_flops * seconds) + 1e-9))


def compute_optimal_params(compute: float, tokens_per_param: float = 20.0) -> float:
    """Parameters of the model a compute budget trains at a fixed tokens-per-parameter ratio (``C = 6 N D``)."""
    return math.sqrt(compute / (6.0 * tokens_per_param))


def years_until_pod_reaches(
    target_compute: float, pod_size: int, hw: Hardware, doubling_years: float, seconds: float = YEAR_S
) -> float:
    """Years until a pod bought today is, through algorithmic progress alone, worth ``target_compute`` of today's.

    Effective compute grows as ``2 ** (t / doubling_years)``; the hardware is fixed, so only algorithms improve.
    """
    if doubling_years <= 0:
        raise ValueError("doubling time must be positive")
    have = pod_compute(pod_size, hw, seconds)
    if have >= target_compute:
        return 0.0
    return doubling_years * math.log2(target_compute / have)
