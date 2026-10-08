"""Synthetic GPU telemetry: per-SM counters plus a chip-level power draw.

Every constant and workload class below is an ASSUMPTION chosen to give plausible relative magnitudes. The
numbers are in arbitrary but consistent units (one "interval" of counters, watts for power). Nothing here
was measured on hardware; the results built on it characterise this model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

N_SM = 32
PEAK_ISSUE = 1000.0  # instructions an SM can issue per interval
FLOPS_PER_FLOP_INSTR = 8.0
P_IDLE = 100.0  # watts
K_FLOP = 1.5e-3  # watts per FLOP counted in the interval
K_MEM = 2.0e-2  # watts per memory access counted in the interval


@dataclass(frozen=True)
class WorkloadClass:
    name: str
    utilisation: float
    flop_instr_fraction: float  # fraction of retired instructions that are floating-point
    mem_fraction: float  # memory accesses per retired instruction
    miss_rate: float  # cache misses per memory access

    @property
    def flops_per_retired(self) -> float:
        return self.flop_instr_fraction * FLOPS_PER_FLOP_INSTR


CLASSES = {
    "dense-matmul": WorkloadClass("dense-matmul", 0.9, 0.70, 0.10, 0.15),
    "decode-memory-bound": WorkloadClass("decode-memory-bound", 0.5, 0.25, 0.50, 0.35),
    "elementwise": WorkloadClass("elementwise", 0.6, 0.10, 0.40, 0.25),
}
# A workload class that is NOT in the calibration set, to expose false positives from over-tight bands.
NOVEL_CLASS = WorkloadClass("sparse-graph", 0.3, 0.04, 0.55, 0.60)


@dataclass(frozen=True)
class Reading:
    """One metering interval as the verifier receives it."""

    issued: np.ndarray
    retired: np.ndarray
    flops: np.ndarray
    mem: np.ndarray
    miss: np.ndarray
    power_w: float  # measured chip power (physical; the attacker cannot edit it)
    agg_flops: tuple[float, ...]  # outputs of the aggregators, each summing a disjoint group of SMs

    def with_counters(self, **arrays: np.ndarray) -> Reading:
        """Replace SM counters and recompute honest aggregator outputs from them (an SM-level attack)."""
        new = replace(self, **arrays)
        return replace(new, agg_flops=aggregate(new.flops, len(self.agg_flops)))


def aggregate(flops: np.ndarray, n_aggregators: int) -> tuple[float, ...]:
    return tuple(float(g.sum()) for g in np.array_split(flops, n_aggregators))


def true_power(flops_total: float, mem_total: float) -> float:
    return P_IDLE + K_FLOP * flops_total + K_MEM * mem_total


def generate_reading(
    cls: WorkloadClass,
    rng: np.random.Generator,
    *,
    active_fraction: float = 1.0,
    power_noise: float = 0.02,
    n_aggregators: int = 2,
) -> Reading:
    n_active = max(1, int(round(N_SM * active_fraction)))
    active = np.zeros(N_SM, dtype=bool)
    active[rng.choice(N_SM, n_active, replace=False)] = True
    util = np.where(active, np.clip(cls.utilisation * rng.uniform(0.9, 1.1, N_SM), 0, 1), 0.0)
    issued = np.round(util * PEAK_ISSUE).astype(np.int64)
    retired = np.floor(issued * rng.uniform(0.92, 1.0, N_SM)).astype(np.int64)
    flops = np.round(retired * cls.flop_instr_fraction * rng.uniform(0.97, 1.03, N_SM)) * FLOPS_PER_FLOP_INSTR
    mem = np.round(retired * cls.mem_fraction * rng.uniform(0.97, 1.03, N_SM)).astype(np.int64)
    miss = np.round(mem * cls.miss_rate * rng.uniform(0.9, 1.1, N_SM)).astype(np.int64)
    power = true_power(float(flops.sum()), float(mem.sum())) * (1.0 + power_noise * rng.standard_normal())
    return Reading(issued, retired, flops.astype(np.float64), mem, miss, float(power), aggregate(flops, n_aggregators))
