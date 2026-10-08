"""Throttle actions and a small job-execution model.

Every performance parameter here is an ASSUMPTION used to compare throttle actions, not a measurement of
any GPU: ``cuda_over_tensor`` (how fast non-tensor units run a tensor workload relative to tensor cores),
``baseline_clock`` (the RFD's 1% example) and ``memory_bw_limit`` (bandwidth left by a limiter).
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass

from .chip import OfflineLicenseChip
from .meters import Meter


class ThrottleAction(str, enum.Enum):
    CLOCK_REDUCTION = "clock-reduction"
    FUNCTIONAL_UNIT_DISABLE = "functional-unit-disable"
    MEMORY_BW_LIMIT = "memory-bandwidth-limit"
    COMPLETE_DISABLE = "complete-disable"


@dataclass(frozen=True)
class HardwareProfile:
    baseline_clock: float = 0.01
    cuda_over_tensor: float = 1 / 16
    memory_bw_limit: float = 0.05

    def __post_init__(self) -> None:
        for name in ("baseline_clock", "cuda_over_tensor", "memory_bw_limit"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass(frozen=True)
class Job:
    name: str
    units: int
    cost: Mapping[Meter, int]  # meter consumption per unit of work
    full_rate: float  # units per virtual second at full speed
    uses_tensor: bool
    memory_bound_fraction: float  # fraction of full-speed time spent waiting on memory

    def __post_init__(self) -> None:
        if self.units < 0 or self.full_rate <= 0 or not 0 <= self.memory_bound_fraction <= 1:
            raise ValueError("invalid job parameters")


def throttle_rate(action: ThrottleAction, job: Job, profile: HardwareProfile) -> float:
    """Fraction of full speed the job runs at once the throttle is engaged."""
    if action is ThrottleAction.CLOCK_REDUCTION:
        return profile.baseline_clock
    if action is ThrottleAction.FUNCTIONAL_UNIT_DISABLE:
        return profile.cuda_over_tensor if job.uses_tensor else 1.0
    if action is ThrottleAction.MEMORY_BW_LIMIT:
        beta = job.memory_bound_fraction
        return 1.0 / ((1.0 - beta) + beta / profile.memory_bw_limit)
    return 0.0


@dataclass(frozen=True)
class JobResult:
    seconds: float
    units_done: int
    completed: bool
    full_speed_units: int
    throttled_units: int


def run_job(
    chip: OfflineLicenseChip, job: Job, action: ThrottleAction, profile: HardwareProfile | None = None
) -> JobResult:
    """Run at full speed until an enforced meter empties, then at the throttled rate (meters stay at zero)."""
    profile = profile or HardwareProfile()
    available = chip.units_available(job.cost)
    full = job.units if available is None else min(job.units, available)
    chip.consume({m: c * full for m, c in job.cost.items()})
    seconds = full / job.full_rate
    rest = job.units - full
    if rest == 0:
        return JobResult(seconds, job.units, True, full, 0)
    rate = throttle_rate(action, job, profile)
    if rate == 0.0:
        return JobResult(seconds, full, False, full, 0)
    return JobResult(seconds + rest / (job.full_rate * rate), job.units, True, full, rest)


# Workload presets: the memory-bound fractions and tensor usage are ASSUMPTIONS for illustration.
def preset_jobs(units: int = 1000) -> dict[str, Job]:
    flops = {Meter.FLOPS: 1}
    return {
        "gaming": Job("gaming", units, flops, 100.0, False, 0.3),
        "ai-training": Job("ai-training", units, flops, 100.0, True, 0.3),
        "ai-inference": Job("ai-inference", units, flops, 100.0, True, 0.8),
    }
