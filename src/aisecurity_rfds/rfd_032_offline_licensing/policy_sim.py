"""Mid-job exhaustion policies, license granularity and stockpile exposure.

``simulate`` is an abstract accounting model, not a model of any deployment: job sizes are lognormal with
median 1 (``sigma`` is an ASSUMPTION), one unit of work takes one unit of time at full speed, and a new license
arrives ``delay`` time units after the balance runs out (the worst case: the operator asks only when empty).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aisecurity_rfds.common.hashing import sha256
from aisecurity_rfds.common.signing import KeyPair

from .chip import OfflineLicenseChip
from .licensing import LicenseAuthority
from .meters import Meter

POLICIES = ("throttle", "checkpoint", "hard-stop")


@dataclass(frozen=True)
class SimParams:
    budget: float  # work units per license
    sigma: float = 0.5
    delay: float = 10.0
    baseline_rate: float = 0.01
    n_jobs: int = 5000


@dataclass(frozen=True)
class SimResult:
    policy: str
    interrupted_fraction: float
    never_completed_fraction: float
    mean_slowdown: float  # elapsed time per unit of useful work, over jobs that completed
    wasted_fraction: float  # work discarded by hard stops, relative to all offered work
    reissues: int


def _run_job(size: float, bal: float, policy: str, p: SimParams) -> tuple[float, float, float, bool, bool, int]:
    """Returns (elapsed, wasted, new balance, interrupted, completed, reissues)."""
    elapsed = wasted = 0.0
    remaining = size
    interrupted = False
    reissues = 0
    while True:
        usable = max(0.0, bal)
        if remaining <= usable:
            return elapsed + remaining, wasted, bal - remaining, interrupted, True, reissues
        interrupted = True
        elapsed += usable
        remaining -= usable
        reissues += 1
        if policy == "checkpoint":
            elapsed += p.delay
        elif policy == "throttle":
            progress = p.baseline_rate * p.delay  # work done at the baseline while waiting for the license
            elapsed += p.delay  # the operator stays blocked until the license arrives
            remaining = max(0.0, remaining - progress)
        else:  # hard-stop: the job is killed and its progress discarded
            wasted += size - remaining
            elapsed += p.delay
            remaining = size
            if size > p.budget:  # a job larger than one license can never finish without checkpoints
                return elapsed, wasted, p.budget, True, False, reissues
        bal = p.budget  # the balance was empty; the new license replaces it
        if remaining == 0.0:
            return elapsed, wasted, bal, True, True, reissues


def simulate(policy: str, params: SimParams, rng: np.random.Generator) -> SimResult:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if params.budget <= 0 or params.delay < 0:
        raise ValueError("budget must be positive and delay non-negative")
    sizes = rng.lognormal(mean=0.0, sigma=params.sigma, size=params.n_jobs)
    bal = params.budget
    interrupted = completed = reissues = 0
    done_work = done_time = wasted_total = 0.0
    for size in sizes:
        elapsed, wasted, bal, was_interrupted, ok, n_re = _run_job(float(size), bal, policy, params)
        interrupted += was_interrupted
        reissues += n_re
        wasted_total += wasted
        if ok:
            completed += 1
            done_work += float(size)
            done_time += elapsed
    n = params.n_jobs
    return SimResult(
        policy,
        interrupted / n,
        1 - completed / n,
        done_time / done_work if done_work else float("inf"),
        wasted_total / float(sizes.sum()),
        reissues,
    )


# ------------------------------------------------------------------ power cycling against a real chip model


@dataclass(frozen=True)
class PowerCycleReport:
    cycles: int
    max_gain: int  # largest (restored - true) balance seen at a power cycle; must be <= 0 to be fail-safe
    max_loss: int  # largest (true - restored) balance, bounded by the checkpoint interval
    checkpoint_writes: int


def power_cycle_experiment(
    budget: int, interval: int, rng: np.random.Generator, *, steps: int = 400, cycle_prob: float = 0.05
) -> PowerCycleReport:
    """Consume a license in random steps while cutting power at random; compare restored and true balances."""
    serial = b"CHIP-POWER"
    authority = LicenseAuthority(KeyPair.from_seed(sha256(b"rfd032 authority")))
    chip = OfflineLicenseChip(
        serial, authority.public_key, enforced=[Meter.FLOPS], checkpoint_interval={Meter.FLOPS: interval}
    )
    chip.apply_license(authority.issue(serial, sequence=1, budgets={Meter.FLOPS: budget}))
    cycles = 0
    max_gain, max_loss = -budget, 0
    for _ in range(steps):
        chip.consume({Meter.FLOPS: int(rng.integers(1, max(2, interval)))})
        if rng.random() < cycle_prob:
            true_before = chip.remaining(Meter.FLOPS)
            chip.power_cycle()
            restored = chip.remaining(Meter.FLOPS)
            cycles += 1
            max_gain = max(max_gain, restored - true_before)
            max_loss = max(max_loss, true_before - restored)
    return PowerCycleReport(cycles, max_gain, max_loss, chip.checkpoint_writes)


def checkpoint_interval_for_endurance(budget: int, writes_allowed: int) -> int:
    """Smallest checkpoint interval that burns a ``budget`` with at most ``writes_allowed`` checkpoint writes."""
    if budget < 0 or writes_allowed < 1:
        raise ValueError("budget must be >= 0 and writes_allowed >= 1")
    return -(-budget // writes_allowed)


# ------------------------------------------------------------------ stockpiling against a real chip model


def _fresh_chip(serial: bytes, cap: int) -> tuple[OfflineLicenseChip, LicenseAuthority]:
    authority = LicenseAuthority(KeyPair.from_seed(sha256(b"rfd032 authority")))
    chip = OfflineLicenseChip(serial, authority.public_key, enforced=[Meter.FLOPS], max_balance={Meter.FLOPS: cap})
    return chip, authority


def drawer_stockpile(k: int, budget: int, cap: int) -> tuple[int, int]:
    """An attacker holds ``k`` pre-issued top-up licenses and applies each as soon as the cap allows.

    Returns ``(total budget consumed, peak balance on the chip)``. The cap limits the burst, not the total.
    """
    serial = b"CHIP-STOCK"
    chip, authority = _fresh_chip(serial, cap)
    licenses = [authority.issue(serial, sequence=s, budgets={Meter.FLOPS: budget}) for s in range(1, k + 1)]
    total = peak = 0
    for lic in licenses:
        if not chip.apply_license(lic).accepted:  # balance-cap: burn down the balance first, then retry
            used = chip.remaining(Meter.FLOPS)
            chip.consume({Meter.FLOPS: used})
            total += used
            if not chip.apply_license(lic).accepted:
                raise RuntimeError("license rejected even with an empty balance (cap below one license)")
        peak = max(peak, chip.remaining(Meter.FLOPS))
    total += chip.remaining(Meter.FLOPS)
    return total, peak


def revocation_outcome(k: int, budget: int, deliver_revocation_first: bool) -> int:
    """Authority issues ``k`` top-ups plus a later ``set`` to 0. Returns budget the attacker consumes in total.

    The revocation only helps if the chip applies it before the stockpiled licenses: once a higher
    sequence number has been applied, every lower-numbered license is stale.
    """
    serial = b"CHIP-REVOKE"
    chip, authority = _fresh_chip(serial, 2**62)
    topups = [authority.issue(serial, sequence=s, budgets={Meter.FLOPS: budget}) for s in range(1, k + 1)]
    revoke = authority.issue(serial, sequence=k + 1, budgets={Meter.FLOPS: 0}, mode="set")
    total = 0
    if deliver_revocation_first:
        chip.apply_license(revoke)
    for lic in topups:
        if chip.apply_license(lic).accepted:
            total += chip.remaining(Meter.FLOPS)
            chip.consume({Meter.FLOPS: chip.remaining(Meter.FLOPS)})
    return total
