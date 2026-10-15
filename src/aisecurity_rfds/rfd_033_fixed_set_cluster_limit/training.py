"""Training across pods under an external bandwidth cap (RFD 033, open question 2).

A closed-form model of one optimiser step of a dense transformer trained over ``K`` pipeline stages x ``R`` data-
parallel replicas, every stage a pod of ``S`` chips. Each pod's traffic leaves through its external links, whose
capacity is ``S * b`` bytes/s (the cap ``b`` applied per chip) or just ``b`` (applied per pod). Per optimiser step,
per pod:

* compute:   ``t_c = 6 N Bt / (K R S F_eff)``                      (6 N FLOPs per token; ``Bt`` tokens per step)
* pipeline:  ``(Bt / R) * 2 d a`` bytes sent, if ``K > 1``         (activation forward + gradient backward per token)
* data par.: ``2 (R-1)/R * (N/K) g / (c H)`` bytes sent, if ``R > 1``  (ring all-reduce of the stage's gradients,
  shrunk by compression ``c``, amortised over ``H`` local steps)

With perfect overlap a step takes ``max(t_c, t_m)``; without, ``t_c + t_m``. The constants 6, 2 and ``2 (R-1)/R``
are checked against executed counts in ``primitives.py`` by the tests; everything else is an *assumption* (see
``Hardware``, ``Run``) and is meant to be varied. Not modelled: latency, pipeline bubbles, stragglers, failures,
sharded optimiser state, mixture-of-experts, and the extra tokens that compression and local steps cost in
practice (the model lets the attacker shrink communication for free, which favours the attacker).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .limiter import TokenBucket

FLOPS_PER_PARAM_TOKEN = 6.0
PIPELINE_VECTORS_PER_TOKEN = 2.0
YEAR_S = 365.25 * 24 * 3600


@dataclass(frozen=True)
class Hardware:
    """Assumed accelerator: ``flops`` is a dense low-precision peak in FLOP/s, ``mfu`` the sustained fraction."""

    flops: float = 1.0e15
    mfu: float = 0.4
    mem_bytes: float = 80e9

    @property
    def effective_flops(self) -> float:
        return self.flops * self.mfu


@dataclass(frozen=True)
class Model:
    n_params: float
    d_model: float
    n_layers: float

    @staticmethod
    def dense(n_params: float, aspect: float = 128.0) -> "Model":
        """A dense decoder with ``N = 12 L d^2`` and ``d = aspect * L`` (the aspect ratio is an assumption)."""
        layers = (n_params / (12.0 * aspect**2)) ** (1.0 / 3.0)
        return Model(n_params, aspect * layers, layers)


@dataclass(frozen=True)
class Run:
    model: Model
    tokens: float
    batch_tokens: float = 16e6
    grad_bytes: float = 2.0
    act_bytes: float = 2.0
    state_bytes_per_param: float = 16.0  # weights + gradients + optimiser state, mixed precision (assumption)

    @property
    def flops(self) -> float:
        return FLOPS_PER_PARAM_TOKEN * self.model.n_params * self.tokens


@dataclass(frozen=True)
class Layout:
    stages: int = 1
    replicas: int = 1
    compression: float = 1.0
    local_steps: int = 1

    @property
    def pods(self) -> int:
        return self.stages * self.replicas


@dataclass(frozen=True)
class External:
    """The cap on traffic leaving a pod: ``per_chip`` scales with pod size, ``per_pod`` does not."""

    per_chip_bps: float
    scope: str = "per_chip"

    def pod_bps(self, pod_size: int) -> float:
        if self.scope == "per_chip":
            return self.per_chip_bps * pod_size
        if self.scope == "per_pod":
            return self.per_chip_bps
        raise ValueError("scope must be 'per_chip' or 'per_pod'")


def ring_factor(p: np.ndarray | float) -> np.ndarray:
    """Bytes each worker sends in a ring all-reduce, as a multiple of the vector: ``2 (P-1) / P``."""
    p = np.asarray(p, dtype=float)
    return 2.0 * (p - 1.0) / p


def min_stages(run: Run, hw: Hardware, pod_size: int) -> int:
    """Fewest pods a model replica must span for its training state to fit in pod memory."""
    return max(1, math.ceil(run.model.n_params * run.state_bytes_per_param / (pod_size * hw.mem_bytes)))


def compute_time(run: Run, hw: Hardware, pod_size: int, stages: int, replicas: np.ndarray | float) -> np.ndarray:
    r = np.asarray(replicas, dtype=float)
    return (
        FLOPS_PER_PARAM_TOKEN * run.model.n_params * run.batch_tokens
        / (stages * r * pod_size * hw.effective_flops)
    )


def pod_bytes_per_step(
    run: Run, stages: int, replicas: np.ndarray | float, compression: float, local_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """(pipeline bytes, data-parallel bytes) one pod sends per optimiser step."""
    r = np.asarray(replicas, dtype=float)
    if stages > 1:
        pipe = (run.batch_tokens / r) * PIPELINE_VECTORS_PER_TOKEN * run.model.d_model * run.act_bytes
    else:
        pipe = np.zeros_like(r)
    dp = np.where(
        r > 1,
        ring_factor(np.maximum(r, 1.0)) * (run.model.n_params / stages) * run.grad_bytes / (compression * local_steps),
        0.0,
    )
    return pipe, dp


def step_time(
    run: Run,
    hw: Hardware,
    pod_size: int,
    ext: External,
    stages: int,
    replicas: np.ndarray | float,
    *,
    compression: float = 1.0,
    local_steps: int = 1,
    overlap: bool = True,
) -> np.ndarray:
    t_c = compute_time(run, hw, pod_size, stages, replicas)
    pipe, dp = pod_bytes_per_step(run, stages, replicas, compression, local_steps)
    t_m = (pipe + dp) / ext.pod_bps(pod_size)
    return np.maximum(t_c, t_m) if overlap else t_c + t_m


def throughput_flops(run: Run, step_s: np.ndarray | float) -> np.ndarray:
    """Useful model FLOP/s: the FLOPs of one step divided by its duration."""
    return FLOPS_PER_PARAM_TOKEN * run.model.n_params * run.batch_tokens / np.asarray(step_s, dtype=float)


def utilisation(run: Run, hw: Hardware, pod_size: int, ext: External, layout: Layout, *, overlap: bool = True) -> float:
    t_c = compute_time(run, hw, pod_size, layout.stages, layout.replicas)
    t = step_time(
        run, hw, pod_size, ext, layout.stages, layout.replicas,
        compression=layout.compression, local_steps=layout.local_steps, overlap=overlap,
    )
    return float(t_c / t)


# ---- break-even bandwidths and ceilings (closed forms) ----------------------------------------------------------


def breakeven_dp_bps(run: Run, hw: Hardware, replicas: int, compression: float = 1.0, local_steps: int = 1) -> float:
    """Per-chip cap at which pure data parallelism spends as long communicating as computing.

    ``(R - 1) g F_eff / (3 c H Bt)``: independent of model size and of pod size.
    """
    return (replicas - 1) * run.grad_bytes * hw.effective_flops / (
        FLOPS_PER_PARAM_TOKEN / 2.0 * compression * local_steps * run.batch_tokens
    )


def breakeven_pp_bps(run: Run, hw: Hardware, stages: int) -> float:
    """Per-chip cap at which a single pipeline (one replica) spends as long communicating as computing.

    ``K F_eff d a / (3 N)``: independent of batch size and of pod size.
    """
    m = run.model
    return stages * hw.effective_flops * m.d_model * run.act_bytes / (FLOPS_PER_PARAM_TOKEN / 2.0 * m.n_params)


def dp_ceiling_flops(run: Run, pod_size: int, per_chip_bps: float, compression: float = 1.0, local_steps: int = 1) -> float:
    """Most useful FLOP/s pure data parallelism can reach however many pods are added: ``3 Bt c H S b / g``."""
    return (
        FLOPS_PER_PARAM_TOKEN / 2.0 * run.batch_tokens * compression * local_steps * pod_size * per_chip_bps
        / run.grad_bytes
    )


def pp_ceiling_flops(run: Run, pod_size: int, per_chip_bps: float) -> float:
    """Most useful FLOP/s one pipeline replica can reach however many stages it has: ``3 N S b / (d a)``."""
    m = run.model
    return FLOPS_PER_PARAM_TOKEN / 2.0 * m.n_params * pod_size * per_chip_bps / (m.d_model * run.act_bytes)


# ---- what an attacker has to buy ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    layout: Layout
    chips: int
    utilisation: float
    time_to_train_s: float


def ideal_chips(run: Run, hw: Hardware, deadline_s: float) -> float:
    """Chips needed to finish by the deadline if communication were free."""
    return run.flops / (deadline_s * hw.effective_flops)


def _k_grid(k_min: int, k_max: int) -> list[int]:
    ks = {k_min}
    k = 1
    while k <= k_max:
        if k >= k_min:
            ks.add(k)
        k *= 2
    return sorted(ks)


def plan(
    run: Run,
    hw: Hardware,
    pod_size: int,
    ext: External,
    deadline_s: float,
    *,
    compression: float = 1.0,
    local_steps: int = 1,
    overlap: bool = True,
    r_max: int = 4096,
    k_candidates: list[int] | None = None,
    k_max: int = 512,
) -> Plan | None:
    """Fewest chips that finish the run by the deadline over any layout in the search grid, or ``None``."""
    needed = run.flops / deadline_s
    k_floor = min_stages(run, hw, pod_size)
    ks = k_candidates if k_candidates is not None else _k_grid(k_floor, k_max)
    replicas = np.arange(1, r_max + 1, dtype=float)
    best: Plan | None = None
    for k in ks:
        if k < k_floor:
            continue
        t = step_time(run, hw, pod_size, ext, k, replicas, compression=compression, local_steps=local_steps, overlap=overlap)
        ok = np.flatnonzero(throughput_flops(run, t) >= needed * (1 - 1e-12))
        if not len(ok):
            continue
        r = int(replicas[ok[0]])
        chips = k * r * pod_size
        if best is None or chips < best.chips:
            layout = Layout(k, r, compression, local_steps)
            best = Plan(
                layout, chips, utilisation(run, hw, pod_size, ext, layout, overlap=overlap),
                float(run.tokens / run.batch_tokens * t[ok[0]]),
            )
    return best


def cost_multiplier(
    run: Run, hw: Hardware, pod_size: int, ext: External, deadline_s: float, **kw: object
) -> float:
    """Chips of the cheapest plan relative to the communication-free ideal (``inf`` if there is no plan)."""
    p = plan(run, hw, pod_size, ext, deadline_s, **kw)  # type: ignore[arg-type]
    return math.inf if p is None else p.chips / ideal_chips(run, hw, deadline_s)


def defeat_bandwidth_bps(
    run: Run,
    hw: Hardware,
    pod_size: int,
    deadline_s: float,
    *,
    multiplier: float = 10.0,
    scope: str = "per_chip",
    **kw: object,
) -> float:
    """The cap (in the given scope) below which the cheapest plan needs more than ``multiplier`` x the ideal chips.

    Found by bisection in log space; the cost multiplier never increases when the cap is raised, so the answer is
    a threshold. Returned in bytes/s.
    """
    lo, hi = 1e3, 1e16
    if cost_multiplier(run, hw, pod_size, External(hi, scope), deadline_s, **kw) > multiplier:
        return math.inf
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if cost_multiplier(run, hw, pod_size, External(mid, scope), deadline_s, **kw) > multiplier:
            lo = mid
        else:
            hi = mid
    return hi


# ---- check of the closed forms against the limiter -------------------------------------------------------------


def simulate_ring_sync_seconds(vector_bytes: float, replicas: int, pod_bps: int, *, burst_bytes: int = 1 << 24) -> float:
    """Time for one ring all-reduce when every pod's egress is a ``TokenBucket`` (all pods are identical).

    ``2 (R-1)`` synchronous phases each push ``vector_bytes / R`` through the bucket; the next phase starts when
    the previous has drained. The first burst goes out at line rate, which is the only difference from the closed
    form ``ring_factor(R) * vector_bytes / pod_bps``.
    """
    bucket = TokenBucket(int(pod_bps), burst_bytes)
    chunk = int(round(vector_bytes / replicas))
    now_ns = 0
    for _ in range(2 * (replicas - 1)):
        now_ns = bucket.stream(chunk, now_ns)
    return now_ns / 1e9
