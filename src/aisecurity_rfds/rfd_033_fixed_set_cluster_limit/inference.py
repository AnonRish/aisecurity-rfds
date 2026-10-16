"""Serving a large model across pods, and how close that is to training it (RFD 033, open question 4).

The RFD asks how to handle "legitimate multi-pod workloads (e.g., large inference)". Under a per-chip external cap
the question is which parallelism a multi-pod server can use. This module compares what each needs from the
external links, per token, using the same constants as ``training.py``: pipeline serving sends one activation
vector per boundary per token; tensor-parallel serving all-reduces two activation vectors per layer per token.
"""

from __future__ import annotations

import math

from .training import PIPELINE_VECTORS_PER_TOKEN, Hardware, Model, ring_factor


def weight_pods(model: Model, pod_size: int, hw: Hardware, weight_bytes: float = 1.0) -> int:
    """Pods whose combined memory holds the weights (activations and caches ignored)."""
    return max(1, math.ceil(model.n_params * weight_bytes / (pod_size * hw.mem_bytes)))


def pipeline_bytes_per_token(model: Model, act_bytes: float = 2.0, *, training: bool = False) -> float:
    """Bytes a middle stage sends per token: one activation vector forward, plus one gradient vector if training."""
    vectors = PIPELINE_VECTORS_PER_TOKEN if training else 1.0
    return vectors * model.d_model * act_bytes


def tensor_parallel_bytes_per_token(model: Model, degree: int, act_bytes: float = 2.0) -> float:
    """Bytes a worker sends per token when the ``degree`` workers are pods: two all-reduces per layer, forward."""
    if degree < 2:
        return 0.0
    return 2.0 * model.n_layers * float(ring_factor(degree)) * model.d_model * act_bytes


def boundary_tokens_per_s(bytes_per_token: float, pod_size: int, per_chip_bps: float) -> float:
    """Tokens per second the external links of one pod can carry at ``bytes_per_token``."""
    return pod_size * per_chip_bps / bytes_per_token


def compute_tokens_per_s(model: Model, chips: int, hw: Hardware) -> float:
    """Tokens per second ``chips`` sustain when each token costs ``2 N`` FLOPs (forward only)."""
    return chips * hw.effective_flops / (2.0 * model.n_params)
