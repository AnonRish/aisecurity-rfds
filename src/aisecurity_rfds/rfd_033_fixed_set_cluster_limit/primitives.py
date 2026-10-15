"""Executable ground truth for the constants the bandwidth analysis relies on.

The analysis in ``training.py`` uses three textbook facts: (1) a dense layer costs 6 FLOPs per parameter per token
to train, (2) a ring all-reduce makes each worker send ``2 (P - 1) / P`` times the vector, and (3) a pipeline
boundary carries one activation vector forward and one gradient vector backward per token. Instead of trusting
memory, this module *runs* a tiny version of each and counts, so a test can assert the constant against the count.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def ring_allreduce(vectors: Sequence[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    """Textbook ring all-reduce (reduce-scatter, then all-gather) of equal-length vectors.

    Returns every worker's final vector and the number of bytes each worker *sent*.
    """
    p = len(vectors)
    if p < 1:
        raise ValueError("need at least one worker")
    n = len(vectors[0])
    if any(len(v) != n for v in vectors) or n % p:
        raise ValueError("vectors must have equal length divisible by the worker count")
    size = n // p
    data = [np.array(v, dtype=np.float64).reshape(p, size) for v in vectors]
    sent = [0] * p
    for step in range(p - 1):  # reduce-scatter
        outgoing = [(i, (i - step) % p, data[i][(i - step) % p].copy()) for i in range(p)]
        for i, chunk_index, chunk in outgoing:
            sent[i] += chunk.nbytes
            data[(i + 1) % p][chunk_index] += chunk
    for step in range(p - 1):  # all-gather
        outgoing = [(i, (i + 1 - step) % p, data[i][(i + 1 - step) % p].copy()) for i in range(p)]
        for i, chunk_index, chunk in outgoing:
            sent[i] += chunk.nbytes
            data[(i + 1) % p][chunk_index] = chunk
    return [d.reshape(-1) for d in data], sent


class CountingMatmul:
    """``a @ b`` that adds ``2 m k n`` to ``flops`` (one multiply and one add per term)."""

    def __init__(self) -> None:
        self.flops = 0

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        (m, k), (k2, n) = a.shape, b.shape
        if k != k2:
            raise ValueError("shape mismatch")
        self.flops += 2 * m * k * n
        return a @ b


def train_step_flops(widths: Sequence[int], tokens: int, rng: np.random.Generator) -> tuple[int, int]:
    """Run one forward and backward pass of a dense stack and return (matmul FLOPs, parameter count).

    Biases and element-wise operations are omitted (they are not matmuls); the gradient with respect to every
    layer's input is computed, which is what the "6 N" rule counts.
    """
    mm = CountingMatmul()
    weights = [rng.standard_normal((a, b)) * 0.1 for a, b in zip(widths[:-1], widths[1:])]
    acts = [rng.standard_normal((tokens, widths[0]))]
    for w in weights:
        acts.append(mm(acts[-1], w))
    grad = rng.standard_normal(acts[-1].shape)
    for w, a_in in zip(reversed(weights), reversed(acts[:-1])):
        _ = mm(a_in.T, grad)  # gradient of the weights
        grad = mm(grad, w.T)  # gradient of the layer input
    return mm.flops, sum(w.size for w in weights)


def pipeline_boundary_bytes(d_model: int, tokens: int, rng: np.random.Generator, dtype: type = np.float16) -> int:
    """Bytes that cross one pipeline boundary for a batch of tokens: activations forward plus gradients backward."""
    forward = rng.standard_normal((tokens, d_model)).astype(dtype)
    backward = rng.standard_normal((tokens, d_model)).astype(dtype)
    return forward.nbytes + backward.nbytes


def transformer_block_params(d_model: int, ffn_mult: int = 4) -> int:
    """Parameters of one decoder block from its weight shapes (attention projections plus the two MLP matrices)."""
    shapes = [(d_model, d_model)] * 4 + [(d_model, ffn_mult * d_model), (ffn_mult * d_model, d_model)]
    return sum(a * b for a, b in shapes)
