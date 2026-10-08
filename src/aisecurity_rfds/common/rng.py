"""Deterministic randomness helpers so every simulation and test is reproducible."""

from __future__ import annotations

import hashlib

import numpy as np


def make_rng(seed: int | None = 0) -> np.random.Generator:
    """Return a numpy Generator. ``seed=None`` gives OS entropy (never used in tests)."""
    return np.random.default_rng(seed)


def derive_seed(*labels: object) -> int:
    """Derive a stable 64-bit seed from arbitrary labels (e.g. device id, trial number)."""
    h = hashlib.sha256()
    for label in labels:
        data = repr(label).encode()
        h.update(len(data).to_bytes(4, "big"))
        h.update(data)
    return int.from_bytes(h.digest()[:8], "big")
