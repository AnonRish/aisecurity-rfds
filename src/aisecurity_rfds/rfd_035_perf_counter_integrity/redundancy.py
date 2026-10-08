"""Distributed redundancy: ``k`` physically separate copies of a counter, compared when read.

A fault that corrupts fewer than ``k`` copies is detected (the copies disagree); with ``k >= 2 t + 1`` the true value
is also recovered by majority against ``t`` corrupted copies. A fault that reaches *every* copy and writes the same
wrong value is invisible. Whether a fault can reach every copy is a layout question, which is what the geometric
Monte Carlo below explores: an EMP pulse is modelled as a disc of corruption of a given radius, centred uniformly on
a unit-square die, that writes one common wrong value into every copy it covers (the attacker's best case).

A global fault such as a supply-voltage glitch reaches all copies sharing the supply, so copies help against it only
if they sit in independent power domains; that is a physical-design assumption this code cannot test.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import combinations

import numpy as np


class RedundantCounter:
    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("need at least one copy")
        self.copies = [0] * k

    def add(self, n: int) -> None:
        self.copies = [c + n for c in self.copies]

    def corrupt(self, indices: Iterable[int], value: int) -> None:
        for i in indices:
            self.copies[i] = value

    def consistent(self) -> bool:
        return len(set(self.copies)) == 1

    def majority(self) -> int | None:
        value, votes = Counter(self.copies).most_common(1)[0]
        return value if votes * 2 > len(self.copies) else None


def exhaustive_detection(k: int, true_value: int = 10, wrong_value: int = 3) -> dict[int, tuple[bool, bool]]:
    """For every reach ``t`` and *every* subset of ``t`` copies: is the fault detected, and is the truth recovered?

    Returns ``{t: (always_detected, always_recovered)}``.
    """
    out: dict[int, tuple[bool, bool]] = {}
    for t in range(1, k + 1):
        detected, recovered = True, True
        for subset in combinations(range(k), t):
            c = RedundantCounter(k)
            c.add(true_value)
            c.corrupt(subset, wrong_value)
            detected &= not c.consistent()
            recovered &= c.majority() == true_value
        out[t] = (detected, recovered)
    return out


def clustered_layout(k: int) -> np.ndarray:
    """All copies within a 0.04-wide cluster at the die centre (one logic block)."""
    angles = 2 * np.pi * np.arange(k) / max(k, 1)
    return 0.5 + 0.02 * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def spread_layout(k: int) -> np.ndarray:
    """Copies spread evenly on a circle of radius 0.35 around the die centre."""
    angles = 2 * np.pi * np.arange(k) / max(k, 1)
    return 0.5 + 0.35 * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def emp_outcomes(positions: np.ndarray, radius: float, n: int, rng: np.random.Generator) -> dict[str, float]:
    """Outcome of a random EMP pulse: ``no effect`` (covers no copy), ``detected`` (some but not all), ``undetected``."""
    centres = rng.uniform(0.0, 1.0, (n, 2))
    covered = np.linalg.norm(centres[:, None, :] - np.asarray(positions)[None, :, :], axis=2) <= radius
    k_hit = covered.sum(axis=1)
    k = len(positions)
    return {
        "no effect": float(np.mean(k_hit == 0)),
        "detected": float(np.mean((k_hit > 0) & (k_hit < k))),
        "undetected": float(np.mean(k_hit == k)),
    }


def layouts(ks: Sequence[int]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k in ks:
        out[f"k={k} clustered"] = clustered_layout(k)
        out[f"k={k} spread"] = spread_layout(k)
    return out
