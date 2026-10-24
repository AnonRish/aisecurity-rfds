"""The physical layer of the attack: what a sensor can say about which experts ran.

Each expert has a signature, a vector of a few measurable features (power level, memory traffic and so on). While a
block runs, the sensor reports the sum of the signatures of the ``k`` experts that were chosen, plus white Gaussian
noise. Two experts with equal signatures cannot be told apart by any sensor; ``classes`` counts the distinct ones.
All of it is assumed: nothing here was measured on a GPU.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.common.stats import McEstimate, wilson_interval

MAX_EXPERTS = 20  # the set index is a table over bitmasks of the experts
EQUAL = 1e-9  # signatures closer than this are one signature: sums of equal parts can differ in the last bit


@dataclass(frozen=True)
class SetSpace:
    """Every way of choosing ``k`` of ``experts`` experts, numbered 0 ... C(experts, k) - 1 in lexicographic order."""

    experts: int
    k: int
    sets: np.ndarray  # (count, k)
    index: np.ndarray  # bitmask of a set -> its number (-1 for a mask that is not a k-subset)

    @classmethod
    def make(cls, experts: int, k: int) -> SetSpace:
        if not 1 <= k <= experts <= MAX_EXPERTS:
            raise ValueError(f"need 1 <= k <= experts <= {MAX_EXPERTS}")
        sets = np.array(list(itertools.combinations(range(experts), k)), dtype=np.int64)
        index = np.full(1 << experts, -1, dtype=np.int64)
        index[(1 << sets).sum(axis=1)] = np.arange(len(sets))
        return cls(experts, k, sets, index)

    @property
    def count(self) -> int:
        return len(self.sets)

    def encode(self, routes: np.ndarray) -> np.ndarray:
        """The number of the set in each ``(..., k)`` array of distinct expert indices."""
        found = self.index[(1 << routes).sum(axis=-1)]
        if (found < 0).any():
            raise ValueError("a route must name k distinct experts")
        return found


@dataclass(frozen=True)
class Signatures:
    vectors: np.ndarray  # (experts, features)

    @classmethod
    def make(cls, experts: int, features: int, classes: int, seed: int, spread: float = 1.0) -> Signatures:
        """``classes`` distinct random signatures, dealt to the experts in turn: expert ``e`` gets class ``e % classes``."""
        if not 1 <= classes <= experts or features < 1 or spread <= 0:
            raise ValueError("need 1 <= classes <= experts, at least one feature and a positive spread")
        base = make_rng(seed).normal(0.0, spread, (classes, features))
        return cls(base[np.arange(experts) % classes])

    @property
    def classes(self) -> int:
        return len(np.unique(np.round(self.vectors, 9), axis=0))

    def of_sets(self, space: SetSpace) -> np.ndarray:
        """The signature of every set: the sum of its experts' signatures, ``(count, features)``."""
        return self.vectors[space.sets].sum(axis=1)

    def drifted(self, relative: float, rng: np.random.Generator) -> Signatures:
        """The same experts measured on another build: every entry moves by Gaussian noise whose sd is ``relative``
        times the root-mean-square signature entry."""
        rms = float(np.sqrt((self.vectors**2).mean()))
        return Signatures(self.vectors + rng.normal(0.0, relative * rms, self.vectors.shape))


def groups_of(set_sigs: np.ndarray) -> np.ndarray:
    """A number for each set such that two sets share a number exactly when their signatures are equal."""
    return np.unique(np.round(set_sigs, 9), axis=0, return_inverse=True)[1].ravel()


def measure(set_sigs: np.ndarray, which: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """One noisy reading per entry of ``which`` (set numbers): the set's signature plus noise of sd ``sigma``."""
    clean = set_sigs[which]
    return clean if sigma == 0 else clean + rng.normal(0.0, sigma, clean.shape)


def decode(readings: np.ndarray, set_sigs: np.ndarray) -> np.ndarray:
    """The set whose signature is nearest each reading (the maximum-likelihood set for equal priors)."""
    d2 = ((readings[:, None, :] - set_sigs[None, :, :]) ** 2).sum(axis=-1)
    return d2.argmin(axis=1)


def pair_errors(set_sigs: np.ndarray, sigma: float) -> np.ndarray:
    """Probability that nearest-signature decoding mistakes set ``i`` for set ``j``, for every pair; the diagonal is 0.

    Sets with equal signatures are confused half the time. With ``sigma = 0`` only equal signatures are confused."""
    d = np.sqrt(((set_sigs[:, None, :] - set_sigs[None, :, :]) ** 2).sum(axis=-1))
    same = d < EQUAL
    out = np.where(same, 0.5, 0.0) if sigma <= 0 else np.where(same, 0.5, sps.norm.sf(d / (2.0 * sigma)))
    np.fill_diagonal(out, 0.0)
    return out


def set_error_bounds(set_sigs: np.ndarray, sigma: float) -> tuple[float, float]:
    """Bounds on the chance of decoding the wrong set, for a uniformly drawn set: the largest pairwise confusion at
    the bottom, their sum (the union bound, capped at one) at the top."""
    pair = pair_errors(set_sigs, sigma)
    return float(pair.max(axis=1).mean()), float(np.minimum(1.0, pair.sum(axis=1)).mean())


def distinct_signatures(set_sigs: np.ndarray) -> int:
    """How many different signatures the sets have between them."""
    return len(np.unique(np.round(set_sigs, 9), axis=0))


def noise_free_ceiling(set_sigs: np.ndarray) -> float:
    """The best accuracy of naming a set from its signature even with no noise: distinct signatures over sets."""
    return distinct_signatures(set_sigs) / len(set_sigs)


def set_accuracy(set_sigs: np.ndarray, sigma: float, trials: int, rng: np.random.Generator) -> McEstimate:
    """Simulated accuracy of nearest-signature decoding of a uniformly drawn set."""
    truth = rng.integers(0, len(set_sigs), trials)
    hits = int((decode(measure(set_sigs, truth, sigma, rng), set_sigs) == truth).sum())
    lo, hi = wilson_interval(hits, trials)
    return McEstimate(hits / trials, lo, hi, trials)


def bits_per_slot(set_count: int) -> float:
    """What one block's route can say at most: log2 of the number of sets."""
    return math.log2(set_count)
