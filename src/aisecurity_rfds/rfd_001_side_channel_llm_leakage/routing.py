"""Mixture-of-experts routing as a leakage channel: which experts ran in a slot is visible as a shift in mean power.

A dense model has no such choice to leak. The router here is a table, ``P(expert set | category)``; the hardware
effect of an expert set is a shift of the slot's mean leakage, the sum of one fixed shift per expert.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.common.stats import McEstimate, wilson_interval

from .analysis import identification_accuracy, identification_error_bounds
from .leakage import Channel


@dataclass(frozen=True)
class MoeRouter:
    sets: tuple[tuple[int, ...], ...]  # every subset of ``k`` experts the router can pick
    probs: np.ndarray  # (categories, sets): P(set | category), each row sums to 1
    levels: np.ndarray  # (sets,): mean leakage shift in LU while that set runs, before any channel gain

    @classmethod
    def make(
        cls,
        seed: int = 0,
        n_experts: int = 8,
        k: int = 2,
        categories: int = 4,
        concentration: float = 0.3,
        expert_spread: float = 0.05,
    ) -> MoeRouter:
        """A random router. Small ``concentration`` means each category favours a few expert sets."""
        rng = make_rng(seed)
        sets = tuple(itertools.combinations(range(n_experts), k))
        shifts = rng.normal(0.0, expert_spread, size=n_experts)
        levels = np.array([shifts[list(s)].sum() for s in sets])
        probs = rng.dirichlet(np.full(len(sets), concentration), size=categories)
        return cls(sets=sets, probs=probs, levels=levels)

    @property
    def categories(self) -> int:
        return self.probs.shape[0]

    def draw(self, truth: np.ndarray, tokens: int, rng: np.random.Generator) -> np.ndarray:
        """For each category in ``truth``, the index of the expert set used by each of ``tokens`` successive tokens."""
        cumulative = self.probs.cumsum(axis=1)
        u = rng.random((len(truth), tokens))
        drawn = (u[..., None] > cumulative[truth][:, None, :]).sum(axis=-1)
        return np.minimum(drawn, self.probs.shape[1] - 1)


def mutual_information(router: MoeRouter) -> float:
    """Bits per token that the expert set reveals about the category, for a uniform prior over categories."""
    p = router.probs
    mean = p.mean(axis=0)
    ratio = np.where(p > 0, p / mean[None, :], 1.0)
    return float((p * np.log2(ratio)).sum(axis=1).mean())


def noise_free_category_accuracy(router: MoeRouter) -> float:
    """Best accuracy from one token when the expert set is seen exactly: the Bayes rule picks the likeliest category."""
    return float(router.probs.max(axis=0).sum() / router.categories)


def slot_noise(channel: Channel) -> float:
    """Standard deviation of one slot's mean level, from the samples the channel takes while a token runs."""
    if channel.samples_per_slot < 1:
        raise ValueError(f"{channel.name} takes {channel.samples_per_slot:g} samples per token: it cannot resolve a slot")
    return channel.noise / math.sqrt(channel.samples_per_slot)


def set_identification_error(router: MoeRouter, channel: Channel) -> tuple[float, float]:
    """Bounds on the chance of misreading which expert set ran in one slot (see ``identification_error_bounds``)."""
    return identification_error_bounds((channel.gain * router.levels)[:, None], slot_noise(channel))


def set_accuracy(router: MoeRouter, channel: Channel, trials: int, rng: np.random.Generator) -> McEstimate:
    """Simulated accuracy of reading which expert set ran in one slot."""
    return identification_accuracy((channel.gain * router.levels)[:, None], slot_noise(channel), trials, rng)


def category_accuracy(
    router: MoeRouter, channel: Channel, tokens: int, trials: int, rng: np.random.Generator
) -> McEstimate:
    """Accuracy of a Bayes attacker who reads one noisy level per token and infers the prompt's category."""
    sigma = slot_noise(channel)
    truth = rng.integers(0, router.categories, size=trials)
    sets = router.draw(truth, tokens, rng)
    mu = channel.gain * router.levels
    y = mu[sets] + rng.normal(0.0, sigma, size=sets.shape)
    like = np.exp(-0.5 * ((y[..., None] - mu) / sigma) ** 2)  # (trials, tokens, sets), constant factors dropped
    per_category = like @ router.probs.T  # (trials, tokens, categories)
    guess = np.log(per_category + 1e-300).sum(axis=1).argmax(axis=1)
    hits = int((guess == truth).sum())
    lo, hi = wilson_interval(hits, trials)
    return McEstimate(hits / trials, lo, hi, trials)
