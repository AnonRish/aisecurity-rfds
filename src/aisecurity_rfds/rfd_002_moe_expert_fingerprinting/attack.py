"""The attacker's side: a lookup table built with white-box access to a model, and a Bayes decoder that names a token
from what a sensor reported about the experts that ran.

The decoder treats the blocks as independent given the token (naive Bayes), takes a uniform prior over tokens and knows
the sensor's signatures and noise. It returns the most probable token; nothing else about the prompt is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aisecurity_rfds.common.stats import McEstimate, wilson_interval

from . import moe
from .signatures import SetSpace, groups_of


@dataclass(frozen=True)
class Profile:
    """``table[layer, set, token]``: the chance that a block ran this set of experts when the token was this one.

    Each ``table[layer, :, token]`` sums to one. It is what running the open model on many contexts teaches."""

    table: np.ndarray


def build_profile(
    cfg: moe.Config,
    params: dict[str, np.ndarray],
    space: SetSpace,
    rng: np.random.Generator,
    per_token: int = 200,
    tau: float = 0.0,
    alpha: float = 0.5,
) -> Profile:
    """Run every token in ``per_token`` random contexts (with routing noise ``tau``, if the attacker knows the model
    is defended) and count the sets that ran; ``alpha`` is the Laplace smoothing added to every count."""
    if per_token < 1 or alpha <= 0:
        raise ValueError("need at least one run per token and positive smoothing")
    tokens = np.repeat(np.arange(cfg.vocab), per_token)
    c = rng.normal(size=(len(tokens), cfg.dim))
    noise = moe.routing_noise(cfg, len(tokens), tau, rng)
    which = space.encode(moe.routes_of(cfg, params, tokens, c, noise))
    counts = np.zeros((cfg.layers, space.count, cfg.vocab))
    for layer in range(cfg.layers):
        np.add.at(counts[layer], (which[:, layer], tokens), 1.0)
    return Profile((counts + alpha) / (per_token + alpha * space.count))


def set_likelihood(
    set_sigs: np.ndarray,
    true_sets: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    true_sigs: np.ndarray | None = None,
) -> np.ndarray:
    """How well each candidate set explains what the sensor reported: ``(n, layers, sets)``, proportional to the
    likelihood of the reading. ``true_sets`` is ``(n, layers)``, the numbers of the sets that actually ran.

    With ``sigma = 0`` the sensor reports the exact signature, so every set with that signature is equally likely and
    the others are impossible. ``true_sigs`` are the signatures of the target when they differ from the attacker's."""
    if sigma == 0:
        if true_sigs is not None:
            raise ValueError("signature drift needs a noisy sensor: with no noise no reading matches the attacker's table")
        groups = groups_of(set_sigs)
        return (groups[None, None, :] == groups[true_sets][..., None]).astype(float)
    reading = (set_sigs if true_sigs is None else true_sigs)[true_sets]
    reading = reading + rng.normal(0.0, sigma, reading.shape)
    d2 = ((reading[:, :, None, :] - set_sigs[None, None, :, :]) ** 2).sum(axis=-1)
    return np.exp(-(d2 - d2.min(axis=-1, keepdims=True)) / (2.0 * sigma**2))


def token_scores(profile: Profile, likelihood: np.ndarray) -> np.ndarray:
    """Log posterior of every token, up to a constant, from the first ``likelihood.shape[1]`` blocks: ``(n, vocab)``."""
    scores = np.zeros((likelihood.shape[0], profile.table.shape[2]))
    for layer in range(likelihood.shape[1]):
        scores += np.log(likelihood[:, layer, :] @ profile.table[layer] + 1e-300)
    return scores


def attack(
    cfg: moe.Config,
    target: dict[str, np.ndarray],
    profile: Profile,
    space: SetSpace,
    set_sigs: np.ndarray,
    sigma: float,
    trials: int,
    rng: np.random.Generator,
    layers: int | None = None,
    tau: float = 0.0,
    repeats: int = 1,
    true_sigs: np.ndarray | None = None,
    tokens: np.ndarray | None = None,
) -> McEstimate:
    """How often the decoder names a token correctly from one position of the target model: a uniformly drawn token
    for each of ``trials`` positions, or the given ``tokens`` (then ``trials`` is ignored).

    ``layers`` is how many blocks the sensor can see (the first ones); ``tau`` is routing noise in the target;
    ``repeats`` is how many times the same token in the same context is observed, each with fresh routing noise and
    sensor noise; ``true_sigs`` lets the target's physical signatures differ from the ones the attacker holds."""
    seen = cfg.layers if layers is None else layers
    if not 1 <= seen <= cfg.layers or repeats < 1:
        raise ValueError("layers must be between 1 and the model's layers, and repeats at least 1")
    tokens = rng.integers(0, cfg.vocab, trials) if tokens is None else np.asarray(tokens)
    trials = len(tokens)
    c = rng.normal(size=(trials, cfg.dim))
    scores = np.zeros((trials, cfg.vocab))
    for _ in range(repeats):
        noise = moe.routing_noise(cfg, trials, tau, rng)
        which = space.encode(moe.routes_of(cfg, target, tokens, c, noise)[:, :seen, :])
        scores += token_scores(profile, set_likelihood(set_sigs, which, sigma, rng, true_sigs))
    hits = int((scores.argmax(axis=1) == tokens).sum())
    lo, hi = wilson_interval(hits, trials)
    return McEstimate(hits / trials, lo, hi, trials)


def route_change(
    cfg: moe.Config,
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    rng: np.random.Generator,
    n: int = 4000,
) -> float:
    """Share of (token, block) positions where models ``a`` and ``b`` choose different experts for the same token in
    the same context."""
    tokens = rng.integers(0, cfg.vocab, n)
    c = rng.normal(size=(n, cfg.dim))
    return float((moe.routes_of(cfg, a, tokens, c) != moe.routes_of(cfg, b, tokens, c)).any(axis=2).mean())


def deterministic_ceiling(cfg: moe.Config, params: dict[str, np.ndarray], layers: int | None = None) -> float:
    """With no context vector every token has one route; tokens that share a route cannot be told apart, so no attacker
    who sees the exact routes of the first ``layers`` blocks names more than this share of the vocabulary."""
    seen = cfg.layers if layers is None else layers
    tokens = np.arange(cfg.vocab)
    routes = moe.routes_of(cfg, params, tokens, np.zeros((cfg.vocab, cfg.dim)))[:, :seen, :]  # no context, whatever ctx is
    return len(np.unique(routes.reshape(cfg.vocab, -1), axis=0)) / cfg.vocab
