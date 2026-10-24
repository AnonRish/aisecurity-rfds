"""Scenarios built from the leakage model: a dense-model attack, a response hidden behind a refusal, and sensors
shared in a cloud."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from aisecurity_rfds.common.rng import derive_seed, make_rng
from aisecurity_rfds.common.stats import McEstimate, wilson_interval

from .analysis import TVLA_THRESHOLD, identification_accuracy, identify, welch_t
from .leakage import Channel, ToyLayer
from .routing import MoeRouter, category_accuracy

CATEGORIES = 4  # the prompt's topic; the vocabulary is split into this many equal blocks, one per topic
OTHER_TENANT_NOISE = 0.005  # LU of extra noise each co-located workload adds to a shared sensor (an assumption)
REFUSAL_TEXT = "I can't help with that."


def _physics(channel: Channel) -> tuple[float, float, float, float]:
    """What determines a channel's behaviour, so that two channels that differ only in name share random draws."""
    return channel.gain, channel.noise_op, channel.noise_out, channel.rate_hz


def token_accuracy(
    layer: ToyLayer, channel: Channel, trials: int, rng: np.random.Generator, repeats: int = 1
) -> McEstimate:
    """Accuracy of naming one token of the vocabulary from the channel's view of the first-layer product."""
    return identification_accuracy(channel.view(layer.templates()), channel.noise, trials, rng, repeats)


def best_effect(layer: ToyLayer, channel: Channel, token: int) -> float:
    """Largest difference, in units of the pooled per-trace standard deviation, between the channel's reading of
    ``token`` and its reading of a random token: the effect size a fixed-versus-random test sees at its best sample."""
    expected = channel.view(layer.templates())
    delta = np.abs(expected[token] - expected.mean(axis=0))
    pooled = np.sqrt(channel.noise**2 + expected.var(axis=0) / 2)  # fixed group: noise only; random group: noise + token
    return float((delta / pooled).max())


def median_best_effect(layer: ToyLayer, channel: Channel) -> float:
    """``best_effect`` for a typical fixed token (the median over the vocabulary)."""
    return float(np.median([best_effect(layer, channel, t) for t in range(layer.vocab)]))


def tvla_flagged(
    layer: ToyLayer, channel: Channel, traces: int, rng: np.random.Generator, fixed: int | None = 0
) -> tuple[int, float]:
    """Fixed-versus-random test on the first-layer product: samples flagged (``|t|`` above the threshold), largest ``|t|``.

    ``fixed`` is the token of the fixed group; None makes both groups random, which is the null case.
    """
    expected = channel.view(layer.templates())
    shape = (traces, expected.shape[1])
    first = rng.integers(0, layer.vocab, size=traces) if fixed is None else np.full(traces, fixed)
    a = expected[first] + rng.normal(0.0, channel.noise, size=shape)
    b = expected[rng.integers(0, layer.vocab, size=traces)] + rng.normal(0.0, channel.noise, size=shape)
    t = np.abs(welch_t(a, b))
    return int((t > TVLA_THRESHOLD).sum()), float(t.max())


def dense_category_accuracy(
    layer: ToyLayer, channel: Channel, tokens: int, trials: int, rng: np.random.Generator
) -> McEstimate:
    """A dense-model attacker names each token of a prompt and takes the category most of them belong to.

    Category ``c`` owns block ``c`` of the vocabulary, and a prompt of that category uses only its tokens.
    """
    if layer.vocab % CATEGORIES:
        raise ValueError("the vocabulary must divide evenly into categories")
    per = layer.vocab // CATEGORIES
    expected = channel.view(layer.templates())
    truth = rng.integers(0, CATEGORIES, size=trials)
    tokens_used = truth[:, None] * per + rng.integers(0, per, size=(trials, tokens))
    observed = expected[tokens_used] + rng.normal(0.0, channel.noise, size=(trials, tokens, expected.shape[1]))
    named = identify(observed.reshape(-1, expected.shape[1]), expected).reshape(trials, tokens)
    votes = ((named // per)[..., None] == np.arange(CATEGORIES)).sum(axis=1)
    hits = int((votes.argmax(axis=1) == truth).sum())
    lo, hi = wilson_interval(hits, trials)
    return McEstimate(hits / trials, lo, hi, trials)


def tokens_for_category(
    router: MoeRouter, channel: Channel, target: float, trials: int, seed: int, limit: int = 1024
) -> int | None:
    """Fewest tokens (powers of two) after which the routing attacker's accuracy reaches ``target``; None past ``limit``."""
    tokens = 1
    while tokens <= limit:
        rng = make_rng(derive_seed("tokens_for_category", _physics(channel), tokens, seed))
        if category_accuracy(router, channel, tokens, trials, rng).p >= target:
            return tokens
        tokens *= 2
    return None


def tokens_for_dense_category(
    layer: ToyLayer, channel: Channel, target: float, trials: int, seed: int, limit: int = 1024
) -> int | None:
    """The same search for the dense-model attacker, who names tokens and votes."""
    tokens = 1
    while tokens <= limit:
        rng = make_rng(derive_seed("tokens_for_dense_category", _physics(channel), tokens, seed))
        if dense_category_accuracy(layer, channel, tokens, trials, rng).p >= target:
            return tokens
        tokens *= 2
    return None


# --- a response hidden behind a refusal ----------------------------------------------------------------------


@dataclass(frozen=True)
class Transcript:
    """What the user is shown, and which tokens the accelerator embedded (looked up) while decoding."""

    shown: str
    decoded: tuple[int, ...]


def respond(mode: str, hidden: Sequence[int]) -> Transcript:
    """A refusal decided *before* decoding does no decoding; one applied *after* decoding replaces finished text."""
    if mode == "pre-check":
        return Transcript(REFUSAL_TEXT, ())
    if mode == "post-hoc filter":
        return Transcript(REFUSAL_TEXT, tuple(hidden))
    if mode == "no refusal":
        return Transcript("<the response>", tuple(hidden))
    raise ValueError(f"unknown mode {mode!r}")


def recovered_fraction(
    layer: ToyLayer, channel: Channel, transcript: Transcript, hidden: Sequence[int], rng: np.random.Generator
) -> float:
    """Fraction of the hidden response's tokens that an attacker names correctly from the decode steps it observed."""
    if not hidden:
        raise ValueError("the hidden response is empty")
    expected = channel.view(layer.templates())
    hits = 0
    for step, token in enumerate(transcript.decoded):
        observed = expected[token] + rng.normal(0.0, channel.noise, size=expected.shape[1])
        hits += int(identify(observed[None, :], expected)[0]) == hidden[step]
    return hits / len(hidden)


def hidden_recovery(
    layer: ToyLayer, channel: Channel, mode: str, responses: int, length: int, rng: np.random.Generator
) -> float:
    """Mean fraction of ``responses`` independent hidden responses of ``length`` tokens that an attacker recovers."""
    total = 0.0
    for _ in range(responses):
        hidden = [int(t) for t in rng.integers(0, layer.vocab, size=length)]
        total += recovered_fraction(layer, channel, respond(mode, hidden), hidden, rng)
    return total / responses
