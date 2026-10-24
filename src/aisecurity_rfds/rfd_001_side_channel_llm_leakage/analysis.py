"""Measurement analysis: leakage detection (Welch t, TVLA), correlation attack, and identification bounds.

Every function takes plain numpy arrays, so the same code can be pointed at measured traces. Here it is only run
on simulated ones.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

from aisecurity_rfds.common.rng import derive_seed, make_rng
from aisecurity_rfds.common.stats import McEstimate, wilson_interval

from .leakage import hw16

TVLA_THRESHOLD = 4.5  # |t| above this is the conventional "leaks" flag for a fixed-versus-random test

HYPOTHESES = np.arange(-128, 128)  # every int8 value an activation can take
_ALL = np.arange(-128, 128)
_PRODUCT_HW = hw16(np.outer(_ALL, _ALL)).astype(float)  # row: activation, column: weight
_SECRET_SD = _PRODUCT_HW.std(axis=1)  # for each activation, the spread of its leakage over uniformly random weights
LEAK_SD = float(np.sqrt((_SECRET_SD**2).mean()))  # root-mean-square of that spread: about 2.5 LU per multiplication


def q_function(x: float) -> float:
    """Upper tail of the standard normal distribution."""
    return float(sps.norm.sf(x))


# --- detection ---------------------------------------------------------------------------------------------


def welch_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Welch's t statistic for every column; the rows of ``a`` and ``b`` are traces."""
    va, vb = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
    return (a.mean(axis=0) - b.mean(axis=0)) / np.sqrt(va / a.shape[0] + vb / b.shape[0])


def tvla_traces_needed(effect: float, power: float = 0.9, threshold: float = TVLA_THRESHOLD) -> int:
    """Traces per group for a fixed-versus-random Welch test to exceed ``threshold`` with probability ``power``.

    ``effect`` is the difference of the two means in units of the per-trace standard deviation. With ``n`` traces
    per group ``t`` is approximately normal with mean ``effect * sqrt(n / 2)`` and variance 1.
    """
    if effect <= 0 or not 0 < power < 1:
        raise ValueError("effect must be positive and power in (0, 1)")
    return math.ceil(2 * (threshold + sps.norm.ppf(power)) ** 2 / effect**2)


# --- correlation attack ------------------------------------------------------------------------------------


def theoretical_rho(snr: float) -> float:
    """Correlation between a perfect prediction and a measurement with the given amplitude SNR."""
    return snr / math.sqrt(1 + snr**2)


def secret_rho(secret: int, sigma: float) -> float:
    """Correlation between a measurement with noise ``sigma`` and the leakage of ``secret * w`` over random weights."""
    sd = float(_SECRET_SD[secret + 128])
    return sd / math.sqrt(sd**2 + sigma**2)


def fisher_traces(rho: float, z: float) -> float:
    """Traces for a correlation ``rho`` to stand ``z`` standard errors above zero, by Fisher's transformation."""
    return 3 + (z / math.atanh(rho)) ** 2


def correlation_detection_traces(rho: float, power: float = 0.9, threshold: float = TVLA_THRESHOLD) -> int:
    """Samples for the correlation with a *known-correct* prediction to be flagged as leakage with this power."""
    return math.ceil(fisher_traces(rho, threshold + sps.norm.ppf(power)))


def correlations(observed: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Pearson correlation of ``observed`` (n values) with each row of ``predictions`` (hypotheses x n)."""
    o = observed - observed.mean()
    p = predictions - predictions.mean(axis=1, keepdims=True)
    denom = np.sqrt((o**2).sum()) * np.sqrt((p**2).sum(axis=1))
    return (p @ o) / np.where(denom == 0, 1.0, denom)


def closest_confusable(secret: int) -> tuple[int, float]:
    """The other activation whose leakage, over uniformly random weights, correlates best with ``secret``'s, and how well.

    A multiplication by ``2 * x`` is a shift of the product by ``x``: it leaks almost the same Hamming weight, so
    the two are hard to tell apart however many samples are taken.
    """
    best, best_r = secret, -1.0
    for h in HYPOTHESES:
        if h == secret or _SECRET_SD[h + 128] == 0:
            continue
        r = float(np.corrcoef(_PRODUCT_HW[secret + 128], _PRODUCT_HW[h + 128])[0, 1])
        if r > best_r:
            best, best_r = int(h), r
    return best, best_r


def cpa_predictions(weights: np.ndarray) -> np.ndarray:
    """Predicted leakage of ``weights * h`` for every hypothesis ``h`` for the activation: (256 x len(weights))."""
    return hw16(weights[None, :].astype(np.int64) * HYPOTHESES[:, None]).astype(float)


def cpa_recover(observed: np.ndarray, weights: np.ndarray) -> int:
    """The activation value whose predicted leakage correlates best with the measurements."""
    return int(HYPOTHESES[np.argmax(correlations(observed, cpa_predictions(weights)))])


def cpa_success_rate(
    sigma: float, samples: int, trials: int, rng: np.random.Generator, secret: int | None = None
) -> McEstimate:
    """How often the attack recovers an int8 activation (a random one unless ``secret`` is given) from ``samples``
    leaking multiplications by known random weights, each measured once with noise ``sigma`` (gain 1)."""
    hits = 0
    for _ in range(trials):
        x = int(rng.integers(-128, 128)) if secret is None else secret
        weights = rng.integers(-128, 128, size=samples)
        observed = hw16(weights * x) + rng.normal(0.0, sigma, size=samples)
        hits += cpa_recover(observed, weights) == x
    lo, hi = wilson_interval(hits, trials)
    return McEstimate(hits / trials, lo, hi, trials)


def correlation_flag_rate(
    secret: int, sigma: float, samples: int, trials: int, rng: np.random.Generator
) -> McEstimate:
    """How often the Fisher-transformed correlation between a measurement of ``secret * w`` (random weights ``w``,
    noise ``sigma``) and the *true* leakage prediction exceeds the detection threshold."""
    hits = 0
    for _ in range(trials):
        weights = rng.integers(-128, 128, size=samples)
        truth = hw16(weights * secret).astype(float)
        observed = truth + rng.normal(0.0, sigma, size=samples)
        r = min(float(correlations(observed, truth[None, :])[0]), 1 - 1e-12)
        hits += math.atanh(r) * math.sqrt(samples - 3) > TVLA_THRESHOLD
    lo, hi = wilson_interval(hits, trials)
    return McEstimate(hits / trials, lo, hi, trials)


LADDER = tuple(round(8 * 1.5**k) for k in range(17))  # sample counts tried by ``samples_to_recover``: 8 ... 5254


def samples_to_recover(
    sigma: float, target: float, trials: int, seed: int, secret: int | None = None
) -> int | None:
    """The first count on ``LADDER`` at which the attack succeeds at least ``target`` of the time, or None."""
    for samples in LADDER:
        rng = make_rng(derive_seed("samples_to_recover", sigma, samples, secret, seed))
        if cpa_success_rate(sigma, samples, trials, rng, secret).p >= target:
            return samples
    return None


# --- identification ----------------------------------------------------------------------------------------


def pair_error(distance: float | np.ndarray, sigma: float) -> float | np.ndarray:
    """Probability that maximum-likelihood detection confuses two hypotheses ``distance`` apart (noise ``sigma``
    per dimension, independent Gaussian). Identical hypotheses are confused half the time."""
    d = np.asarray(distance, dtype=float)
    out = np.where(d == 0, 0.5, 0.0) if sigma <= 0 else np.where(d == 0, 0.5, sps.norm.sf(d / (2 * sigma)))
    return float(out) if out.ndim == 0 else out


def distances(expected: np.ndarray) -> np.ndarray:
    """Euclidean distances between the rows of ``expected``."""
    diff = expected[:, None, :] - expected[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def identification_error_bounds(expected: np.ndarray, sigma: float) -> tuple[float, float]:
    """Bounds on the error of identifying which row of ``expected`` was observed, with a uniform prior.

    The error given hypothesis ``i`` is at least the largest pairwise error and at most their sum (union bound).
    """
    pair = np.asarray(pair_error(distances(expected), sigma))
    np.fill_diagonal(pair, 0.0)
    return float(pair.max(axis=1).mean()), float(np.minimum(1.0, pair.sum(axis=1)).mean())


def identify(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Maximum-likelihood row of ``expected`` for each row of ``observed`` (the nearest one)."""
    d2 = (observed**2).sum(1, keepdims=True) - 2 * observed @ expected.T + (expected**2).sum(1)[None, :]
    return d2.argmin(axis=1)


def identification_accuracy(
    expected: np.ndarray, sigma: float, trials: int, rng: np.random.Generator, repeats: int = 1
) -> McEstimate:
    """Simulated identification accuracy; ``repeats`` identical runs are averaged before the decision."""
    truth = rng.integers(0, len(expected), size=trials)
    observed = expected[truth] + rng.normal(0.0, sigma / math.sqrt(repeats), size=(trials, expected.shape[1]))
    hits = int((identify(observed, expected) == truth).sum())
    lo, hi = wilson_interval(hits, trials)
    return McEstimate(hits / trials, lo, hi, trials)


def noise_free_identification_accuracy(expected: np.ndarray) -> float:
    """The best accuracy any attacker can reach even when the readings carry no noise at all, with a uniform prior.

    Rows that are exactly equal cannot be told apart: one row of each group is named correctly and the others never
    are, so the ceiling is the number of distinct rows over the number of rows.
    """
    return len(np.unique(expected, axis=0)) / len(expected)


def repeats_needed(expected: np.ndarray, sigma: float, target: float, limit: int = 10**7) -> int | None:
    """Fewest averaged runs for which the union bound guarantees accuracy of at least ``target``."""
    if not 0 < target < 1:
        raise ValueError("target must be in (0, 1)")

    def guaranteed(n: int) -> bool:
        return 1 - identification_error_bounds(expected, sigma / math.sqrt(n))[1] >= target

    if not guaranteed(limit):
        return None
    lo, hi = 1, 1
    while not guaranteed(hi):
        lo, hi = hi, hi * 2
    while lo < hi:
        mid = (lo + hi) // 2
        if guaranteed(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
