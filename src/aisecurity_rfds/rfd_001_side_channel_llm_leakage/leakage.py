"""The leakage model: Hamming-weight leakage of int8 multiply-accumulates seen through a measurement channel.

Two assumptions carry everything else, and both are stated here rather than measured:

* each multiplication leaks the Hamming weight of its 16-bit two's-complement product, one sample per operation
  (the usual first-order model of CMOS switching power);
* a channel observes the *mean* leakage over a window of ``m = OP_RATE_HZ / rate_hz`` consecutive operations, with
  white noise added before that averaging (``noise_op``, it averages down) and after it (``noise_out``, it does not).

Units are arbitrary "leakage units" (LU): the data-dependent part of one multiplication has a standard deviation of
about 2.5 LU. Nothing here was measured on a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from aisecurity_rfds.common.rng import make_rng

OP_RATE_HZ = 1e9  # multiply-accumulates per second in one stream of the toy accelerator
TOKEN_SLOT_S = 0.02  # one decode step: the time one token's expert choice stays on the hardware

POPCOUNT16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)


def hw16(values: np.ndarray | int) -> np.ndarray:
    """Hamming weight of the low 16 bits of each value, which for a negative number is its two's-complement form."""
    return POPCOUNT16[np.asarray(values, dtype=np.int64) & 0xFFFF]


def mac_leakage(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Leakage of every multiplication in ``weights @ x``: an array of the same shape as ``weights``."""
    products = weights.astype(np.int64) * x.astype(np.int64)[None, :]
    return hw16(products).astype(float)


@dataclass(frozen=True)
class ToyLayer:
    """A first layer with known int8 weights and an int8 embedding for each token of a small vocabulary."""

    weights: np.ndarray
    embeddings: np.ndarray

    @classmethod
    def make(cls, seed: int = 0, vocab: int = 64, rows: int = 12, width: int = 24) -> ToyLayer:
        rng = make_rng(seed)
        return cls(
            weights=rng.integers(-128, 128, size=(rows, width)),
            embeddings=rng.integers(-128, 128, size=(vocab, width)),
        )

    @property
    def vocab(self) -> int:
        return self.embeddings.shape[0]

    @property
    def ops(self) -> int:
        return self.weights.size

    def template(self, token: int) -> np.ndarray:
        """The leakage vector of the product of the weights with this token's embedding, one entry per operation."""
        return mac_leakage(self.weights, self.embeddings[token]).ravel()

    def templates(self) -> np.ndarray:
        return np.stack([self.template(t) for t in range(self.vocab)])


@dataclass(frozen=True)
class Channel:
    """A way of observing the leakage: a gain, two noise sources, and a sample rate."""

    name: str
    gain: float
    noise_op: float
    noise_out: float
    rate_hz: float

    def __post_init__(self) -> None:
        if self.gain <= 0 or self.rate_hz <= 0 or self.noise_op < 0 or self.noise_out < 0:
            raise ValueError("gain and rate must be positive and noise levels must not be negative")

    @property
    def window(self) -> int:
        """Operations averaged into one output sample (1 when the channel is at least as fast as the operations)."""
        return max(1, round(OP_RATE_HZ / self.rate_hz))

    @property
    def noise(self) -> float:
        """Standard deviation of the noise on one output sample, after the per-operation noise has been averaged."""
        return math.sqrt(self.noise_op**2 / self.window + self.noise_out**2)

    @property
    def samples_per_slot(self) -> float:
        return self.rate_hz * TOKEN_SLOT_S

    def view(self, signal: np.ndarray) -> np.ndarray:
        """What the channel would report with no noise: block means of the last axis, zero-padded to whole windows.

        A window longer than the signal therefore reports ``sum / window``: the rest of the window carries no
        token-dependent leakage, so the signal is diluted by ``len(signal) / window``.
        """
        m, n = self.window, signal.shape[-1]
        if m >= n:  # one window holds the whole signal: nothing to pad, and the dilution is exactly n / m
            return self.gain * signal.sum(axis=-1, keepdims=True) / m
        blocks = -(-n // m)
        padded = np.pad(signal, [(0, 0)] * (signal.ndim - 1) + [(0, blocks * m - n)])
        return self.gain * padded.reshape(*signal.shape[:-1], blocks, m).mean(axis=-1)

    def observe(self, signal: np.ndarray, rng: np.random.Generator, repeats: int = 1) -> np.ndarray:
        """One noisy observation, averaged over ``repeats`` identical runs (noise falls as the square root)."""
        clean = self.view(signal)
        return clean + rng.normal(0.0, self.noise / math.sqrt(repeats), size=clean.shape)

    def with_tenants(self, tenants: int, other_noise: float) -> Channel:
        """The same sensor shared with ``tenants - 1`` other workloads, each adding ``other_noise`` at the output."""
        if tenants < 1:
            raise ValueError("a sensor has at least one tenant")
        if tenants == 1:
            return self
        extra = math.sqrt(self.noise_out**2 + (tenants - 1) * other_noise**2)
        return replace(self, name=f"{self.name} x{tenants}", noise_out=extra)


# Assumed channels. The values are round numbers chosen so that the model has something to rank, not measurements.
SCOPE = Channel("power rail, oscilloscope", gain=1.0, noise_op=12.0, noise_out=0.0, rate_hz=1e9)
EM_PROBE = Channel("EM near-field probe", gain=0.4, noise_op=4.0, noise_out=0.0, rate_hz=1e9)
ACOUSTIC = Channel("acoustic, microphone", gain=0.2, noise_op=5.0, noise_out=0.002, rate_hz=5e4)
SOFTWARE = Channel("software power sensor", gain=1.0, noise_op=0.0, noise_out=0.01, rate_hz=50.0)
THERMAL = Channel("thermal sensor", gain=0.05, noise_op=0.0, noise_out=0.004, rate_hz=10.0)
CHANNELS = (SCOPE, EM_PROBE, ACOUSTIC, SOFTWARE, THERMAL)
