"""Simulated physically unclonable function (PUF) and a code-offset fuzzy extractor.

The PUF model: each cell has a latent analog value (e.g. a capacitance deviation) drawn once per
device from N(0, 1). A measurement reports ``sign(latent + environment + noise)``. Cells with small
|latent| are unstable, as in real PUFs. Environment shifts (temperature, aging) move each cell by a
per-cell random coefficient. ``perturb`` models physical intrusion.

The fuzzy extractor is the classic code-offset construction with a repetition code, followed by
hashing for privacy amplification. IMPORTANT LIMITS: it is only information-theoretically sound if
raw bits are i.i.d. uniform (true of this simulation, not of real PUFs, which have bias and
correlation), and the repetition code is chosen for clarity, not efficiency. This is a research
model, not a production key-derivation scheme.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

import numpy as np

from .hashing import ct_equal, domain_hash
from .rng import derive_seed, make_rng


class SimulatedPuf:
    def __init__(
        self,
        device_seed: int,
        n_cells: int = 2048,
        noise_sigma: float = 0.1,
        temp_coeff_sigma: float = 0.002,
        aging_sigma_per_year: float = 0.01,
    ):
        rng = make_rng(derive_seed("simulated-puf", device_seed))
        self.n_cells = n_cells
        self.noise_sigma = noise_sigma
        self._latent = rng.normal(size=n_cells)
        self._temp_coeff = rng.normal(scale=temp_coeff_sigma, size=n_cells)
        self._aging = rng.normal(scale=aging_sigma_per_year, size=n_cells)

    def measure(
        self,
        *,
        temperature_c: float = 25.0,
        age_years: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """One noisy raw-bit measurement (uint8 array of 0/1)."""
        rng = rng if rng is not None else make_rng(None)
        effective = (
            self._latent
            + self._temp_coeff * (temperature_c - 25.0)
            + self._aging * age_years
        )
        noisy = effective + rng.normal(scale=self.noise_sigma, size=self.n_cells)
        return (noisy > 0).astype(np.uint8)

    def reference_response(self) -> np.ndarray:
        """Noise-free response at nominal conditions."""
        return (self._latent > 0).astype(np.uint8)

    def perturb(
        self,
        fraction: float,
        *,
        strength: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Model physical intrusion: shift ``fraction`` of the cells' latent values by N(0, strength)."""
        rng = rng if rng is not None else make_rng(None)
        mask = rng.random(self.n_cells) < fraction
        shifts = rng.normal(scale=strength, size=self.n_cells)
        self._latent = np.where(mask, self._latent + shifts, self._latent)


def bit_error_rate(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.mean(a[:n] != b[:n]))


@dataclass(frozen=True)
class HelperData:
    """Public helper data. Safe to store openly under the assumptions in the module docstring."""

    offset: bytes
    check: bytes
    key_bits: int
    repetition: int


class FuzzyExtractor:
    """Code-offset fuzzy extractor with an odd-length repetition code."""

    def __init__(self, key_bits: int = 128, repetition: int = 15):
        if repetition % 2 == 0:
            raise ValueError("repetition must be odd so majority voting has no ties")
        self.key_bits = key_bits
        self.repetition = repetition

    @property
    def n_cells_required(self) -> int:
        return self.key_bits * self.repetition

    def enroll(
        self,
        raw_bits: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        context: bytes = b"",
    ) -> tuple[bytes, HelperData]:
        n = self.n_cells_required
        if len(raw_bits) < n:
            raise ValueError(f"need at least {n} raw bits, got {len(raw_bits)}")
        if rng is None:
            raw = np.frombuffer(secrets.token_bytes(self.key_bits // 8 + 1), dtype=np.uint8)
            secret_bits = np.unpackbits(raw)[: self.key_bits]
        else:
            secret_bits = rng.integers(0, 2, size=self.key_bits, dtype=np.uint8)
        codeword = np.repeat(secret_bits, self.repetition)
        offset = (raw_bits[:n].astype(np.uint8) ^ codeword).astype(np.uint8)
        key = self._derive_key(secret_bits, context)
        helper = HelperData(
            offset=np.packbits(offset).tobytes(),
            check=domain_hash("aisec.puf.check", key),
            key_bits=self.key_bits,
            repetition=self.repetition,
        )
        return key, helper

    def reproduce(
        self, raw_bits: np.ndarray, helper: HelperData, *, context: bytes = b""
    ) -> tuple[bytes, bool]:
        """Return ``(key, ok)``. ``ok`` is False when the recovered key fails the stored check.

        When ``ok`` is False the returned key is a *different* key (the decoded bits are wrong), which
        is exactly the behaviour RFDs 036/041 want from a tampered enclosure.
        """
        n = helper.key_bits * helper.repetition
        if len(raw_bits) < n:
            raise ValueError(f"need at least {n} raw bits, got {len(raw_bits)}")
        offset = np.unpackbits(np.frombuffer(helper.offset, dtype=np.uint8))[:n]
        noisy = (raw_bits[:n].astype(np.uint8) ^ offset).reshape(helper.key_bits, helper.repetition)
        decoded = (noisy.sum(axis=1) * 2 > helper.repetition).astype(np.uint8)
        key = self._derive_key(decoded, context)
        ok = ct_equal(domain_hash("aisec.puf.check", key), helper.check)
        return key, ok

    @staticmethod
    def _derive_key(secret_bits: np.ndarray, context: bytes) -> bytes:
        return domain_hash("aisec.puf.key", context, np.packbits(secret_bits).tobytes())
