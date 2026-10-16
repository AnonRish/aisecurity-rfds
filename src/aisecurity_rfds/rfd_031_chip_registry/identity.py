"""Chip identifiers and how a verifier checks that a chip holds one (RFD 031, technical requirement 1).

The RFD names three styles of identifier. This module models each and measures what the registry can and cannot learn
from it:

* a device key in fuses (``DeviceKeyChip``): the chip signs a verifier's challenge together with a monotonic counter, so
  the verifier learns that the chip holds the key without seeing it. If the key is extracted, the copy is
  indistinguishable by signature; the counter is what can reveal two chips using one key (``CounterTracker``);
* a PUF-derived key (``PufChip``): the key is derived at each boot from a noisy PUF and public helper data, so no key is
  stored, but a chip whose PUF has drifted too far cannot attest (a false reject);
* an adhesive glitter pattern photographed at registration (``random_tag``, ``photograph``, ``match_fraction``): it
  identifies a package, not the silicon, and its error rates depend on how large the photographic noise is against the
  matching radius (``glitter_error_rates``, ``simulate_glitter``).

Everything is a model: the PUF is the simulated one in ``aisecurity_rfds.common.puf`` and the glitter geometry is an
idealised unit torus with independent Gaussian placement noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy import stats as sps

from aisecurity_rfds.common.hashing import domain_hash, u64
from aisecurity_rfds.common.puf import FuzzyExtractor, SimulatedPuf, bit_error_rate
from aisecurity_rfds.common.rng import derive_seed, make_rng
from aisecurity_rfds.common.signing import KeyPair, verify
from aisecurity_rfds.common.stats import McEstimate, monte_carlo_probability

FRESH = "fresh"
REGRESSION = "regression"


def chip_id_of(public_key: bytes) -> bytes:
    """A chip's registry identifier: a domain-separated hash of its device public key."""
    return domain_hash("rfd031.chip-id", public_key)


def party_key(name: str) -> KeyPair:
    """A deterministic signing key for a named party. For demonstrations and tests only: anyone can recompute it."""
    return KeyPair.from_seed(domain_hash("rfd031.party-key", name.encode()))


@dataclass(frozen=True)
class Attestation:
    chip_id: bytes
    challenge: bytes
    counter: int
    signature: bytes


def attestation_message(chip_id: bytes, challenge: bytes, counter: int) -> bytes:
    return domain_hash("rfd031.attestation", chip_id, challenge, u64(counter))


def verify_attestation(public_key: bytes, challenge: bytes, attestation: Attestation) -> bool:
    """Did the holder of the key behind ``public_key`` sign exactly this challenge?"""
    return (
        attestation.chip_id == chip_id_of(public_key)
        and attestation.challenge == challenge
        and verify(
            public_key,
            attestation_message(attestation.chip_id, attestation.challenge, attestation.counter),
            attestation.signature,
        )
    )


class DeviceKeyChip:
    """A chip with a device key in fuses and an attestation counter that only goes up."""

    def __init__(self, seed: bytes, counter: int = 0) -> None:
        self._seed = seed
        self._key = KeyPair.from_seed(domain_hash("rfd031.device-key", seed))
        self.public_key = self._key.public_bytes
        self.chip_id = chip_id_of(self.public_key)
        self._counter = counter

    @property
    def counter(self) -> int:
        return self._counter

    def attest(self, challenge: bytes) -> Attestation:
        self._counter += 1
        message = attestation_message(self.chip_id, challenge, self._counter)
        return Attestation(self.chip_id, challenge, self._counter, self._key.sign(message))

    def clone(self, advance: int = 0) -> DeviceKeyChip:
        """What invasive key extraction yields: another chip with the same key and the counter as it stands now.

        ``advance`` lets a careful copier start its counter ahead of the original's.
        """
        if advance < 0:
            raise ValueError("a copy cannot start behind the original")
        return DeviceKeyChip(self._seed, self._counter + advance)


class CounterTracker:
    """The registry's memory of the highest attestation counter seen for each chip."""

    def __init__(self) -> None:
        self._last: dict[bytes, int] = {}

    def last(self, chip_id: bytes) -> int:
        return self._last.get(chip_id, 0)

    def observe(self, attestation: Attestation) -> str:
        if attestation.counter > self.last(attestation.chip_id):
            self._last[attestation.chip_id] = attestation.counter
            return FRESH
        return REGRESSION


def run_clone_sequence(pattern: str, advance: int = 0) -> bool:
    """Feed attestations by the original (``O``) and a copy (``C``) to a tracker; True if any is a regression.

    The tracker is told of the original's state at the moment of copying, so the first event of either chip is fresh.
    """
    if set(pattern) - {"O", "C"}:
        raise ValueError("pattern may contain only O and C")
    original = DeviceKeyChip(b"clone-experiment")
    copy = original.clone(advance)
    tracker = CounterTracker()
    for i, who in enumerate(pattern):
        chip = original if who == "O" else copy
        if tracker.observe(chip.attest(b"challenge-%d" % i)) == REGRESSION:
            return True
    return False


def clone_detection_probability(events: int, clone_share: float, careful: bool = False) -> float:
    """P(a tracker flags a copy) when each of ``events`` attestations is the copy's with probability ``clone_share``.

    A copy that starts at the original's counter is flagged at the first switch between the two chips. A careful copy
    that starts far enough ahead is flagged only when the original attests after the copy has: use of the original
    first and the copy afterwards (a transplant) goes unnoticed.
    """
    if events < 0 or not 0.0 <= clone_share <= 1.0:
        raise ValueError("events must be non-negative and clone_share in [0, 1]")
    q = clone_share
    if not careful:
        return 0.0 if events == 0 else 1.0 - q**events - (1.0 - q) ** events
    quiet = sum((1.0 - q) ** a * q ** (events - a) for a in range(events + 1))
    return 1.0 - quiet


def clone_detection_by_enumeration(events: int, clone_share: float, careful: bool = False) -> float:
    """The same probability by running every one of the 2^events sequences through a real tracker and chips."""
    advance = 10**6 if careful else 0
    total = 0.0
    for pattern in product("OC", repeat=events):
        copies = pattern.count("C")
        if run_clone_sequence("".join(pattern), advance):
            total += clone_share**copies * (1.0 - clone_share) ** (events - copies)
    return total


class PufChip:
    """A chip whose attestation key is rederived from its PUF at every boot, using public helper data."""

    def __init__(self, seed: int, repetition: int = 15) -> None:
        self._extractor = FuzzyExtractor(128, repetition)
        self._puf = SimulatedPuf(derive_seed("rfd031.puf-chip", seed), n_cells=self._extractor.n_cells_required)
        rng = make_rng(derive_seed("rfd031.puf-enrol", seed))
        key, self.helper = self._extractor.enroll(self._puf.measure(rng=rng), rng=rng)
        self.public_key = KeyPair.from_seed(domain_hash("rfd031.puf-key", key)).public_bytes
        self.chip_id = chip_id_of(self.public_key)
        self._counter = 0

    @property
    def counter(self) -> int:
        return self._counter

    def attest(
        self, challenge: bytes, rng: np.random.Generator, temperature_c: float = 25.0, age_years: float = 0.0
    ) -> Attestation | None:
        """Attest to ``challenge``, or return None when the PUF no longer reproduces the enrolled key."""
        raw = self._puf.measure(temperature_c=temperature_c, age_years=age_years, rng=rng)
        key, reproduced = self._extractor.reproduce(raw, self.helper)
        if not reproduced:
            return None
        self._counter += 1
        signer = KeyPair.from_seed(domain_hash("rfd031.puf-key", key))
        return Attestation(
            self.chip_id, challenge, self._counter, signer.sign(attestation_message(self.chip_id, challenge, self._counter))
        )


@dataclass(frozen=True)
class PufSeparation:
    """Bit-error rates between measurements of one chip and between different chips (raw responses, no extractor)."""

    condition: str
    intra_mean: float
    intra_worst: float
    false_reject_rate: float
    inter_mean: float


def puf_separation(
    devices: int, n_cells: int, threshold: float, conditions: dict[str, tuple[float, float]], seed: int = 0
) -> list[PufSeparation]:
    """Compare a noisy enrolment measurement with later ones, for the same chip and for other chips.

    ``conditions`` maps a label to (temperature in C, age in years). A later measurement of the same chip that is more
    than ``threshold`` away from its enrolment counts as a false reject.
    """
    if devices < 2:
        raise ValueError("need at least two devices")
    rng = make_rng(derive_seed("rfd031.puf-separation", seed))
    pufs = [SimulatedPuf(derive_seed("rfd031.puf-device", seed, d), n_cells=n_cells) for d in range(devices)]
    enrolled = [p.measure(rng=rng) for p in pufs]
    rows = []
    for label, (temperature, age) in conditions.items():
        later = [p.measure(temperature_c=temperature, age_years=age, rng=rng) for p in pufs]
        intra = np.array([bit_error_rate(e, m) for e, m in zip(enrolled, later)])
        inter = np.array([bit_error_rate(enrolled[(d + 1) % devices], later[d]) for d in range(devices)])
        rows.append(PufSeparation(label, float(intra.mean()), float(intra.max()), float((intra > threshold).mean()), float(inter.mean())))
    return rows


def puf_false_accept_probability(n_cells: int, threshold: float) -> float:
    """P(an unrelated chip's response is within ``threshold`` of an enrolled one): a binomial tail at p = 1/2."""
    return float(sps.binom.cdf(int(np.floor(threshold * n_cells)), n_cells, 0.5))


# ---- adhesive glitter patterns -------------------------------------------------------------------------------------


def random_tag(n_flakes: int, rng: np.random.Generator) -> np.ndarray:
    """``n_flakes`` flake positions, uniform on the unit torus."""
    return rng.random((n_flakes, 2))


def photograph(tag: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """A photograph of a tag: every flake seen, each position perturbed by isotropic Gaussian noise of scale ``sigma``."""
    return (tag + rng.normal(scale=sigma, size=tag.shape)) % 1.0


def match_fraction(enrolled: np.ndarray, photo: np.ndarray, radius: float) -> float:
    """Share of enrolled flakes with a photographed flake within ``radius`` (torus distance)."""
    delta = np.abs(enrolled[:, None, :] - photo[None, :, :])
    delta = np.minimum(delta, 1.0 - delta)
    nearest = (delta**2).sum(axis=2).min(axis=1)
    return float(np.mean(nearest <= radius**2))


def glitter_error_rates(n_flakes: int, sigma: float, radius: float, threshold: float) -> tuple[float, float]:
    """(false reject rate, false accept rate) of "at least ``threshold`` of the flakes match", by an independence approximation.

    Each enrolled flake is matched by its own photographed flake with probability 1 - exp(-r^2 / 2 sigma^2), or by one
    of the other flakes by chance with probability pi r^2 each; flake matches are treated as independent, which the
    simulation in ``simulate_glitter`` checks.
    """
    if not (0.0 < radius < 0.5 and sigma > 0.0 and 0.0 < threshold <= 1.0 and n_flakes >= 1):
        raise ValueError("require 0 < radius < 0.5, sigma > 0, 0 < threshold <= 1 and at least one flake")
    need = int(np.ceil(threshold * n_flakes - 1e-12))
    chance = 1.0 - (1.0 - np.pi * radius**2) ** n_flakes
    own = 1.0 - np.exp(-(radius**2) / (2.0 * sigma**2))
    genuine = 1.0 - (1.0 - own) * (1.0 - np.pi * radius**2) ** (n_flakes - 1)
    frr = float(sps.binom.cdf(need - 1, n_flakes, genuine))
    far = float(sps.binom.sf(need - 1, n_flakes, chance))
    return frr, far


def simulate_glitter(
    n_flakes: int, sigma: float, radius: float, threshold: float, trials: int, rng: np.random.Generator
) -> tuple[McEstimate, McEstimate]:
    """Monte Carlo (false reject, false accept) for the same decision rule as ``glitter_error_rates``."""

    def reject(r: np.random.Generator) -> bool:
        tag = random_tag(n_flakes, r)
        return match_fraction(tag, photograph(tag, sigma, r), radius) < threshold - 1e-12

    def accept(r: np.random.Generator) -> bool:
        tag = random_tag(n_flakes, r)
        other = random_tag(n_flakes, r)
        return match_fraction(tag, photograph(other, sigma, r), radius) >= threshold - 1e-12

    return monte_carlo_probability(reject, trials, rng), monte_carlo_probability(accept, trials, rng)
