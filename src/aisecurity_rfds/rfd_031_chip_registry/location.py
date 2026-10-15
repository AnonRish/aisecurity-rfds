"""Location in a registry: commitments that hide a facility until it is opened, and audits that sample for ghosts (RFD 031).

Two of the RFD's open questions are about location. Whether to record it at all is a policy choice, so this module
only models the options and what each reveals: nothing but a jurisdiction (a "logical location"), a salted commitment
that the owner opens to an inspector, or the facility in plaintext. Whether a logical location is enough is a question
about what an audit can catch, which is a sampling calculation: if some of an owner's declared chips are not where
they are declared (or do not exist), how many must an inspector challenge to see one?
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from aisecurity_rfds.common.hashing import ct_equal, domain_hash
from aisecurity_rfds.common.stats import McEstimate, hypergeom_miss_probability, monte_carlo_probability, samples_needed_hypergeom

from .identity import DeviceKeyChip, verify_attestation


def commit_location(facility: str, salt: bytes) -> bytes:
    """A binding commitment to a facility name; hiding only as far as the salt is unguessable."""
    return domain_hash("rfd031.location", facility.encode(), salt)


@dataclass(frozen=True)
class Opening:
    facility: str
    salt: bytes


def verify_opening(commitment: bytes, opening: Opening) -> bool:
    return ct_equal(commit_location(opening.facility, opening.salt), commitment)


def guess_facility(commitment: bytes, candidates: Sequence[str], salt_bytes: int) -> Opening | None:
    """Dictionary attack: try every candidate facility with every possible salt of ``salt_bytes`` bytes."""
    if salt_bytes < 0:
        raise ValueError("salt length cannot be negative")
    for facility in candidates:
        for value in range(256**salt_bytes):
            salt = value.to_bytes(salt_bytes, "big")
            if ct_equal(commit_location(facility, salt), commitment):
                return Opening(facility, salt)
    return None


def dictionary_work(candidates: int, salt_bytes: int) -> int:
    """Hash evaluations needed to exhaust ``candidates`` facility names against a ``salt_bytes``-byte salt."""
    return candidates * 256**salt_bytes


def distinct_commitments(facilities: Sequence[str], salts: Sequence[bytes]) -> int:
    """How many different commitments an observer of the registry sees for chips at the given facilities."""
    return len({commit_location(f, s) for f, s in zip(facilities, salts)})


# ---- ghost-inventory audits ----------------------------------------------------------------------------------------


def detection_probability(total: int, ghosts: int, sampled: int) -> float:
    """P(at least one of ``ghosts`` non-existent or misplaced chips is among ``sampled`` of ``total`` declared)."""
    return 1.0 - hypergeom_miss_probability(total, ghosts, sampled)


def chips_to_inspect(total: int, ghosts: int, confidence: float) -> int:
    return samples_needed_hypergeom(total, ghosts, confidence)


def simulate_detection(total: int, ghosts: int, sampled: int, trials: int, rng: np.random.Generator) -> McEstimate:
    """Monte Carlo of the same event: draw ``sampled`` declared chips uniformly without replacement."""

    def trial(r: np.random.Generator) -> bool:
        return bool((r.choice(total, size=sampled, replace=False) < ghosts).any())

    return monte_carlo_probability(trial, trials, rng)


class Facility:
    """A site that physically holds some chips; an inspector there can challenge a chip that is present."""

    def __init__(self, chips: Sequence[DeviceKeyChip]) -> None:
        self._chips = {c.chip_id: c for c in chips}

    def inspect(self, chip_id: bytes, public_key: bytes, challenge: bytes) -> bool:
        """True when the declared chip is here and signs the challenge; False when it cannot be produced."""
        chip = self._chips.get(chip_id)
        return chip is not None and verify_attestation(public_key, challenge, chip.attest(challenge))


def audit_declared(
    declared: Sequence[tuple[bytes, bytes]],
    facility: Facility,
    sample: Sequence[int],
    challenge_for: Callable[[bytes], bytes],
) -> list[int]:
    """Challenge the chips at ``sample`` positions of ``declared`` ((chip id, public key) pairs); return the failures."""
    return [i for i in sample if not facility.inspect(declared[i][0], declared[i][1], challenge_for(declared[i][0]))]
