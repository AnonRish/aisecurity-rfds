"""Signed compute licenses: the object whose replay RFD 034 wants to prevent.

Security-relevant choices:

* The Ed25519 signature covers every field (device, sequence, id, budget, challenge) through a
  domain-separated, length-prefixed hash, so fields cannot be re-parsed or recombined.
* ``content_digest`` excludes the signature and ``full_digest`` includes it. They differ exactly when a
  second valid encoding of the same signed content exists (signature malleability, lenient parsers),
  which is why replay logs should key on the content digest.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from aisecurity_rfds.common.hashing import domain_hash, u64
from aisecurity_rfds.common.signing import KeyPair

_DOMAIN = "rfd034/license/v1"
LICENSE_ID_BYTES = 16


@dataclass(frozen=True)
class License:
    device_id: bytes
    sequence: int
    license_id: bytes
    budget_flops: int
    challenge: bytes
    signature: bytes

    def content_digest(self) -> bytes:
        """The 32-byte value the authority signs (everything except the signature)."""
        return domain_hash(
            _DOMAIN, self.device_id, u64(self.sequence), self.license_id, u64(self.budget_flops), self.challenge
        )

    def full_digest(self) -> bytes:
        """Digest of the license as transmitted, signature included."""
        return domain_hash(_DOMAIN + "/full", self.content_digest(), self.signature)

    def with_signature(self, signature: bytes) -> License:
        return dataclasses.replace(self, signature=signature)


class LicenseAuthority:
    """Issues licenses; its key pair stands in for the private half of the key fused into the chip."""

    def __init__(self, key: KeyPair, rng: np.random.Generator) -> None:
        self._key = key
        self._rng = rng

    @property
    def public_key(self) -> bytes:
        return self._key.public_bytes

    def issue(
        self,
        device_id: bytes,
        *,
        sequence: int,
        budget_flops: int,
        challenge: bytes = b"",
        license_id: bytes | None = None,
    ) -> License:
        if sequence < 0 or budget_flops < 0:
            raise ValueError("sequence and budget_flops must be non-negative")
        if license_id is None:
            license_id = bytes(self._rng.integers(0, 256, LICENSE_ID_BYTES, dtype=np.uint8))
        if len(license_id) != LICENSE_ID_BYTES:
            raise ValueError(f"license_id must be {LICENSE_ID_BYTES} bytes")
        unsigned = License(device_id, sequence, license_id, budget_flops, challenge, b"")
        return unsigned.with_signature(self._key.sign(unsigned.content_digest()))
