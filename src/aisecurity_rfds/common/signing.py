"""Ed25519 signing wrappers (via ``cryptography``). Deterministic keys are available for tests."""

from __future__ import annotations

from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


@dataclass(frozen=True)
class KeyPair:
    private: Ed25519PrivateKey
    public_bytes: bytes  # 32-byte raw public key

    @classmethod
    def generate(cls) -> "KeyPair":
        return cls._from_private(Ed25519PrivateKey.generate())

    @classmethod
    def from_seed(cls, seed: bytes) -> "KeyPair":
        """Deterministic key from a 32-byte seed (tests and simulations only)."""
        if len(seed) != 32:
            raise ValueError("seed must be exactly 32 bytes")
        return cls._from_private(Ed25519PrivateKey.from_private_bytes(seed))

    @classmethod
    def _from_private(cls, private: Ed25519PrivateKey) -> "KeyPair":
        pub = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return cls(private=private, public_bytes=pub)

    def sign(self, message: bytes) -> bytes:
        return self.private.sign(message)


def verify(public_bytes: bytes, message: bytes, signature: bytes) -> bool:
    """Return True iff ``signature`` is a valid Ed25519 signature of ``message``."""
    try:
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(signature, message)
    except (InvalidSignature, ValueError):
        return False
    return True
