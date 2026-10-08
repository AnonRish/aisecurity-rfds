"""Hash helpers: plain SHA-256, unambiguous domain-separated hashing, constant-time compare."""

from __future__ import annotations

import hashlib
import hmac
import struct


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def domain_hash(domain: str, *parts: bytes) -> bytes:
    """Domain-separated SHA-256 over length-prefixed parts.

    Length prefixes make the encoding injective: ``domain_hash(d, b"ab", b"c")`` differs from
    ``domain_hash(d, b"a", b"bc")``, and different domains never collide by construction of the
    prefix (up to SHA-256 collision resistance).
    """
    h = hashlib.sha256()
    d = domain.encode()
    h.update(struct.pack(">I", len(d)))
    h.update(d)
    for part in parts:
        h.update(struct.pack(">Q", len(part)))
        h.update(part)
    return h.digest()


def ct_equal(a: bytes, b: bytes) -> bool:
    """Constant-time equality for secrets, MACs and digests."""
    return hmac.compare_digest(a, b)


def u64(n: int) -> bytes:
    return n.to_bytes(8, "big")


def u32(n: int) -> bytes:
    return n.to_bytes(4, "big")
