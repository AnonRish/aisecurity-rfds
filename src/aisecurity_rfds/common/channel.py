"""X25519 key agreement, HKDF-SHA256 and AES-256-GCM sessions with counter nonces.

Nonce discipline is the point: AES-GCM is catastrophically broken by nonce reuse (RFD 021), so
senders use a strictly increasing counter and refuse to wrap, and receivers reject replays and
reordering. Keys are derived per direction so the two directions never share a nonce space.
"""

from __future__ import annotations

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class NonceExhaustedError(Exception):
    """The sender hit its message budget for this key; a fresh session key is required."""


class ReplayError(Exception):
    """A received message carried a counter that is not strictly greater than the last accepted."""


class DecryptionError(Exception):
    """Authentication failed: wrong key, modified ciphertext/AAD, or truncated message."""


def x25519_generate() -> tuple[X25519PrivateKey, bytes]:
    priv = X25519PrivateKey.generate()
    pub = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return priv, pub


def x25519_exchange(private: X25519PrivateKey, peer_public: bytes) -> bytes:
    return private.exchange(X25519PublicKey.from_public_bytes(peer_public))


def hkdf_sha256(ikm: bytes, *, salt: bytes = b"", info: bytes = b"", length: int = 32) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt or None, info=info).derive(ikm)


_COUNTER_LEN = 8


class AeadSender:
    def __init__(self, key: bytes, *, max_messages: int = 2**40):
        if len(key) != 32:
            raise ValueError("AES-256-GCM key must be 32 bytes")
        self._aead = AESGCM(key)
        self._counter = 0
        self._max = max_messages

    @property
    def messages_sent(self) -> int:
        return self._counter

    def seal(self, plaintext: bytes, aad: bytes = b"") -> bytes:
        if self._counter >= self._max:
            raise NonceExhaustedError("message budget exhausted for this key")
        counter = self._counter
        self._counter += 1  # incremented before use: a failure can never cause nonce reuse
        header = counter.to_bytes(_COUNTER_LEN, "big")
        nonce = bytes(4) + header
        return header + self._aead.encrypt(nonce, plaintext, aad)


class AeadReceiver:
    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("AES-256-GCM key must be 32 bytes")
        self._aead = AESGCM(key)
        self._last = -1

    def open(self, blob: bytes, aad: bytes = b"") -> bytes:
        if len(blob) < _COUNTER_LEN + 16:
            raise DecryptionError("message too short")
        header, ct = blob[:_COUNTER_LEN], blob[_COUNTER_LEN:]
        counter = int.from_bytes(header, "big")
        nonce = bytes(4) + header
        try:
            plaintext = self._aead.decrypt(nonce, ct, aad)
        except InvalidTag as exc:
            raise DecryptionError("authentication failed") from exc
        if counter <= self._last:  # checked after auth so forged counters cannot burn the window
            raise ReplayError(f"counter {counter} <= last accepted {self._last}")
        self._last = counter
        return plaintext


def derive_session(
    shared_secret: bytes, context: bytes, *, initiator: bool, max_messages: int = 2**40
) -> tuple[AeadSender, AeadReceiver]:
    """Derive per-direction keys from a shared secret; returns (sender, receiver) for this side."""
    k_i2r = hkdf_sha256(shared_secret, salt=b"aisec.channel", info=context + b"|i2r")
    k_r2i = hkdf_sha256(shared_secret, salt=b"aisec.channel", info=context + b"|r2i")
    send_key, recv_key = (k_i2r, k_r2i) if initiator else (k_r2i, k_i2r)
    return AeadSender(send_key, max_messages=max_messages), AeadReceiver(recv_key)
