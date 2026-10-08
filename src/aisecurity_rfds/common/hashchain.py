"""Hash-chained, optionally signed append-only log with external-anchor (checkpoint) verification.

Each entry commits to its index, the previous entry hash and its payload, so any deletion,
reordering or modification is detected by :func:`verify_chain`. A *checkpoint* (length + head hash,
optionally signed) held outside the device detects rollback and wholesale replacement, which a
self-contained chain cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .hashing import ct_equal, domain_hash, u64
from .signing import KeyPair, verify

GENESIS = bytes(32)


@dataclass(frozen=True)
class ChainEntry:
    index: int
    prev_hash: bytes
    payload: bytes
    entry_hash: bytes
    signature: bytes | None = None


@dataclass(frozen=True)
class Checkpoint:
    log_id: bytes
    length: int
    head: bytes
    signature: bytes | None = None


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    reason: str = "ok"

    def __bool__(self) -> bool:
        return self.ok


def compute_entry_hash(log_id: bytes, index: int, prev_hash: bytes, payload: bytes) -> bytes:
    return domain_hash("aisec.hashchain.entry", log_id, u64(index), prev_hash, payload)


def _checkpoint_digest(log_id: bytes, length: int, head: bytes) -> bytes:
    return domain_hash("aisec.hashchain.checkpoint", log_id, u64(length), head)


class HashChainLog:
    def __init__(self, log_id: bytes = b"", signer: KeyPair | None = None):
        self.log_id = log_id
        self._signer = signer
        self._entries: list[ChainEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[ChainEntry, ...]:
        return tuple(self._entries)

    @property
    def head(self) -> bytes:
        return self._entries[-1].entry_hash if self._entries else GENESIS

    def append(self, payload: bytes) -> ChainEntry:
        index = len(self._entries)
        prev = self.head
        h = compute_entry_hash(self.log_id, index, prev, payload)
        sig = self._signer.sign(h) if self._signer else None
        entry = ChainEntry(index, prev, payload, h, sig)
        self._entries.append(entry)
        return entry

    def checkpoint(self) -> Checkpoint:
        digest = _checkpoint_digest(self.log_id, len(self), self.head)
        sig = self._signer.sign(digest) if self._signer else None
        return Checkpoint(self.log_id, len(self), self.head, sig)


def verify_chain(
    entries: Sequence[ChainEntry],
    *,
    log_id: bytes = b"",
    public_key: bytes | None = None,
    anchor: Checkpoint | None = None,
    require_signatures: bool = False,
) -> VerifyResult:
    """Verify structure, hashes, optional per-entry signatures and an optional external anchor."""
    prev = GENESIS
    for i, e in enumerate(entries):
        if e.index != i:
            return VerifyResult(False, f"entry {i}: index {e.index} not contiguous")
        if not ct_equal(e.prev_hash, prev):
            return VerifyResult(False, f"entry {i}: broken link to previous entry")
        if not ct_equal(e.entry_hash, compute_entry_hash(log_id, i, prev, e.payload)):
            return VerifyResult(False, f"entry {i}: hash mismatch (payload or header modified)")
        if public_key is not None:
            if e.signature is None:
                if require_signatures:
                    return VerifyResult(False, f"entry {i}: missing signature")
            elif not verify(public_key, e.entry_hash, e.signature):
                return VerifyResult(False, f"entry {i}: bad signature")
        elif require_signatures and e.signature is None:
            return VerifyResult(False, f"entry {i}: missing signature")
        prev = e.entry_hash

    if anchor is not None:
        if anchor.log_id != log_id:
            return VerifyResult(False, "anchor is for a different log")
        if public_key is not None:
            if anchor.signature is None or not verify(
                public_key,
                _checkpoint_digest(anchor.log_id, anchor.length, anchor.head),
                anchor.signature,
            ):
                return VerifyResult(False, "anchor signature invalid")
        if len(entries) < anchor.length:
            return VerifyResult(False, "log shorter than anchor (truncation or rollback)")
        if anchor.length > 0 and not ct_equal(entries[anchor.length - 1].entry_hash, anchor.head):
            return VerifyResult(False, "log does not extend anchor (history rewritten)")
        if anchor.length == 0 and not ct_equal(anchor.head, GENESIS):
            return VerifyResult(False, "anchor head inconsistent with empty log")
    return VerifyResult(True)
