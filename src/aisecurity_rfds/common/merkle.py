"""Append-only Merkle log with RFC 6962 / RFC 9162 inclusion and consistency proofs.

Tree shape and hashing follow RFC 6962 section 2.1: leaves are hashed with a 0x00 prefix, interior
nodes with 0x01, and a tree of ``n`` leaves splits at the largest power of two strictly below ``n``.
Verification routines follow the iterative algorithms of RFC 9162 sections 2.1.3.2 and 2.1.4.2.

SECURITY NOTE: a Merkle root does not commit to the tree size. Two trees of different sizes can
share a proof shape for some leaves, so ``verify_inclusion(..., size, ..., root)`` only means
something when ``(size, root)`` arrive together from an authenticated source (for example a signed
checkpoint, as in Certificate Transparency). Always sign or otherwise bind the pair.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

_LEAF = b"\x00"
_NODE = b"\x01"
EMPTY_ROOT = hashlib.sha256(b"").digest()


def leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(_LEAF + data).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE + left + right).digest()


def _split(n: int) -> int:
    """Largest power of two strictly less than ``n`` (requires n >= 2)."""
    return 1 << ((n - 1).bit_length() - 1)


def _mth(hashes: Sequence[bytes]) -> bytes:
    n = len(hashes)
    if n == 0:
        return EMPTY_ROOT
    if n == 1:
        return hashes[0]
    k = _split(n)
    return node_hash(_mth(hashes[:k]), _mth(hashes[k:]))


def _path(m: int, hashes: Sequence[bytes]) -> list[bytes]:
    n = len(hashes)
    if n == 1:
        return []
    k = _split(n)
    if m < k:
        return _path(m, hashes[:k]) + [_mth(hashes[k:])]
    return _path(m - k, hashes[k:]) + [_mth(hashes[:k])]


def _subproof(m: int, hashes: Sequence[bytes], complete: bool) -> list[bytes]:
    n = len(hashes)
    if m == n:
        return [] if complete else [_mth(hashes)]
    k = _split(n)
    if m <= k:
        return _subproof(m, hashes[:k], complete) + [_mth(hashes[k:])]
    return _subproof(m - k, hashes[k:], False) + [_mth(hashes[:k])]


class MerkleLog:
    """Append-only Merkle log. ``root()`` is O(log n); proofs are O(n) to build, O(log n) to check."""

    def __init__(self, leaves: Sequence[bytes] = ()):
        self._hashes: list[bytes] = []
        self._frontier: list[tuple[int, bytes]] = []  # perfect subtrees, largest first
        for leaf in leaves:
            self.append(leaf)

    def __len__(self) -> int:
        return len(self._hashes)

    def append(self, data: bytes) -> int:
        index = len(self._hashes)
        h = leaf_hash(data)
        self._hashes.append(h)
        self._frontier.append((1, h))
        while len(self._frontier) >= 2 and self._frontier[-1][0] == self._frontier[-2][0]:
            s2, h2 = self._frontier.pop()
            s1, h1 = self._frontier.pop()
            self._frontier.append((s1 + s2, node_hash(h1, h2)))
        return index

    def root(self, size: int | None = None) -> bytes:
        n = len(self) if size is None else size
        if not 0 <= n <= len(self):
            raise ValueError(f"size {n} out of range 0..{len(self)}")
        if n == len(self):
            if n == 0:
                return EMPTY_ROOT
            r = self._frontier[-1][1]
            for _, h in reversed(self._frontier[:-1]):
                r = node_hash(h, r)
            return r
        return _mth(self._hashes[:n])

    def inclusion_proof(self, index: int, size: int | None = None) -> list[bytes]:
        n = len(self) if size is None else size
        if not 0 <= index < n <= len(self):
            raise ValueError("index/size out of range")
        return _path(index, self._hashes[:n])

    def consistency_proof(self, first: int, second: int | None = None) -> list[bytes]:
        n = len(self) if second is None else second
        if not 1 <= first <= n <= len(self):
            raise ValueError("require 1 <= first <= second <= len(log)")
        return _subproof(first, self._hashes[:n], True)


def verify_inclusion(
    leaf_data: bytes, index: int, size: int, proof: Sequence[bytes], root: bytes
) -> bool:
    """Check that ``leaf_data`` is leaf ``index`` of the size-``size`` tree with Merkle root ``root``."""
    if not 0 <= index < size:
        return False
    fn, sn = index, size - 1
    r = leaf_hash(leaf_data)
    for p in proof:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = node_hash(p, r)
            if not fn & 1:
                while not (fn & 1) and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            r = node_hash(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


def verify_consistency(
    first: int,
    second: int,
    first_root: bytes,
    second_root: bytes,
    proof: Sequence[bytes],
) -> bool:
    """Check that the size-``first`` tree is a prefix of the size-``second`` tree (append-only)."""
    if first < 1 or first > second:
        return False
    if first == second:
        return len(proof) == 0 and first_root == second_root
    path = list(proof)
    if first & (first - 1) == 0:  # exact power of two
        path = [first_root] + path
    if not path:
        return False
    fn, sn = first - 1, second - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1
    fr = sr = path[0]
    for c in path[1:]:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            fr = node_hash(c, fr)
            sr = node_hash(c, sr)
            if not fn & 1:
                while not (fn & 1) and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            sr = node_hash(sr, c)
        fn >>= 1
        sn >>= 1
    return fr == first_root and sr == second_root and sn == 0
