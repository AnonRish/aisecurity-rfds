"""A sparse Merkle map: a verifiable key-to-value store with proofs of presence and of absence (RFD 031, registry database).

Keys are placed by the first ``DEPTH`` bits of a hash of the key, so the tree has a fixed shape and its root depends
only on its contents, not on the order of insertion. Empty subtrees hash to precomputed defaults, so only the paths
to stored keys are held in memory and a proof carries only the siblings that are not defaults. A proof for a key that
is absent is the same object as a proof for a key that is present, checked against an empty leaf, which is what lets
a client verify "this chip is not registered" instead of taking the operator's word for it.
"""

from __future__ import annotations

from dataclasses import dataclass

from aisecurity_rfds.common.hashing import sha256

DEPTH = 128


def _leaf(key: bytes, value: bytes) -> bytes:
    return sha256(b"\x00" + len(key).to_bytes(4, "big") + key + value)


def _node(left: bytes, right: bytes) -> bytes:
    return sha256(b"\x01" + left + right)


EMPTY_LEAF = sha256(b"\x02")
_DEFAULTS: list[bytes] = [EMPTY_LEAF]
for _ in range(DEPTH):
    _DEFAULTS.append(_node(_DEFAULTS[-1], _DEFAULTS[-1]))


def path_of(key: bytes) -> int:
    """The leaf position of ``key``: ``DEPTH`` bits of a domain-separated hash of it."""
    return int.from_bytes(sha256(b"rfd031.smt.path" + key)[: DEPTH // 8], "big")


@dataclass(frozen=True)
class MapProof:
    """Siblings from the leaf upwards; bit ``l`` of ``bitmap`` is set when the level-``l`` sibling is not the default."""

    bitmap: int
    siblings: tuple[bytes, ...]

    def size_bytes(self) -> int:
        return DEPTH // 8 + 32 * len(self.siblings)


class SparseMerkleMap:
    def __init__(self) -> None:
        self._nodes: dict[tuple[int, int], bytes] = {}  # (level, index) -> hash; level 0 holds the leaves
        self._items: dict[int, tuple[bytes, bytes]] = {}

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: bytes) -> bytes | None:
        item = self._items.get(path_of(key))
        return item[1] if item is not None and item[0] == key else None

    def set(self, key: bytes, value: bytes) -> None:
        p = path_of(key)
        held = self._items.get(p)
        if held is not None and held[0] != key:
            raise ValueError("two distinct keys share a leaf position")
        self._items[p] = (key, value)
        h = _leaf(key, value)
        self._nodes[(0, p)] = h
        for level in range(DEPTH):
            sibling = self._nodes.get((level, (p >> level) ^ 1), _DEFAULTS[level])
            h = _node(sibling, h) if (p >> level) & 1 else _node(h, sibling)
            self._nodes[(level + 1, p >> (level + 1))] = h

    def root(self) -> bytes:
        return self._nodes.get((DEPTH, 0), _DEFAULTS[DEPTH])

    def prove(self, key: bytes) -> MapProof:
        p = path_of(key)
        bitmap = 0
        siblings = []
        for level in range(DEPTH):
            sibling = self._nodes.get((level, (p >> level) ^ 1))
            if sibling is not None:
                bitmap |= 1 << level
                siblings.append(sibling)
        return MapProof(bitmap, tuple(siblings))


def verify_proof(root: bytes, key: bytes, value: bytes | None, proof: MapProof) -> bool:
    """True when ``value`` (``None`` for absent) is what the map with this ``root`` holds under ``key``."""
    if proof.bitmap >> DEPTH or bin(proof.bitmap).count("1") != len(proof.siblings):
        return False
    p = path_of(key)
    h = EMPTY_LEAF if value is None else _leaf(key, value)
    supplied = iter(proof.siblings)
    for level in range(DEPTH):
        sibling = next(supplied) if (proof.bitmap >> level) & 1 else _DEFAULTS[level]
        h = _node(sibling, h) if (p >> level) & 1 else _node(h, sibling)
    return h == root
