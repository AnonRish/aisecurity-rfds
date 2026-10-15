"""Witnesses and quorums: how many parties must collude before an operator can show two different registries (RFD 031).

The operator signs every tree head, so a client that trusts the operator trusts it not to show different clients
different histories. A witness removes that trust: it cosigns a head only if the head extends the last head it
cosigned (a Merkle consistency proof) and never cosigns two different heads of the same size. A client then accepts a
head only when at least ``threshold`` of ``n`` known witnesses have cosigned it.

A witness checks the log, not the map: it cannot tell whether the map root is what replaying the log produces. That is
the auditor's job (``registry.audit_log``). The analysis here is therefore about *equivocation* (two conflicting
heads), not about wrong state in a single head.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

from aisecurity_rfds.common.hashing import domain_hash
from aisecurity_rfds.common.merkle import verify_consistency
from aisecurity_rfds.common.signing import KeyPair, verify

from .registry import Registry, SignedTreeHead


class WitnessRefused(Exception):
    """A witness declined to cosign; ``code`` says why."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def cosign_message(sth: SignedTreeHead) -> bytes:
    return domain_hash("rfd031.cosign", sth.message())


class Witness:
    def __init__(self, key: KeyPair, operator_public_key: bytes, max_skew: int = 60) -> None:
        self._key = key
        self._operator = operator_public_key
        self._max_skew = max_skew
        self.head: SignedTreeHead | None = None

    @property
    def public_key(self) -> bytes:
        return self._key.public_bytes

    def cosign(self, sth: SignedTreeHead, proof: Sequence[bytes], now: int) -> bytes:
        """Cosign ``sth`` if it is genuine, not from the future, and extends the last head this witness cosigned."""
        if not sth.verify(self._operator):
            raise WitnessRefused("head-signature")
        if sth.timestamp - now > self._max_skew:
            raise WitnessRefused("future-head")
        last = self.head
        if last is not None:
            if sth.timestamp < last.timestamp:
                raise WitnessRefused("timestamp-regression")
            if sth.size < last.size:
                raise WitnessRefused("shrinking")
            if sth.size == last.size:
                if (sth.log_root, sth.map_root) != (last.log_root, last.map_root):
                    raise WitnessRefused("fork")
            elif last.size > 0 and not verify_consistency(last.size, sth.size, last.log_root, sth.log_root, list(proof)):
                raise WitnessRefused("inconsistent")
        self.head = sth
        return self._key.sign(cosign_message(sth))


class CompromisedWitness(Witness):
    """A witness that has been bought or hacked: it cosigns anything the operator shows it."""

    def cosign(self, sth: SignedTreeHead, proof: Sequence[bytes], now: int) -> bytes:
        self.head = sth
        return self._key.sign(cosign_message(sth))


@dataclass(frozen=True)
class CosignedHead:
    sth: SignedTreeHead
    cosignatures: tuple[tuple[bytes, bytes], ...]  # (witness public key, signature over cosign_message)


@dataclass(frozen=True)
class Quorum:
    witnesses: frozenset[bytes]
    threshold: int

    def accepts(self, head: CosignedHead, operator_public_key: bytes) -> bool:
        """True when the operator signed the head and ``threshold`` distinct known witnesses cosigned it."""
        if not head.sth.verify(operator_public_key):
            return False
        message = cosign_message(head.sth)
        valid = {pk for pk, sig in head.cosignatures if pk in self.witnesses and verify(pk, message, sig)}
        return len(valid) >= self.threshold


def gather_cosignatures(registry: Registry, sth: SignedTreeHead, witnesses: Sequence[Witness], now: int) -> CosignedHead:
    """Ask each witness to cosign ``sth``, supplying it a consistency proof from the last head it cosigned."""
    signatures = []
    for witness in witnesses:
        first = witness.head.size if witness.head is not None else 0
        proof = registry.consistency_proof(first, sth.size) if 0 < first < sth.size else []
        try:
            signatures.append((witness.public_key, witness.cosign(sth, proof, now)))
        except WitnessRefused:
            continue
    return CosignedHead(sth, tuple(signatures))


def min_colluders_to_fork(n: int, t: int) -> int:
    """Fewest colluding witnesses with which an operator gets two conflicting heads, each cosigned by ``t`` of ``n``.

    Each honest witness cosigns at most one of two conflicting heads, so the two quorums can only overlap in colluders:
    their union holds at most ``n`` witnesses and each holds at least ``t``, which forces an overlap of ``2t - n``.
    """
    if not 1 <= t <= n:
        raise ValueError("require 1 <= t <= n")
    return max(0, 2 * t - n)


def liveness_margin(n: int, t: int) -> int:
    """How many witnesses may be offline before no new head can gather a quorum."""
    if not 1 <= t <= n:
        raise ValueError("require 1 <= t <= n")
    return n - t


def fork_by_enumeration(
    n: int, t: int, colluders: int, operator_public_key: bytes, head_a: SignedTreeHead, head_b: SignedTreeHead, now: int
) -> bool:
    """Try every way of showing two same-size conflicting heads to ``n`` real witnesses; True if any gets both accepted.

    Each honest witness is shown one head first, then asked for the other; compromised witnesses cosign both.
    """
    if not 0 <= colluders <= n:
        raise ValueError("colluders must be between 0 and n")
    honest = n - colluders
    for first_seen in product("AB", repeat=honest):
        witnesses: list[Witness] = [
            Witness(KeyPair.from_seed(domain_hash("rfd031.test-witness", bytes([i]))), operator_public_key)
            for i in range(honest)
        ]
        witnesses += [
            CompromisedWitness(KeyPair.from_seed(domain_hash("rfd031.test-witness", bytes([honest + i]))), operator_public_key)
            for i in range(colluders)
        ]
        quorum = Quorum(frozenset(w.public_key for w in witnesses), t)
        signatures: dict[str, list[tuple[bytes, bytes]]] = {"A": [], "B": []}
        for i, witness in enumerate(witnesses):
            order = [first_seen[i], "B" if first_seen[i] == "A" else "A"] if i < honest else ["A", "B"]
            for label in order:
                try:
                    sig = witness.cosign(head_a if label == "A" else head_b, [], now)
                except WitnessRefused:
                    continue
                signatures[label].append((witness.public_key, sig))
        accepted_a = quorum.accepts(CosignedHead(head_a, tuple(signatures["A"])), operator_public_key)
        accepted_b = quorum.accepts(CosignedHead(head_b, tuple(signatures["B"])), operator_public_key)
        if accepted_a and accepted_b:
            return True
    return False
