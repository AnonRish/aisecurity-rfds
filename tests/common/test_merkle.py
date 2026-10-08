import hashlib

import pytest

from aisecurity_rfds.common import merkle
from aisecurity_rfds.common.merkle import (
    EMPTY_ROOT,
    MerkleLog,
    _mth,
    leaf_hash,
    verify_consistency,
    verify_inclusion,
)


def leaves(n: int) -> list[bytes]:
    return [f"leaf-{i}".encode() for i in range(n)]


def test_empty_and_single_leaf_roots():
    assert MerkleLog().root() == hashlib.sha256(b"").digest() == EMPTY_ROOT
    # RFC 6962: leaf hash = SHA-256(0x00 || data); a one-leaf tree's root is that leaf hash.
    assert leaf_hash(b"") == hashlib.sha256(b"\x00").digest()
    assert MerkleLog([b""]).root() == hashlib.sha256(b"\x00").digest()


def test_matches_certificate_transparency_reference_vectors():
    # Tree-head vectors from the RFC 6962 reference test data (certificate-transparency-go / -rs).
    data = [b"", b"\x00", b"\x10", b"\x20\x21", b"\x30\x31", b"\x40\x41\x42\x43",
            bytes(range(0x50, 0x58)), bytes(range(0x60, 0x70))]
    roots = [
        "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
        "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
        "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77",
        "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
        "4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4",
        "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef",
        "ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c",
        "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
    ]
    log = MerkleLog()
    for i, d in enumerate(data):
        log.append(d)
        assert log.root().hex() == roots[i], f"size {i + 1}"


def test_frontier_root_matches_recursive_definition_for_all_sizes():
    log = MerkleLog()
    for n in range(0, 71):
        if n:
            log.append(f"leaf-{n - 1}".encode())
        expected = _mth([leaf_hash(x) for x in leaves(n)])
        assert log.root() == expected, n
        assert log.root(n) == expected


def test_historical_roots_are_stable_as_log_grows():
    log = MerkleLog(leaves(40))
    for n in range(0, 41):
        assert log.root(n) == MerkleLog(leaves(n)).root()


def test_inclusion_proofs_exhaustive():
    for n in range(1, 41):
        log = MerkleLog(leaves(n))
        root = log.root()
        for i in range(n):
            proof = log.inclusion_proof(i)
            assert verify_inclusion(leaves(n)[i], i, n, proof, root), (n, i)
            # wrong leaf data, wrong index, wrong size, wrong root must all fail
            assert not verify_inclusion(b"forged", i, n, proof, root)
            assert not verify_inclusion(leaves(n)[i], (i + 1) % n, n, proof, root) or n == 1
            assert not verify_inclusion(leaves(n)[i], i, n, proof, bytes(32))


def test_root_does_not_commit_to_size_so_size_must_be_authenticated():
    # Property of RFC 6962 trees (not a bug): the root hash alone does not bind the tree size.
    # A proof with the same path shape verifies under a different claimed size. Callers MUST
    # authenticate (size, root) together, e.g. in a signed checkpoint, as Certificate Transparency does.
    log = MerkleLog(leaves(3))
    proof = log.inclusion_proof(0)
    assert verify_inclusion(leaves(3)[0], 0, 3, proof, log.root())
    assert verify_inclusion(leaves(3)[0], 0, 4, proof, log.root())  # same shape, different claimed size
    assert not verify_inclusion(leaves(3)[0], 0, 8, proof, log.root())  # different shape is rejected


def test_inclusion_proof_for_historical_size():
    log = MerkleLog(leaves(33))
    proof = log.inclusion_proof(5, size=20)
    assert verify_inclusion(leaves(33)[5], 5, 20, proof, log.root(20))
    assert not verify_inclusion(leaves(33)[5], 5, 20, proof, log.root(33))


def test_inclusion_proof_rejects_extra_or_missing_nodes():
    log = MerkleLog(leaves(13))
    proof = log.inclusion_proof(4)
    root = log.root()
    assert not verify_inclusion(leaves(13)[4], 4, 13, proof + [bytes(32)], root)
    assert not verify_inclusion(leaves(13)[4], 4, 13, proof[:-1], root)


def test_consistency_proofs_exhaustive():
    log = MerkleLog(leaves(40))
    for n in range(1, 41):
        for m in range(1, n + 1):
            proof = log.consistency_proof(m, n)
            assert verify_consistency(m, n, log.root(m), log.root(n), proof), (m, n)


def test_consistency_detects_rewritten_history():
    honest = MerkleLog(leaves(20))
    forked = MerkleLog([b"evil"] + leaves(20)[1:])  # same size, first entry rewritten
    for m in range(1, 20):
        proof = forked.consistency_proof(m, 20)
        # the forked log is internally consistent but does NOT extend the honest size-m root
        assert not verify_consistency(m, 20, honest.root(m), forked.root(20), proof)


def test_consistency_rejects_corrupted_proof_and_bad_arguments():
    log = MerkleLog(leaves(37))
    proof = log.consistency_proof(11, 37)
    assert verify_consistency(11, 37, log.root(11), log.root(37), proof)
    bad = list(proof)
    bad[0] = hashlib.sha256(bad[0]).digest()
    assert not verify_consistency(11, 37, log.root(11), log.root(37), bad)
    assert not verify_consistency(11, 37, log.root(11), log.root(37), proof[:-1])
    assert not verify_consistency(0, 37, log.root(0), log.root(37), proof)
    assert not verify_consistency(38, 37, log.root(37), log.root(37), [])
    assert verify_consistency(37, 37, log.root(37), log.root(37), [])
    assert not verify_consistency(37, 37, log.root(37), log.root(37), [bytes(32)])


def test_domain_separation_leaf_vs_node():
    # A node hash must never be accepted as a leaf: classic second-preimage defence.
    a, b = leaf_hash(b"a"), leaf_hash(b"b")
    fake_leaf_data = a + b
    two_leaf_root = MerkleLog([b"a", b"b"]).root()
    assert leaf_hash(fake_leaf_data) != two_leaf_root
    assert merkle.node_hash(a, b) == two_leaf_root


def test_argument_validation():
    log = MerkleLog(leaves(5))
    with pytest.raises(ValueError):
        log.inclusion_proof(5)
    with pytest.raises(ValueError):
        log.root(6)
    with pytest.raises(ValueError):
        log.consistency_proof(0)
    with pytest.raises(ValueError):
        log.consistency_proof(3, 9)
