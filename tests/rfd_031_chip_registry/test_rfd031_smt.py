"""The sparse Merkle map: its root is a function of its contents, and its proofs say what is and is not in it."""

from __future__ import annotations

import math

import pytest

from aisecurity_rfds.common.hashing import sha256
from aisecurity_rfds.rfd_031_chip_registry import smt
from aisecurity_rfds.rfd_031_chip_registry.smt import DEPTH, EMPTY_LEAF, MapProof, SparseMerkleMap, path_of, verify_proof


def keys(n: int, tag: bytes = b"k") -> list[bytes]:
    return [sha256(tag + str(i).encode()) for i in range(n)]


def filled(n: int) -> SparseMerkleMap:
    m = SparseMerkleMap()
    for i, k in enumerate(keys(n)):
        m.set(k, b"value-%d" % i)
    return m


def test_an_empty_map_has_the_default_root_for_the_full_depth():
    m = SparseMerkleMap()
    assert m.root() == smt._DEFAULTS[DEPTH]
    assert len(m) == 0


def test_default_hashes_are_built_up_from_the_empty_leaf():
    assert len(smt._DEFAULTS) == DEPTH + 1
    assert smt._DEFAULTS[0] == EMPTY_LEAF
    for level in range(DEPTH):
        assert smt._DEFAULTS[level + 1] == smt._node(smt._DEFAULTS[level], smt._DEFAULTS[level])
    assert len(set(smt._DEFAULTS)) == DEPTH + 1


def test_leaves_and_nodes_are_hashed_in_different_domains():
    assert smt._leaf(b"a", b"b") != smt._node(b"a", b"b")
    assert smt._leaf(b"", b"") != EMPTY_LEAF
    # a leaf whose preimage looks like two child hashes must still differ from the node over them
    key, value = b"k" * 28, b"v" * 32
    left = len(key).to_bytes(4, "big") + key
    assert len(left) == len(value) == 32
    assert smt._leaf(key, value) != smt._node(left, value)


def test_a_leaf_commits_to_where_the_key_ends_and_the_value_begins():
    assert smt._leaf(b"ab", b"c") != smt._leaf(b"a", b"bc")


def test_the_left_and_right_child_are_not_interchangeable():
    assert smt._node(b"l", b"r") != smt._node(b"r", b"l")


def test_the_root_does_not_depend_on_the_order_of_insertion(rng):
    items = [(k, b"v%d" % i) for i, k in enumerate(keys(40))]
    a, b = SparseMerkleMap(), SparseMerkleMap()
    for k, v in items:
        a.set(k, v)
    for i in rng.permutation(len(items)):
        b.set(*items[int(i)])
    assert a.root() == b.root()
    assert len(a) == len(b) == 40


def test_the_root_depends_on_every_value_and_every_key():
    base = filled(10)
    other = filled(10)
    assert base.root() == other.root()
    other.set(keys(10)[3], b"changed")
    assert other.root() != base.root()
    third = filled(10)
    third.set(sha256(b"one more"), b"x")
    assert third.root() != base.root()


def test_overwriting_a_key_replaces_its_value_and_restoring_it_restores_the_root():
    m = filled(8)
    key = keys(8)[2]
    original_root, original = m.root(), m.get(key)
    m.set(key, b"other")
    assert m.get(key) == b"other" and len(m) == 8 and m.root() != original_root
    m.set(key, original)
    assert m.root() == original_root


def test_get_returns_the_value_for_present_keys_and_none_for_absent_ones():
    m = filled(5)
    assert m.get(keys(5)[4]) == b"value-4"
    assert m.get(sha256(b"absent")) is None


def test_get_does_not_confuse_a_key_with_another_at_the_same_position(monkeypatch):
    monkeypatch.setattr(smt, "path_of", lambda key: 7)
    m = SparseMerkleMap()
    m.set(b"first", b"v")
    assert m.get(b"first") == b"v"
    assert m.get(b"second") is None


def test_two_keys_at_the_same_position_are_refused(monkeypatch):
    monkeypatch.setattr(smt, "path_of", lambda key: 5)
    m = SparseMerkleMap()
    m.set(b"a", b"1")
    with pytest.raises(ValueError, match="share a leaf position"):
        m.set(b"b", b"2")
    m.set(b"a", b"3")  # the same key may be rewritten
    assert m.get(b"a") == b"3"


def test_positions_are_depth_bits_of_a_domain_separated_hash():
    p = path_of(b"key")
    assert 0 <= p < 2**DEPTH
    assert p == int.from_bytes(sha256(b"rfd031.smt.path" + b"key")[: DEPTH // 8], "big")
    assert path_of(b"key") != path_of(b"key2")


def test_a_proof_of_presence_verifies_for_every_key(rng):
    m = filled(60)
    for i, k in enumerate(keys(60)):
        assert verify_proof(m.root(), k, b"value-%d" % i, m.prove(k))


def test_a_proof_of_absence_verifies_and_the_same_proof_cannot_claim_a_value():
    m = filled(60)
    for i in range(20):
        k = sha256(b"absent-%d" % i)
        proof = m.prove(k)
        assert verify_proof(m.root(), k, None, proof)
        assert not verify_proof(m.root(), k, b"value-0", proof)


def test_a_present_key_cannot_be_claimed_absent_nor_given_another_value():
    m = filled(30)
    k = keys(30)[7]
    proof = m.prove(k)
    assert not verify_proof(m.root(), k, None, proof)
    assert not verify_proof(m.root(), k, b"value-8", proof)
    assert not verify_proof(m.root(), k, b"", proof)


def test_a_proof_for_one_key_does_not_prove_another():
    m = filled(30)
    a, b = keys(30)[1], keys(30)[2]
    assert not verify_proof(m.root(), b, b"value-2", m.prove(a))


def test_a_proof_does_not_verify_against_another_root():
    m = filled(30)
    k = keys(30)[0]
    proof = m.prove(k)
    assert verify_proof(m.root(), k, b"value-0", proof)
    m.set(sha256(b"new"), b"v")
    assert not verify_proof(m.root(), k, b"value-0", proof)
    assert verify_proof(m.root(), k, b"value-0", m.prove(k))


def test_a_proof_with_a_changed_sibling_fails():
    m = filled(30)
    k = keys(30)[0]
    proof = m.prove(k)
    for i in range(len(proof.siblings)):
        siblings = list(proof.siblings)
        siblings[i] = sha256(siblings[i])
        assert not verify_proof(m.root(), k, b"value-0", MapProof(proof.bitmap, tuple(siblings)))


def test_a_proof_whose_bitmap_and_siblings_disagree_is_refused():
    m = filled(30)
    k = keys(30)[0]
    proof = m.prove(k)
    assert not verify_proof(m.root(), k, b"value-0", MapProof(proof.bitmap, proof.siblings[:-1]))
    assert not verify_proof(m.root(), k, b"value-0", MapProof(proof.bitmap, proof.siblings + (sha256(b"x"),)))
    assert not verify_proof(m.root(), k, b"value-0", MapProof(0, proof.siblings))
    assert not verify_proof(m.root(), k, b"value-0", MapProof(proof.bitmap ^ 1, proof.siblings))


def test_a_bitmap_with_bits_beyond_the_depth_is_refused():
    m = filled(30)
    k = keys(30)[0]
    proof = m.prove(k)
    extra = MapProof(proof.bitmap | (1 << DEPTH), proof.siblings + (sha256(b"x"),))
    assert not verify_proof(m.root(), k, b"value-0", extra)


def test_a_map_with_one_key_needs_no_siblings():
    m = SparseMerkleMap()
    m.set(b"only", b"v")
    proof = m.prove(b"only")
    assert proof.siblings == () and proof.bitmap == 0
    assert proof.size_bytes() == DEPTH // 8
    assert verify_proof(m.root(), b"only", b"v", proof)


def test_an_empty_map_proves_absence_with_an_empty_proof():
    m = SparseMerkleMap()
    proof = m.prove(b"anything")
    assert proof == MapProof(0, ())
    assert verify_proof(m.root(), b"anything", None, proof)


def test_proof_size_is_the_bitmap_plus_32_bytes_per_sibling():
    m = filled(20)
    proof = m.prove(keys(20)[0])
    assert proof.size_bytes() == 16 + 32 * len(proof.siblings)
    assert len(proof.siblings) == bin(proof.bitmap).count("1")


@pytest.mark.parametrize("n", [16, 256, 2048])
def test_proofs_carry_about_log2_n_siblings(n: int):
    m = filled(n)
    counts = [len(m.prove(k).siblings) for k in keys(n)[:40]]
    mean = sum(counts) / len(counts)
    assert math.log2(n) - 1.5 <= mean <= math.log2(n) + 2.0


def test_every_sibling_in_a_proof_is_a_stored_node_and_default_siblings_are_omitted():
    m = filled(12)
    k = keys(12)[0]
    proof = m.prove(k)
    p = path_of(k)
    expected_bits = [lvl for lvl in range(DEPTH) if (lvl, (p >> lvl) ^ 1) in m._nodes]
    assert [lvl for lvl in range(DEPTH) if (proof.bitmap >> lvl) & 1] == expected_bits
    assert list(proof.siblings) == [m._nodes[(lvl, (p >> lvl) ^ 1)] for lvl in expected_bits]
