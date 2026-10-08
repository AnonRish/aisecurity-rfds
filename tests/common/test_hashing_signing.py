from aisecurity_rfds.common.hashing import ct_equal, domain_hash, sha256, u32, u64
from aisecurity_rfds.common.rng import derive_seed
from aisecurity_rfds.common.signing import KeyPair, verify


def test_domain_hash_is_injective_over_part_boundaries_and_domains():
    assert domain_hash("d", b"ab", b"c") != domain_hash("d", b"a", b"bc")
    assert domain_hash("d", b"abc") != domain_hash("d", b"abc", b"")
    assert domain_hash("d1", b"x") != domain_hash("d2", b"x")
    assert domain_hash("d", b"x") == domain_hash("d", b"x")
    assert len(domain_hash("d")) == 32


def test_misc_helpers():
    assert ct_equal(sha256(b"a"), sha256(b"a")) and not ct_equal(sha256(b"a"), sha256(b"b"))
    assert u64(1) == b"\x00" * 7 + b"\x01" and u32(258) == b"\x00\x00\x01\x02"
    assert derive_seed("x", 1) == derive_seed("x", 1)


def test_signing_roundtrip_and_rejection():
    kp = KeyPair.from_seed(bytes(range(32)))
    sig = kp.sign(b"message")
    assert verify(kp.public_bytes, b"message", sig)
    assert not verify(kp.public_bytes, b"messagE", sig)
    assert not verify(kp.public_bytes, b"message", sig[:-1] + bytes([sig[-1] ^ 1]))
    assert not verify(b"short", b"message", sig)  # malformed key is rejected, not raised
    other = KeyPair.from_seed(bytes(range(1, 33)))
    assert not verify(other.public_bytes, b"message", sig)


def test_deterministic_keys_and_validation():
    import pytest

    assert KeyPair.from_seed(bytes(32)).public_bytes == KeyPair.from_seed(bytes(32)).public_bytes
    assert KeyPair.generate().public_bytes != KeyPair.generate().public_bytes
    with pytest.raises(ValueError):
        KeyPair.from_seed(b"short")
