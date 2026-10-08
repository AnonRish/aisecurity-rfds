from dataclasses import replace

from aisecurity_rfds.common.hashchain import HashChainLog, verify_chain
from aisecurity_rfds.common.signing import KeyPair


def build(n=10, signer=None, log_id=b"dev-1"):
    log = HashChainLog(log_id=log_id, signer=signer)
    for i in range(n):
        log.append(f"event-{i}".encode())
    return log


def test_valid_chain_verifies_with_and_without_signatures():
    assert verify_chain(build().entries, log_id=b"dev-1")
    kp = KeyPair.from_seed(bytes(range(32)))
    signed = build(signer=kp)
    assert verify_chain(signed.entries, log_id=b"dev-1", public_key=kp.public_bytes, require_signatures=True)


def test_modified_payload_detected():
    entries = list(build().entries)
    entries[4] = replace(entries[4], payload=b"tampered")
    res = verify_chain(entries, log_id=b"dev-1")
    assert not res and "entry 4" in res.reason


def test_deleted_and_reordered_entries_detected():
    entries = list(build().entries)
    assert not verify_chain(entries[:3] + entries[4:], log_id=b"dev-1")
    swapped = entries[:]
    swapped[2], swapped[3] = swapped[3], swapped[2]
    assert not verify_chain(swapped, log_id=b"dev-1")


def test_recomputed_chain_after_edit_is_caught_only_by_an_external_anchor():
    original = build()
    anchor = original.checkpoint()
    # Attacker rewrites history consistently from entry 3 on (self-consistent chain, different content).
    forged = HashChainLog(log_id=b"dev-1")
    for i, e in enumerate(original.entries):
        forged.append(b"rewritten" if i == 3 else e.payload)
    assert verify_chain(forged.entries, log_id=b"dev-1")  # internally valid...
    res = verify_chain(forged.entries, log_id=b"dev-1", anchor=anchor)
    assert not res and "rewritten" in res.reason  # ...but contradicts the anchor


def test_rollback_and_truncation_detected_by_anchor():
    log = build(10)
    anchor = log.checkpoint()
    assert not verify_chain(log.entries[:6], log_id=b"dev-1", anchor=anchor)
    assert verify_chain(log.entries, log_id=b"dev-1", anchor=anchor)
    log.append(b"later")
    assert verify_chain(log.entries, log_id=b"dev-1", anchor=anchor)  # honest growth is fine


def test_signature_failures():
    kp = KeyPair.from_seed(bytes(range(32)))
    other = KeyPair.from_seed(bytes(range(1, 33)))
    signed = build(signer=kp)
    assert not verify_chain(signed.entries, log_id=b"dev-1", public_key=other.public_bytes)
    stripped = [replace(e, signature=None) for e in signed.entries]
    assert not verify_chain(stripped, log_id=b"dev-1", public_key=kp.public_bytes, require_signatures=True)
    assert not verify_chain(stripped, log_id=b"dev-1", require_signatures=True)


def test_signed_anchor_is_authenticated():
    kp = KeyPair.from_seed(bytes(range(32)))
    other = KeyPair.from_seed(bytes(range(1, 33)))
    log = build(signer=kp)
    anchor = log.checkpoint()
    assert verify_chain(log.entries, log_id=b"dev-1", public_key=kp.public_bytes, anchor=anchor)
    forged_anchor = replace(anchor, length=anchor.length - 1)  # signature no longer matches
    assert not verify_chain(log.entries, log_id=b"dev-1", public_key=kp.public_bytes, anchor=forged_anchor)
    assert not verify_chain(log.entries, log_id=b"dev-1", public_key=other.public_bytes, anchor=anchor)


def test_log_id_binds_entries_to_their_log():
    entries = build(5, log_id=b"dev-1").entries
    assert not verify_chain(entries, log_id=b"dev-2")


def test_empty_log_and_anchor():
    empty = HashChainLog(log_id=b"x")
    assert verify_chain(empty.entries, log_id=b"x", anchor=empty.checkpoint())
