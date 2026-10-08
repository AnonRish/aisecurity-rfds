import pytest

from aisecurity_rfds.common.channel import (
    AeadReceiver,
    AeadSender,
    DecryptionError,
    NonceExhaustedError,
    ReplayError,
    derive_session,
    x25519_exchange,
    x25519_generate,
)


def make_pair(context=b"ctx"):
    a_priv, a_pub = x25519_generate()
    b_priv, b_pub = x25519_generate()
    sa = x25519_exchange(a_priv, b_pub)
    sb = x25519_exchange(b_priv, a_pub)
    assert sa == sb
    return derive_session(sa, context, initiator=True), derive_session(sb, context, initiator=False)


def test_bidirectional_roundtrip():
    (a_send, a_recv), (b_send, b_recv) = make_pair()
    assert b_recv.open(a_send.seal(b"hello", b"aad"), b"aad") == b"hello"
    assert a_recv.open(b_send.seal(b"world"), b"") == b"world"


def test_replay_and_reorder_rejected():
    (a_send, _), (_, b_recv) = make_pair()
    m0, m1, m2 = (a_send.seal(f"m{i}".encode()) for i in range(3))
    assert b_recv.open(m0) == b"m0"
    assert b_recv.open(m2) == b"m2"  # skipping ahead is allowed (packet loss)
    with pytest.raises(ReplayError):
        b_recv.open(m2)  # replay
    with pytest.raises(ReplayError):
        b_recv.open(m1)  # stale / reordered


def test_tampering_detected():
    (a_send, _), (_, b_recv) = make_pair()
    blob = bytearray(a_send.seal(b"payload", b"aad"))
    flipped = bytes(blob[:-1] + bytes([blob[-1] ^ 1]))
    with pytest.raises(DecryptionError):
        b_recv.open(flipped, b"aad")
    with pytest.raises(DecryptionError):
        b_recv.open(bytes(blob), b"other-aad")
    with pytest.raises(DecryptionError):
        b_recv.open(bytes(blob[:10]), b"aad")
    # forged counter bytes are authenticated too (they are part of the nonce)
    forged = bytes(blob)
    forged = (999).to_bytes(8, "big") + forged[8:]
    with pytest.raises(DecryptionError):
        b_recv.open(forged, b"aad")
    # a failed attempt must not have consumed the replay window
    assert b_recv.open(bytes(blob), b"aad") == b"payload"


def test_nonces_never_repeat_and_budget_is_enforced():
    sender = AeadSender(bytes(32), max_messages=1000)
    nonces = set()
    for _ in range(1000):
        nonces.add(sender.seal(b"x")[:8])
    assert len(nonces) == 1000
    with pytest.raises(NonceExhaustedError):
        sender.seal(b"x")
    assert sender.messages_sent == 1000


def test_directions_use_different_keys():
    (a_send, a_recv), _ = make_pair()
    with pytest.raises(DecryptionError):
        a_recv.open(a_send.seal(b"reflection attack"))  # cannot reflect own message back at self


def test_context_binding_and_key_validation():
    (a_send, _), _ = make_pair(b"ctx-1")
    a_priv, a_pub = x25519_generate()
    b_priv, b_pub = x25519_generate()
    secret = x25519_exchange(a_priv, b_pub)
    _, recv_wrong_ctx = derive_session(x25519_exchange(b_priv, a_pub), b"ctx-2", initiator=False)
    send, _ = derive_session(secret, b"ctx-1", initiator=True)
    with pytest.raises(DecryptionError):
        recv_wrong_ctx.open(send.seal(b"x"))
    with pytest.raises(ValueError):
        AeadSender(b"short")
    with pytest.raises(ValueError):
        AeadReceiver(b"short")
