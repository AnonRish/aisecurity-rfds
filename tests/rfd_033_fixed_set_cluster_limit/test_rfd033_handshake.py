"""Link establishment between pod members, and the attacks on it (RFD 033: spoofing, replay, stale membership)."""

from __future__ import annotations

import dataclasses

import pytest

from aisecurity_rfds.common.channel import DecryptionError, ReplayError
from aisecurity_rfds.common.signing import verify
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit import pod as podmod
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit.cluster import establish
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit.pod import (
    MAX_PENDING,
    Chip,
    Finish,
    HandshakeError,
    Hello1,
    LinkDownError,
    LinkMode,
    Manufacturer,
    PodManifest,
)


def pod(n: int = 4, *, interval: int = 10) -> tuple[Manufacturer, list[Chip], PodManifest]:
    mfr = Manufacturer(b"hs")
    chips = [mfr.make_chip(pod_cap=8, reverify_interval=interval) for _ in range(n)]
    return mfr, chips, mfr.form_pod(chips, label=1)


def run(a: Chip, b: Chip):
    h1 = a.begin()
    h2 = b.respond(h1)
    fin = a.complete(h2)
    b.finalize(fin)
    return h1, h2, fin


def transcript(h1, h2) -> bytes:
    return podmod._transcript(h1.pod_id, h1.device, h2.device, h1.eph, h2.eph, h1.nonce, h2.nonce)


def test_members_establish_a_high_link_and_exchange_authenticated_data_both_ways():
    _, (a, b, *_), _ = pod()
    run(a, b)
    assert a.link_mode(b.device_id) is LinkMode.HIGH and b.link_mode(a.device_id) is LinkMode.HIGH
    assert b.unseal(a.device_id, a.seal(b.device_id, b"gradient shard")) == b"gradient shard"
    assert a.unseal(b.device_id, b.seal(a.device_id, b"activations")) == b"activations"
    assert a.high_peers() == [b.device_id]


def test_a_chip_of_another_pod_cannot_claim_membership_of_this_one():
    mfr, chips, _ = pod(3)
    outsiders = [mfr.make_chip(pod_cap=8) for _ in range(3)]
    mfr.form_pod(outsiders, label=2)
    victim, outsider = chips[0], outsiders[0]
    _, pub = podmod.x25519_generate()
    claim = Hello1(victim.pod_id, outsider.device_id, pub, b"n" * 16)
    with pytest.raises(HandshakeError, match="not in this chip's manifest"):
        victim.respond(claim)
    assert not establish(outsider, victim) and not establish(victim, outsider)


def test_a_hello_that_names_a_different_pod_is_refused():
    mfr, chips, _ = pod(3)
    others = [mfr.make_chip(pod_cap=8) for _ in range(3)]
    mfr.form_pod(others, label=2)
    with pytest.raises(HandshakeError, match="different pod"):
        chips[0].respond(others[0].begin())


def test_reflecting_a_chips_own_hello_back_at_it_is_refused():
    _, (a, *_), _ = pod()
    with pytest.raises(HandshakeError, match="own identity"):
        a.respond(a.begin())


def test_changing_the_initiators_identity_in_flight_is_caught_by_the_initiator():
    _, (a, b, c, _), _ = pod()
    h1 = a.begin()
    swapped = dataclasses.replace(h1, device=c.device_id)  # the attacker relabels a's hello as c's
    h2 = b.respond(swapped)
    with pytest.raises(HandshakeError, match="bad responder signature"):
        a.complete(h2)
    assert a.link_mode(b.device_id) is LinkMode.LOW


def test_a_forged_responder_signature_is_refused():
    _, (a, b, *_), _ = pod()
    h2 = b.respond(a.begin())
    forged = dataclasses.replace(h2, signature=bytes(64))
    with pytest.raises(HandshakeError, match="bad responder signature"):
        a.complete(forged)


def test_replaying_a_finished_handshake_creates_no_session():
    _, (a, b, *_), _ = pod()
    h1, _, fin = run(a, b)
    with pytest.raises(HandshakeError):
        b.finalize(fin)  # same finish again
    b.respond(h1)  # the attacker replays the first message: the responder answers with a fresh nonce and key
    with pytest.raises(HandshakeError, match="no handshake in progress"):
        b.finalize(fin)  # but the recorded finish belongs to the old transcript


def test_the_responders_signature_cannot_be_presented_as_the_initiators():
    _, (a, b, *_), _ = pod()
    h1 = a.begin()
    h2 = b.respond(h1)
    with pytest.raises(HandshakeError, match="bad initiator signature"):
        b.finalize(Finish(transcript(h1, h2), h2.signature))
    assert b.link_mode(a.device_id) is LinkMode.LOW


def test_an_unknown_reply_is_refused():
    _, (a, b, c, _), _ = pod()
    h2 = b.respond(a.begin())
    with pytest.raises(HandshakeError, match="no handshake in progress"):
        c.complete(h2)


def test_tampered_or_replayed_traffic_on_a_live_link_is_rejected():
    _, (a, b, *_), _ = pod()
    run(a, b)
    blob = a.seal(b.device_id, b"payload")
    flipped = bytearray(blob)
    flipped[-1] ^= 1
    with pytest.raises(DecryptionError):
        b.unseal(a.device_id, bytes(flipped))
    assert b.unseal(a.device_id, blob) == b"payload"
    with pytest.raises(ReplayError):
        b.unseal(a.device_id, blob)


def test_every_handshake_derives_fresh_keys():
    _, (a, b, *_), _ = pod()
    run(a, b)
    old = a.seal(b.device_id, b"session one")
    run(a, b)
    with pytest.raises(DecryptionError):
        b.unseal(a.device_id, old)
    assert b.unseal(a.device_id, a.seal(b.device_id, b"session two")) == b"session two"


def test_links_expire_without_reverification_and_recover_after_a_new_handshake():
    _, (a, b, *_), _ = pod(interval=3)
    run(a, b)
    a.tick(2)
    assert a.link_mode(b.device_id) is LinkMode.HIGH
    a.tick(1)
    assert a.link_mode(b.device_id) is LinkMode.LOW
    with pytest.raises(LinkDownError):
        a.seal(b.device_id, b"late")
    assert establish(a, b)
    assert a.link_mode(b.device_id) is LinkMode.HIGH


def test_time_cannot_be_wound_back():
    _, (a, *_), _ = pod()
    with pytest.raises(ValueError):
        a.tick(-1)


def test_tamper_drops_every_link_and_blocks_new_ones():
    _, (a, b, c, _), _ = pod()
    run(a, b)
    run(a, c)
    a.tamper("lid opened")
    assert a.high_peers() == []
    assert not establish(a, b) and not establish(b, a)
    with pytest.raises(LinkDownError):
        a.seal(b.device_id, b"x")


def test_a_chip_outside_any_pod_can_neither_start_nor_answer():
    mfr, chips, _ = pod(3)
    loose = mfr.make_chip(pod_cap=8)
    with pytest.raises(HandshakeError, match="not a current pod member"):
        loose.begin()
    with pytest.raises(HandshakeError, match="not a current pod member"):
        loose.respond(chips[0].begin())


def test_limitation_a_removed_chip_keeps_linking_with_members_that_have_not_updated_yet():
    mfr, (a, b, c, x), m1 = pod(4)
    m2 = mfr.sign_manifest(m1.pod_id, 2, [a.device_id, b.device_id, c.device_id])  # x is removed
    a.provision(m2)
    assert not establish(a, x)  # an updated member refuses x ...
    assert establish(b, x) and b.link_mode(x.device_id) is LinkMode.HIGH  # ... a stale one still accepts it


def test_manifest_gossip_closes_the_stale_window_after_one_handshake_with_an_updated_member():
    mfr, (a, b, c, x), m1 = pod(4)
    m2 = mfr.sign_manifest(m1.pod_id, 2, [a.device_id, b.device_id, c.device_id])
    a.provision(m2)
    assert establish(b, x)
    assert establish(a, b)  # a's hello carries m2; b adopts it
    assert b.manifest == m2 and b.generation == 2
    assert b.link_mode(x.device_id) is LinkMode.LOW  # the existing link to the removed chip stops being high
    assert not establish(b, x) and not establish(x, b)
    assert c.manifest == m1  # c has not heard yet, which is the remaining exposure


def test_gossip_ignores_manifests_that_are_old_forged_for_another_pod_or_do_not_list_the_chip():
    mfr, (a, b, c, d), m1 = pod(4)
    other = Manufacturer(b"other")
    forged = other.sign_manifest(m1.pod_id, 9, list(m1.members))
    elsewhere = mfr.sign_manifest(mfr.pod_id_for(77), 9, list(m1.members))
    without_b = mfr.sign_manifest(m1.pod_id, 9, [a.device_id, c.device_id, d.device_id])
    for bad in (forged, elsewhere, without_b, m1):
        b._maybe_adopt(bad)
        assert b.manifest == m1 and b.generation == 1


def test_peers_with_different_generations_link_as_long_as_each_lists_the_other():
    mfr, (a, b, *_), m1 = pod(4)
    a.provision(mfr.sign_manifest(m1.pod_id, 2, list(m1.members)))
    assert a.generation == 2 and b.generation == 1
    assert establish(b, a)


def test_an_existing_link_stops_being_high_when_the_peer_leaves_the_manifest():
    mfr, (a, b, c, x), m1 = pod(4)
    run(a, x)
    assert a.link_mode(x.device_id) is LinkMode.HIGH
    a.provision(mfr.sign_manifest(m1.pod_id, 2, [a.device_id, b.device_id, c.device_id]))
    assert a.link_mode(x.device_id) is LinkMode.LOW
    with pytest.raises(LinkDownError):
        a.unseal(x.device_id, b"\x00" * 40)


def test_half_open_handshakes_are_bounded_and_the_oldest_is_dropped():
    _, (a, b, *_), _ = pod()
    first = a.begin()
    h2_first = b.respond(first)
    for _ in range(MAX_PENDING + 5):
        a.begin()
    assert len(a._pending_init) == MAX_PENDING
    with pytest.raises(HandshakeError, match="no handshake in progress"):
        a.complete(h2_first)
    for _ in range(MAX_PENDING + 5):
        b.respond(a.begin())
    assert len(b._pending_resp) == MAX_PENDING


def test_a_chip_whose_membership_was_invalidated_mid_handshake_cannot_finish_it():
    _, (a, b, *_), _ = pod()
    h2 = b.respond(a.begin())
    a.tamper("probe")
    with pytest.raises(HandshakeError):
        a.complete(h2)


def test_a_responder_signature_is_not_valid_as_an_initiator_signature_under_the_same_key():
    # Role separation is redundant with refusing a chip's own identity in this model; this pins the primitive property.
    _, (a, b, *_), _ = pod()
    h1 = a.begin()
    h2 = b.respond(h1)
    th = transcript(h1, h2)
    assert verify(b.device_id, podmod._role_digest("responder", th), h2.signature)
    assert not verify(b.device_id, podmod._role_digest("initiator", th), h2.signature)
    assert podmod._role_digest("initiator", th) != podmod._role_digest("responder", th)
