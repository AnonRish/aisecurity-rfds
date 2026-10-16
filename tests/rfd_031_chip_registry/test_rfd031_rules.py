"""The rules of ``apply_request``: who may change a record, when, and into what."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aisecurity_rfds.common.hashing import u64
from aisecurity_rfds.rfd_031_chip_registry import identity as I
from aisecurity_rfds.rfd_031_chip_registry import registry as R
from aisecurity_rfds.rfd_031_chip_registry.scenario import EMBARGOED, World

WINDOW = R.Policy(frozenset()).max_transfer_window


def code(action) -> str:
    """The rejection code of ``action``; fails the test if the rules accept."""
    with pytest.raises(R.Rejected) as info:
        action()
    return info.value.code


def enrol_request(chip, **over):
    fields = {"public_key": chip.public_key, "jurisdiction": "CA"}
    fields.update(over)
    return R.make_request(R.ENROLL, chip.chip_id, 0, **fields)


def fresh_chip() -> I.DeviceKeyChip:
    return I.DeviceKeyChip(b"rules-chip")


# ---- requests ------------------------------------------------------------------------------------------------------


def test_make_request_sorts_fields_and_encodes_values():
    r = R.make_request("k", b"id", 3, zeta=b"raw", alpha="text", mid=7)
    assert r.fields == (("alpha", b"text"), ("mid", u64(7)), ("zeta", b"raw"))
    assert r.get("mid") == u64(7) and r.get("alpha") == b"text"
    assert R.make_request("k", b"id", 3, b=1, a=2) == R.make_request("k", b"id", 3, a=2, b=1)


def test_a_missing_field_is_rejected_unless_a_default_is_given():
    r = R.make_request("k", b"id", 1)
    assert r.get("x", b"d") == b"d" and r.get("x", b"") == b""
    assert code(lambda: r.get("x")) == "bad-field"


def test_the_digest_covers_every_part_of_the_request():
    base = R.make_request("kind", b"id", 1, a="x", b="y")
    variants = [
        R.make_request("kind2", b"id", 1, a="x", b="y"),
        R.make_request("kind", b"id2", 1, a="x", b="y"),
        R.make_request("kind", b"id", 2, a="x", b="y"),
        R.make_request("kind", b"id", 1, a="z", b="y"),
        R.make_request("kind", b"id", 1, a="x"),
        R.make_request("kind", b"id", 1, a="x", b="y", c="w"),
        R.make_request("kind", b"id", 1, a="y", b="x"),
        R.make_request("kind", b"id", 1, **{"a": "x", "b2": "y"}),
    ]
    digests = {base.digest()} | {v.digest() for v in variants}
    assert len(digests) == len(variants) + 1


def test_the_digest_does_not_confuse_where_one_field_ends_and_the_next_begins():
    assert R.make_request("k", b"i", 1, a=b"bc").digest() != R.make_request("k", b"i", 1, ab=b"c").digest()
    assert R.make_request("k", b"i", 1, a=b"", b=b"x").digest() != R.make_request("k", b"i", 1, a=b"x", b=b"").digest()


def test_a_signature_counts_only_for_the_key_it_was_made_with():
    alice, bob = I.party_key("alice"), I.party_key("bob")
    request = R.make_request("k", b"id", 1)
    signed = R.sign_request(request, alice)
    assert signed.signed_by(alice.public_bytes) and not signed.signed_by(bob.public_bytes)
    relabelled = R.SignedRequest(request, ((bob.public_bytes, signed.signatures[0][1]),))
    assert not relabelled.signed_by(bob.public_bytes) and not relabelled.signed_by(alice.public_bytes)
    assert R.sign_request(request).signed_by(alice.public_bytes) is False


def test_a_signature_does_not_carry_over_to_an_edited_request():
    alice = I.party_key("alice")
    signed = R.sign_request(R.make_request("k", b"id", 1, a="x"), alice)
    edited = R.SignedRequest(R.make_request("k", b"id", 1, a="y"), signed.signatures)
    assert not edited.signed_by(alice.public_bytes)


def test_several_signatures_may_accompany_one_request():
    alice, bob = I.party_key("alice"), I.party_key("bob")
    signed = R.sign_request(R.make_request("k", b"id", 1), alice, bob)
    assert signed.signed_by(alice.public_bytes) and signed.signed_by(bob.public_bytes)
    assert signed.encode() != R.sign_request(R.make_request("k", b"id", 1), alice).encode()


def test_a_records_encoding_changes_with_every_field():
    w = World.new()
    base = w.record(w.enroll("c"))
    changes = dict(
        chip_id=b"x" * 32,
        public_key=b"y" * 32,
        manufacturer=b"m" * 32,
        owner=b"o" * 32,
        status=R.IN_TRANSIT,
        jurisdiction="GB",
        location_commitment=b"l" * 32,
        version=base.version + 1,
        pending_buyer=b"p" * 32,
        pending_until=5,
        retired_reason="lost",
    )
    encodings = {base.encode()} | {replace(base, **{k: v}).encode() for k, v in changes.items()}
    assert len(encodings) == len(changes) + 1


def test_a_log_entrys_encoding_changes_with_its_index_time_request_and_record():
    w = World.new()
    entry = w.submit(R.sign_request(enrol_request(fresh_chip()), w.fab))
    variants = [
        replace(entry, index=entry.index + 1),
        replace(entry, timestamp=entry.timestamp + 1),
        replace(entry, record_after=replace(entry.record_after, jurisdiction="GB")),
        replace(entry, signed=R.sign_request(enrol_request(fresh_chip(), jurisdiction="US"), w.fab)),
    ]
    assert len({entry.encode()} | {v.encode() for v in variants}) == 5


# ---- enrolment -----------------------------------------------------------------------------------------------------


def test_enrolment_creates_an_active_record_owned_by_the_manufacturer():
    w = World.new()
    chip = w.enroll("a", jurisdiction="CA")
    r = w.record(chip)
    assert r == R.Record(
        chip_id=chip.chip_id,
        public_key=chip.public_key,
        manufacturer=w.fab.public_bytes,
        owner=w.fab.public_bytes,
        status=R.ACTIVE,
        jurisdiction="CA",
        location_commitment=b"",
        version=1,
    )


def test_enrolment_may_carry_a_location_commitment():
    w = World.new()
    chip = w.enroll("a", location=b"\x07" * 32)
    assert w.record(chip).location_commitment == b"\x07" * 32


def test_any_approved_manufacturer_may_enrol_and_is_recorded():
    w = World.new()
    other = I.party_key("fab-2")
    policy = R.Policy(frozenset({w.fab.public_bytes, other.public_bytes}))
    chip = fresh_chip()
    record = R.apply_request(None, R.sign_request(enrol_request(chip), other), 100, policy)
    assert record.manufacturer == other.public_bytes == record.owner
    both = R.apply_request(None, R.sign_request(enrol_request(chip), w.mallory, w.fab), 100, policy)
    assert both.manufacturer == w.fab.public_bytes


def test_enrolment_needs_a_valid_approved_signature():
    w = World.new()
    chip = fresh_chip()
    assert code(lambda: R.apply_request(None, R.sign_request(enrol_request(chip), w.mallory), 1, w.policy)) == "unauthorised"
    assert code(lambda: R.apply_request(None, R.sign_request(enrol_request(chip)), 1, w.policy)) == "unauthorised"
    forged = R.SignedRequest(enrol_request(chip), ((w.fab.public_bytes, w.mallory.sign(enrol_request(chip).digest())),))
    assert code(lambda: R.apply_request(None, forged, 1, w.policy)) == "unauthorised"


def test_the_chip_id_must_be_the_hash_of_the_chips_public_key():
    w = World.new()
    chip = fresh_chip()
    wrong = R.make_request(R.ENROLL, b"\x11" * 32, 0, public_key=chip.public_key, jurisdiction="CA")
    assert code(lambda: R.apply_request(None, R.sign_request(wrong, w.fab), 1, w.policy)) == "bad-chip-id"
    short = chip.public_key[:31]
    shortened = R.make_request(R.ENROLL, I.chip_id_of(short), 0, public_key=short, jurisdiction="CA")
    assert code(lambda: R.apply_request(None, R.sign_request(shortened, w.fab), 1, w.policy)) == "bad-chip-id"
    missing = R.make_request(R.ENROLL, chip.chip_id, 0, jurisdiction="CA")
    assert code(lambda: R.apply_request(None, R.sign_request(missing, w.fab), 1, w.policy)) == "bad-field"


def test_a_chip_cannot_be_enrolled_twice_or_from_a_nonzero_version():
    w = World.new()
    chip = w.enroll("a")
    again = R.sign_request(enrol_request(chip), w.fab)
    assert code(lambda: R.apply_request(w.record(chip), again, 1, w.policy)) == "duplicate"
    fresh = fresh_chip()
    versioned = R.make_request(R.ENROLL, fresh.chip_id, 1, public_key=fresh.public_key, jurisdiction="CA")
    assert code(lambda: R.apply_request(None, R.sign_request(versioned, w.fab), 1, w.policy)) == "stale-version"


def test_the_jurisdiction_must_be_present_short_and_not_embargoed():
    w = World.new()
    chip = fresh_chip()

    def enrol(**over):
        return R.apply_request(None, R.sign_request(enrol_request(chip, **over), w.fab), 1, w.policy)

    assert enrol(jurisdiction="x" * 64).jurisdiction == "x" * 64
    assert enrol(jurisdiction="x").jurisdiction == "x"
    assert code(lambda: enrol(jurisdiction="x" * 65)) == "bad-field"
    assert code(lambda: enrol(jurisdiction="")) == "bad-field"
    assert code(lambda: enrol(jurisdiction=EMBARGOED)) == "embargoed"
    assert code(lambda: enrol(jurisdiction=b"\xff\xfe")) == "bad-field"
    no_jurisdiction = R.make_request(R.ENROLL, chip.chip_id, 0, public_key=chip.public_key)
    assert code(lambda: R.apply_request(None, R.sign_request(no_jurisdiction, w.fab), 1, w.policy)) == "bad-field"


def test_a_location_commitment_is_empty_or_32_bytes():
    w = World.new()
    chip = fresh_chip()

    def enrol(**over):
        return R.apply_request(None, R.sign_request(enrol_request(chip, **over), w.fab), 1, w.policy)

    assert enrol(location=b"a" * 32).location_commitment == b"a" * 32
    assert enrol(location=b"").location_commitment == b""
    for bad in (b"a" * 31, b"a" * 33, b"a"):
        assert code(lambda bad=bad: enrol(location=bad)) == "bad-field"


def test_requests_that_name_a_field_twice_or_an_unknown_kind_are_refused():
    w = World.new()
    chip = w.enroll("a")
    twice = R.Request(R.RELOCATE, chip.chip_id, 1, (("jurisdiction", b"US"), ("jurisdiction", b"GB")))
    assert code(lambda: R.apply_request(w.record(chip), R.sign_request(twice, w.fab), 1, w.policy)) == "bad-field"
    unknown = R.make_request("teleport", chip.chip_id, 1)
    assert code(lambda: R.apply_request(w.record(chip), R.sign_request(unknown, w.fab), 1, w.policy)) == "unknown-kind"
    assert code(lambda: R.apply_request(None, R.sign_request(unknown, w.fab), 1, w.policy)) == "unknown-kind"


def test_every_kind_but_enrolment_needs_an_existing_chip():
    w = World.new()
    for kind in (R.START_TRANSFER, R.COMPLETE_TRANSFER, R.CANCEL_TRANSFER, R.RELOCATE, R.RETIRE):
        signed = R.sign_request(R.make_request(kind, b"\x01" * 32, 1), w.fab)
        assert code(lambda signed=signed: R.apply_request(None, signed, 1, w.policy)) == "unknown-chip"
    other = w.enroll("a")
    mismatched = R.sign_request(R.make_request(R.RELOCATE, b"\x01" * 32, 1, jurisdiction="US"), w.fab)
    assert code(lambda: R.apply_request(w.record(other), mismatched, 1, w.policy)) == "unknown-chip"


# ---- versions ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("delta", [-1, 1, 5])
@pytest.mark.parametrize("kind", [R.START_TRANSFER, R.RELOCATE, R.RETIRE])
def test_a_request_for_another_version_of_an_active_chip_is_stale(kind, delta):
    w = World.new()
    chip = w.enroll("a")
    version = w.record(chip).version
    signed = R.sign_request(R.make_request(kind, chip.chip_id, version + delta, buyer=w.alice.public_bytes, expires=9, jurisdiction="US", reason="lost"), w.fab)
    assert code(lambda: R.apply_request(w.record(chip), signed, 1, w.policy)) == "stale-version"


@pytest.mark.parametrize("delta", [-1, 1])
@pytest.mark.parametrize("kind", [R.COMPLETE_TRANSFER, R.CANCEL_TRANSFER])
def test_a_request_for_another_version_of_a_chip_in_transit_is_stale(kind, delta):
    w = World.new()
    chip = w.enroll("a")
    w.offer(chip, w.fab, w.alice)
    version = w.record(chip).version
    signed = R.sign_request(R.make_request(kind, chip.chip_id, version + delta), w.alice)
    assert code(lambda: R.apply_request(w.record(chip), signed, w.clock, w.policy)) == "stale-version"


def test_staleness_is_reported_before_authorisation():
    w = World.new()
    chip = w.enroll("a")
    signed = R.sign_request(R.make_request(R.RELOCATE, chip.chip_id, 99, jurisdiction="US"), w.mallory)
    assert code(lambda: R.apply_request(w.record(chip), signed, 1, w.policy)) == "stale-version"


# ---- start-transfer ------------------------------------------------------------------------------------------------


def test_an_owner_can_put_a_chip_up_for_transfer():
    w = World.new()
    chip = w.enroll("a")
    before = w.record(chip)
    deadline = w.clock + 500  # the scenario helper counts the window from the clock before it submits
    entry = w.offer(chip, w.fab, w.alice, window=500)
    after = entry.record_after
    assert after == replace(
        before, status=R.IN_TRANSIT, pending_buyer=w.alice.public_bytes, pending_until=deadline, version=2
    )
    assert after.owner == w.fab.public_bytes  # ownership does not move until completion


def test_only_the_owner_can_start_a_transfer_and_that_is_checked_before_the_state():
    w = World.new()
    chip = w.enroll("a")
    assert code(lambda: w.offer(chip, w.mallory, w.bob)) == "unauthorised"
    assert code(lambda: w.offer(chip, w.alice, w.bob)) == "unauthorised"
    w.offer(chip, w.fab, w.alice)
    assert code(lambda: w.offer(chip, w.mallory, w.bob)) == "unauthorised"  # not "bad-state"
    assert code(lambda: w.offer(chip, w.fab, w.bob)) == "bad-state"


def test_the_buyer_must_be_another_partys_public_key():
    w = World.new()
    chip = w.enroll("a")
    version = w.record(chip).version

    def start(buyer):
        request = R.make_request(R.START_TRANSFER, chip.chip_id, version, buyer=buyer, expires=100 + 3600)
        return R.apply_request(w.record(chip), R.sign_request(request, w.fab), 100, w.policy)

    assert start(w.alice.public_bytes).pending_buyer == w.alice.public_bytes
    assert code(lambda: start(w.fab.public_bytes)) == "bad-field"
    assert code(lambda: start(w.alice.public_bytes[:31])) == "bad-field"
    assert code(lambda: start(b"")) == "bad-field"
    missing = R.make_request(R.START_TRANSFER, chip.chip_id, version, expires=3700)
    assert code(lambda: R.apply_request(w.record(chip), R.sign_request(missing, w.fab), 100, w.policy)) == "bad-field"


def test_an_offer_must_expire_in_the_future_and_within_the_policy_window():
    w = World.new()
    chip = w.enroll("a")
    version, now = w.record(chip).version, 1000

    def start(expires):
        request = R.make_request(R.START_TRANSFER, chip.chip_id, version, buyer=w.alice.public_bytes, expires=expires)
        return R.apply_request(w.record(chip), R.sign_request(request, w.fab), now, w.policy)

    assert start(now + 1).pending_until == now + 1
    assert start(now + WINDOW).pending_until == now + WINDOW
    for bad in (now, now - 1, 0, now + WINDOW + 1):
        assert code(lambda bad=bad: start(bad)) == "bad-window"
    no_expiry = R.make_request(R.START_TRANSFER, chip.chip_id, version, buyer=w.alice.public_bytes)
    assert code(lambda: R.apply_request(w.record(chip), R.sign_request(no_expiry, w.fab), now, w.policy)) == "bad-field"


def test_the_policy_window_is_a_parameter():
    w = World.new()
    chip = w.enroll("a")
    short = replace(w.policy, max_transfer_window=60)
    request = R.make_request(R.START_TRANSFER, chip.chip_id, 1, buyer=w.alice.public_bytes, expires=161)
    assert code(lambda: R.apply_request(w.record(chip), R.sign_request(request, w.fab), 100, short)) == "bad-window"
    assert R.apply_request(w.record(chip), R.sign_request(request, w.fab), 101, short).pending_until == 161


# ---- complete-transfer ---------------------------------------------------------------------------------------------


def test_completion_hands_the_chip_to_the_buyer_and_clears_what_belonged_to_the_seller():
    w = World.new()
    chip = w.enroll("a", location=b"\x05" * 32)
    w.offer(chip, w.fab, w.alice)
    pending = w.record(chip)
    entry = w.submit(w.completion(chip, w.alice, jurisdiction="US"))
    after = entry.record_after
    assert after == replace(
        pending,
        owner=w.alice.public_bytes,
        status=R.ACTIVE,
        jurisdiction="US",
        location_commitment=b"",
        pending_buyer=b"",
        pending_until=0,
        version=pending.version + 1,
    )
    assert after.manufacturer == w.fab.public_bytes and after.public_key == chip.public_key


def test_only_the_named_buyer_can_complete_and_not_the_seller():
    w = World.new()
    chip = w.enroll("a")
    w.offer(chip, w.fab, w.alice)
    for stranger in (w.mallory, w.bob, w.fab):
        assert code(lambda stranger=stranger: w.submit(w.completion(chip, stranger))) == "unauthorised"
    w.submit(w.completion(chip, w.alice))


def test_completion_needs_a_pending_transfer():
    w = World.new()
    chip = w.enroll("a")
    assert code(lambda: w.submit(w.completion(chip, w.alice))) == "bad-state"


def test_the_buyer_can_complete_up_to_the_expiry_and_not_after():
    w = World.new()
    chip = w.enroll("a")
    w.offer(chip, w.fab, w.alice, window=100)
    pending = w.record(chip)
    signed = w.completion(chip, w.alice)
    assert R.apply_request(pending, signed, pending.pending_until, w.policy).owner == w.alice.public_bytes
    assert code(lambda: R.apply_request(pending, signed, pending.pending_until + 1, w.policy)) == "expired"


def test_completion_into_an_embargoed_or_blank_jurisdiction_is_refused():
    w = World.new()
    chip = w.enroll("a")
    w.offer(chip, w.fab, w.alice)
    assert code(lambda: w.submit(w.completion(chip, w.alice, jurisdiction=EMBARGOED))) == "embargoed"
    assert code(lambda: w.submit(w.completion(chip, w.alice, jurisdiction=""))) == "bad-field"
    assert w.record(chip).status == R.IN_TRANSIT


def test_completion_needs_the_chip_to_sign_this_transfers_challenge():
    w = World.new()
    chip = w.enroll("a")
    w.offer(chip, w.fab, w.alice)
    pending = w.record(chip)
    good = chip.attest(R.transfer_challenge(pending))

    def complete(counter=None, signature=None, **drop):
        fields = {"jurisdiction": "US", "att_counter": good.counter if counter is None else counter, "att_signature": signature or good.signature}
        for name in drop:
            fields.pop(name)
        return R.apply_request(pending, R.sign_request(R.make_request(R.COMPLETE_TRANSFER, chip.chip_id, pending.version, **fields), w.alice), w.clock, w.policy)

    assert complete().owner == w.alice.public_bytes
    assert code(lambda: complete(counter=good.counter + 1)) == "bad-attestation"
    assert code(lambda: complete(signature=bytes(64))) == "bad-attestation"
    assert code(lambda: complete(signature=chip.attest(b"another challenge").signature)) == "bad-attestation"
    assert code(lambda: complete(signature=I.DeviceKeyChip(b"other").attest(R.transfer_challenge(pending)).signature)) == "bad-attestation"
    assert code(lambda: complete(att_counter=None)) == "bad-field"
    assert code(lambda: complete(att_signature=None)) == "bad-field"
    assert code(lambda: complete(jurisdiction=None)) == "bad-field"


def test_the_challenge_binds_chip_version_both_parties_and_the_deadline():
    w = World.new()
    chip = w.enroll("a")
    w.offer(chip, w.fab, w.alice)
    r = w.record(chip)
    base = R.transfer_challenge(r)
    changed = [
        replace(r, chip_id=b"x" * 32),
        replace(r, version=r.version + 1),
        replace(r, owner=b"y" * 32),
        replace(r, pending_buyer=b"z" * 32),
        replace(r, pending_until=r.pending_until + 1),
    ]
    assert len({base} | {R.transfer_challenge(c) for c in changed}) == 6


# ---- cancel-transfer -----------------------------------------------------------------------------------------------


def test_the_buyer_can_decline_at_any_time_and_the_chip_returns_to_active_unchanged():
    w = World.new()
    chip = w.enroll("a", location=b"\x03" * 32)
    before = w.record(chip)
    w.offer(chip, w.fab, w.alice)
    entry = w.submit(R.sign_request(R.make_request(R.CANCEL_TRANSFER, chip.chip_id, w.record(chip).version), w.alice))
    assert entry.record_after == replace(before, version=before.version + 2)
    assert entry.record_after.location_commitment == b"\x03" * 32


def test_the_owner_can_withdraw_only_after_the_offer_has_lapsed():
    w = World.new()
    chip = w.enroll("a")
    w.offer(chip, w.fab, w.alice, window=100)
    pending = w.record(chip)
    signed = R.sign_request(R.make_request(R.CANCEL_TRANSFER, chip.chip_id, pending.version), w.fab)
    assert code(lambda: R.apply_request(pending, signed, pending.pending_until - 1, w.policy)) == "not-expired"
    assert code(lambda: R.apply_request(pending, signed, pending.pending_until, w.policy)) == "not-expired"
    after = R.apply_request(pending, signed, pending.pending_until + 1, w.policy)
    assert after.status == R.ACTIVE and after.pending_buyer == b"" and after.pending_until == 0
    assert after.owner == w.fab.public_bytes and after.version == pending.version + 1


def test_a_stranger_cannot_cancel_and_a_cancel_needs_a_pending_transfer():
    w = World.new()
    chip = w.enroll("a")
    version = w.record(chip).version
    cancel = R.sign_request(R.make_request(R.CANCEL_TRANSFER, chip.chip_id, version), w.fab)
    assert code(lambda: R.apply_request(w.record(chip), cancel, w.clock, w.policy)) == "bad-state"
    w.offer(chip, w.fab, w.alice, window=100)
    stranger = R.sign_request(R.make_request(R.CANCEL_TRANSFER, chip.chip_id, w.record(chip).version), w.mallory)
    assert code(lambda: R.apply_request(w.record(chip), stranger, w.clock + 10_000, w.policy)) == "unauthorised"


# ---- relocate ------------------------------------------------------------------------------------------------------


def test_the_owner_can_change_the_jurisdiction_and_the_location_commitment():
    w = World.new()
    chip = w.enroll("a", location=b"\x01" * 32)
    entry = w.relocate(chip, w.fab, "US", location=b"\x02" * 32)
    assert (entry.record_after.jurisdiction, entry.record_after.location_commitment) == ("US", b"\x02" * 32)
    assert entry.record_after.version == 2 and entry.record_after.owner == w.fab.public_bytes
    cleared = w.relocate(chip, w.fab, "GB")
    assert cleared.record_after.location_commitment == b""


def test_relocation_is_the_owners_alone_and_only_while_active():
    w = World.new()
    chip = w.enroll("a")
    assert code(lambda: w.relocate(chip, w.mallory, "US")) == "unauthorised"
    assert code(lambda: w.relocate(chip, w.fab, EMBARGOED)) == "embargoed"
    assert code(lambda: w.relocate(chip, w.fab, "")) == "bad-field"
    assert code(lambda: w.relocate(chip, w.fab, "US", location=b"short")) == "bad-field"
    w.offer(chip, w.fab, w.alice)
    assert code(lambda: w.relocate(chip, w.fab, "US")) == "bad-state"


# ---- retire --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("reason", R.RETIRE_REASONS)
def test_retirement_is_final_and_records_why(reason):
    w = World.new()
    chip = w.enroll("a", location=b"\x04" * 32)
    entry = w.retire(chip, w.fab, reason, witnessed=True)
    after = entry.record_after
    assert (after.status, after.retired_reason, after.location_commitment, after.version) == (R.RETIRED, reason, b"", 2)
    assert after.owner == w.fab.public_bytes
    for kind in (R.START_TRANSFER, R.COMPLETE_TRANSFER, R.CANCEL_TRANSFER, R.RELOCATE, R.RETIRE):
        signed = R.sign_request(R.make_request(kind, chip.chip_id, 2, buyer=w.alice.public_bytes, expires=w.clock + 9, jurisdiction="US", reason="lost"), w.fab)
        assert code(lambda signed=signed: R.apply_request(after, signed, w.clock, w.policy)) == "retired"


def test_retirement_needs_the_owner_a_known_reason_and_no_pending_transfer():
    w = World.new()
    chip = w.enroll("a")
    assert code(lambda: w.retire(chip, w.mallory, "lost")) == "unauthorised"
    bad = R.sign_request(R.make_request(R.RETIRE, chip.chip_id, 1, reason="vanished"), w.fab)
    assert code(lambda: R.apply_request(w.record(chip), bad, 1, w.policy)) == "bad-field"
    missing = R.sign_request(R.make_request(R.RETIRE, chip.chip_id, 1), w.fab)
    assert code(lambda: R.apply_request(w.record(chip), missing, 1, w.policy)) == "bad-field"
    w.offer(chip, w.fab, w.alice)
    assert code(lambda: w.retire(chip, w.fab, "lost")) == "bad-state"


def test_destruction_needs_an_accredited_witness_when_the_policy_names_some():
    w = World.new()
    chip = w.enroll("a")
    assert code(lambda: w.retire(chip, w.fab, "destroyed")) == "unwitnessed"
    request = R.make_request(R.RETIRE, chip.chip_id, 1, reason="destroyed")
    assert code(lambda: R.apply_request(w.record(chip), R.sign_request(request, w.fab, w.mallory), 1, w.policy)) == "unwitnessed"
    assert R.apply_request(w.record(chip), R.sign_request(request, w.fab, w.recycler), 1, w.policy).status == R.RETIRED
    for reason in ("recycled", "repurposed", "lost"):
        other = w.enroll("other-" + reason)
        assert w.retire(other, w.fab, reason).record_after.status == R.RETIRED  # no witness needed


def test_destruction_needs_no_witness_when_the_policy_names_none():
    w = World.new(require_destruction_witness=False)
    chip = w.enroll("a")
    assert w.retire(chip, w.fab, "destroyed").record_after.retired_reason == "destroyed"


def test_a_retired_chip_cannot_be_enrolled_again():
    w = World.new()
    chip = w.enroll("a")
    w.retire(chip, w.fab, "lost")
    assert code(lambda: R.apply_request(w.record(chip), R.sign_request(enrol_request(chip), w.fab), 1, w.policy)) == "duplicate"


def test_policy_defaults():
    p = R.Policy(frozenset())
    assert p.max_transfer_window == 14 * 24 * 3600
    assert p.embargoed == frozenset() and p.destruction_witnesses == frozenset()
