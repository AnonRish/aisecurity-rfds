"""Provisioning, updates, rollback, release and tamper response of a pod-bound chip (RFD 033)."""

from __future__ import annotations

import dataclasses

import pytest

from aisecurity_rfds.common.nvm import WriteVerifyError
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit import pod as podmod
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit.cluster import establish
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit.pod import (
    GENERATION_FUSES,
    Chip,
    LinkMode,
    Manufacturer,
    PodManifest,
    ProvisionError,
    manifest_digest,
)


def make(n: int, *, cap: int = 8, bind: bool = True, seed: bytes = b"t") -> tuple[Manufacturer, list[Chip]]:
    mfr = Manufacturer(seed)
    return mfr, [mfr.make_chip(pod_cap=cap, bind_in_fuses=bind) for _ in range(n)]


def test_form_pod_installs_one_signed_manifest_in_every_member():
    mfr, chips = make(4)
    m = mfr.form_pod(chips, label=1)
    assert m.generation == 1 and len(m.members) == 4
    for c in chips:
        assert c.is_member and c.manifest == m
        assert c.pod_id == m.pod_id == c.bound_pod
        assert c.generation == 1


def test_manifest_round_trips_through_bytes_and_malformed_blobs_are_rejected():
    mfr, chips = make(3)
    m = mfr.form_pod(chips, label=1)
    assert PodManifest.from_bytes(m.to_bytes()) == m
    with pytest.raises(ValueError):
        PodManifest.from_bytes(m.to_bytes()[:-1])
    with pytest.raises(ValueError):
        PodManifest.from_bytes(m.to_bytes() + b"\x00")
    with pytest.raises(ValueError):
        PodManifest.from_bytes(b"short")


def test_a_manifest_signed_by_anyone_but_the_manufacturer_is_refused():
    mfr, chips = make(3)
    attacker = Manufacturer(b"attacker")
    forged = attacker.sign_manifest(mfr.pod_id_for(1), 1, [c.device_id for c in chips])
    for c in chips:
        with pytest.raises(ProvisionError, match="signature"):
            c.provision(forged)
        assert not c.is_member and c.generation == 0


def test_flipping_one_bit_of_the_member_list_invalidates_the_signature():
    mfr, chips = make(3)
    good = mfr.sign_manifest(mfr.pod_id_for(1), 1, [c.device_id for c in chips])
    key = bytearray(good.members[0])
    key[0] ^= 1
    bad = dataclasses.replace(good, members=tuple(sorted((bytes(key), *good.members[1:]))))
    with pytest.raises(ProvisionError):
        chips[1].provision(bad)
    chips[1].provision(good)  # the untouched manifest is accepted


def test_a_manifest_that_does_not_list_the_chip_is_refused():
    mfr, chips = make(4)
    m = mfr.sign_manifest(mfr.pod_id_for(1), 1, [c.device_id for c in chips[:3]])
    with pytest.raises(ProvisionError, match="not listed"):
        chips[3].provision(m)


def test_a_consumer_chip_with_cap_one_refuses_any_larger_pod_even_if_validly_signed():
    mfr, chips = make(2, cap=1)
    both = mfr.sign_manifest(mfr.pod_id_for(1), 1, [c.device_id for c in chips])
    for c in chips:
        with pytest.raises(ProvisionError, match="cap"):
            c.provision(both)
    solo = mfr.sign_manifest(mfr.pod_id_for(2), 1, [chips[0].device_id])
    chips[0].provision(solo)
    assert chips[0].is_member and chips[0].manifest.members == (chips[0].device_id,)


def test_unsorted_or_duplicate_member_lists_are_refused_even_with_a_valid_signature():
    mfr, chips = make(3)
    pod_id = mfr.pod_id_for(1)
    ids = [c.device_id for c in chips]
    for members in (tuple(sorted(ids, reverse=True)), (*sorted(ids), sorted(ids)[0])):
        digest = manifest_digest(pod_id, 1, members)
        m = PodManifest(pod_id, 1, members, mfr._key.sign(digest))
        with pytest.raises(ProvisionError, match="sorted"):
            chips[0].provision(m)


def test_a_chip_bound_to_one_pod_refuses_a_signed_manifest_for_another():
    mfr, chips = make(4)
    mfr.form_pod(chips[:2], label=1)
    careless = mfr.sign_manifest(mfr.pod_id_for(2), 5, [chips[0].device_id, chips[2].device_id, chips[3].device_id])
    with pytest.raises(ProvisionError, match="different pod"):
        chips[0].provision(careless)
    assert chips[0].pod_id == mfr.pod_id_for(1)


def test_generation_must_strictly_increase():
    mfr, chips = make(3)
    m1 = mfr.form_pod(chips, label=1)
    for c in chips:
        with pytest.raises(ProvisionError, match="stale"):
            c.provision(m1)  # equal generation: replay
    m2 = mfr.sign_manifest(m1.pod_id, 2, list(m1.members))
    chips[0].provision(m2)
    with pytest.raises(ProvisionError, match="stale"):
        chips[0].provision(m1)  # older generation after an update
    assert chips[0].generation == 2


def test_flash_rollback_cannot_bring_back_a_superseded_manifest():
    mfr, chips = make(3)
    m1 = mfr.form_pod(chips, label=1)
    before = chips[0].flash.snapshot()
    m2 = mfr.sign_manifest(m1.pod_id, 2, list(m1.members))
    chips[0].provision(m2)
    chips[0].flash.physical_restore(before)
    chips[0].boot()
    assert not chips[0].is_member  # fails closed: the old manifest is below the fuse floor
    assert chips[0].generation == 2
    chips[0].provision(mfr.sign_manifest(m1.pod_id, 3, list(m1.members)))
    assert chips[0].is_member


def test_a_blank_flash_image_leaves_a_chip_out_of_the_pod_and_old_manifests_stay_dead():
    mfr, chips = make(3)
    m1 = mfr.form_pod(chips, label=1)
    chips[0].provision(mfr.sign_manifest(m1.pod_id, 2, list(m1.members)))
    chips[0].flash.physical_restore({})
    chips[0].boot()
    assert not chips[0].is_member
    with pytest.raises(ProvisionError, match="stale"):
        chips[0].provision(m1)


def test_a_crash_between_the_flash_write_and_the_fuse_burn_is_repaired_at_boot():
    mfr, chips = make(2)
    m = mfr.sign_manifest(mfr.pod_id_for(1), 3, [c.device_id for c in chips])
    c = chips[0]
    c._manifest_rec.write(m.to_bytes())  # power lost right here, before any fuse was burned
    assert c.generation == 0 and c.bound_pod is None
    c.boot()
    assert c.is_member and c.generation == 3 and c.bound_pod == m.pod_id


def test_tamper_invalidates_membership_and_a_flash_restore_cannot_undo_it():
    mfr, chips = make(3)
    m1 = mfr.form_pod(chips, label=1)
    image = chips[0].flash.snapshot()
    chips[0].tamper("mesh broken")
    assert not chips[0].is_member and chips[0].generation == 2 and chips[0].tamper_log == ["mesh broken"]
    chips[0].flash.physical_restore(image)
    chips[0].boot()
    assert not chips[0].is_member
    with pytest.raises(ProvisionError, match="stale"):
        chips[0].provision(m1)
    chips[0].provision(mfr.sign_manifest(m1.pod_id, 3, list(m1.members)))  # re-provisioned at the facility
    assert chips[0].is_member


def test_limitation_starving_the_burn_supply_lets_a_flash_restore_undo_a_tamper_response():
    mfr, chips = make(3)
    mfr.form_pod(chips, label=1)
    image = chips[0].flash.snapshot()
    chips[0].fuses.hv_disrupted = True
    chips[0].tamper("mesh broken")
    assert chips[0].generation == 1 and "fuse burn did not take effect" in chips[0].tamper_log
    chips[0].flash.physical_restore(image)
    chips[0].boot()
    assert chips[0].is_member  # the documented weakness: invalidation is only as irreversible as the fuse burn


def test_tamper_with_an_exhausted_counter_kills_the_chip():
    mfr, chips = make(2)
    c = chips[0]
    c._burn_to(GENERATION_FUSES)
    assert c.generation == GENERATION_FUSES
    c.tamper("last resort")
    assert c.dead
    m = mfr.sign_manifest(mfr.pod_id_for(1), GENERATION_FUSES, [x.device_id for x in chips])
    with pytest.raises(ProvisionError):
        c.provision(m)
    c.boot()
    assert not c.is_member


def test_a_generation_beyond_the_fuse_counter_is_refused():
    mfr, chips = make(2)
    m = mfr.sign_manifest(mfr.pod_id_for(1), GENERATION_FUSES + 1, [c.device_id for c in chips])
    with pytest.raises(ProvisionError, match="exceeds"):
        chips[0].provision(m)
    assert chips[0].generation == 0


def test_a_failed_burn_is_reported_and_leaves_the_chip_out_of_the_pod():
    mfr, chips = make(2)
    m = mfr.sign_manifest(mfr.pod_id_for(1), 1, [c.device_id for c in chips])
    chips[0].fuses.hv_disrupted = True
    with pytest.raises(WriteVerifyError):
        chips[0].provision(m)
    assert not chips[0].is_member
    chips[0].boot()
    assert not chips[0].is_member  # boot cannot repair the fuses either, so it does not trust flash alone
    chips[0].fuses.hv_disrupted = False
    chips[0].boot()
    assert chips[0].is_member and chips[0].generation == 1
    assert chips[0].bound_pod == m.pod_id  # the half-finished binding was completed too


def test_corrupt_flash_fails_closed():
    mfr, chips = make(2)
    mfr.form_pod(chips, label=1)
    for slot in ("manifest.A", "manifest.B"):
        chips[0].flash.write(slot, b"\xff" * 300)
    chips[0].boot()
    assert not chips[0].is_member


def test_replacing_a_failed_member_isolates_it_once_the_new_manifest_spreads():
    mfr, chips = make(5)
    pod, spare = chips[:4], chips[4]
    m1 = mfr.form_pod(pod, label=1)
    failed = pod[3]
    m2 = mfr.replace_member(m1, failed.device_id, spare)
    assert m2.generation == 2 and failed.device_id not in m2.members and spare.device_id in m2.members
    for c in (*pod[:3], spare):
        c.provision(m2)
    assert all(establish(spare, c) for c in pod[:3])
    assert not establish(pod[0], failed)  # updated members no longer accept the retired chip
    assert not establish(failed, spare)
    with pytest.raises(ValueError):
        mfr.form_pod([failed], label=9)  # retired chips cannot be assigned again


def test_the_manufacturer_will_not_use_an_assigned_chip_as_a_replacement():
    mfr, chips = make(6)
    m1 = mfr.form_pod(chips[:3], label=1)
    mfr.form_pod(chips[3:5], label=2)
    with pytest.raises(ValueError):
        mfr.replace_member(m1, chips[2].device_id, chips[3])


def test_release_lets_a_flash_bound_chip_join_a_new_pod_but_not_return_to_the_old_one():
    mfr, chips = make(4, bind=False)
    m1 = mfr.form_pod(chips[:2], label=1)
    c = chips[0]
    c.release(mfr.release_chip(c))
    assert not c.is_member and c.bound_pod is None
    m2 = mfr.form_pod([c, chips[2]], label=2)
    assert c.is_member and c.pod_id == m2.pod_id and m2.generation > m1.generation
    with pytest.raises(ProvisionError):
        c.provision(m1)


def test_a_release_cannot_be_replayed_misdirected_or_forged():
    mfr, chips = make(3, bind=False)
    mfr.form_pod(chips[:2], label=1)
    c, other = chips[0], chips[1]
    release = mfr.release_chip(c)
    with pytest.raises(ProvisionError, match="another device"):
        other.release(release)
    c.release(release)
    with pytest.raises(ProvisionError):
        c.release(release)  # nothing left to release, and the generation is spent
    mfr2 = Manufacturer(b"attacker")
    forged = mfr2.sign_release(other.device_id, other.bound_pod, other.generation + 1)
    with pytest.raises(ProvisionError, match="signature"):
        other.release(forged)
    assert other.is_member


def test_a_fuse_bound_chip_cannot_be_released_at_all():
    mfr, chips = make(2, bind=True)
    mfr.form_pod(chips, label=1)
    r = mfr.release_chip(chips[0])
    with pytest.raises(ProvisionError, match="cannot be released"):
        chips[0].release(r)
    assert chips[0].is_member


def test_limitation_a_rolled_back_flash_binding_lets_a_careless_facility_move_a_chip_but_fuses_do_not():
    for bind, expect_moved in ((False, True), (True, False)):
        mfr, chips = make(4, bind=bind)
        blank = chips[0].flash.snapshot()
        mfr.form_pod(chips[:2], label=1)
        chips[0].flash.physical_restore(blank)
        chips[0].boot()
        careless = mfr.sign_manifest(mfr.pod_id_for(2), 5, [chips[0].device_id, chips[2].device_id])
        if expect_moved:
            chips[0].provision(careless)
            assert chips[0].pod_id == mfr.pod_id_for(2)
        else:
            with pytest.raises(ProvisionError, match="different pod"):
                chips[0].provision(careless)


def test_pod_ids_and_device_keys_are_distinct_and_the_registry_knows_every_chip():
    mfr, chips = make(6)
    assert len({c.device_id for c in chips}) == 6
    assert set(mfr.registry) == {c.device_id for c in chips}
    assert len({mfr.pod_id_for(i) for i in range(50)}) == 50


def test_link_mode_defaults_to_low_for_strangers():
    mfr, chips = make(3)
    mfr.form_pod(chips, label=1)
    assert chips[0].link_mode(chips[1].device_id) is LinkMode.LOW
    assert chips[0].link_mode(b"\x01" * 32) is LinkMode.LOW


def test_a_set_dead_fuse_alone_makes_a_chip_refuse_every_manifest_and_stay_out_at_boot():
    mfr, chips = make(2)
    c = chips[0]
    c.fuses.burn(podmod._DEAD_FUSE)  # not through the tamper path: the counter is nowhere near exhausted
    assert c.dead and c.generation == 0
    m = mfr.sign_manifest(mfr.pod_id_for(1), 1, [x.device_id for x in chips])
    with pytest.raises(ProvisionError, match="dead"):
        c.provision(m)
    c.boot()
    assert not c.is_member
    chips[1].provision(m)  # the same manifest is fine for a chip that is alive
    assert chips[1].is_member


def test_a_signed_manifest_for_another_pod_written_straight_into_flash_is_ignored_by_a_fuse_bound_chip():
    mfr, chips = make(4, bind=True)
    mfr.form_pod(chips[:2], label=1)
    victim = chips[0]
    careless = mfr.sign_manifest(mfr.pod_id_for(2), 5, [victim.device_id, chips[2].device_id])
    victim._manifest_rec.write(careless.to_bytes())  # an attacker's physical flash write: no provisioning code runs
    victim.boot()
    assert not victim.is_member
    assert victim.bound_pod == mfr.pod_id_for(1)
    assert victim.generation == 1  # and nothing was burned on the strength of the injected manifest


def test_an_old_release_cannot_eject_a_chip_that_later_rejoined_the_same_pod():
    mfr, chips = make(3, bind=False)
    m1 = mfr.form_pod(chips, label=1)
    c = chips[0]
    old_release = mfr.release_chip(c)
    c.release(old_release)
    assert not c.is_member and c.bound_pod is None
    c.provision(mfr.sign_manifest(m1.pod_id, old_release.generation + 1, list(m1.members)))  # same pod, later generation
    assert c.is_member and c.bound_pod == m1.pod_id
    with pytest.raises(ProvisionError, match="stale"):
        c.release(old_release)
    assert c.is_member


def test_a_tamper_response_under_a_starved_burn_supply_still_survives_an_ordinary_reboot():
    mfr, chips = make(3)
    mfr.form_pod(chips, label=1)
    c = chips[0]
    c.fuses.hv_disrupted = True
    c.tamper("mesh broken")
    assert c.generation == 1  # the burn did not take
    c.boot()  # a plain power cycle: nobody restored a flash image
    assert not c.is_member
