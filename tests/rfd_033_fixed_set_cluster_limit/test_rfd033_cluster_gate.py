"""Fleet-level cluster sizes, the key-extraction bridge, and the three enforcement designs (RFD 033 attack table)."""

from __future__ import annotations

import pytest

from aisecurity_rfds.common.channel import ReplayError
from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit.cluster import (
    bridge_cluster_size,
    build_fleet,
    components,
    establish,
    hbw_edges,
    impersonate,
    largest_hbw_cluster,
)
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit.gate import (
    Capability,
    Design,
    Frame,
    KeyGatedReceiver,
    ProtocolOnlyReceiver,
    attempt,
    bypass_matrix,
    compromise_firmware,
)
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit.pod import LinkDownError, LinkMode, Manufacturer, ProvisionError


def test_honest_pods_are_exactly_the_high_bandwidth_clusters():
    fleet = build_fleet(3, 4)
    comps = components(len(fleet.chips), hbw_edges(fleet.chips))
    assert sorted(len(c) for c in comps) == [4, 4, 4]
    assert {frozenset(c) for c in comps} == {frozenset(p) for p in fleet.pods}


def test_consumer_pods_of_one_never_form_a_cluster():
    fleet = build_fleet(6, 1)
    assert largest_hbw_cluster(fleet.chips) == 1
    careless = fleet.mfr.sign_manifest(fleet.mfr.pod_id_for(99), 5, [fleet.chips[0].device_id, fleet.chips[1].device_id])
    for chip in fleet.chips[:2]:
        with pytest.raises(ProvisionError, match="cap"):
            chip.provision(careless)
    assert largest_hbw_cluster(fleet.chips) == 1


def test_a_self_signed_super_manifest_is_refused_by_every_chip():
    fleet = build_fleet(3, 4)
    attacker = Manufacturer(b"attacker")
    everyone = attacker.sign_manifest(attacker.pod_id_for(1), 50, [c.device_id for c in fleet.chips])
    refused = 0
    for chip in fleet.chips:
        with pytest.raises(ProvisionError):
            chip.provision(everyone)
        refused += 1
    assert refused == 12 and largest_hbw_cluster(fleet.chips) == 4


def test_splicing_one_pods_manifest_into_another_pods_chips_is_refused():
    fleet = build_fleet(2, 4)
    for chip in fleet.chips[4:]:
        with pytest.raises(ProvisionError, match="not listed"):
            chip.provision(fleet.manifests[0])
    assert largest_hbw_cluster(fleet.chips) == 4


@pytest.mark.parametrize("pods_hit", [1, 2, 3, 4])
def test_extracting_one_key_from_each_of_m_pods_joins_exactly_m_pods(pods_hit: int):
    fleet = build_fleet(4, 4)
    extracted = [fleet.pods[p][0] for p in range(pods_hit)]
    assert bridge_cluster_size(fleet, extracted) == 4 * pods_hit


def test_extracting_several_keys_from_one_pod_adds_nothing_beyond_that_pod():
    fleet = build_fleet(3, 4)
    assert bridge_cluster_size(fleet, fleet.pods[0]) == 4
    assert bridge_cluster_size(fleet, []) == 4


def test_the_bridge_links_with_the_victim_pod_while_the_genuine_chip_is_still_in_it():
    fleet = build_fleet(2, 3)
    victim, neighbour = fleet.chips[0], fleet.chips[1]
    emulator = impersonate(fleet.mfr.public, victim.device, fleet.manifests[0], serial=900, pod_cap=3)
    assert establish(emulator, neighbour) and establish(victim, neighbour)
    assert neighbour.high_peers() == sorted([victim.device_id])  # same identity, so the neighbour cannot tell them apart


def test_an_emulator_without_the_device_key_cannot_join():
    from aisecurity_rfds.common.signing import KeyPair

    fleet = build_fleet(2, 3)
    fake_key = KeyPair.from_seed(make_rng(5).bytes(32))
    with pytest.raises(ProvisionError, match="not listed"):
        impersonate(fleet.mfr.public, fake_key, fleet.manifests[0], serial=901, pod_cap=3)


def test_enforcement_matrix_matches_the_expected_verdicts():
    expected = {
        Design.FIRMWARE_LIMITER: {Capability.NONE: False, Capability.SPOOF_FRAMES: True,
                                  Capability.MODIFY_FIRMWARE: True, Capability.EXTRACT_KEYS: False},
        Design.PROTOCOL_ONLY: {Capability.NONE: False, Capability.SPOOF_FRAMES: True,
                               Capability.MODIFY_FIRMWARE: True, Capability.EXTRACT_KEYS: True},
        Design.KEY_GATED: {Capability.NONE: False, Capability.SPOOF_FRAMES: False,
                           Capability.MODIFY_FIRMWARE: False, Capability.EXTRACT_KEYS: True},
    }
    got = {d: {} for d in Design}
    for o in bypass_matrix():
        got[o.design][o.capability] = o.bypassed
    assert got == expected


def test_key_gating_is_the_only_design_that_survives_firmware_rewriting_and_frame_injection():
    survivors = [
        d for d in Design
        if not attempt(d, Capability.MODIFY_FIRMWARE).bypassed and not attempt(d, Capability.SPOOF_FRAMES).bypassed
    ]
    assert survivors == [Design.KEY_GATED]


def test_a_rewritten_firmware_cannot_produce_or_accept_session_traffic():
    fleet = build_fleet(2, 3)
    a1, a2, b1 = fleet.chips[0], fleet.chips[1], fleet.chips[3]
    establish(a1, a2)
    compromise_firmware(a1)
    compromise_firmware(b1)
    assert a1.link_mode(b1.device_id) is LinkMode.HIGH  # the software check now lies ...
    with pytest.raises(LinkDownError):
        b1.seal(a1.device_id, b"x")  # ... but there is no session key to encrypt with
    assert not KeyGatedReceiver(a1).accept_hbw(Frame(b1.device_id, b"\x00" * 64))
    assert ProtocolOnlyReceiver(a1).accept_hbw(Frame(b1.device_id, b"x"))  # the protocol-only design has no such barrier


def test_a_captured_key_gated_frame_cannot_be_replayed():
    fleet = build_fleet(1, 3)
    a, b = fleet.chips[0], fleet.chips[1]
    establish(a, b)
    blob = b.seal(a.device_id, b"shard")
    rx = KeyGatedReceiver(a)
    assert rx.accept_hbw(Frame(b.device_id, blob))
    assert not rx.accept_hbw(Frame(b.device_id, blob))
    with pytest.raises(ReplayError):
        a.unseal(b.device_id, blob)


def test_components_fuses_entities_that_are_the_same_device():
    comps = components(6, [(0, 1), (2, 3)], same_entity=[[1, 2]])
    assert comps == [{0, 1, 2, 3}, {4}, {5}]
    assert components(6, [(0, 1), (2, 3)]) == [{0, 1}, {2, 3}, {4}, {5}]


def test_fleet_pod_caps_default_to_the_pod_size():
    fleet = build_fleet(2, 5)
    assert all(c._cap == 5 for c in fleet.chips)
    bigger = build_fleet(2, 5, pod_cap=8)
    assert all(c._cap == 8 for c in bigger.chips)


def test_flash_bound_fleets_behave_like_fuse_bound_ones_for_honest_traffic():
    fleet = build_fleet(2, 4, bind_in_fuses=False)
    assert largest_hbw_cluster(fleet.chips) == 4
