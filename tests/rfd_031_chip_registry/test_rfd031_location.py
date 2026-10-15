"""Location commitments and ghost-inventory audits."""

from __future__ import annotations

from math import comb

import pytest

from aisecurity_rfds.common.hashing import domain_hash, sha256
from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.rfd_031_chip_registry import identity as I
from aisecurity_rfds.rfd_031_chip_registry import location as L


def test_a_commitment_opens_only_to_the_facility_and_salt_it_was_made_with():
    salt = bytes(range(16))
    c = L.commit_location("Facility-A", salt)
    assert len(c) == 32
    assert L.verify_opening(c, L.Opening("Facility-A", salt))
    assert not L.verify_opening(c, L.Opening("Facility-B", salt))
    assert not L.verify_opening(c, L.Opening("Facility-A", salt[:-1] + b"\xff"))
    assert not L.verify_opening(c, L.Opening("Facility-A", b""))
    assert not L.verify_opening(sha256(c), L.Opening("Facility-A", salt))


def test_a_commitment_is_domain_separated_and_does_not_blur_facility_and_salt():
    assert L.commit_location("ab", b"c") != L.commit_location("a", b"bc")
    assert L.commit_location("a", b"") != L.commit_location("", b"a")
    assert L.commit_location("a", b"s") == domain_hash("rfd031.location", b"a", b"s")
    assert L.commit_location("a", b"s") != domain_hash("rfd031.other", b"a", b"s")


def test_the_same_facility_with_different_salts_gives_unlinkable_commitments():
    assert L.commit_location("Facility-A", b"salt-1") != L.commit_location("Facility-A", b"salt-2")


def test_distinct_commitments_counts_what_an_observer_could_group():
    facilities = ["A", "A", "B", "B", "B", "C"]
    assert L.distinct_commitments(facilities, [b"A", b"A", b"B", b"B", b"B", b"C"]) == 3
    assert L.distinct_commitments(facilities, [bytes([i]) for i in range(6)]) == 6
    assert L.distinct_commitments(facilities, [b"same"] * 6) == 3
    assert L.distinct_commitments([], []) == 0


def test_a_short_salt_falls_to_a_dictionary_and_the_right_opening_is_recovered():
    names = [f"Facility-{i:03d}" for i in range(50)]
    for salt_bytes in (0, 1):
        salt = bytes([0xA7]) * salt_bytes
        target = L.commit_location(names[37], salt)
        found = L.guess_facility(target, names, salt_bytes)
        assert found == L.Opening(names[37], salt)
        assert L.verify_opening(target, found)


def test_the_dictionary_attack_fails_when_the_facility_is_not_listed_or_the_salt_is_longer():
    names = [f"Facility-{i:03d}" for i in range(20)]
    assert L.guess_facility(L.commit_location("Elsewhere", b""), names, 0) is None
    assert L.guess_facility(L.commit_location(names[3], b"\x01\x02"), names, 1) is None
    with pytest.raises(ValueError):
        L.guess_facility(b"x" * 32, names, -1)


def test_dictionary_work_grows_with_candidates_and_256_per_salt_byte():
    assert L.dictionary_work(100, 0) == 100
    assert L.dictionary_work(100, 1) == 25_600
    assert L.dictionary_work(100, 2) == 6_553_600
    assert L.dictionary_work(1, 16) == 2**128
    assert L.dictionary_work(0, 3) == 0


def test_detection_probability_is_the_exact_hypergeometric_complement():
    for total, ghosts, sampled in ((20, 3, 5), (50, 10, 7), (12, 12, 1), (30, 1, 29)):
        expected = 1.0 - comb(total - ghosts, sampled) / comb(total, sampled)
        assert L.detection_probability(total, ghosts, sampled) == pytest.approx(expected, abs=1e-12)


def test_detection_probability_at_the_edges():
    assert L.detection_probability(100, 0, 50) == 0.0
    assert L.detection_probability(100, 10, 0) == 0.0
    assert L.detection_probability(100, 10, 91) == 1.0  # more chips than the honest ones: a ghost must be drawn
    assert L.detection_probability(100, 10, 90) < 1.0
    assert L.detection_probability(100, 100, 1) == 1.0


def test_detection_rises_with_the_sample_and_with_the_number_of_ghosts():
    sample_series = [L.detection_probability(1000, 20, m) for m in range(0, 200, 10)]
    assert all(b >= a for a, b in zip(sample_series, sample_series[1:]))
    ghost_series = [L.detection_probability(1000, g, 50) for g in range(0, 100, 5)]
    assert all(b >= a for a, b in zip(ghost_series, ghost_series[1:]))


@pytest.mark.parametrize("total,ghosts,confidence", [(10_000, 500, 0.95), (10_000, 100, 0.95), (10_000, 10, 0.95), (1000, 50, 0.99), (200, 12, 0.9)])
def test_the_chips_to_inspect_is_the_smallest_sample_that_reaches_the_confidence(total, ghosts, confidence):
    m = L.chips_to_inspect(total, ghosts, confidence)
    assert L.detection_probability(total, ghosts, m) >= confidence
    assert L.detection_probability(total, ghosts, m - 1) < confidence


def test_the_known_audit_sizes():
    assert [L.chips_to_inspect(10_000, g, 0.95) for g in (1000, 500, 100, 10)] == [29, 59, 294, 2588]


def test_simulation_agrees_with_the_closed_form_and_is_reproducible():
    exact = L.detection_probability(500, 30, 20)
    estimate = L.simulate_detection(500, 30, 20, 20_000, make_rng(3))
    assert estimate.contains(exact) and estimate.trials == 20_000
    assert estimate == L.simulate_detection(500, 30, 20, 20_000, make_rng(3))
    assert L.simulate_detection(500, 0, 20, 500, make_rng(3)).p == 0.0
    assert L.simulate_detection(500, 500, 1, 500, make_rng(3)).p == 1.0


def make_site(n: int, absent: int):
    chips = [I.DeviceKeyChip(b"site-%d" % i) for i in range(n)]
    declared = [(c.chip_id, c.public_key) for c in chips]
    return chips, declared, L.Facility(chips[absent:])


def challenge(chip_id: bytes) -> bytes:
    return domain_hash("test-audit", chip_id)


def test_a_facility_produces_the_chips_it_holds_and_not_the_ones_it_does_not():
    chips, declared, site = make_site(10, absent=3)
    assert site.inspect(chips[5].chip_id, chips[5].public_key, b"c")
    assert not site.inspect(chips[1].chip_id, chips[1].public_key, b"c")  # declared but absent
    assert not site.inspect(b"\x00" * 32, chips[5].public_key, b"c")  # an unknown id
    assert not site.inspect(chips[5].chip_id, chips[6].public_key, b"c")  # the wrong public key on record


def test_inspecting_a_chip_makes_it_attest_with_the_given_challenge():
    chips, _, site = make_site(3, absent=0)
    before = chips[0].counter
    assert site.inspect(chips[0].chip_id, chips[0].public_key, b"c")
    assert chips[0].counter == before + 1


def test_an_audit_returns_exactly_the_absent_chips_it_sampled():
    _, declared, site = make_site(40, absent=6)
    sample = [0, 3, 5, 6, 20, 39]
    assert L.audit_declared(declared, site, sample, challenge) == [0, 3, 5]
    assert L.audit_declared(declared, site, [10, 11], challenge) == []
    assert L.audit_declared(declared, site, [], challenge) == []


def test_an_audit_challenges_each_sampled_chip_with_its_own_challenge():
    chips, declared, site = make_site(5, absent=0)
    seen = []

    def spy(chip_id):
        seen.append(chip_id)
        return challenge(chip_id)

    L.audit_declared(declared, site, [4, 1], spy)
    assert seen == [chips[4].chip_id, chips[1].chip_id]
