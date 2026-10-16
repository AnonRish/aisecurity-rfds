"""Chip identifiers: attestation, the counter tripwire for copied keys, PUF-derived keys, and glitter tags."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest
from scipy import stats as sps

from aisecurity_rfds.common.hashing import domain_hash, sha256
from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.common.signing import KeyPair, verify
from aisecurity_rfds.rfd_031_chip_registry import identity as I


def test_a_chip_id_is_a_domain_separated_hash_of_its_public_key():
    chip = I.DeviceKeyChip(b"x")
    assert chip.chip_id == domain_hash("rfd031.chip-id", chip.public_key)
    assert len(chip.chip_id) == 32 and chip.chip_id != sha256(chip.public_key)
    assert I.chip_id_of(b"a") != I.chip_id_of(b"b")


def test_party_keys_are_deterministic_and_distinct():
    assert I.party_key("alice").public_bytes == I.party_key("alice").public_bytes
    assert I.party_key("alice").public_bytes != I.party_key("bob").public_bytes


def test_chips_with_different_seeds_have_different_keys_and_equal_seeds_the_same():
    assert I.DeviceKeyChip(b"a").public_key == I.DeviceKeyChip(b"a").public_key
    assert I.DeviceKeyChip(b"a").public_key != I.DeviceKeyChip(b"b").public_key


def test_an_attestation_verifies_for_its_challenge_and_not_another():
    chip = I.DeviceKeyChip(b"x")
    att = chip.attest(b"challenge")
    assert I.verify_attestation(chip.public_key, b"challenge", att)
    assert not I.verify_attestation(chip.public_key, b"other", att)
    assert att.challenge == b"challenge" and att.chip_id == chip.chip_id


def test_an_attestation_does_not_verify_under_another_chips_key():
    a, b = I.DeviceKeyChip(b"a"), I.DeviceKeyChip(b"b")
    att = a.attest(b"c")
    assert not I.verify_attestation(b.public_key, b"c", att)


def test_an_attestation_cannot_be_edited_without_the_key():
    chip = I.DeviceKeyChip(b"x")
    att = chip.attest(b"c")
    bumped = I.Attestation(att.chip_id, att.challenge, att.counter + 1, att.signature)
    other_id = I.Attestation(b"\x00" * 32, att.challenge, att.counter, att.signature)
    flipped = I.Attestation(att.chip_id, att.challenge, att.counter, bytes([att.signature[0] ^ 1]) + att.signature[1:])
    for forged in (bumped, other_id, flipped):
        assert not I.verify_attestation(chip.public_key, b"c", forged)


def test_a_valid_signature_over_another_chip_id_is_not_an_attestation_by_this_key():
    key = KeyPair.from_seed(b"k" * 32)
    other = I.chip_id_of(b"\x01" * 32)
    att = I.Attestation(other, b"c", 1, key.sign(I.attestation_message(other, b"c", 1)))
    assert verify(key.public_bytes, I.attestation_message(other, b"c", 1), att.signature)  # the signature itself is genuine
    assert not I.verify_attestation(key.public_bytes, b"c", att)  # but the id it names is not the hash of this key
    own_id = I.chip_id_of(key.public_bytes)
    genuine = I.Attestation(own_id, b"c", 1, key.sign(I.attestation_message(own_id, b"c", 1)))
    assert I.verify_attestation(key.public_bytes, b"c", genuine)  # the same construction with its own id verifies


def test_the_attestation_message_binds_chip_challenge_and_counter():
    m = I.attestation_message(b"id", b"challenge", 1)
    assert m != I.attestation_message(b"id2", b"challenge", 1)
    assert m != I.attestation_message(b"id", b"challenge2", 1)
    assert m != I.attestation_message(b"id", b"challenge", 2)


def test_the_counter_rises_by_one_with_every_attestation():
    chip = I.DeviceKeyChip(b"x")
    assert chip.counter == 0
    assert [chip.attest(b"c").counter for _ in range(4)] == [1, 2, 3, 4]
    assert chip.counter == 4


def test_a_copy_has_the_same_identity_and_starts_at_the_originals_counter():
    chip = I.DeviceKeyChip(b"x")
    chip.attest(b"c")
    copy = chip.clone()
    assert copy.public_key == chip.public_key and copy.chip_id == chip.chip_id and copy.counter == 1
    assert chip.clone(advance=5).counter == 6
    att = copy.attest(b"c")
    assert I.verify_attestation(chip.public_key, b"c", att)  # indistinguishable by signature
    assert chip.counter == 1  # the copy's use does not touch the original


def test_a_copy_cannot_start_behind_the_original():
    with pytest.raises(ValueError):
        I.DeviceKeyChip(b"x").clone(advance=-1)


def test_the_tracker_accepts_rising_counters_and_flags_repeats_and_drops():
    chip = I.DeviceKeyChip(b"x")
    tracker = I.CounterTracker()
    assert tracker.last(chip.chip_id) == 0
    first, second = chip.attest(b"1"), chip.attest(b"2")
    assert tracker.observe(second) == I.FRESH and tracker.last(chip.chip_id) == 2
    assert tracker.observe(first) == I.REGRESSION  # lower
    assert tracker.observe(second) == I.REGRESSION  # equal: a replay
    assert tracker.last(chip.chip_id) == 2  # a regression does not move the mark


def test_the_tracker_keeps_chips_apart():
    a, b = I.DeviceKeyChip(b"a"), I.DeviceKeyChip(b"b")
    tracker = I.CounterTracker()
    for _ in range(3):
        tracker.observe(a.attest(b"c"))
    assert tracker.last(b.chip_id) == 0
    assert tracker.observe(b.attest(b"c")) == I.FRESH


@pytest.mark.parametrize("pattern", ["O", "C", "OO", "CC", "OOO", "CCCC"])
def test_use_by_only_one_of_the_two_chips_is_never_flagged(pattern):
    assert not I.run_clone_sequence(pattern)
    assert not I.run_clone_sequence(pattern, advance=100)


@pytest.mark.parametrize("pattern", ["OC", "CO", "OOC", "COO", "OCO", "CCO", "OCC"])
def test_a_copy_that_starts_level_is_caught_at_the_first_switch(pattern):
    assert I.run_clone_sequence(pattern)


@pytest.mark.parametrize(
    "pattern,caught",
    [("OC", False), ("OOCC", False), ("OOOC", False), ("CO", True), ("OCO", True), ("COC", True), ("CCO", True), ("OCCO", True)],
)
def test_a_copy_that_starts_ahead_is_caught_only_when_the_original_follows_it(pattern, caught):
    assert I.run_clone_sequence(pattern, advance=1000) is caught


def test_a_clone_sequence_may_contain_only_the_two_chips():
    with pytest.raises(ValueError):
        I.run_clone_sequence("OXC")


def test_an_empty_sequence_flags_nothing():
    assert I.run_clone_sequence("") is False


@pytest.mark.parametrize("events", range(0, 7))
@pytest.mark.parametrize("q", [0.05, 0.3, 0.5, 0.9])
@pytest.mark.parametrize("careful", [False, True])
def test_the_closed_forms_equal_a_sum_over_every_sequence(events, q, careful):
    closed = I.clone_detection_probability(events, q, careful)
    assert closed == pytest.approx(I.clone_detection_by_enumeration(events, q, careful), abs=1e-12)


def test_the_closed_forms_at_known_points():
    assert I.clone_detection_probability(2, 0.5) == pytest.approx(0.5)
    assert I.clone_detection_probability(2, 0.5, careful=True) == pytest.approx(0.25)
    assert I.clone_detection_probability(5, 0.2) == pytest.approx(1 - 0.2**5 - 0.8**5)
    assert I.clone_detection_probability(1, 0.3) == 0.0
    assert I.clone_detection_probability(1, 0.3, careful=True) == 0.0


@pytest.mark.parametrize("careful", [False, True])
def test_a_copy_that_is_never_or_always_used_is_never_caught(careful):
    for events in (0, 1, 5, 12):
        assert I.clone_detection_probability(events, 0.0, careful) == pytest.approx(0.0)
        assert I.clone_detection_probability(events, 1.0, careful) == pytest.approx(0.0)


def test_a_careful_copy_is_never_easier_to_catch_than_a_level_one():
    for events in range(1, 25):
        for q in (0.01, 0.1, 0.3, 0.5, 0.8):
            assert I.clone_detection_probability(events, q, True) <= I.clone_detection_probability(events, q, False) + 1e-12


def test_detection_rises_with_the_number_of_attestations_seen():
    for careful in (False, True):
        series = [I.clone_detection_probability(m, 0.1, careful) for m in range(1, 40)]
        assert all(b >= a for a, b in zip(series, series[1:]))
        assert series[-1] > 0.95


def test_clone_probability_rejects_nonsense():
    with pytest.raises(ValueError):
        I.clone_detection_probability(-1, 0.5)
    with pytest.raises(ValueError):
        I.clone_detection_probability(3, 1.5)
    with pytest.raises(ValueError):
        I.clone_detection_probability(3, -0.1)


def test_a_puf_chip_attests_and_its_key_is_stable_across_boots():
    chip = I.PufChip(1)
    rng = make_rng(0)
    first = chip.attest(b"c", rng)
    second = chip.attest(b"c", rng)
    assert first is not None and second is not None
    assert I.verify_attestation(chip.public_key, b"c", first)
    assert I.verify_attestation(chip.public_key, b"c", second)
    assert (first.counter, second.counter) == (1, 2)
    assert chip.chip_id == I.chip_id_of(chip.public_key)
    assert I.PufChip(1).public_key == chip.public_key
    assert I.PufChip(2).public_key != chip.public_key


def test_a_puf_chip_with_too_little_redundancy_cannot_attest_under_stress():
    chip = I.PufChip(1, repetition=3)
    rng = make_rng(0)
    stressed = [chip.attest(b"c", rng, 125.0, 20.0) for _ in range(60)]
    assert 0 < sum(a is None for a in stressed)
    assert sum(a is not None for a in stressed) < 30  # fewer than half of the boots yield the key
    assert chip.counter == sum(a is not None for a in stressed)  # a failed boot is not an attestation


def test_more_redundancy_survives_conditions_that_defeat_less():
    rng = make_rng(1)
    rates = {}
    for repetition in (5, 15):
        chip = I.PufChip(3, repetition=repetition)
        rates[repetition] = sum(chip.attest(b"c", rng, 85.0, 5.0) is not None for _ in range(100))
    assert rates[15] == 100
    assert rates[5] < 90


def test_puf_responses_of_the_same_chip_are_close_and_of_different_chips_are_far():
    rows = I.puf_separation(100, 1024, 0.25, {"nominal": (25.0, 0.0), "hot and old": (125.0, 20.0)}, seed=1)
    nominal, hot = rows
    assert 0.03 < nominal.intra_mean < 0.06 and nominal.intra_worst < 0.10
    assert hot.intra_mean > nominal.intra_mean + 0.03
    for row in rows:
        assert abs(row.inter_mean - 0.5) < 0.01
        assert row.false_reject_rate == 0.0


def test_a_threshold_inside_the_noise_rejects_genuine_responses():
    (row,) = I.puf_separation(50, 1024, 0.01, {"nominal": (25.0, 0.0)}, seed=1)
    assert row.false_reject_rate == 1.0


def test_puf_separation_is_reproducible_and_needs_two_devices():
    cond = {"a": (25.0, 0.0)}
    assert I.puf_separation(20, 256, 0.25, cond, seed=4) == I.puf_separation(20, 256, 0.25, cond, seed=4)
    assert I.puf_separation(20, 256, 0.25, cond, seed=4) != I.puf_separation(20, 256, 0.25, cond, seed=5)
    with pytest.raises(ValueError):
        I.puf_separation(1, 256, 0.25, cond)


def test_the_puf_false_accept_probability_is_a_binomial_tail():
    assert I.puf_false_accept_probability(8, 0.25) == pytest.approx((1 + 8 + 28) / 256)
    assert I.puf_false_accept_probability(8, 0.0) == pytest.approx(1 / 256)
    assert I.puf_false_accept_probability(1024, 0.25) == pytest.approx(float(sps.binom.cdf(256, 1024, 0.5)))
    assert I.puf_false_accept_probability(256, 0.25) > I.puf_false_accept_probability(1024, 0.25)
    assert I.puf_false_accept_probability(1024, 0.15) < I.puf_false_accept_probability(1024, 0.25)


def test_random_tags_are_on_the_unit_torus_and_reproducible():
    tag = I.random_tag(50, make_rng(3))
    assert tag.shape == (50, 2) and tag.min() >= 0.0 and tag.max() < 1.0
    assert np.array_equal(tag, I.random_tag(50, make_rng(3)))
    assert not np.array_equal(tag, I.random_tag(50, make_rng(4)))


def test_a_photograph_keeps_every_flake_on_the_torus_and_moves_it_by_about_sigma():
    rng = make_rng(5)
    tag = I.random_tag(4000, rng)
    photo = I.photograph(tag, 0.01, rng)
    assert photo.min() >= 0.0 and photo.max() < 1.0
    delta = (photo - tag + 0.5) % 1.0 - 0.5
    assert abs(delta.std() - 0.01) < 0.001


def test_an_identical_photograph_matches_every_flake():
    tag = I.random_tag(30, make_rng(1))
    assert I.match_fraction(tag, tag.copy(), 1e-9) == 1.0


def test_matching_wraps_around_the_edges_of_the_torus():
    tag = np.array([[0.999, 0.5]])
    photo = np.array([[0.001, 0.5]])
    assert I.match_fraction(tag, photo, 0.01) == 1.0
    assert I.match_fraction(tag, photo, 0.001) == 0.0


def test_a_flake_matches_only_inside_the_radius():
    tag = np.array([[0.5, 0.5]])
    assert I.match_fraction(tag, np.array([[0.5, 0.51]]), 0.0101) == 1.0
    assert I.match_fraction(tag, np.array([[0.5, 0.51]]), 0.0099) == 0.0
    assert I.match_fraction(tag, np.array([[0.508, 0.508]]), 0.0115) == 1.0  # diagonal distance 0.0113
    assert I.match_fraction(tag, np.array([[0.508, 0.508]]), 0.0110) == 0.0
    assert I.match_fraction(tag, np.array([[0.75, 0.5]]), 0.25) == 1.0  # exactly on the radius (0.25 squared is exact)
    assert I.match_fraction(tag, np.array([[0.75, 0.5]]), 0.2499) == 0.0


def test_match_fraction_counts_enrolled_flakes():
    tag = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.2], [0.3, 0.7]])
    photo = np.array([[0.1, 0.1], [0.5, 0.5], [0.0, 0.0]])
    assert I.match_fraction(tag, photo, 0.01) == 0.5


def test_unrelated_tags_do_not_match():
    rng = make_rng(2)
    fractions = [I.match_fraction(I.random_tag(30, rng), I.random_tag(30, rng), 0.012) for _ in range(200)]
    assert max(fractions) <= 0.1


def test_glitter_rates_for_a_single_flake_are_closed_forms():
    sigma, radius = 0.01, 0.02
    frr, far = I.glitter_error_rates(1, sigma, radius, 1.0)
    assert frr == pytest.approx(math.exp(-(radius**2) / (2 * sigma**2)))
    assert far == pytest.approx(math.pi * radius**2)


def test_the_acceptance_count_is_rounded_up_without_float_noise():
    # 0.28 * 25 is 7.000000000000001 in floating point: 7 flakes must be enough, not 8
    assert 0.28 * 25 > 7.0
    n, sigma, radius = 25, 0.012, 0.012
    own = 1.0 - math.exp(-(radius**2) / (2.0 * sigma**2))
    genuine = 1.0 - (1.0 - own) * (1.0 - math.pi * radius**2) ** (n - 1)
    frr, _ = I.glitter_error_rates(n, sigma, radius, 0.28)
    assert frr == pytest.approx(float(sps.binom.cdf(6, n, genuine)))
    assert frr != pytest.approx(float(sps.binom.cdf(7, n, genuine)), rel=1e-3)


def test_a_noisier_camera_rejects_more_and_a_larger_radius_accepts_more_of_both():
    base = I.glitter_error_rates(30, 0.005, 0.012, 0.8)
    assert I.glitter_error_rates(30, 0.007, 0.012, 0.8)[0] > base[0]
    wide = I.glitter_error_rates(30, 0.005, 0.02, 0.8)
    assert wide[0] < base[0] and wide[1] > base[1]
    assert I.glitter_error_rates(30, 0.005, 0.012, 0.95)[0] > base[0]
    assert I.glitter_error_rates(30, 0.005, 0.012, 0.95)[1] < base[1]


def test_glitter_rates_reject_nonsense_parameters():
    for args in ((0, 0.01, 0.01, 0.5), (10, 0.0, 0.01, 0.5), (10, 0.01, 0.0, 0.5), (10, 0.01, 0.5, 0.5), (10, 0.01, 0.01, 0.0), (10, 0.01, 0.01, 1.5)):
        with pytest.raises(ValueError):
            I.glitter_error_rates(*args)


@pytest.mark.parametrize("sigma", [0.005, 0.006, 0.007])
def test_the_false_reject_model_agrees_with_simulated_photographs(sigma):
    rng = make_rng(11)
    frr, _ = I.glitter_error_rates(30, sigma, 0.012, 0.8)
    simulated, _ = I.simulate_glitter(30, sigma, 0.012, 0.8, 2000, rng)
    assert simulated.contains(frr)


def test_the_false_accept_model_is_close_to_simulation_where_it_can_be_measured():
    rng = make_rng(12)
    _, far = I.glitter_error_rates(10, 0.05, 0.15, 0.6)
    _, simulated = I.simulate_glitter(10, 0.05, 0.15, 0.6, 3000, rng)
    assert abs(simulated.p - far) < 0.04


def test_simulated_glitter_is_reproducible():
    a = I.simulate_glitter(20, 0.006, 0.012, 0.8, 200, make_rng(1))
    assert a == I.simulate_glitter(20, 0.006, 0.012, 0.8, 200, make_rng(1))


def test_enumeration_helper_uses_a_start_far_enough_ahead():
    # careful copies must really start ahead: a pattern the original ends is caught only when the original follows
    assert I.clone_detection_by_enumeration(2, 0.5, careful=True) == pytest.approx(0.25)
    assert I.clone_detection_by_enumeration(2, 0.5, careful=False) == pytest.approx(0.5)
    assert sum(1 for _ in product("OC", repeat=3)) == 8
