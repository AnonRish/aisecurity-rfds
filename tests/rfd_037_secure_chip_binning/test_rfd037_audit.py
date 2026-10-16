"""Sampling audits for absent circuits: exact detection probabilities, evasion budgets and the design-file blind spot."""

from __future__ import annotations

import itertools
import math

import pytest

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.common.stats import hypergeom_miss_probability, samples_needed_hypergeom
from aisecurity_rfds.rfd_037_secure_chip_binning import audit

LOT = 100_000


@pytest.mark.parametrize("lot, cheated, inspected", [(LOT, 1000, 298), (50, 3, 10), (20, 0, 5), (20, 20, 1), (30, 29, 2)])
def test_miss_probability_equals_the_shared_hypergeometric_when_detection_is_perfect(lot, cheated, inspected):
    assert audit.miss_probability(lot, cheated, inspected, 1.0) == pytest.approx(
        hypergeom_miss_probability(lot, cheated, inspected), rel=1e-9, abs=1e-12
    )


def test_miss_probability_with_imperfect_detection_equals_enumeration_of_every_possible_sample():
    lot, cheated, inspected, detect = 10, 3, 4, 0.3
    marked = set(range(cheated))
    samples = list(itertools.combinations(range(lot), inspected))
    expected = sum((1 - detect) ** len(marked.intersection(s)) for s in samples) / len(samples)
    assert audit.miss_probability(lot, cheated, inspected, detect) == pytest.approx(expected, rel=1e-12)
    assert audit.miss_probability(lot, cheated, inspected, 0.0) == pytest.approx(1.0)
    assert audit.miss_probability(lot, cheated, 0, 0.9) == pytest.approx(1.0)
    assert audit.miss_probability(lot, cheated, lot, 0.5) == pytest.approx(0.5**cheated)


def test_a_larger_audit_or_better_detection_always_lowers_the_miss_probability():
    by_size = [audit.miss_probability(LOT, 100, n, 0.7) for n in (0, 100, 1000, 5000, 20000)]
    assert all(a > c for a, c in zip(by_size, by_size[1:]))
    by_detection = [audit.miss_probability(LOT, 100, 1000, d) for d in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(a > c for a, c in zip(by_detection, by_detection[1:]))
    by_cheating = [audit.miss_probability(LOT, k, 1000, 1.0) for k in (1, 10, 100, 1000)]
    assert all(a > c for a, c in zip(by_cheating, by_cheating[1:]))


@pytest.mark.parametrize("cheated, confidence, detect", [(1000, 0.95, 1.0), (100, 0.95, 0.5), (10, 0.9, 0.9), (2, 0.5, 1.0)])
def test_inspections_needed_is_the_exact_threshold(cheated: int, confidence: float, detect: float):
    needed = audit.inspections_needed(LOT, cheated, confidence, detect)
    assert needed is not None
    assert audit.miss_probability(LOT, cheated, needed, detect) <= 1 - confidence
    assert audit.miss_probability(LOT, cheated, needed - 1, detect) > 1 - confidence
    if detect == 1.0:
        assert needed == samples_needed_hypergeom(LOT, cheated, confidence)


def test_a_threshold_that_is_an_exact_tie_is_not_pushed_up_by_rounding():
    # One cheated chip is missed with probability (lot - n) / lot, so these answers are exact ties.
    assert audit.inspections_needed(LOT, 1, 0.5) == 50_000
    assert audit.inspections_needed(LOT, 1, 0.95) == 95_000
    assert audit.inspections_needed(1000, 1, 0.9) == 900
    assert audit.evasion_budget(LOT, 50_000, 0.5) == 1
    assert audit.evasion_budget(LOT, 95_000, 0.05) == 1
    # a hair either side of the tie moves the answer by one chip, so the slack is not hiding real differences
    assert audit.inspections_needed(LOT, 1, 0.5 + 1e-5) == 50_001
    assert audit.evasion_budget(LOT, 50_000, 0.5 + 1e-5) == 0


def test_no_amount_of_inspection_suffices_when_detection_is_too_weak():
    assert audit.inspections_needed(LOT, 10, 0.95, 0.2) is None
    enough = audit.inspections_needed(LOT, 10, 0.95, 0.3)
    assert enough is not None and enough > LOT // 2
    for bad in ((LOT, 0, 0.95), (LOT, 10, 0.0), (LOT, 10, 1.0)):
        with pytest.raises(ValueError):
            audit.inspections_needed(*bad)


def test_the_evasion_budget_is_the_largest_cheat_that_still_escapes():
    for inspected, tolerated in ((300, 0.05), (300, 0.5), (3000, 0.05), (3000, 0.5)):
        k = audit.evasion_budget(LOT, inspected, tolerated)
        assert audit.miss_probability(LOT, k, inspected) >= tolerated
        assert audit.miss_probability(LOT, k + 1, inspected) < tolerated
    assert audit.evasion_budget(LOT, 300, 0.05) > audit.evasion_budget(LOT, 3000, 0.05) > audit.evasion_budget(LOT, 30000, 0.05)
    assert audit.evasion_budget(LOT, 300, 0.05) > audit.evasion_budget(LOT, 300, 0.5)
    assert audit.evasion_budget(LOT, 0, 0.5) == LOT
    with pytest.raises(ValueError):
        audit.evasion_budget(LOT, 300, 0.0)


def test_a_weak_inspection_lets_far_more_cheating_escape():
    perfect = audit.evasion_budget(LOT, 3000, 0.05, detect=1.0)
    weak = audit.evasion_budget(LOT, 3000, 0.05, detect=0.2)
    assert weak > 4 * perfect


def test_destroyed_fraction_is_inspected_over_lot_and_the_inputs_are_checked():
    assert audit.destroyed_fraction(LOT, 300) == pytest.approx(0.003)
    assert audit.destroyed_fraction(LOT, 0) == 0.0
    assert audit.destroyed_fraction(10, 10) == 1.0
    for bad in ((10, 11), (10, -1)):
        with pytest.raises(ValueError):
            audit.destroyed_fraction(*bad)
    with pytest.raises(ValueError):
        audit.miss_probability(LOT, 10, 10, 1.5)
    with pytest.raises(ValueError):
        audit.miss_probability(LOT, LOT + 1, 10, 1.0)
    with pytest.raises(ValueError):
        audit.miss_probability(0, 0, 0, 1.0)


def test_repeated_lots_compound_the_chance_of_never_being_caught():
    once = audit.miss_probability(LOT, 100, 300, 1.0)
    assert audit.repeated_miss_probability(LOT, 100, 300, 1.0, 10) == pytest.approx(once**10)
    assert audit.repeated_miss_probability(LOT, 100, 300, 1.0, 0) == 1.0
    assert audit.repeated_miss_probability(LOT, 100, 300, 1.0, 10) < 0.05 < once
    with pytest.raises(ValueError):
        audit.repeated_miss_probability(LOT, 100, 300, 1.0, -1)


def test_a_simulated_audit_finds_cheating_as_often_as_the_closed_form_says():
    lot = audit.Lot(2000, frozenset(range(20)), "reference-design")
    trials = 3000
    estimate = audit.estimate_detection(lot, 100, 0.6, trials, make_rng(77))
    exact = 1.0 - audit.miss_probability(2000, 20, 100, 0.6)
    assert abs(estimate.p - exact) < 4 * math.sqrt(exact * (1 - exact) / trials)
    assert estimate.trials == trials


def test_the_chance_of_detecting_one_cheated_chip_is_the_detection_probability():
    lot = audit.Lot(1, frozenset({0}), "reference-design")
    estimate = audit.estimate_detection(lot, 1, 0.25, 4000, make_rng(5))
    assert abs(estimate.p - 0.25) < 4 * math.sqrt(0.25 * 0.75 / 4000)
    assert audit.estimate_detection(lot, 1, 1.0, 200, make_rng(5)).p == 1.0
    assert audit.estimate_detection(lot, 1, 0.0, 200, make_rng(5)).p == 0.0


def test_design_file_comparison_reads_only_what_the_manufacturer_declares():
    cheating = audit.Lot(1000, frozenset(range(1000)), "reference-design")
    assert audit.design_files_match("reference-design", cheating)
    assert not audit.design_files_match("reference-design", audit.Lot(1000, frozenset(), "a-different-design"))
    assert audit.sample_audit(cheating, 1000, 1.0, make_rng(1)), "physical inspection is independent of the declaration"


def test_an_honest_lot_never_fails_a_physical_audit_and_oversized_audits_are_refused():
    honest = audit.Lot(500, frozenset(), "reference-design")
    assert not any(audit.sample_audit(honest, 500, 1.0, make_rng(seed)) for seed in range(20))
    with pytest.raises(ValueError):
        audit.sample_audit(honest, 501, 1.0, make_rng(0))
    with pytest.raises(ValueError):
        audit.miss_probability(10, 11, 5)
