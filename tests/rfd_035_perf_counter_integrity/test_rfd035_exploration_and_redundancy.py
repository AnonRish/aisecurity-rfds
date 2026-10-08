"""Bounded exhaustive exploration of the readout code (including deliberately broken variants), and redundancy."""

from __future__ import annotations

import numpy as np
import pytest

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.rfd_035_perf_counter_integrity import monotonic, redundancy
from aisecurity_rfds.rfd_035_perf_counter_integrity.readout import ReadoutBlock, ReadoutVerifier, Verdict


def violation_names(report):
    return {name for name, _ in report.violations}


def test_no_invariant_is_violated_over_every_sequence_of_hiding_actions():
    report = monotonic.explore(monotonic.ALPHABET_HIDING, 6)
    assert report.violations == []
    assert report.transitions > 15_000
    assert report.accepted_readouts > 1_000 and report.mature_windows > 100


def test_no_invariant_is_violated_with_replays_drops_and_bad_tickets():
    report = monotonic.explore(monotonic.ALPHABET_PROTOCOL, 5)
    assert report.violations == []
    assert report.transitions > 30_000
    assert {"replay", "gap"} <= set(report.detected)  # these inputs were generated and refused, not skipped


def test_exploration_finds_the_shortest_trace_that_hides_a_whole_mature_window():
    report = monotonic.explore(monotonic.ALPHABET_HIDING, 6)
    assert report.max_window_hidden_fraction == pytest.approx(1.0)
    assert report.witness == ("work1", "read", "glitch_clear", "work1", "read")


def test_a_stall_also_hides_a_whole_window_and_is_accepted():
    report = monotonic.explore(("work1", "read", "stall_toggle"), 5)
    assert report.violations == []
    assert report.max_window_hidden_fraction == pytest.approx(1.0)
    assert report.witness is not None and "stall_toggle" in report.witness


def test_without_glitches_or_stalls_no_window_is_ever_hidden():
    report = monotonic.explore(("work1", "work2", "read", "legit_reset"), 6)
    assert report.violations == []
    assert report.max_window_hidden_fraction == pytest.approx(0.0)
    assert report.mature_windows > 100


def test_glitches_that_do_not_regain_the_old_value_are_always_detected():
    report = monotonic.explore(("work1", "read", "glitch_clear"), 5)
    assert report.detected["implausible-delta"] > 0
    assert report.violations == []


def test_exploration_is_deterministic():
    a = monotonic.explore(monotonic.ALPHABET_HIDING, 5)
    b = monotonic.explore(monotonic.ALPHABET_HIDING, 5)
    assert (a.transitions, a.accepted_readouts, dict(a.detected), a.witness) == (b.transitions, b.accepted_readouts, dict(b.detected), b.witness)


def test_explore_rejects_unknown_actions():
    with pytest.raises(ValueError):
        monotonic.explore(("work1", "teleport"), 2)


# --- the explorer must be able to fail: broken variants violate the invariants it checks ---------------------------


class AcceptsReplays(ReadoutVerifier):
    def accept(self, ro):
        verdict = super().accept(ro)
        return Verdict(True) if verdict.reason == "replay" else verdict


class BlockAcceptsAnyTicket(ReadoutBlock):
    def _ticket_problem(self, ticket):
        return None


class SkipsRegressionCheck(ReadoutVerifier):
    def accept(self, ro):
        verdict = super().accept(ro)
        if verdict.reason != "implausible-delta":
            return verdict
        # Pretend the plausibility check did not exist: take the readout at face value.
        self.seq, self.head, self.ticks = ro.seq, ro.chain_hash(), ro.ticks
        self.credited = [c + ((x - b) & 0xFF) for c, x, b in zip(self.credited, ro.counters, self.last_counters)]
        self.last_counters = list(ro.counters)
        return Verdict(True)


class CreditsTwiceTheDelta(ReadoutVerifier):
    """Over-credits: every accepted readout is credited double."""

    def accept(self, ro):
        before = list(self.credited)
        verdict = super().accept(ro)
        if verdict.ok:
            self.credited = [b + 2 * (c - b) for b, c in zip(before, self.credited)]
        return verdict


class LosesTheWorkInClosingReadouts(ReadoutVerifier):
    """Under-credits: the work reported in a closing readout (the last window before a reset) is thrown away."""

    def accept(self, ro):
        before = list(self.credited)
        verdict = super().accept(ro)
        if verdict.ok and ro.closing:
            self.credited = before
        return verdict


def test_a_verifier_that_accepts_replays_is_caught_by_the_exploration():
    report = monotonic.explore(("work1", "read", "replay"), 4, verifier_cls=AcceptsReplays)
    assert "replay-accepted" in violation_names(report)


def test_a_block_that_accepts_any_ticket_is_caught_by_the_exploration():
    report = monotonic.explore(("work1", "forged_ticket", "future_ticket"), 3, block_cls=BlockAcceptsAnyTicket)
    assert {"invalid-ticket-accepted", "unauthorised-epoch-change"} & violation_names(report)


def test_a_verifier_without_the_plausibility_check_accepts_a_regression_and_is_caught():
    report = monotonic.explore(("work1", "read", "glitch_clear"), 5, verifier_cls=SkipsRegressionCheck)
    assert {"regression-accepted", "over-credit"} & violation_names(report)


def test_a_block_that_forgets_the_epoch_on_reset_is_caught():
    class ForgetsEpoch(ReadoutBlock):
        def reset(self, ticket):
            closing = super().reset(ticket)
            self.epoch -= 1
            return closing

    assert monotonic.explore(("work1", "read", "legit_reset"), 4).violations == []  # the correct block passes
    report = monotonic.explore(("work1", "read", "legit_reset"), 4, block_cls=ForgetsEpoch)
    assert {"epoch-desync", "unauthorised-epoch-change"} <= violation_names(report)


def test_the_correct_implementation_trips_none_of_the_named_invariants():
    for alphabet, depth in ((("work1", "read", "legit_reset", "glitch_clear"), 5), (("work1", "read", "replay"), 4)):
        assert monotonic.explore(alphabet, depth).violations == []


def test_over_crediting_is_reported_as_over_credit():
    report = monotonic.explore(("work1", "read"), 3, verifier_cls=CreditsTwiceTheDelta)
    assert "over-credit" in violation_names(report)


def test_losing_work_in_a_closing_readout_is_reported_as_a_conservation_violation():
    report = monotonic.explore(("work1", "read", "legit_reset"), 4, verifier_cls=LosesTheWorkInClosingReadouts)
    assert "conservation" in violation_names(report)
    assert "over-credit" not in violation_names(report)  # it under-credits, so only the conservation law notices


def test_a_verifier_with_a_huge_rate_bound_accepts_regressions():
    report = monotonic.explore(("work1", "read", "glitch_clear"), 5, max_work=10**6)
    assert "regression-accepted" in violation_names(report)


# --- counter width -------------------------------------------------------------------------------------------


def test_counter_width_arithmetic():
    assert monotonic.bits_needed(2**64 - 1) == 64
    assert monotonic.bits_needed(2**64) == 65
    assert monotonic.bits_needed(10**26) == 87
    assert monotonic.seconds_to_wrap(64, 1e15) / 3600 == pytest.approx(5.124, abs=0.001)
    assert monotonic.seconds_to_wrap(87, 1e15) / (3600 * 24 * 365) > 4000  # years


def test_safe_gap_and_aliasing_are_consistent():
    for width, rate in ((8, 2), (8, 20), (16, 100)):
        gap = monotonic.safe_gap(width, rate)
        assert rate * gap < 2**width <= rate * (gap + 1) + rate
        assert monotonic.aliased_work(width, rate * gap) == 0
        assert monotonic.aliased_work(width, 2**width + 5) == 2**width


# --- redundancy ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_any_fault_short_of_all_copies_is_detected_and_majority_recovers_below_half(k):
    result = redundancy.exhaustive_detection(k)
    for t, (always_detected, always_recovered) in result.items():
        assert always_detected == (t < k)
        assert always_recovered == (2 * t < k)


def test_a_single_copy_cannot_detect_anything():
    c = redundancy.RedundantCounter(1)
    c.add(10)
    c.corrupt([0], 3)
    assert c.consistent() and c.majority() == 3


def test_an_even_split_has_no_majority():
    two = redundancy.RedundantCounter(2)
    two.add(10)
    two.corrupt([0], 3)
    assert two.majority() is None  # one copy against one: nobody wins
    four = redundancy.RedundantCounter(4)
    four.add(10)
    four.corrupt([0, 1], 3)
    assert four.majority() is None
    four.corrupt([0], 10)
    assert four.majority() == 10  # three against one


def test_redundant_counter_requires_a_copy():
    with pytest.raises(ValueError):
        redundancy.RedundantCounter(0)


def test_single_copy_undetected_probability_matches_the_disc_area():
    radius, n = 0.1, 200_000
    out = redundancy.emp_outcomes(redundancy.clustered_layout(1), radius, n, make_rng(41))
    p = np.pi * radius**2
    assert abs(out["undetected"] - p) < 4 * np.sqrt(p * (1 - p) / n)
    assert out["detected"] == 0.0


def test_spreading_copies_removes_undetected_emp_outcomes_at_small_radius():
    for k in (2, 3, 4):
        spread = redundancy.emp_outcomes(redundancy.spread_layout(k), 0.15, 60_000, make_rng(42))
        clustered = redundancy.emp_outcomes(redundancy.clustered_layout(k), 0.15, 60_000, make_rng(42))
        assert spread["undetected"] == 0.0 and spread["detected"] > 0.1
        assert clustered["undetected"] > 0.04 and clustered["detected"] < spread["detected"]


def test_a_large_enough_pulse_defeats_even_spread_copies():
    out = redundancy.emp_outcomes(redundancy.spread_layout(3), 0.9, 20_000, make_rng(43))
    assert out["undetected"] > 0.5


def test_emp_outcome_probabilities_sum_to_one():
    out = redundancy.emp_outcomes(redundancy.spread_layout(3), 0.3, 10_000, make_rng(44))
    assert sum(out.values()) == pytest.approx(1.0)
    assert redundancy.layouts((2, 3)).keys() == {"k=2 clustered", "k=2 spread", "k=3 clustered", "k=3 spread"}
