"""Lockout mechanisms against attacker budgets, computed by running the chip models (RFD 037, mechanisms 1 to 4)."""

from __future__ import annotations

import itertools
import math

import pytest

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.common.stats import monte_carlo_probability
from aisecurity_rfds.rfd_037_secure_chip_binning import lockout
from aisecurity_rfds.rfd_037_secure_chip_binning.mechanisms import (
    BASE_FEATURE,
    FEATURE,
    AntifuseRail,
    Attacker,
    Authority,
    FirmwareFlag,
    FuseGate,
    Outcome,
    PufBound,
    SeparateDie,
    Token,
    TokenGate,
    evaluate,
    force_decision,
)

B, V, S = Outcome.BLOCKED, Outcome.VISIBLE, Outcome.SILENT
RANK = {B: 0, V: 1, S: 2}


class Stub:
    """A chip whose answers are fixed, to test how ``evaluate`` classifies them."""

    name = "stub"

    def __init__(self, works: bool, reports_off: bool) -> None:
        self._works, self._reports_off = works, reports_off

    def attack(self, attacker: Attacker) -> None:
        return None

    def feature_works(self) -> bool:
        return self._works

    def reports_disabled(self) -> bool:
        return self._reports_off


def test_evaluate_separates_blocked_visible_and_silent():
    assert evaluate(lambda: Stub(False, True), Attacker()) is B
    assert evaluate(lambda: Stub(False, False), Attacker()) is B
    assert evaluate(lambda: Stub(True, False), Attacker()) is V
    assert evaluate(lambda: Stub(True, True), Attacker()) is S


def test_attack_budgets_cannot_be_negative():
    with pytest.raises(ValueError):
        Attacker(glitches=-1)
    with pytest.raises(ValueError):
        Attacker(edits=-1)


def test_force_decision_spends_faults_first_and_reports_the_edits_left_over():
    assert force_decision(Attacker(), 1) == (False, -1)
    assert force_decision(Attacker(glitches=1), 1) == (True, 0)
    assert force_decision(Attacker(edits=1), 1) == (True, 0)
    assert force_decision(Attacker(edits=2), 1) == (True, 1)
    assert force_decision(Attacker(glitches=1), 3) == (False, -1)
    assert force_decision(Attacker(glitches=1, edits=1), 3) == (True, 0)
    assert force_decision(Attacker(glitches=5, edits=1), 3) == (True, 1)


def test_a_firmware_flag_falls_to_software_alone_and_the_forged_report_hides_it():
    assert evaluate(FirmwareFlag, Attacker()) is S
    clean = FirmwareFlag()
    assert not clean.feature_works() and clean.reports_disabled()


def test_a_single_fuse_read_falls_to_one_fault_or_one_edit_and_the_chip_reports_it():
    factory = lambda: FuseGate(1)  # noqa: E731
    assert evaluate(factory, Attacker()) is B
    assert evaluate(factory, Attacker(glitches=1)) is V
    assert evaluate(factory, Attacker(edits=1)) is V


def test_hiding_a_fuse_bypass_costs_one_more_edit():
    factory = lambda: FuseGate(1)  # noqa: E731
    assert evaluate(factory, Attacker(edits=2)) is S
    assert evaluate(factory, Attacker(glitches=1, edits=1)) is S
    assert evaluate(factory, Attacker(glitches=5)) is V, "faults alone cannot touch the report"


def test_a_majority_of_three_fuses_needs_two_flipped_reads():
    factory = lambda: FuseGate(3)  # noqa: E731
    assert evaluate(factory, Attacker(glitches=1)) is B
    assert evaluate(factory, Attacker(edits=1)) is B
    assert evaluate(factory, Attacker(glitches=2)) is V
    assert evaluate(factory, Attacker(edits=2)) is V
    assert evaluate(factory, Attacker(glitches=1, edits=1)) is V
    assert evaluate(factory, Attacker(edits=3)) is S
    assert evaluate(factory, Attacker(glitches=2, edits=1)) is S


@pytest.mark.parametrize("copies", [1, 3, 5, 7])
def test_the_cost_of_a_fuse_bypass_grows_with_the_number_of_copies(copies: int):
    need = copies // 2 + 1
    factory = lambda: FuseGate(copies)  # noqa: E731
    assert lockout.minimum_budget(factory, "glitches") == need
    assert lockout.minimum_budget(factory, "edits") == need
    assert lockout.minimum_budget(factory, "edits", silent=True) == need + 1


def test_fuse_copies_must_be_a_positive_odd_number():
    for bad in (0, 2, 4, -1):
        with pytest.raises(ValueError):
            FuseGate(bad)


def test_a_fault_is_transient_and_a_reconnected_fuse_is_permanent():
    faulted = FuseGate(1)
    faulted.attack(Attacker(glitches=1))
    assert faulted.feature_works()
    faulted.reboot()
    assert not faulted.feature_works()
    edited = FuseGate(1)
    edited.attack(Attacker(edits=1))
    edited.reboot()
    assert edited.feature_works()


def test_one_edit_on_the_antifuse_rail_bricks_the_chip_and_two_hide_it():
    factory = lambda: AntifuseRail(1)  # noqa: E731
    assert evaluate(factory, Attacker(edits=1)) is B
    assert evaluate(factory, Attacker(edits=2)) is S
    assert evaluate(factory, Attacker(glitches=9)) is B, "faults cannot reconnect a cut rail"
    one = AntifuseRail(1)
    one.attack(Attacker(edits=1))
    assert one.bricked and not one.feature_works()


@pytest.mark.parametrize("reads", [1, 2, 3, 4])
def test_the_self_test_must_be_fooled_on_every_read_so_its_read_count_sets_the_fault_cost(reads: int):
    factory = lambda: AntifuseRail(reads)  # noqa: E731
    assert evaluate(factory, Attacker(edits=1, glitches=reads - 1)) is B
    assert evaluate(factory, Attacker(edits=1, glitches=reads)) is S
    assert evaluate(factory, Attacker(edits=2)) is S


def test_the_antifuse_self_test_needs_at_least_one_read():
    with pytest.raises(ValueError):
        AntifuseRail(0)


def test_software_cannot_forge_a_token_signature():
    for variant in ("unbound", "bound", "epoch"):
        assert evaluate(lambda v=variant: TokenGate(v), Attacker()) is B
        gate = TokenGate(variant)
        assert not gate.accepts(Token(0, bytes(64)))
        assert not gate.accepts(Token(1, bytes(range(64))))


def test_a_token_signed_by_another_authority_or_with_a_flipped_bit_is_rejected():
    gate = TokenGate("bound", Authority("legitimate"))
    good = Authority("legitimate").mint("bound", FEATURE, b"chip-A", 0)
    assert gate.accepts(good)
    flipped = Token(good.epoch, bytes([good.signature[0] ^ 1]) + good.signature[1:])
    assert not gate.accepts(flipped)
    assert not gate.accepts(Authority("someone-else").mint("bound", FEATURE, b"chip-A", 0))
    assert Authority("a").public_key != Authority("b").public_key


def test_an_unbound_token_enables_every_chip_and_a_bound_token_enables_one():
    other = Attacker(other_device_token=True)
    assert evaluate(lambda: TokenGate("unbound"), other) is V
    assert evaluate(lambda: TokenGate("bound"), other) is B
    assert evaluate(lambda: TokenGate("epoch"), other) is B
    assert Authority.message("unbound", FEATURE, b"chip-A", 0) == Authority.message("unbound", FEATURE, b"chip-B", 5)
    assert Authority.message("bound", FEATURE, b"chip-A", 0) != Authority.message("bound", FEATURE, b"chip-B", 0)
    assert Authority.message("bound", FEATURE, b"chip-A", 0) == Authority.message("bound", FEATURE, b"chip-A", 9)
    assert Authority.message("epoch", FEATURE, b"chip-A", 0) != Authority.message("epoch", FEATURE, b"chip-A", 1)
    with pytest.raises(ValueError):
        Authority.message("imaginary", FEATURE, b"chip-A", 0)


def test_a_device_bound_token_stays_valid_after_a_downgrade_until_the_epoch_counter_advances():
    old = Attacker(old_token=True)
    assert evaluate(lambda: TokenGate("unbound"), old) is V
    assert evaluate(lambda: TokenGate("bound"), old) is V
    assert evaluate(lambda: TokenGate("epoch"), old) is B
    authority = Authority()
    gate = TokenGate("epoch", authority)
    assert not gate.accepts(authority.mint("epoch", FEATURE, b"chip-A", 0))
    assert gate.accepts(authority.mint("epoch", FEATURE, b"chip-A", 1))
    assert gate.accepts(authority.mint("epoch", FEATURE, b"chip-A", 2))


def test_a_leaked_signing_key_opens_every_token_design_but_not_a_puf_bound_feature():
    leaked = Attacker(manufacturer_secrets=True)
    for variant in ("unbound", "bound", "epoch"):
        assert evaluate(lambda v=variant: TokenGate(v), leaked) is V
    assert evaluate(PufBound, leaked) is B
    chip = PufBound()
    assert set(chip.database) == {BASE_FEATURE}, "the database holds keys only for purchased features"


def test_a_puf_bound_feature_falls_only_to_a_readout_of_the_puf_or_to_an_edit():
    assert evaluate(PufBound, Attacker()) is B
    assert evaluate(PufBound, Attacker(other_device_token=True, old_token=True)) is B
    assert evaluate(PufBound, Attacker(puf_readout=True)) is V
    assert evaluate(PufBound, Attacker(puf_readout=True, edits=1)) is S
    assert evaluate(PufBound, Attacker(edits=1)) is V


def test_a_puf_bound_key_is_device_specific_and_the_purchased_feature_unlocks():
    chip, other = PufBound(seed=1), PufBound(seed=2)
    assert chip.accepts(BASE_FEATURE, chip.database[BASE_FEATURE])
    assert chip.accepts(FEATURE, chip.invasive_key(FEATURE))
    assert not chip.accepts(FEATURE, other.invasive_key(FEATURE))
    assert not chip.accepts(BASE_FEATURE, other.database[BASE_FEATURE])
    assert not chip.accepts(FEATURE, chip.database[BASE_FEATURE])
    assert not chip.accepts(FEATURE, bytes(16))
    assert not chip.accepts(FEATURE, None)


def test_a_single_decision_gate_falls_to_one_fault_and_redundancy_raises_the_price():
    for factory in (lambda: TokenGate("epoch"), PufBound):
        assert evaluate(factory, Attacker(glitches=1)) is V
    assert evaluate(lambda: TokenGate("epoch", decision_reads=3), Attacker(glitches=1)) is B
    assert evaluate(lambda: TokenGate("epoch", decision_reads=3), Attacker(glitches=2)) is V
    assert evaluate(lambda: PufBound(decision_reads=3), Attacker(glitches=1)) is B
    assert evaluate(lambda: PufBound(decision_reads=3), Attacker(glitches=2)) is V


def test_a_separate_die_blocks_every_budget_and_only_the_wrong_die_fails():
    everything = Attacker(
        glitches=50, edits=50, other_device_token=True, old_token=True, manufacturer_secrets=True, puf_readout=True
    )
    assert evaluate(SeparateDie, everything) is B
    assert evaluate(lambda: SeparateDie(blocks_present=131), everything) is B
    assert evaluate(lambda: SeparateDie(blocks_present=132), Attacker()) is S, "a die that has the circuits is no lockout"


def test_the_matrix_has_a_row_per_mechanism_and_a_cell_per_attacker():
    cells = lockout.matrix()
    assert set(cells) == set(lockout.MECHANISMS)
    assert all(set(row) == set(lockout.PROFILES) for row in cells.values())
    assert all(outcome is S for outcome in cells["firmware_flag"].values())
    assert all(outcome is B for outcome in cells["separate_die"].values())
    for name, row in cells.items():
        if name not in ("firmware_flag", "separate_die"):
            assert row["soft"] is B, name
            assert any(outcome is not B for outcome in row.values()), name


def test_only_a_circuit_edit_ever_hides_a_bypass_from_the_chips_own_report():
    for name, factory in lockout.MECHANISMS.items():
        if name == "firmware_flag":
            continue
        for glitches, other, old, mfr, puf in itertools.product(range(4), *[(False, True)] * 4):
            outcome = evaluate(factory, Attacker(glitches, 0, other, old, mfr, puf))
            assert outcome is not S, (name, glitches, other, old, mfr, puf)


def test_more_budget_never_makes_an_attack_worse():
    flags = ("other_device_token", "old_token", "manufacturer_secrets", "puf_readout")
    for name, factory in lockout.MECHANISMS.items():
        for glitches, edits, *booleans in itertools.product(range(4), range(4), *[(False, True)] * 4):
            base = dict(zip(flags, booleans))
            here = RANK[evaluate(factory, Attacker(glitches, edits, **base))]
            bigger = [Attacker(glitches + 1, edits, **base), Attacker(glitches, edits + 1, **base)]
            bigger += [Attacker(glitches, edits, **{**base, f: True}) for f in flags if not base[f]]
            for attacker in bigger:
                assert RANK[evaluate(factory, attacker)] >= here, (name, attacker)


def test_the_budget_search_agrees_with_running_the_attacker_directly():
    for name, factory in lockout.MECHANISMS.items():
        for resource in ("glitches", "edits"):
            for silent in (False, True):
                found = lockout.minimum_budget(factory, resource, silent)
                wanted = (S,) if silent else (V, S)
                if found is None:
                    assert all(evaluate(factory, Attacker(**{resource: n})) not in wanted for n in range(9)), name
                    continue
                assert evaluate(factory, Attacker(**{resource: found})) in wanted, (name, resource, silent)
                if found > 0:
                    assert evaluate(factory, Attacker(**{resource: found - 1})) not in wanted, (name, resource, silent)


def test_budget_search_rejects_unknown_resources_and_respects_its_limit():
    with pytest.raises(ValueError):
        lockout.minimum_budget(FirmwareFlag, "tokens")
    assert lockout.minimum_budget(lambda: FuseGate(7), "edits", limit=3) is None
    assert lockout.minimum_budget(lambda: FuseGate(7), "edits", limit=4) == 4
    assert lockout.minimum_budget(SeparateDie, "edits") is None
    assert lockout.budgets(FirmwareFlag) == lockout.Budgets(0, 0, 0)


def test_random_fault_bypass_probability_matches_exact_binomial_values():
    p = 0.05
    assert lockout.glitch_bypass_probability(3, p, 1) == pytest.approx(3 * p * p * (1 - p) + p**3, rel=1e-12)
    assert lockout.glitch_bypass_probability(1, p, 1) == pytest.approx(p)
    assert lockout.glitch_bypass_probability(3, p, 0) == 0.0
    assert lockout.glitch_bypass_probability(3, 0.0, 1000) == 0.0
    assert lockout.glitch_bypass_probability(3, 1.0, 1) == 1.0
    per_boot = lockout.glitch_bypass_probability(5, p, 1)
    assert lockout.glitch_bypass_probability(5, p, 40) == pytest.approx(1 - (1 - per_boot) ** 40)
    for bad in ((2, 0.1, 1), (0, 0.1, 1), (3, -0.1, 1), (3, 1.1, 1), (3, 0.1, -1)):
        with pytest.raises(ValueError):
            lockout.glitch_bypass_probability(*bad)


def test_more_redundant_reads_lower_the_chance_that_random_faults_flip_the_vote():
    for p in (0.01, 0.05, 0.2):
        values = [lockout.glitch_bypass_probability(k, p, 1) for k in (1, 3, 5, 7, 9)]
        assert all(a > c for a, c in zip(values, values[1:]))


def test_simulated_random_fault_boots_agree_with_the_closed_form():
    gate = FuseGate(3)
    n = 50_000
    estimate = monte_carlo_probability(lambda r: gate.boot_with_random_faults(r, 0.05), n, make_rng(2024))
    exact = lockout.glitch_bypass_probability(3, 0.05, 1)
    assert abs(estimate.p - exact) < 4 * math.sqrt(exact * (1 - exact) / n)
    no_faults = monte_carlo_probability(lambda r: gate.boot_with_random_faults(r, 0.0), 2_000, make_rng(1))
    assert no_faults.p == 0.0


def test_puf_keys_fail_more_often_under_stress_and_with_less_error_correction():
    devices = 120
    nominal = {r: lockout.puf_field_failures(r, 25.0, 0.0, devices, seed=3).failures for r in (3, 5, 9, 15)}
    assert nominal[3] > nominal[5] > nominal[9] >= nominal[15]
    assert nominal[15] == 0
    harsh = lockout.puf_field_failures(5, 125.0, 20.0, devices, seed=3).failures
    assert harsh > 3 * nominal[5]


def test_the_field_failure_bound_is_a_clopper_pearson_upper_end():
    zero = lockout.FieldFailure(15, 25.0, 0.0, 0, 300)
    assert zero.upper_bound == pytest.approx(1 - 0.025 ** (1 / 300), rel=1e-6)
    some = lockout.FieldFailure(5, 25.0, 0.0, 30, 300)
    assert some.upper_bound > 30 / 300
    with pytest.raises(ValueError):
        lockout.puf_field_failures(15, 25.0, 0.0, 0)


def test_the_formatting_helpers_use_one_letter_per_outcome_and_a_dash_for_none():
    assert [lockout.format_cell(o) for o in (B, V, S)] == ["B", "V", "S"]
    assert lockout.budget_text(None) == "-" and lockout.budget_text(3) == "3"
