"""The tier-change model checker: upgrade, downgrade, die replacement and token replay (RFD 037, open questions 3 and 5)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aisecurity_rfds.rfd_037_secure_chip_binning import rma

BARE, BOUND, REVOKE, NOSYNC, PER_TIER = rma.DESIGNS


def reachable(design: rma.Design) -> set[rma.World]:
    """Every reachable world, by a search written separately from ``rma.explore``."""
    seen = {rma.initial_world(design)}
    stack = list(seen)
    while stack:
        world = stack.pop()
        for _, nxt in rma.successors(design, world):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def step(design: rma.Design, world: rma.World, label: str) -> rma.World:
    options = dict(rma.successors(design, world))
    assert label in options, f"{label} is not available; options: {sorted(options)}"
    return options[label]


def test_bare_tokens_let_another_chips_token_raise_a_low_tier_chip_in_one_step():
    result = rma.explore(BARE)
    assert result.violations["I1"].trace == ("replay(0,token 1:2@0)",)
    assert result.holds("I2")
    assert "runs at tier 2" in result.violations["I1"].detail


def test_device_binding_alone_still_lets_a_token_from_before_a_downgrade_come_back():
    result = rma.explore(BOUND)
    assert result.violations["I1"].trace == ("downgrade(1,0)", "replay(1,token 1:2@0)")
    assert result.holds("I2")


def test_binding_with_revocation_and_a_synced_registry_has_no_violation_anywhere_in_the_model():
    result = rma.explore(REVOKE)
    assert result.complete
    assert result.holds("I1") and result.holds("I2")
    assert result.states > 1000
    assert result.states == len(reachable(REVOKE))


def test_a_registry_that_is_not_updated_breaks_both_invariants_at_the_first_sale():
    result = rma.explore(NOSYNC)
    assert result.violations["I1"].trace == ("upgrade(0,1)",)
    assert result.violations["I2"].trace == ("upgrade(0,1)",)


def test_per_tier_dies_hold_the_invariants_but_cannot_be_upgraded_in_place():
    result = rma.explore(PER_TIER)
    assert result.complete and result.holds("I1") and result.holds("I2")
    assert not rma.in_place_upgrade_possible(PER_TIER)
    assert all(rma.in_place_upgrade_possible(d) for d in (BARE, BOUND, REVOKE, NOSYNC))
    world = rma.initial_world(PER_TIER)
    assert all(not label.startswith(("upgrade", "downgrade")) for label, _ in rma.successors(PER_TIER, world))


def test_the_physical_cap_overrides_a_token_that_is_otherwise_valid():
    low_die = rma.fresh_chip(PER_TIER, 0)
    assert low_die.cap == 0
    assert not rma.accepts(PER_TIER, low_die, 0, 1, 2, 0)
    binned = rma.fresh_chip(BARE, 0)
    assert binned.cap == 2
    assert rma.accepts(BARE, binned, 0, 1, 2, 0)
    assert rma.fresh_chip(PER_TIER, 2).cap == 2


def test_acceptance_rules_follow_the_design_switches():
    chip = replace(rma.fresh_chip(REVOKE, 0), epoch=1)
    assert not rma.accepts(REVOKE, chip, 0, 1, 2, 0), "wrong chip"
    assert not rma.accepts(REVOKE, chip, 0, 0, 2, 0), "older epoch"
    assert rma.accepts(REVOKE, chip, 0, 0, 2, 1)
    assert rma.accepts(REVOKE, chip, 0, 0, 2, 2)
    assert rma.accepts(BOUND, chip, 0, 0, 2, 0), "no revocation: an older epoch is fine"
    assert not rma.accepts(BOUND, chip, 0, 1, 2, 0)
    assert rma.accepts(BARE, chip, 0, 1, 2, 0)


@pytest.mark.parametrize("design", rma.DESIGNS, ids=lambda d: d.name)
def test_the_search_finds_the_same_shortest_violations_as_enumerating_every_action_sequence(design: rma.Design):
    brute = rma.brute_force_depths(design, 4)
    result = rma.explore(design)
    for invariant in ("I1", "I2"):
        violation = result.violations.get(invariant)
        if violation is not None and len(violation.trace) <= 4:
            assert brute[invariant] == len(violation.trace)
        else:
            assert invariant not in brute


@pytest.mark.parametrize("design", rma.DESIGNS, ids=lambda d: d.name)
def test_an_independent_search_visits_the_same_number_of_worlds(design: rma.Design):
    assert rma.explore(design).states == len(reachable(design))


def test_a_full_epoch_counter_removes_in_place_downgrade_but_not_replacement():
    full = replace(rma.fresh_chip(REVOKE, 2), epoch=rma.MAX_EPOCH)
    world = (rma.fresh_chip(REVOKE, 0), full)
    labels = {label for label, _ in rma.successors(REVOKE, world)}
    assert not any(label.startswith("downgrade(1") for label in labels)
    assert {"swap(1,0)", "swap(1,1)"} <= labels
    assert any(label.startswith("downgrade") for label, _ in rma.successors(REVOKE, (replace(full, epoch=0), full)))


def test_a_swap_gives_the_slot_a_fresh_die_and_drops_the_old_chips_tokens():
    world = step(REVOKE, rma.initial_world(REVOKE), "swap(1,0)")
    assert world[1] == rma.Chip(cap=2, entitled=0, registry=0, active=0, epoch=0, issued=frozenset())
    per_tier = step(PER_TIER, rma.initial_world(PER_TIER), "swap(1,0)")
    assert per_tier[1].cap == 0
    assert per_tier[0] == rma.initial_world(PER_TIER)[0]


@pytest.mark.parametrize("design", [BARE, BOUND])
def test_a_replay_only_fires_when_it_raises_the_tier_and_only_changes_the_target_chip(design: rma.Design):
    count = 0
    for world in reachable(design):
        for label, nxt in rma.successors(design, world):
            if not label.startswith("replay"):
                continue
            count += 1
            target = int(label[len("replay(")])
            assert nxt[target].active > world[target].active
            assert replace(nxt[target], active=world[target].active) == world[target]
            assert all(nxt[i] == world[i] for i in range(len(world)) if i != target)
    assert count > 0


@pytest.mark.parametrize("design", [REVOKE, NOSYNC, PER_TIER], ids=lambda d: d.name)
def test_replay_is_never_possible_once_old_tokens_die_or_the_silicon_cannot_reach_the_tier(design: rma.Design):
    for world in reachable(design):
        assert not any(label.startswith("replay") for label, _ in rma.successors(design, world))


def test_a_leased_unlock_can_be_sold_and_ended_repeatedly_without_breaking_either_invariant():
    design = REVOKE
    world = rma.initial_world(design)
    world = step(design, world, "upgrade(0,2)")
    assert world[0].registry == 2 and world[0].active == 2
    world = step(design, world, "downgrade(0,0)")
    assert world[0] == replace(world[0], registry=0, active=0, entitled=0, epoch=1)
    assert not any(label.startswith("replay(0") for label, _ in rma.successors(design, world))
    world = step(design, world, "upgrade(0,2)")
    assert (2, 0) in world[0].issued and (2, 1) in world[0].issued
    world = step(design, world, "downgrade(0,1)")
    assert world[0].epoch == 2
    assert not any(label.startswith("replay(0") for label, _ in rma.successors(design, world))
    assert rma.violations(world) == []
    assert not any(label.startswith("downgrade(0") for label, _ in rma.successors(design, step(design, world, "upgrade(0,2)")))


def test_invariant_checks_flag_exactly_the_stated_conditions():
    fine = rma.Chip(2, 1, 1, 1, 0, frozenset())
    assert rma.violations((fine,)) == []
    assert [i for i, _ in rma.violations((replace(fine, active=2),))] == ["I1"]
    assert [i for i, _ in rma.violations((replace(fine, registry=2, active=1),))] == ["I2"]
    assert sorted(i for i, _ in rma.violations((replace(fine, registry=0, entitled=2, active=1),))) == ["I1", "I2"]
    assert rma.violations((replace(fine, entitled=1, registry=1, active=0),)) == []


def test_a_depth_bounded_search_is_marked_incomplete_and_sees_fewer_worlds():
    full = rma.explore(REVOKE)
    shallow = rma.explore(REVOKE, depth=2)
    assert not shallow.complete and shallow.depth == 2
    assert 1 < shallow.states < full.states
    assert full.complete and full.depth > 2


def test_the_search_refuses_to_start_from_a_world_that_already_breaks_an_invariant(monkeypatch):
    monkeypatch.setattr(rma, "initial_world", lambda design: (rma.Chip(0, 0, 0, 1, 0, frozenset()),))
    with pytest.raises(ValueError, match="initial world"):
        rma.explore(BARE)
