"""Signed readout protocol: forgery, replay, reordering, resets and tickets, wrap-around, and end-to-end scenarios."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from aisecurity_rfds.common.hashing import sha256
from aisecurity_rfds.common.signing import KeyPair
from aisecurity_rfds.rfd_035_perf_counter_integrity import device_sim, monotonic
from aisecurity_rfds.rfd_035_perf_counter_integrity.readout import (
    GENESIS,
    Readout,
    ReadoutBlock,
    ReadoutVerifier,
    ResetAuthority,
    ResetTicket,
    TicketRefused,
    protocol_overhead,
)

DEV = b"GPU-0001"


@pytest.fixture
def rig():
    authority = ResetAuthority(KeyPair.from_seed(sha256(b"test authority")))
    device = KeyPair.from_seed(sha256(b"test device"))
    attacker = KeyPair.from_seed(sha256(b"test attacker"))
    block = ReadoutBlock(DEV, device, authority.public_bytes)
    verifier = ReadoutVerifier(DEV, device.public_bytes, authority.public_bytes, max_rate_per_tick=(1000, 1000))
    return SimpleNamespace(authority=authority, device=device, attacker=attacker, block=block, verifier=verifier)


def step(rig, flops=100, mem=10):
    rig.block.advance(1, [flops, mem])
    return rig.block.read()


def test_honest_chain_is_accepted_and_credit_equals_work(rig):
    total_f = total_m = 0
    for i in range(1, 11):
        total_f, total_m = total_f + 10 * i, total_m + i
        ro = step(rig, 10 * i, i)
        assert rig.verifier.accept(ro).ok
    assert rig.verifier.credited == [total_f, total_m]
    assert rig.verifier.seq == 10 and rig.verifier.epoch == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epoch", 1),
        ("seq", 2),
        ("ticks", 99),
        ("closing", True),
        ("counters", (1, 1)),
        ("prev_hash", b"\x05" * 32),
        ("device_id", b"GPU-0002"),
    ],
)
def test_signature_covers_every_field_so_edits_in_transit_are_rejected(rig, field, value):
    ro = step(rig)
    forged = dataclasses.replace(ro, **{field: value})
    verdict = rig.verifier.accept(forged)
    assert not verdict.ok and verdict.reason in {"bad-signature", "wrong-device"}
    assert rig.verifier.accept(ro).ok  # the genuine readout still verifies: rejection left no state behind


def test_readout_signed_by_another_key_is_rejected(rig):
    impostor = ReadoutBlock(DEV, rig.attacker, rig.authority.public_bytes)
    impostor.advance(1, [5, 5])
    assert rig.verifier.accept(impostor.read()).reason == "bad-signature"


def test_replay_gap_and_time_regression_have_distinct_reasons(rig):
    first = step(rig)
    assert rig.verifier.accept(first).ok
    assert rig.verifier.accept(first).reason == "replay"
    step(rig)  # produced but never delivered
    third = step(rig)
    assert rig.verifier.accept(third).reason == "gap"
    # A second readout at the same tick is refused even when it chains correctly.
    block = ReadoutBlock(DEV, rig.device, rig.authority.public_bytes)
    verifier = ReadoutVerifier(DEV, rig.device.public_bytes, rig.authority.public_bytes, max_rate_per_tick=(1000, 1000))
    block.advance(1, [1, 1])
    assert verifier.accept(block.read()).ok
    assert verifier.accept(block.read()).reason == "time-regression"


def test_rolled_back_or_cloned_block_breaks_the_hash_chain(rig):
    assert rig.verifier.accept(step(rig)).ok
    rig.block.head = b"\x07" * 32  # a block restored from a different history
    assert rig.verifier.accept(step(rig)).reason == "chain-break"


def test_clearing_the_registers_is_rejected_as_an_implausible_delta(rig):
    for _ in range(3):
        assert rig.verifier.accept(step(rig)).ok
    before = (list(rig.verifier.credited), rig.verifier.seq, rig.verifier.head)
    rig.block.glitch_clear_registers()
    assert rig.verifier.accept(step(rig, 0, 0)).reason == "implausible-delta"
    assert (rig.verifier.credited, rig.verifier.seq, rig.verifier.head) == before


def test_impossible_upward_jumps_are_rejected_too(rig):
    assert rig.verifier.accept(step(rig)).ok
    rig.block.registers[0] += 10**9  # physical corruption far above the maximum rate
    assert rig.verifier.accept(step(rig)).reason == "implausible-delta"


def test_clock_gating_stall_is_accepted_by_the_protocol_but_caught_by_power():
    outcome = device_sim.simulate_device("clock-gating-stall", seed=0)
    assert outcome.first_rejection is None and outcome.intervals_accepted == 40
    assert outcome.power_flagged and outcome.hidden_fraction > 0.4


def test_glitch_timed_to_regain_the_old_value_is_accepted_documented_limitation(rig):
    assert rig.verifier.accept(step(rig, 10, 0)).ok  # counter 10 credited
    rig.block.advance(1, [10, 0])  # counter 20
    rig.block.glitch_clear_registers()  # the attacker clears it ...
    rig.block.advance(1, [10, 0])  # ... and 10 more units of real work regain the old value
    assert rig.verifier.accept(rig.block.read()).ok
    work_done_in_window = 20
    credited_in_window = rig.verifier.credited[0] - 10
    assert credited_in_window == 0 < work_done_in_window  # the whole window is invisible to the protocol


def test_legitimate_reset_loses_no_work_and_starts_a_new_epoch(rig):
    assert rig.verifier.accept(step(rig, 10, 1)).ok
    rig.block.advance(1, [7, 2])
    ticket = rig.authority.issue(DEV, rig.block.epoch, rig.block.ticks + 5)
    closing = rig.block.reset(ticket)
    assert closing.closing and closing.counters == (17, 3) and rig.block.registers == [0, 0]
    assert rig.verifier.accept(closing).ok and rig.verifier.closed
    assert rig.verifier.accept(step(rig, 5, 1)).ok
    assert rig.verifier.epoch == 1 and not rig.verifier.closed
    assert rig.verifier.credited == [22, 4]


@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        ("forged", "bad-ticket-signature"),
        ("other-device", "ticket-for-another-device"),
        ("future-epoch", "ticket-epoch-mismatch"),
        ("expired", "ticket-expired"),
    ],
)
def test_block_refuses_bad_tickets_and_is_left_unchanged(rig, kind, reason):
    rig.block.advance(3, [30, 3])
    epoch, ticks = rig.block.epoch, rig.block.ticks
    if kind == "forged":
        t = ResetTicket(DEV, epoch, 99, 10**9, b"")
        ticket = dataclasses.replace(t, signature=rig.attacker.sign(t.digest()))
    elif kind == "other-device":
        ticket = rig.authority.issue(b"GPU-0002", epoch, ticks + 10)
    elif kind == "future-epoch":
        ticket = rig.authority.issue(DEV, epoch + 1, ticks + 10)
    else:
        ticket = rig.authority.issue(DEV, epoch, ticks - 1)
    before = rig.block.snapshot()
    with pytest.raises(TicketRefused) as exc:
        rig.block.reset(ticket)
    assert exc.value.reason == reason
    assert rig.block.snapshot() == before


def test_used_ticket_cannot_be_replayed_in_a_later_epoch(rig):
    ticket = rig.authority.issue(DEV, 0, 10_000)
    rig.block.advance(1, [1, 1])
    assert rig.verifier.accept(rig.block.reset(ticket)).ok
    with pytest.raises(TicketRefused, match="ticket-epoch-mismatch"):
        rig.block.reset(ticket)


def test_a_hoarded_ticket_still_yields_a_closing_readout_with_the_true_final_counters(rig):
    ticket = rig.authority.issue(DEV, 0, 10_000)  # issued early, used late
    for _ in range(25):
        assert rig.verifier.accept(step(rig, 40, 4)).ok
    rig.block.advance(1, [40, 4])
    closing = rig.block.reset(ticket)
    assert closing.counters == (26 * 40, 26 * 4)
    assert rig.verifier.accept(closing).ok and rig.verifier.credited == [26 * 40, 26 * 4]


def test_verifier_independently_rejects_a_closing_readout_with_a_bad_ticket(rig):
    """Second line of defence: even a signed closing readout is refused if its ticket does not check out."""
    rig.block.advance(1, [5, 5])
    good_ro = rig.block.read()
    assert rig.verifier.accept(good_ro).ok

    def closing_with(ticket):
        unsigned = Readout(DEV, 0, 2, 2, True, (6, 6), rig.block.head, ticket, b"")
        return dataclasses.replace(unsigned, signature=rig.device.sign(unsigned.digest()))

    rig.block.advance(1, [1, 1])
    forged = ResetTicket(DEV, 0, 5, 10**9, b"")
    forged = dataclasses.replace(forged, signature=rig.attacker.sign(forged.digest()))
    assert rig.verifier.accept(closing_with(forged)).reason == "bad-ticket-signature"
    assert rig.verifier.accept(closing_with(None)).reason == "closing-without-ticket"
    wrong_epoch = rig.authority.issue(DEV, 4, 10**9)
    assert rig.verifier.accept(closing_with(wrong_epoch)).reason == "ticket-not-for-this-epoch"
    open_with_ticket = dataclasses.replace(closing_with(rig.authority.issue(DEV, 0, 10**9)), closing=False)
    open_with_ticket = dataclasses.replace(open_with_ticket, signature=rig.device.sign(open_with_ticket.digest()))
    assert rig.verifier.accept(open_with_ticket).reason == "ticket-on-open-readout"


def test_epoch_cannot_advance_without_a_closing_readout(rig):
    assert rig.verifier.accept(step(rig)).ok
    rig.block.epoch = 1  # state corruption in the block, no authorised reset
    assert rig.verifier.accept(step(rig)).reason == "epoch-jump"


def test_counters_after_a_reset_must_be_plausible_from_zero(rig):
    ticket = rig.authority.issue(DEV, 0, 10_000)
    rig.block.advance(1, [1, 1])
    assert rig.verifier.accept(rig.block.reset(ticket)).ok
    rig.block.advance(1, [1, 1])
    rig.block.registers[0] = 10**7  # corruption right after the reset
    assert rig.verifier.accept(rig.block.read()).reason == "implausible-delta"


def test_counters_wrap_correctly_within_the_safe_gap_and_alias_beyond_it():
    authority = ResetAuthority(KeyPair.from_seed(sha256(b"w authority")))
    device = KeyPair.from_seed(sha256(b"w device"))
    width, rate = 8, 20
    gap = monotonic.safe_gap(width, rate)
    assert gap == 12

    def fresh(**kw):
        b = ReadoutBlock(DEV, device, authority.public_bytes, n_counters=1, width_bits=width)
        v = ReadoutVerifier(DEV, device.public_bytes, authority.public_bytes, max_rate_per_tick=(rate,), n_counters=1, width_bits=width, **kw)
        return b, v

    b, v = fresh()
    b.advance(12, [240])
    assert v.accept(b.read()).ok and v.credited == [240]
    b.advance(1, [20])  # 240 + 20 = 260 wraps to 4, but the modular delta is still 20
    assert b.registers == [4]
    assert v.accept(b.read()).ok and v.credited == [260]
    # Beyond the safe gap a whole wrap is invisible: 260 units look like 4.
    b, v = fresh()
    b.advance(20, [260])
    assert v.accept(b.read()).ok and v.credited == [4]
    assert monotonic.aliased_work(width, 260) == 256
    # A verifier that enforces the safe gap refuses that late readout instead.
    b, v = fresh(max_gap_ticks=gap)
    b.advance(20, [260])
    assert v.accept(b.read()).reason == "stale"


def test_stale_readouts_are_flagged_when_a_gap_limit_is_set(rig):
    limited = ReadoutVerifier(DEV, rig.device.public_bytes, rig.authority.public_bytes, max_rate_per_tick=(10**6, 10**6), max_gap_ticks=5)
    rig.block.advance(5, [1, 1])
    assert limited.accept(rig.block.read()).ok
    rig.block.advance(6, [1, 1])
    assert limited.accept(rig.block.read()).reason == "stale"


def test_malformed_counter_vectors_are_rejected(rig):
    ro = step(rig)
    short = dataclasses.replace(ro, counters=(1,))
    short = dataclasses.replace(short, signature=rig.device.sign(short.digest()))
    assert rig.verifier.accept(short).reason == "malformed"


EXPECTED = {
    "honest": "no attack",
    "legitimate-reset": "no attack",
    "ticket-hoarded-then-used": "no attack",
    "firmware-edits-counters": "caught by protocol (bad-signature)",
    "firmware-replays-old-readout": "caught by protocol (replay)",
    "firmware-drops-a-readout": "caught by protocol (gap)",
    "firmware-forges-reset-ticket": "refused by block (bad-ticket-signature)",
    "firmware-replays-used-ticket": "refused by block (ticket-epoch-mismatch)",
    "firmware-uses-expired-ticket": "refused by block (ticket-expired)",
    "glitch-clears-registers": "caught by protocol (implausible-delta)",
    "clock-gating-stall": "caught by power cross-check",
    "undercount-1pct": "MISSED",
    "undercount-5pct": "caught by power cross-check",
    "undercount-30pct": "caught by power cross-check",
}


def test_every_scenario_is_covered_by_an_expectation():
    assert set(EXPECTED) == set(device_sim.SCENARIOS)


@pytest.mark.parametrize("scenario", list(EXPECTED))
def test_scenario_outcome(scenario):
    outcome = device_sim.simulate_device(scenario, seed=0)
    assert outcome.verdict.startswith(EXPECTED[scenario]), outcome


def test_honest_like_scenarios_credit_exactly_the_true_work_across_seeds():
    for scenario in ("honest", "legitimate-reset", "ticket-hoarded-then-used"):
        for seed in range(8):
            o = device_sim.simulate_device(scenario, seed=seed)
            assert o.credited_flops == o.true_flops and o.intervals_accepted == 40
            assert o.first_rejection is None and o.block_refusal is None


def test_undercount_detection_rate_over_seeds_matches_the_analytic_formula():
    h, n = 0.02, 150
    flagged = sum(device_sim.simulate_device("undercount-1pct", seed=s, undercount=h).power_flagged for s in range(n))
    expected = 1.0 - device_sim.power_miss_probability(h, 40)
    assert 0.6 < expected < 0.95  # a regime where the formula is informative
    assert abs(flagged / n - expected) < 4 * np.sqrt(expected * (1 - expected) / n) + 0.02


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError):
        device_sim.simulate_device("not-a-scenario")


def test_protocol_overhead_is_plain_arithmetic():
    one = protocol_overhead(1, 1.0)
    assert one["readout_bytes"] == 8 + 3 * 8 + 1 + 2 * 8 + 32 + 64 == 145
    assert one["closing_readout_bytes"] == 145 + 8 + 3 * 8 + 64
    big = protocol_overhead(100_000, 2.0)
    assert big["bytes_per_second_total"] == pytest.approx(145 * 2 * 100_000)
    assert big["signature_verifications_per_second"] == 200_000
    assert protocol_overhead(1, 1.0, n_counters=4)["readout_bytes"] == 145 + 16


def test_genesis_hash_is_all_zero_and_first_readout_chains_to_it(rig):
    assert GENESIS == b"\x00" * 32
    assert step(rig).prev_hash == GENESIS


def test_simulation_is_deterministic_per_seed():
    assert device_sim.simulate_device("undercount-5pct", seed=3) == device_sim.simulate_device("undercount-5pct", seed=3)
    a = device_sim.simulate_device("honest", seed=1)
    b = device_sim.simulate_device("honest", seed=2)
    assert a.true_flops != b.true_flops


def test_the_protocol_consumes_no_randomness():
    """Two blocks fed the same work produce byte-identical readouts (Ed25519 signatures are deterministic)."""
    a = ReadoutBlock(DEV, KeyPair.from_seed(sha256(b"k")), b"x" * 32)
    b = ReadoutBlock(DEV, KeyPair.from_seed(sha256(b"k")), b"x" * 32)
    for blk in (a, b):
        blk.advance(1, [3, 4])
    assert a.read() == b.read()
