"""Bounded exhaustive exploration of the readout protocol, and counter-width arithmetic.

``explore`` runs the *real* ``ReadoutBlock`` / ``ReadoutVerifier`` / ``ResetAuthority`` code (with an ideal signature
scheme substituted for speed) through every sequence of adversary and world actions up to a depth, and checks
invariants after each action. It is bounded model checking of this implementation, not a proof about hardware: the
counters are tiny, the depth is small, and the signature scheme is assumed unforgeable.

Invariants checked on every accepted readout:
  * no over-credit: work credited never exceeds work actually done;
  * no retroactive erasure: credited work never decreases;
  * conservation: ``true work - credited work == work erased by glitches + work lost while stalled``;
  * a regression of a counter within an epoch is never accepted;
  * the epoch advances only through an authorised reset and the verifier stays in step with the block;
Invariants checked on refused or replayed inputs:
  * a replayed readout is never accepted twice;
  * forged, replayed and wrong-epoch tickets are refused by the block, which is left unchanged.

The exploration also *finds* what the protocol cannot prevent: accepted traces in which a mature window's work is
entirely hidden (a stall, or a glitch timed to regain the old value).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from .readout import IdealSigner, Readout, ReadoutBlock, ReadoutVerifier, ResetAuthority, ResetTicket, TicketRefused, ideal_verify

DEVICE = b"DEV-MC01"
OK, NA, DETECTED = "ok", "na", "detected"

ALPHABET_HIDING = ("work1", "work2", "read", "glitch_clear", "stall_toggle")
ALPHABET_PROTOCOL = (
    "work1",
    "read",
    "read_drop",
    "replay",
    "legit_reset",
    "forged_ticket",
    "replayed_ticket",
    "future_ticket",
    "glitch_clear",
    "stall_toggle",
)


class _World:
    """The protocol under exploration plus ghost bookkeeping that the verifier never sees."""

    def __init__(
        self,
        width_bits: int,
        max_work: int,
        block_cls: type[ReadoutBlock] = ReadoutBlock,
        verifier_cls: type[ReadoutVerifier] = ReadoutVerifier,
    ) -> None:
        self.authority = ResetAuthority(IdealSigner(b"authority-key"))
        device = IdealSigner(b"device-key")
        self.forger = IdealSigner(b"attacker-key")
        self.block = block_cls(DEVICE, device, self.authority.public_bytes, n_counters=1, width_bits=width_bits, verify_fn=ideal_verify)
        self.verifier = verifier_cls(
            DEVICE,
            device.public_bytes,
            self.authority.public_bytes,
            max_rate_per_tick=(max_work,),
            n_counters=1,
            width_bits=width_bits,
            verify_fn=ideal_verify,
        )
        self.true_work = 0
        self.lost = 0
        self.stalled_work = 0
        self.credited_prev = 0
        self.true_prev = 0
        self.resets = 0
        self.glitch_since_accept = False
        self.last_delivered: Readout | None = None
        self.last_ticket: ResetTicket | None = None
        self.violations: list[str] = []
        self.window_fraction: float | None = None  # hidden fraction of the window just accepted (mature windows only)
        self.accepted = False  # the last action delivered a readout the verifier accepted

    # --- snapshot / restore ---
    def snapshot(self) -> tuple:
        ghost = (self.true_work, self.lost, self.stalled_work, self.credited_prev, self.true_prev, self.resets, self.glitch_since_accept)
        return (self.block.snapshot(), self.verifier.snapshot(), self.authority.snapshot(), ghost, self.last_delivered, self.last_ticket)

    def restore(self, s: tuple) -> None:
        block, verifier, authority, ghost, self.last_delivered, self.last_ticket = s
        self.block.restore(block)
        self.verifier.restore(verifier)
        self.authority.restore(authority)
        (self.true_work, self.lost, self.stalled_work, self.credited_prev, self.true_prev, self.resets, self.glitch_since_accept) = ghost

    # --- delivery with invariant checks ---
    def _deliver(self, ro: Readout) -> tuple[str, str]:
        v = self.verifier
        prev_counter, was_closed, prev_epoch = v.last_counters[0], v.closed, v.epoch
        verdict = v.accept(ro)
        if not verdict.ok:
            return DETECTED, verdict.reason
        self.last_delivered = ro
        self.accepted = True
        credited = v.credited[0]
        if credited > self.true_work:
            self.violations.append("over-credit")
        if credited < self.credited_prev:
            self.violations.append("credit-decreased")
        if self.true_work - credited != self.lost + self.stalled_work:
            self.violations.append("conservation")
        if v.epoch + int(v.closed) != self.block.epoch:
            self.violations.append("epoch-desync")
        if self.block.epoch != self.resets:
            self.violations.append("unauthorised-epoch-change")
        if not was_closed and ro.epoch == prev_epoch and ro.counters[0] < prev_counter:
            self.violations.append("regression-accepted")
        window_true = self.true_work - self.true_prev
        self.window_fraction = None
        if window_true > 0 and self.credited_prev > 0:
            self.window_fraction = 1.0 - (credited - self.credited_prev) / window_true
        self.credited_prev, self.true_prev, self.glitch_since_accept = credited, self.true_work, False
        return OK, ""

    def _expect_refusal(self, ticket: ResetTicket) -> tuple[str, str]:
        before = self.block.snapshot()
        try:
            self.block.reset(ticket)
        except TicketRefused:
            if self.block.snapshot() != before:
                self.violations.append("block-changed-by-refused-ticket")
            return OK, ""
        self.violations.append("invalid-ticket-accepted")
        return OK, ""

    def apply(self, action: str) -> tuple[str, str]:
        self.violations = []
        self.window_fraction = None
        self.accepted = False
        if action.startswith("work"):
            w = int(action[4:])
            self.block.advance(1, [w])
            self.true_work += w
            if self.block.stalled:
                self.stalled_work += w
            return OK, ""
        if action == "read":
            self.block.advance(1, [0])  # readouts are separated in time: one idle tick, then the sample
            return self._deliver(self.block.read())
        if action == "read_drop":
            self.block.advance(1, [0])
            self.block.read()  # produced, never delivered
            return OK, ""
        if action == "replay":
            if self.last_delivered is None:
                return NA, ""
            kind, reason = self._deliver(self.last_delivered)
            if kind == OK:
                self.violations.append("replay-accepted")
            return kind, reason
        if action == "glitch_clear":
            self.lost += self.block.registers[0]
            self.block.glitch_clear_registers()
            self.glitch_since_accept = True
            return OK, ""
        if action == "stall_toggle":
            self.block.set_stalled(not self.block.stalled)
            return OK, ""
        if action == "legit_reset":
            self.block.advance(1, [0])
            ticket = self.authority.issue(DEVICE, self.block.epoch, self.block.ticks + 1000)
            closing = self.block.reset(ticket)
            self.last_ticket = ticket
            self.resets += 1
            return self._deliver(closing)
        if action == "forged_ticket":
            forged = ResetTicket(DEVICE, self.block.epoch, 777, 10**9, b"")
            return self._expect_refusal(ResetTicket(DEVICE, self.block.epoch, 777, 10**9, self.forger.sign(forged.digest())))
        if action == "replayed_ticket":
            if self.last_ticket is None:
                return NA, ""
            return self._expect_refusal(self.last_ticket)
        if action == "future_ticket":
            return self._expect_refusal(self.authority.issue(DEVICE, self.block.epoch + 1, self.block.ticks + 1000))
        raise ValueError(action)


@dataclass
class ExploreReport:
    alphabet: tuple[str, ...]
    depth: int
    transitions: int = 0
    accepted_readouts: int = 0
    mature_windows: int = 0  # accepted readouts that closed a window with work done and work already credited
    detected: Counter = field(default_factory=Counter)  # rejection reason -> traces ended by it
    violations: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    max_window_hidden_fraction: float = 0.0
    witness: tuple[str, ...] | None = None  # shortest accepted trace achieving ``max_window_hidden_fraction``


def explore(
    alphabet: Sequence[str] = ALPHABET_HIDING,
    depth: int = 6,
    *,
    width_bits: int = 8,
    max_work: int = 2,
    block_cls: type[ReadoutBlock] = ReadoutBlock,
    verifier_cls: type[ReadoutVerifier] = ReadoutVerifier,
) -> ExploreReport:
    """Depth-first search over every action sequence of length <= ``depth`` (traces end when the verifier rejects).

    ``block_cls`` / ``verifier_cls`` exist so tests can explore deliberately broken variants and confirm that the
    invariants really do fail for them.
    """
    world = _World(width_bits, max_work, block_cls, verifier_cls)
    report = ExploreReport(tuple(alphabet), depth)

    def dfs(path: tuple[str, ...]) -> None:
        if len(path) == depth:
            return
        snap = world.snapshot()
        for action in alphabet:
            kind, reason = world.apply(action)
            if kind == NA:
                world.restore(snap)
                continue
            report.transitions += 1
            here = (*path, action)
            for v in world.violations:
                report.violations.append((v, here))
            if kind == DETECTED:
                report.detected[reason] += 1
            else:
                report.accepted_readouts += int(world.accepted)
                if world.window_fraction is not None:
                    report.mature_windows += 1
                    wf = world.window_fraction
                    better = wf > report.max_window_hidden_fraction + 1e-12
                    shorter = abs(wf - report.max_window_hidden_fraction) <= 1e-12 and report.witness and len(here) < len(report.witness)
                    if better or shorter:
                        report.max_window_hidden_fraction, report.witness = wf, here
                dfs(here)
            world.restore(snap)

    dfs(())
    return report


# --- counter width -------------------------------------------------------------------------------------------


def bits_needed(max_count: int) -> int:
    """Smallest counter width (bits) that can hold ``max_count`` without wrapping."""
    return max(1, int(max_count).bit_length())


def seconds_to_wrap(width_bits: int, rate_per_second: float) -> float:
    return float(2**width_bits) / rate_per_second


def safe_gap(width_bits: int, max_rate_per_tick: int) -> int:
    """Longest interval (ticks) between readouts for which ``max_rate * gap < 2**width`` still holds."""
    return (2**width_bits - 1) // max_rate_per_tick


def aliased_work(width_bits: int, work_between_reads: int) -> int:
    """Work a modular-delta verifier cannot see: whole wraps of the counter between two readouts."""
    return (work_between_reads >> width_bits) << width_bits
