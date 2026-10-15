"""Mechanism-by-attacker results computed by running the chip models in ``mechanisms``.

Nothing in the matrix is typed in: every cell is the outcome of ``evaluate`` on a fresh chip. The summary gives
the smallest glitch and edit budgets that make each feature work, found by search.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from scipy import stats as sps

from aisecurity_rfds.common.puf import FuzzyExtractor, SimulatedPuf
from aisecurity_rfds.common.rng import derive_seed, make_rng
from aisecurity_rfds.common.stats import clopper_pearson

from .mechanisms import (
    AntifuseRail,
    Attacker,
    FirmwareFlag,
    FuseGate,
    Lockout,
    Outcome,
    PufBound,
    SeparateDie,
    TokenGate,
    evaluate,
)

MECHANISMS: dict[str, Callable[[], Lockout]] = {
    "firmware_flag": FirmwareFlag,
    "fuse_x1": lambda: FuseGate(1),
    "fuse_x3": lambda: FuseGate(3),
    "antifuse_rail_st1": lambda: AntifuseRail(1),
    "antifuse_rail_st3": lambda: AntifuseRail(3),
    "token_unbound": lambda: TokenGate("unbound"),
    "token_bound": lambda: TokenGate("bound"),
    "token_bound_epoch": lambda: TokenGate("epoch"),
    "puf_bound": PufBound,
    "separate_die": SeparateDie,
}

PROFILES: dict[str, Attacker] = {
    "soft": Attacker(),
    "g1": Attacker(glitches=1),
    "g3": Attacker(glitches=3),
    "e1": Attacker(edits=1),
    "e2": Attacker(edits=2),
    "e3": Attacker(edits=3),
    "e1g1": Attacker(edits=1, glitches=1),
    "e1g3": Attacker(edits=1, glitches=3),
    "tok_other": Attacker(other_device_token=True),
    "tok_old": Attacker(old_token=True),
    "mfr": Attacker(manufacturer_secrets=True),
    "puf": Attacker(puf_readout=True),
}

CODES = {Outcome.BLOCKED: "B", Outcome.VISIBLE: "V", Outcome.SILENT: "S"}


def matrix() -> dict[str, dict[str, Outcome]]:
    """Outcome of every attacker profile against every mechanism."""
    return {m: {p: evaluate(factory, a) for p, a in PROFILES.items()} for m, factory in MECHANISMS.items()}


def minimum_budget(factory: Callable[[], Lockout], resource: str, silent: bool = False, limit: int = 8) -> int | None:
    """Smallest budget of ``resource`` ('glitches' or 'edits') that makes the feature work, or None up to ``limit``.

    With ``silent=True`` the chip's own report must also still say the feature is off.
    """
    if resource not in ("glitches", "edits"):
        raise ValueError("resource must be 'glitches' or 'edits'")
    for budget in range(limit + 1):
        outcome = evaluate(factory, Attacker(**{resource: budget}))
        if outcome is Outcome.SILENT or (outcome is Outcome.VISIBLE and not silent):
            return budget
    return None


@dataclass(frozen=True)
class Budgets:
    glitches_to_work: int | None
    edits_to_work: int | None
    edits_to_hide: int | None


def budgets(factory: Callable[[], Lockout]) -> Budgets:
    return Budgets(
        minimum_budget(factory, "glitches"), minimum_budget(factory, "edits"), minimum_budget(factory, "edits", True)
    )


def glitch_bypass_probability(copies: int, p: float, boots: int) -> float:
    """P(at least one of ``boots`` boots flips a ``copies``-read majority vote when each read is corrupted w.p. p)."""
    if copies < 1 or copies % 2 == 0 or not 0.0 <= p <= 1.0 or boots < 0:
        raise ValueError("copies must be odd and positive, p in [0, 1], boots non-negative")
    per_boot = float(sps.binom.sf(copies // 2, copies, p))
    return 1.0 - (1.0 - per_boot) ** boots


@dataclass(frozen=True)
class FieldFailure:
    repetition: int
    temperature_c: float
    age_years: float
    failures: int
    devices: int

    @property
    def upper_bound(self) -> float:
        """Upper end of the two-sided 95% Clopper-Pearson interval for the per-device failure probability."""
        return clopper_pearson(self.failures, self.devices)[1]


def puf_field_failures(
    repetition: int, temperature_c: float, age_years: float, devices: int, seed: int = 0, key_bits: int = 128
) -> FieldFailure:
    """Devices (of ``devices`` simulated) whose key cannot be reproduced after enrolment at 25 C, age zero.

    This is the cost of the PUF-bound mechanism the RFD calls "PUF stability requirements": a legitimately
    purchased feature that stops unlocking. Both the enrolment and the later measurement are noisy.
    """
    if devices < 1:
        raise ValueError("devices must be positive")
    extractor = FuzzyExtractor(key_bits, repetition)
    failures = 0
    for d in range(devices):
        puf = SimulatedPuf(derive_seed("rfd037.field", seed, d), n_cells=extractor.n_cells_required)
        rng = make_rng(derive_seed("rfd037.field-rng", seed, d))
        _, helper = extractor.enroll(puf.measure(rng=rng), rng=rng)
        _, ok = extractor.reproduce(puf.measure(temperature_c=temperature_c, age_years=age_years, rng=rng), helper)
        failures += 0 if ok else 1
    return FieldFailure(repetition, temperature_c, age_years, failures, devices)


def format_cell(outcome: Outcome) -> str:
    return CODES[outcome]


def budget_text(value: int | None) -> str:
    return "-" if value is None else str(value)
