"""Prints every number the README quotes: ``python -m aisecurity_rfds.rfd_037_secure_chip_binning``."""

from __future__ import annotations

from collections.abc import Sequence

from aisecurity_rfds.common.rng import derive_seed, make_rng
from aisecurity_rfds.common.stats import monte_carlo_probability

from . import audit, binning, lockout, rma
from .mechanisms import FuseGate

D0_SWEEP = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
GAPS = (18, 12, 6, 3)
LOT = 100_000


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    print(line(headers))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print(line(row))
    print()


def milli(cost: float) -> str:
    """Cost per chip in thousandths of a wafer."""
    return f"{1000.0 * cost:.2f}"


def percent(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}%"


def tiers_with_gap(gap: int) -> tuple[binning.Tier, ...]:
    return (
        binning.Tier("top", 132, 0.2),
        binning.Tier("mid", 132 - gap, 0.3),
        binning.Tier("low", 132 - 2 * gap, 0.5),
    )


def section_binning() -> None:
    die, tiers = binning.ILLUSTRATIVE_DIE, binning.ILLUSTRATIVE_TIERS
    print("== 1. Separate dies versus binning (open question 1) ==")
    print(
        f"Assumed die: {die.uncore_mm2:.0f} mm2 uncore + {die.n_blocks} blocks of {die.block_mm2:.0f} mm2"
        f" = {die.area_mm2:.0f} mm2, {binning.dies_per_wafer(die.area_mm2)} dies per 300 mm wafer."
    )
    print(
        "Assumed tiers (blocks, demand): "
        + ", ".join(f"{t.name} {t.blocks} {100 * t.share:.0f}%" for t in tiers)
        + ". Clustering alpha 3. Costs are thousandths of a wafer per chip."
    )
    proc = binning.Process(1.0)
    pmf = binning.good_block_pmf(die, proc)
    dies = binning.sample_good_blocks(die, proc, 40_000, make_rng(derive_seed("rfd037", "yield-check")))
    print(
        f"Model check at D0 1.0: uncore survival closed form {binning.region_survival(die.uncore_mm2, proc):.4f},"
        f" quadrature {pmf.sum():.4f}, Monte Carlo {(dies >= 0).mean():.4f}."
    )
    hard = binning.Process(4.0)
    pmf_hard = binning.good_block_pmf(die, hard)
    dies_hard = binning.sample_good_blocks(die, hard, 40_000, make_rng(derive_seed("rfd037", "yield-check-hard")))
    print(
        f"Model check at D0 4.0: P(top-capable) quadrature {pmf_hard[tiers[0].blocks:].sum():.4f},"
        f" Monte Carlo {(dies_hard >= tiers[0].blocks).mean():.4f}."
    )
    print()

    rows = []
    for d0 in D0_SWEEP:
        p = binning.Process(d0)
        binned = binning.plan_binning(die, p, tiers)
        spares = binning.best_safe_spares(die, p, tiers)
        separate = binning.plan_separate(die, p, tiers, spares)
        hybrid = binning.plan_hybrid(die, p, tiers, spares)
        rows.append(
            [
                f"{d0:g}",
                f"{binned.units_per_wafer:.1f}",
                percent(binned.latent_top_share),
                milli(binned.cost_per_unit),
                f"{spares[0]}, {spares[1]}",
                milli(separate.cost_per_unit),
                milli(hybrid.cost_per_unit),
                f"{100.0 * (hybrid.cost_per_unit / binned.cost_per_unit - 1.0):+.1f}%",
            ]
        )
    print("Binned die versus separate dies (best safe spares for mid, low) and the hybrid, by defect density D0 per cm2:")
    table(
        ["D0", "binned chips/wafer", "latent top share", "binned cost", "spares (mid, low)", "separate cost", "hybrid cost", "hybrid premium"],
        rows,
    )

    rows = []
    for gap in GAPS:
        tr = tiers_with_gap(gap)
        premiums = [
            binning.silicon_premium(die, binning.Process(d0), tr, "hybrid") for d0 in (0.1, 1.0, 4.0)
        ]
        crossing = binning.breakeven_defect_density(die, tr)
        rows.append(
            [
                str(gap),
                f"{binning.max_safe_spares(tr)[1]}",
                *(f"{100.0 * x:+.1f}%" for x in premiums),
                "none up to 20" if crossing is None else f"{crossing:.2f}",
            ]
        )
    print("Hybrid premium over binning when the tiers are closer together (mid and low are gap and 2 x gap blocks below top):")
    table(["tier gap (blocks)", "safe spares", "premium at D0 0.1", "at D0 1", "at D0 4", "break-even D0"], rows)

    rows = []
    for alpha in (0.5, 1.0, 3.0, 100.0):
        p = binning.Process(1.0, alpha)
        binned = binning.plan_binning(die, p, tiers)
        rows.append(
            [
                f"{alpha:g}",
                f"{binning.region_survival(die.uncore_mm2, p):.3f}",
                percent(binned.latent_top_share, 2),
                f"{100.0 * binning.silicon_premium(die, p, tiers, 'hybrid'):+.1f}%",
            ]
        )
    print("Sensitivity to defect clustering at D0 1 (small alpha is strongly clustered, large alpha is close to Poisson):")
    table(["alpha", "uncore survival", "latent top share", "hybrid premium"], rows)

    p = binning.Process(0.1)
    binned_cost = binning.plan_binning(die, p, tiers).cost_per_unit
    hybrid_cost = binning.secure_cost(die, p, tiers, "hybrid")
    rows = []
    for nre in (1_000, 10_000, 100_000):
        breakeven = binning.breakeven_volume(nre, binned_cost, hybrid_cost)
        totals = [
            binning.total_cost_per_unit(hybrid_cost, nre, volume) / binned_cost - 1.0
            for volume in (100_000, 1_000_000, 10_000_000)
        ]
        rows.append([f"{nre:,}", f"{breakeven:,.0f}", *(f"{100.0 * x:+.1f}%" for x in totals)])
    print("At D0 0.1, extra one-off cost (NRE, in wafer equivalents) of the two extra designs, spread over volume:")
    table(["extra NRE", "break-even chips", "premium at 100,000", "at 1,000,000", "at 10,000,000"], rows)


def section_lockout() -> None:
    print("== 2. Lockout mechanisms against attacker budgets (RFD mechanisms 1 to 4) ==")
    print("B = feature stays off, V = feature works and the chip's report says so, S = works and the report says off.")
    print(
        "Attackers: soft = privileged software only; gN = N fault injections; eN = N circuit edits;"
        " e1g1 = one of each; tok_other = a token minted for another chip; tok_old = this chip's token from before"
        " a downgrade; mfr = the manufacturer's signing key and database; puf = reads the chip's PUF response."
    )
    print()
    cells = lockout.matrix()
    profiles = list(lockout.PROFILES)
    table(
        ["mechanism", *profiles],
        [[name, *(lockout.format_cell(row[p]) for p in profiles)] for name, row in cells.items()],
    )

    rows = []
    for name, factory in lockout.MECHANISMS.items():
        b = lockout.budgets(factory)
        rows.append(
            [
                name,
                lockout.budget_text(b.glitches_to_work),
                lockout.budget_text(b.edits_to_work),
                lockout.budget_text(b.edits_to_hide),
            ]
        )
    print("Smallest budget that works (a dash means none up to 8):")
    table(["mechanism", "fault injections to work", "edits to work", "edits to hide"], rows)

    rows = []
    for reads in (1, 3, 5, 7):
        rows.append(
            [
                str(reads),
                f"{lockout.glitch_bypass_probability(reads, 0.01, 1):.2e}",
                f"{lockout.glitch_bypass_probability(reads, 0.01, 1000):.4f}",
                f"{lockout.glitch_bypass_probability(reads, 0.05, 1):.2e}",
                f"{lockout.glitch_bypass_probability(reads, 0.05, 1000):.4f}",
            ]
        )
    print("Redundant fuse reads against random faults (each read corrupted with probability p per boot):")
    table(["reads", "p 0.01, one boot", "p 0.01, 1000 boots", "p 0.05, one boot", "p 0.05, 1000 boots"], rows)
    gate = FuseGate(3)
    estimate = monte_carlo_probability(
        lambda r: gate.boot_with_random_faults(r, 0.05), 50_000, make_rng(derive_seed("rfd037", "glitch-mc"))
    )
    print(
        f"Check, 3 reads at p 0.05: closed form {lockout.glitch_bypass_probability(3, 0.05, 1):.5f},"
        f" Monte Carlo {estimate.p:.5f} (95% interval {estimate.lo:.5f} to {estimate.hi:.5f}).\n"
    )

    conditions = ((25.0, 0.0), (85.0, 5.0), (125.0, 20.0))
    rows = []
    for repetition in (3, 5, 9, 15):
        row = [str(repetition)]
        for temperature, age in conditions:
            result = lockout.puf_field_failures(repetition, temperature, age, 300, seed=1)
            row.append(f"{result.failures} of {result.devices} (<= {percent(result.upper_bound)})")
        rows.append(row)
    print("PUF-bound key that fails to reproduce after enrolment at 25 C (simulated devices; bound is the upper end of a two-sided 95% Clopper-Pearson interval):")
    table(["repetition", "25 C, new", "85 C, 5 years", "125 C, 20 years"], rows)


def section_rma() -> None:
    print("== 3. Tier changes after sale (open questions 3 and 5) ==")
    rows = []
    traces = []
    for design in rma.DESIGNS:
        result = rma.explore(design)
        cells = []
        for invariant in ("I1", "I2"):
            violation = result.violations.get(invariant)
            cells.append("holds" if violation is None else f"broken in {len(violation.trace)}")
            if violation is not None:
                traces.append(f"{design.name} / {invariant}: " + " -> ".join(violation.trace))
        rows.append(
            [
                design.name,
                f"{result.states:,}",
                "yes" if result.complete else "no",
                "yes" if rma.in_place_upgrade_possible(design) else "no",
                *cells,
            ]
        )
    print("Exhaustive search of every reachable state (two chips, three tiers, epoch counter of two):")
    table(["design", "states", "all explored", "in-place upgrade", "I1 runs <= registry", "I2 registry = entitlement"], rows)
    print("Shortest counterexamples:")
    for trace in traces:
        print(f"  {trace}")
    print()


def section_audit() -> None:
    print("== 4. Auditing that restricted circuits are absent (open question 2) ==")
    print(f"Lot of {LOT:,} chips; 'detect' is the assumed chance that inspecting one cheated chip reveals it.")
    detects = (1.0, 0.9, 0.5, 0.2)
    rows = []
    for cheated in (1000, 100, 10):
        row = [f"{cheated:,}", percent(cheated / LOT, 2)]
        for d in detects:
            needed = audit.inspections_needed(LOT, cheated, 0.95, d)
            row.append("none" if needed is None else f"{needed:,}")
        row.append(percent(audit.destroyed_fraction(LOT, audit.inspections_needed(LOT, cheated, 0.95, 1.0)), 2))
        rows.append(row)
    print("Chips to inspect to find cheating with 95% confidence:")
    table(
        ["cheated chips", "share of lot", "detect 1.0", "detect 0.9", "detect 0.5", "detect 0.2", "lot destroyed at 1.0"],
        rows,
    )

    rows = []
    for inspected in (300, 3000):
        rows.append(
            [
                f"{inspected:,}",
                percent(audit.destroyed_fraction(LOT, inspected), 1),
                f"{audit.evasion_budget(LOT, inspected, 0.05):,}",
                f"{audit.evasion_budget(LOT, inspected, 0.5):,}",
            ]
        )
    print("Cheated chips that go unnoticed (perfect detection per inspected chip):")
    table(["inspected", "lot destroyed", "escape with probability >= 5%", "escape with probability >= 50%"], rows)

    once = audit.miss_probability(LOT, 100, 300, 1.0)
    print(
        f"Cheating 100 chips per lot with 300 inspected: missed with probability {once:.4f} in one lot,"
        f" {audit.repeated_miss_probability(LOT, 100, 300, 1.0, 10):.4f} across 10 lots."
    )
    lot = audit.Lot(2000, frozenset(range(20)), "reference-design")
    estimate = audit.estimate_detection(lot, 100, 0.6, 2000, make_rng(derive_seed("rfd037", "audit-mc")))
    print(
        f"Design-file comparison passes a cheating lot that declares the reference design: {audit.design_files_match('reference-design', lot)}."
    )
    print(
        f"Check, 20 of 2,000 cheated, 100 inspected, detect 0.6: closed form {1.0 - audit.miss_probability(2000, 20, 100, 0.6):.4f},"
        f" Monte Carlo {estimate.p:.4f} (95% interval {estimate.lo:.4f} to {estimate.hi:.4f})."
    )


def main() -> None:
    print("RFD 037, Secure Chip Binning and Feature Lockout: results of models, not measurements of any chip or fab.")
    print()
    section_binning()
    section_lockout()
    section_rma()
    section_audit()


if __name__ == "__main__":
    main()
