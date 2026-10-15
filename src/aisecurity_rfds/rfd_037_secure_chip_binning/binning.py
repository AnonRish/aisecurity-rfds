"""Yield and cost model: one binned die versus dies built for each tier (RFD 037, open question 1).

Everything here is a model. The numbers it produces follow from the parameters in ``Die``, ``Process`` and
``Tier``, which are illustrative assumptions, not measurements of any product or any fab. Costs are in wafer
equivalents (one wafer start costs 1.0) so that no price has to be invented.

The defect model is the standard clustered ("negative binomial") one: all regions of one die share a defect
density multiplier ``g`` drawn from Gamma(alpha, 1/alpha), which has mean 1, and given ``g`` each region's defect
count is Poisson. A die is made of one *uncore* region that must be defect free (anything fatal in it kills the
die) and ``n_blocks`` identical blocks (the repeated compute units) that may each be defective and fused off.

Binning: one die design is made, tested, and sold in the highest tier whose block count it still reaches. Dies
that reach a tier's count are therefore fused down into lower tiers when demand needs them. ``latent_top_share``
measures what the RFD worries about: the share of chips sold *below* the top tier whose silicon could have run
at the top tier. It is computed for the allocation that minimises it (best case for governance).

Two designs avoid that latent capability. ``plan_separate`` gives every lower tier its own die with only as many
blocks as that tier sells, plus optional spare blocks for yield (spares are latent capability too, so they must
stay below the next tier's block count: ``max_safe_spares``). ``plan_hybrid`` also sells top dies that fell short
of the top block count as lower tiers, which is safe because their missing blocks are defective, not fused.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import optimize, special, stats


class InfeasibleError(ValueError):
    """No sellable die exists under the given parameters (for example at an extreme defect density)."""


@dataclass(frozen=True)
class Die:
    """A die: one uncore region plus ``n_blocks`` identical repeated blocks. Areas in mm^2."""

    uncore_mm2: float
    block_mm2: float
    n_blocks: int

    def __post_init__(self) -> None:
        if self.uncore_mm2 <= 0 or self.block_mm2 <= 0:
            raise ValueError("areas must be positive")
        if self.n_blocks < 1:
            raise ValueError("a die needs at least one block")

    @property
    def area_mm2(self) -> float:
        return self.uncore_mm2 + self.n_blocks * self.block_mm2


@dataclass(frozen=True)
class Process:
    """Mean defect density per cm^2 and the clustering parameter (smaller alpha means stronger clustering)."""

    d0_per_cm2: float
    alpha: float = 3.0
    wafer_mm: float = 300.0

    def __post_init__(self) -> None:
        if self.d0_per_cm2 < 0:
            raise ValueError("defect density cannot be negative")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.wafer_mm <= 0:
            raise ValueError("wafer diameter must be positive")


@dataclass(frozen=True)
class Tier:
    """A product tier: it sells chips with exactly ``blocks`` enabled blocks and ``share`` of the demand."""

    name: str
    blocks: int
    share: float


# Assumptions, chosen only so that the die resembles a large accelerator die; not taken from any datasheet.
ILLUSTRATIVE_DIE = Die(uncore_mm2=224.0, block_mm2=4.0, n_blocks=144)
ILLUSTRATIVE_TIERS = (Tier("top", 132, 0.2), Tier("mid", 114, 0.3), Tier("low", 96, 0.5))


def check_tiers(tiers: Sequence[Tier], die: Die | None = None) -> None:
    """Tiers must be listed from the most to the least capable, with positive shares summing to one."""
    if not tiers:
        raise ValueError("at least one tier is required")
    if any(t.share <= 0 for t in tiers):
        raise ValueError("tier shares must be positive")
    if abs(sum(t.share for t in tiers) - 1.0) > 1e-9:
        raise ValueError("tier shares must sum to 1")
    if any(a.blocks <= b.blocks for a, b in zip(tiers, tiers[1:])):
        raise ValueError("tiers must be listed with strictly decreasing block counts")
    if tiers[-1].blocks < 1:
        raise ValueError("a tier needs at least one block")
    if die is not None and tiers[0].blocks > die.n_blocks:
        raise ValueError("the top tier needs more blocks than the die has")


def dies_per_wafer(area_mm2: float, wafer_mm: float = 300.0) -> int:
    """Gross dies per wafer: wafer area over die area, minus the edge loss term (a standard approximation)."""
    if area_mm2 <= 0:
        raise ValueError("die area must be positive")
    n = math.pi * (wafer_mm / 2.0) ** 2 / area_mm2 - math.pi * wafer_mm / math.sqrt(2.0 * area_mm2)
    return max(0, math.floor(n))


@lru_cache(maxsize=64)
def _laguerre(alpha: float, nodes: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Gauss-Laguerre nodes and weights for the weight x^(alpha-1) e^-x, weights divided by Gamma(alpha)."""
    x, w = special.roots_genlaguerre(nodes, alpha - 1.0)
    return x, w / special.gamma(alpha)


def region_survival(area_mm2: float, proc: Process) -> float:
    """P(a region of this area has no defect) under the clustered model: (1 + lambda/alpha)^-alpha."""
    lam = proc.d0_per_cm2 * area_mm2 / 100.0
    return float((1.0 + lam / proc.alpha) ** (-proc.alpha))


@lru_cache(maxsize=4096)
def good_block_pmf(die: Die, proc: Process) -> np.ndarray:
    """``pmf[k]`` = P(the uncore is good and exactly ``k`` blocks are good), for k = 0..n_blocks.

    Computed by integrating the binomial over the shared density multiplier with Gauss-Laguerre quadrature whose
    nodes are scaled to the die's total defect exponent, so that the integrand for "every region good" is constant
    and its integral is exact; the tests compare the rest with closed forms, an independent integration and Monte
    Carlo. The entries sum to the uncore survival probability, not to one. The array is cached and read-only.
    """
    x, w = _laguerre(proc.alpha)
    scale = 1.0 + proc.d0_per_cm2 * die.area_mm2 / 100.0 / proc.alpha
    g = x / (proc.alpha * scale)
    w = w * scale ** (-proc.alpha) * np.exp(x * (1.0 - 1.0 / scale))
    q = np.exp(-(proc.d0_per_cm2 * die.block_mm2 / 100.0) * g)
    u = np.exp(-(proc.d0_per_cm2 * die.uncore_mm2 / 100.0) * g)
    k = np.arange(die.n_blocks + 1)
    pmf = (w * u) @ stats.binom.pmf(k[None, :], die.n_blocks, q[:, None])
    pmf.flags.writeable = False
    return pmf


def sample_good_blocks(die: Die, proc: Process, n: int, rng: np.random.Generator) -> np.ndarray:
    """Monte Carlo of the same model: good-block counts of ``n`` dies, with -1 where the uncore is dead."""
    g = rng.gamma(proc.alpha, 1.0 / proc.alpha, size=n)
    uncore_ok = rng.poisson(proc.d0_per_cm2 * die.uncore_mm2 / 100.0 * g) == 0
    defects = rng.poisson(proc.d0_per_cm2 * die.block_mm2 / 100.0 * g[:, None], size=(n, die.n_blocks))
    good = (defects == 0).sum(axis=1)
    return np.where(uncore_ok, good, -1)


@dataclass(frozen=True)
class Binned:
    """Result of ``plan_binning``. ``supply[j]`` is dies per wafer that reach tier j's block count."""

    units_per_wafer: float
    cost_per_unit: float
    binding_tier: str
    latent_top_share: float
    supply: tuple[float, ...]


def plan_binning(die: Die, proc: Process, tiers: Sequence[Tier], wafer_cost: float = 1.0) -> Binned:
    """Most units per wafer a single binned die can supply in the demanded tier mix, and what that costs.

    Tier j can only use dies that reach ``blocks_j``, and tiers 1..j together can only use dies that reach the
    smallest of their counts, so units <= supply_j / (share_1 + ... + share_j) for every j (Hall's condition for
    this nested structure; the tests compare it with a linear program). Every die that can be sold is sold.
    """
    check_tiers(tiers, die)
    pmf = good_block_pmf(die, proc)
    n_dies = dies_per_wafer(die.area_mm2, proc.wafer_mm)
    supply = tuple(float(n_dies * pmf[t.blocks :].sum()) for t in tiers)
    cum_share = np.cumsum([t.share for t in tiers])
    limits = [s / c for s, c in zip(supply, cum_share)]
    units = min(limits)
    if units <= 0:
        raise InfeasibleError("no die reaches the lowest tier at this defect density")
    binding = tiers[int(np.argmin(limits))].name
    lower_demand = (1.0 - tiers[0].share) * units
    if lower_demand <= 0:
        latent = 0.0
    else:
        non_top_supply = supply[-1] - supply[0]
        latent = max(0.0, lower_demand - non_top_supply) / lower_demand
    return Binned(units, wafer_cost / units, binding, latent, supply)


@dataclass(frozen=True)
class TierDie:
    """One tier's own die in a separate-dies plan."""

    name: str
    blocks_present: int
    area_mm2: float
    dies_per_wafer: int
    sellable_fraction: float
    cost_per_unit: float


@dataclass(frozen=True)
class Separate:
    """Result of ``plan_separate``: the demand-weighted cost and each tier's die."""

    cost_per_unit: float
    per_tier: tuple[TierDie, ...]


def max_safe_spares(tiers: Sequence[Tier]) -> tuple[int, ...]:
    """Most spare blocks each lower tier's own die may carry while staying below the next tier up.

    Entry 0 (the top tier) is unconstrained and reported as the maximum int for clarity.
    """
    check_tiers(tiers)
    return (2**31 - 1, *(a.blocks - b.blocks - 1 for a, b in zip(tiers, tiers[1:])))


def plan_separate(
    die: Die, proc: Process, tiers: Sequence[Tier], spares: int | Sequence[int] = 0, wafer_cost: float = 1.0
) -> Separate:
    """Cost of giving every lower tier its own die, with the top tier keeping the full ``die``.

    ``spares`` is the number of spare blocks on each lower tier's die (one number for all, or one per lower tier).
    A lower tier's die may not carry as many blocks as the tier above sells, or the chip could be raised to it.
    """
    check_tiers(tiers, die)
    lower = len(tiers) - 1
    per_tier_spares = [spares] * lower if isinstance(spares, int) else list(spares)
    if len(per_tier_spares) != lower:
        raise ValueError("one spares value per lower tier is required")
    if any(s < 0 for s in per_tier_spares):
        raise ValueError("spares cannot be negative")
    limits = max_safe_spares(tiers)[1:]
    if any(s > m for s, m in zip(per_tier_spares, limits)):
        raise ValueError("spare blocks would let a lower-tier die reach the tier above it")
    rows = []
    total = 0.0
    for idx, tier in enumerate(tiers):
        present = die.n_blocks if idx == 0 else tier.blocks + per_tier_spares[idx - 1]
        tier_die = Die(die.uncore_mm2, die.block_mm2, present)
        n_dies = dies_per_wafer(tier_die.area_mm2, proc.wafer_mm)
        ok = float(good_block_pmf(tier_die, proc)[tier.blocks :].sum())
        sellable = n_dies * ok
        if sellable <= 0:
            raise InfeasibleError(f"no sellable {tier.name} die at this defect density")
        cost = wafer_cost / sellable
        rows.append(TierDie(tier.name, present, tier_die.area_mm2, n_dies, ok, cost))
        total += tier.share * cost
    return Separate(total, tuple(rows))


def best_safe_spares(die: Die, proc: Process, tiers: Sequence[Tier]) -> tuple[int, ...]:
    """For each lower tier, the spare-block count within the safe limit that minimises that tier's die cost."""
    check_tiers(tiers, die)
    best_per_tier = []
    for tier, limit in zip(tiers[1:], max_safe_spares(tiers)[1:]):
        best, best_cost = 0, math.inf
        for spare in range(limit + 1):
            tier_die = Die(die.uncore_mm2, die.block_mm2, tier.blocks + spare)
            sellable = dies_per_wafer(tier_die.area_mm2, proc.wafer_mm) * float(
                good_block_pmf(tier_die, proc)[tier.blocks :].sum()
            )
            cost = math.inf if sellable <= 0 else 1.0 / sellable
            if cost < best_cost:
                best, best_cost = spare, cost
        best_per_tier.append(best)
    return tuple(best_per_tier)


@dataclass(frozen=True)
class Hybrid:
    """Result of ``plan_hybrid``, per unit of total demand.

    ``top_wafers`` are started to meet the top tier's demand; their dies that fall short of the top block count
    are sold lower (they cannot be raised: the blocks are defective, not fused). ``separate_wafers`` covers the
    rest of each lower tier with its own die. ``surplus`` is by-product dies for which there was no demand.
    """

    cost_per_unit: float
    top_wafers: float
    separate_wafers: tuple[float, ...]
    surplus: float


def plan_hybrid(
    die: Die, proc: Process, tiers: Sequence[Tier], spares: int | Sequence[int] = 0, wafer_cost: float = 1.0
) -> Hybrid:
    """Salvage without latent capability: sell defective top dies lower, and cover the rest with smaller dies.

    A die that has fewer than ``tiers[0].blocks`` good blocks can never run at the top tier, so selling it lower is
    safe. A die that reaches the top count is only ever sold at the top. Lower-tier demand that the by-products
    cannot meet is served by each tier's own die, built as in ``plan_separate``.
    """
    separate = plan_separate(die, proc, tiers, spares, wafer_cost)
    pmf = good_block_pmf(die, proc)
    n_dies = dies_per_wafer(die.area_mm2, proc.wafer_mm)
    top_dies_per_wafer = n_dies * float(pmf[tiers[0].blocks :].sum())
    if top_dies_per_wafer <= 0:
        raise InfeasibleError("no die reaches the top tier at this defect density")
    top_wafers = tiers[0].share / top_dies_per_wafer
    pool = 0.0
    extra = []
    for idx in range(1, len(tiers)):
        band = n_dies * float(pmf[tiers[idx].blocks : tiers[idx - 1].blocks].sum())
        pool += top_wafers * band
        used = min(pool, tiers[idx].share)
        pool -= used
        shortfall = tiers[idx].share - used
        extra.append(shortfall * separate.per_tier[idx].cost_per_unit / wafer_cost)
    total = (top_wafers + sum(extra)) * wafer_cost
    return Hybrid(total, top_wafers, tuple(extra), pool)


def total_cost_per_unit(silicon_per_unit: float, nre_wafer_equivalents: float, volume_units: float) -> float:
    """Silicon cost per unit plus a fixed one-off cost spread over the units sold."""
    if volume_units <= 0 or nre_wafer_equivalents < 0:
        raise ValueError("volume must be positive and NRE non-negative")
    return silicon_per_unit + nre_wafer_equivalents / volume_units


def breakeven_volume(extra_nre: float, silicon_binned: float, silicon_secure: float) -> float:
    """Volume above which the secure design is cheaper overall; infinity when it is dearer in silicon."""
    saving = silicon_binned - silicon_secure
    if saving <= 0:
        return math.inf
    return extra_nre / saving


def secure_cost(die: Die, proc: Process, tiers: Sequence[Tier], design: str = "hybrid") -> float:
    """Silicon cost per unit of a design with no latent top capability, using the cheapest safe spares."""
    spares = best_safe_spares(die, proc, tiers)
    if design == "separate":
        return plan_separate(die, proc, tiers, spares).cost_per_unit
    if design == "hybrid":
        return plan_hybrid(die, proc, tiers, spares).cost_per_unit
    raise ValueError(f"unknown design {design!r}")


def silicon_premium(die: Die, proc: Process, tiers: Sequence[Tier], design: str = "hybrid") -> float:
    """(secure - binned) / binned silicon cost per unit; negative when the secure design is cheaper in silicon."""
    return secure_cost(die, proc, tiers, design) / plan_binning(die, proc, tiers).cost_per_unit - 1.0


def breakeven_defect_density(
    die: Die,
    tiers: Sequence[Tier],
    alpha: float = 3.0,
    design: str = "hybrid",
    lo: float = 0.01,
    hi: float = 20.0,
    points: int = 60,
) -> float | None:
    """Lowest defect density in [lo, hi] at which the silicon premium changes sign, or None if it never does."""

    def premium(d0: float) -> float:
        try:
            return silicon_premium(die, Process(d0, alpha), tiers, design)
        except InfeasibleError:
            return math.inf

    grid = [float(d) for d in np.geomspace(lo, hi, points)]
    values = [premium(d) for d in grid]
    for (a, fa), (b, fb) in zip(zip(grid, values), zip(grid[1:], values[1:])):
        if math.isfinite(fa) and math.isfinite(fb) and (fa < 0) != (fb < 0):
            return float(optimize.brentq(premium, a, b, xtol=1e-9))
    return None
