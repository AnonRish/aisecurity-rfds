"""The yield and cost model: closed forms, quadrature, Monte Carlo, linear programs and the designs (RFD 037, open question 1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, optimize, special, stats

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.rfd_037_secure_chip_binning import binning as b

DIE, TIERS = b.ILLUSTRATIVE_DIE, b.ILLUSTRATIVE_TIERS


def tiers_with_gap(gap: int) -> tuple[b.Tier, ...]:
    return (b.Tier("top", 132, 0.2), b.Tier("mid", 132 - gap, 0.3), b.Tier("low", 132 - 2 * gap, 0.5))


def tiers_with_shares(gap: int, shares: tuple[float, float, float]) -> tuple[b.Tier, ...]:
    return (b.Tier("top", 132, shares[0]), b.Tier("mid", 132 - gap, shares[1]), b.Tier("low", 132 - 2 * gap, shares[2]))


def band_supply(die: b.Die, proc: b.Process, tiers) -> list[float]:
    """Dies per wafer whose good-block count lies in [blocks_i, blocks_(i-1)): band 0 is top-capable."""
    pmf = b.good_block_pmf(die, proc)
    n_dies = b.dies_per_wafer(die.area_mm2, proc.wafer_mm)
    edges = [die.n_blocks + 1] + [t.blocks for t in tiers]
    return [float(n_dies * pmf[edges[i + 1] : edges[i]].sum()) for i in range(len(tiers))]


def binning_lp(die: b.Die, proc: b.Process, tiers, fixed_units: float | None = None):
    """Transportation LP for binning. Variables x[band, tier] (a die of a band can serve its own tier or lower) and U.

    Maximises units per wafer U, or with ``fixed_units`` minimises the top-capable dies sold below the top tier.
    """
    n = len(tiers)
    supply = band_supply(die, proc, tiers)
    pairs = [(band, tier) for band in range(n) for tier in range(band, n)]
    index = {pair: i for i, pair in enumerate(pairs)}
    n_vars = len(pairs) + 1
    a_eq, b_eq = [], []
    for tier in range(n):
        row = np.zeros(n_vars)
        for band in range(tier + 1):
            row[index[(band, tier)]] = 1.0
        if fixed_units is None:
            row[-1] = -tiers[tier].share
            b_eq.append(0.0)
        else:
            b_eq.append(tiers[tier].share * fixed_units)
        a_eq.append(row)
    a_ub, b_ub = [], []
    for band in range(n):
        row = np.zeros(n_vars)
        for tier in range(band, n):
            row[index[(band, tier)]] = 1.0
        a_ub.append(row)
        b_ub.append(supply[band])
    cost = np.zeros(n_vars)
    if fixed_units is None:
        cost[-1] = -1.0
    else:
        for tier in range(1, n):
            cost[index[(0, tier)]] = 1.0
    result = optimize.linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    assert result.status == 0, result.message
    return result


def random_instance(seed: int):
    rng = make_rng(seed)
    n_tiers = int(rng.integers(2, 5))
    blocks = sorted((int(x) for x in rng.choice(np.arange(60, 141), size=n_tiers, replace=False)), reverse=True)
    shares = rng.dirichlet(np.ones(n_tiers) * 2.0)
    shares = np.round(shares, 6)
    shares[-1] = 1.0 - shares[:-1].sum()
    tiers = tuple(b.Tier(f"t{i}", blocks[i], float(shares[i])) for i in range(n_tiers))
    proc = b.Process(float(rng.uniform(0.05, 6.0)), float(rng.uniform(0.5, 5.0)))
    return DIE, proc, tiers


def test_dies_per_wafer_matches_hand_computation_and_falls_with_area():
    # pi * 150^2 / 100 - pi * 300 / sqrt(200) = 706.86 - 66.64 = 640.2
    assert b.dies_per_wafer(100.0) == 640
    # 88.36 - 23.56 = 64.8
    assert b.dies_per_wafer(800.0) == 64
    assert b.dies_per_wafer(1_000_000.0) == 0
    counts = [b.dies_per_wafer(a) for a in (50.0, 100.0, 200.0, 400.0, 800.0)]
    assert counts == sorted(counts, reverse=True)
    assert b.dies_per_wafer(800.0, wafer_mm=450.0) > b.dies_per_wafer(800.0)


def test_region_survival_is_the_negative_binomial_and_beats_poisson_when_clustered():
    proc = b.Process(1.0, 3.0)
    assert b.region_survival(100.0, proc) == pytest.approx((4 / 3) ** -3, rel=1e-12)
    poisson = math.exp(-1.0)
    assert b.region_survival(100.0, proc) > poisson
    assert b.region_survival(100.0, b.Process(1.0, 1e7)) == pytest.approx(poisson, rel=1e-5)
    assert b.region_survival(100.0, b.Process(1.0, 0.5)) > b.region_survival(100.0, proc)


@pytest.mark.parametrize("alpha", [0.3, 0.5, 1.0, 3.0, 10.0, 100.0])
def test_quadrature_reproduces_the_closed_form_survival_of_the_whole_die_and_of_the_uncore(alpha: float):
    for d0 in (0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0):
        proc = b.Process(d0, alpha)
        pmf = b.good_block_pmf(DIE, proc)
        assert pmf[-1] == pytest.approx(b.region_survival(DIE.area_mm2, proc), abs=1e-12)
        assert pmf.sum() == pytest.approx(b.region_survival(DIE.uncore_mm2, proc), abs=1e-12)
        assert np.all(pmf >= 0.0)


@pytest.mark.parametrize("alpha, d0", [(3.0, 4.0), (0.5, 4.0), (1.0, 8.0), (10.0, 4.0)])
def test_every_entry_of_the_pmf_agrees_with_an_independent_integration(alpha: float, d0: float):
    """Adaptive integration over ln(g) of the gamma density times the binomial, using no quadrature rule."""
    proc = b.Process(d0, alpha)
    pmf = b.good_block_pmf(DIE, proc)
    lam_block = d0 * DIE.block_mm2 / 100.0
    lam_uncore = d0 * DIE.uncore_mm2 / 100.0

    def integrand(s: float, k: int) -> float:
        g = math.exp(s)
        log_density = alpha * math.log(alpha) + alpha * s - alpha * g - special.gammaln(alpha)
        return math.exp(log_density - lam_uncore * g) * float(stats.binom.pmf(k, DIE.n_blocks, math.exp(-lam_block * g)))

    edges = [-60.0 / alpha, -30.0 / alpha, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
    for k in (60, 100, 120, 132, 140, 143):
        exact = sum(
            integrate.quad(integrand, lo, hi, args=(k,), epsabs=1e-16, epsrel=1e-12, limit=400)[0]
            for lo, hi in zip(edges, edges[1:])
        )
        assert pmf[k] == pytest.approx(exact, abs=1e-12, rel=1e-8)


def test_the_pmf_agrees_with_a_monte_carlo_of_the_same_defect_model(rng: np.random.Generator):
    proc = b.Process(4.0, 3.0)
    n = 40_000
    pmf = b.good_block_pmf(DIE, proc)
    sample = b.sample_good_blocks(DIE, proc, n, rng)
    dead = float(np.mean(sample < 0))
    expected_dead = 1.0 - pmf.sum()
    assert abs(dead - expected_dead) < 4 * math.sqrt(expected_dead * (1 - expected_dead) / n)
    for blocks in (132, 114, 96):
        p = float(pmf[blocks:].sum())
        estimate = float(np.mean(sample >= blocks))
        assert abs(estimate - p) < 4 * math.sqrt(p * (1 - p) / n) + 1e-4


def test_with_no_defects_every_die_is_perfect_and_every_lower_chip_is_top_capable():
    proc = b.Process(0.0)
    pmf = b.good_block_pmf(DIE, proc)
    assert pmf[-1] == pytest.approx(1.0, abs=1e-12)
    binned = b.plan_binning(DIE, proc, TIERS)
    assert binned.units_per_wafer == pytest.approx(b.dies_per_wafer(DIE.area_mm2))
    assert binned.latent_top_share == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(8))
def test_binning_units_per_wafer_equal_a_linear_program_over_all_allocations(seed: int):
    die, proc, tiers = random_instance(seed)
    planned = b.plan_binning(die, proc, tiers)
    lp = binning_lp(die, proc, tiers)
    assert planned.units_per_wafer == pytest.approx(-lp.fun, rel=1e-7)
    limits = [s / c for s, c in zip(planned.supply, np.cumsum([t.share for t in tiers]))]
    assert planned.binding_tier == tiers[int(np.argmin(limits))].name


@pytest.mark.parametrize("seed", range(8))
def test_latent_top_share_is_the_least_any_allocation_can_achieve(seed: int):
    die, proc, tiers = random_instance(100 + seed)
    planned = b.plan_binning(die, proc, tiers)
    lp = binning_lp(die, proc, tiers, fixed_units=planned.units_per_wafer * (1.0 - 1e-9))
    lower_demand = (1.0 - tiers[0].share) * planned.units_per_wafer
    assert planned.latent_top_share * lower_demand == pytest.approx(lp.fun, abs=1e-6 * max(1.0, lower_demand))


def test_latent_share_is_total_at_low_defect_density_and_falls_as_defects_rise():
    shares, units = [], []
    for d0 in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0):
        binned = b.plan_binning(DIE, b.Process(d0), TIERS)
        shares.append(binned.latent_top_share)
        units.append(binned.units_per_wafer)
    assert shares[0] > 0.9999 and shares[3] > 0.999
    assert 0.9 < shares[-1] < 0.95
    assert all(a >= c - 1e-12 for a, c in zip(shares, shares[1:]))
    assert all(a > c for a, c in zip(units, units[1:]))


def test_when_the_top_tier_is_the_bottleneck_no_top_capable_die_is_sold_lower():
    tiers = (b.Tier("top", 132, 0.99), b.Tier("mid", 114, 0.005), b.Tier("low", 96, 0.005))
    binned = b.plan_binning(DIE, b.Process(4.0), tiers)
    assert binned.binding_tier == "top"
    assert binned.latent_top_share == 0.0
    assert binned.units_per_wafer == pytest.approx(binned.supply[0] / 0.99)


def test_max_safe_spares_and_plan_separate_refuse_a_die_that_reaches_the_tier_above():
    assert b.max_safe_spares(TIERS)[1:] == (17, 17)
    plan = b.plan_separate(DIE, b.Process(0.5), TIERS, spares=17)
    assert [t.blocks_present for t in plan.per_tier] == [144, 131, 113]
    with pytest.raises(ValueError, match="reach the tier above"):
        b.plan_separate(DIE, b.Process(0.5), TIERS, spares=18)
    with pytest.raises(ValueError, match="one spares value"):
        b.plan_separate(DIE, b.Process(0.5), TIERS, spares=[1, 2, 3])
    with pytest.raises(ValueError, match="negative"):
        b.plan_separate(DIE, b.Process(0.5), TIERS, spares=[-1, 0])


@pytest.mark.parametrize("d0", [0.05, 0.5, 2.0, 4.0])
def test_a_lower_tier_die_never_has_as_many_blocks_as_the_tier_above_sells(d0: float):
    proc = b.Process(d0)
    spares = b.best_safe_spares(DIE, proc, TIERS)
    plan = b.plan_separate(DIE, proc, TIERS, spares)
    for above, below in zip(TIERS, plan.per_tier[1:]):
        assert below.blocks_present < above.blocks


def test_best_safe_spares_is_the_cheapest_choice_by_brute_force():
    proc = b.Process(0.5)
    best = b.best_safe_spares(DIE, proc, TIERS)
    for position in (0, 1):
        costs = []
        for spare in range(18):
            spares = [0, 0]
            spares[position] = spare
            costs.append(b.plan_separate(DIE, proc, TIERS, spares).per_tier[position + 1].cost_per_unit)
        assert best[position] == int(np.argmin(costs))
    assert best[0] > 0 and best[1] > 0, "some spares always pay for themselves at this defect density"


def hybrid_lp_cost(die: b.Die, proc: b.Process, tiers, spares) -> float:
    """Cheapest way to meet demand: top wafers fixed by top demand, by-product dies placed optimally, rest separate."""
    n = len(tiers)
    separate = b.plan_separate(die, proc, tiers, spares)
    supply = band_supply(die, proc, tiers)
    top_wafers = tiers[0].share / supply[0]
    lower = range(1, n)
    pairs = [(band, tier) for band in lower for tier in range(band, n)]
    index = {pair: i for i, pair in enumerate(pairs)}
    n_vars = len(pairs) + len(lower)
    cost = np.zeros(n_vars)
    cost[len(pairs) :] = 1.0
    a_ub, b_ub = [], []
    for tier in lower:  # served by by-products plus separate wafers: -(sum x + sellable * W) <= -share
        row = np.zeros(n_vars)
        for band in range(1, tier + 1):
            row[index[(band, tier)]] = -1.0
        tier_die = separate.per_tier[tier]
        row[len(pairs) + tier - 1] = -tier_die.dies_per_wafer * tier_die.sellable_fraction
        a_ub.append(row)
        b_ub.append(-tiers[tier].share)
    for band in lower:  # by-product supply of this band from the fixed top wafers
        row = np.zeros(n_vars)
        for tier in range(band, n):
            row[index[(band, tier)]] = 1.0
        a_ub.append(row)
        b_ub.append(top_wafers * supply[band])
    result = optimize.linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=(0, None), method="highs")
    assert result.status == 0, result.message
    return top_wafers + result.fun


@pytest.mark.parametrize("gap, d0", [(18, 0.5), (18, 4.0), (6, 1.0), (6, 4.0), (3, 2.0), (3, 4.0)])
def test_hybrid_cost_equals_a_linear_program_over_all_placements_of_by_product_dies(gap: int, d0: float):
    tiers = tiers_with_gap(gap)
    proc = b.Process(d0)
    spares = b.best_safe_spares(DIE, proc, tiers)
    assert b.plan_hybrid(DIE, proc, tiers, spares).cost_per_unit == pytest.approx(
        hybrid_lp_cost(DIE, proc, tiers, spares), rel=1e-9
    )


@pytest.mark.parametrize(
    "gap, shares, d0",
    [(18, (0.98, 0.01, 0.01), 4.0), (6, (0.98, 0.01, 0.01), 2.0), (18, (0.9, 0.05, 0.05), 4.0), (6, (0.9, 0.09, 0.01), 4.0)],
)
def test_hybrid_matches_the_linear_program_when_by_products_outrun_lower_tier_demand(gap, shares, d0):
    tiers = tiers_with_shares(gap, shares)
    proc = b.Process(d0)
    spares = b.best_safe_spares(DIE, proc, tiers)
    hybrid = b.plan_hybrid(DIE, proc, tiers, spares)
    assert hybrid.cost_per_unit == pytest.approx(hybrid_lp_cost(DIE, proc, tiers, spares), rel=1e-9)
    assert hybrid.surplus >= 0.0
    assert all(wafers >= 0.0 for wafers in hybrid.separate_wafers)


def test_by_products_beyond_a_tiers_demand_are_carried_down_and_the_rest_is_surplus():
    proc = b.Process(4.0)
    plenty = tiers_with_shares(18, (0.98, 0.01, 0.01))
    hybrid = b.plan_hybrid(DIE, proc, plenty, b.best_safe_spares(DIE, proc, plenty))
    assert hybrid.separate_wafers == (0.0, 0.0), "by-products meet both lower tiers"
    assert hybrid.surplus > 0.0
    assert hybrid.cost_per_unit == pytest.approx(hybrid.top_wafers), "so the top wafers are the whole cost"
    # mid demand is met with by-products to spare, and the spare ones go on to cover part of the low tier
    carry = tiers_with_shares(18, (0.9, 0.05, 0.05))
    spares = b.best_safe_spares(DIE, proc, carry)
    hybrid = b.plan_hybrid(DIE, proc, carry, spares)
    separate = b.plan_separate(DIE, proc, carry, spares)
    assert hybrid.separate_wafers[0] == 0.0 and hybrid.surplus == 0.0
    low = separate.per_tier[2]
    low_from_separate = hybrid.separate_wafers[1] * low.dies_per_wafer * low.sellable_fraction
    assert 0.0 < low_from_separate < carry[2].share, "the carried by-products cover part of the low tier"


@pytest.mark.parametrize("shares, d0", [((0.98, 0.01, 0.01), 4.0), ((0.9, 0.05, 0.05), 4.0), ((0.2, 0.3, 0.5), 1.0), ((0.6, 0.05, 0.35), 4.0)])
def test_every_lower_tier_unit_comes_from_a_by_product_or_a_separate_die(shares, d0):
    tiers = tiers_with_shares(6, shares)
    proc = b.Process(d0)
    spares = b.best_safe_spares(DIE, proc, tiers)
    hybrid = b.plan_hybrid(DIE, proc, tiers, spares)
    separate = b.plan_separate(DIE, proc, tiers, spares)
    pmf = b.good_block_pmf(DIE, proc)
    by_products = hybrid.top_wafers * b.dies_per_wafer(DIE.area_mm2) * float(pmf[tiers[-1].blocks : tiers[0].blocks].sum())
    from_separate = sum(
        wafers * tier_die.dies_per_wafer * tier_die.sellable_fraction
        for wafers, tier_die in zip(hybrid.separate_wafers, separate.per_tier[1:])
    )
    assert by_products - hybrid.surplus + from_separate == pytest.approx(1.0 - tiers[0].share, rel=1e-9)


def test_separate_dies_without_defects_cost_one_over_dies_per_wafer_weighted_by_demand():
    plan = b.plan_separate(DIE, b.Process(0.0), TIERS, 0)
    areas = [DIE.area_mm2] + [DIE.uncore_mm2 + t.blocks * DIE.block_mm2 for t in TIERS[1:]]
    per_wafer = [b.dies_per_wafer(area) for area in areas]
    assert per_wafer == [64, 78, 89]
    assert [d.dies_per_wafer for d in plan.per_tier] == per_wafer
    assert [d.cost_per_unit for d in plan.per_tier] == pytest.approx([1.0 / n for n in per_wafer])
    assert plan.cost_per_unit == pytest.approx(0.2 / 64 + 0.3 / 78 + 0.5 / 89)
    assert b.plan_separate(DIE, b.Process(0.0), TIERS, 0, wafer_cost=3.0).cost_per_unit == pytest.approx(3.0 * plan.cost_per_unit)
    binned = b.plan_binning(DIE, b.Process(0.0), TIERS)
    assert binned.units_per_wafer == pytest.approx(64.0) and binned.cost_per_unit == pytest.approx(1.0 / 64)


@pytest.mark.parametrize("gap, d0", [(3, 1.0), (3, 4.0), (6, 0.5), (6, 4.0)])
def test_best_safe_spares_can_sit_at_the_largest_safe_count(gap: int, d0: float):
    tiers = tiers_with_gap(gap)
    proc = b.Process(d0)
    limits = b.max_safe_spares(tiers)[1:]
    best = b.best_safe_spares(DIE, proc, tiers)
    for position, limit in enumerate(limits):
        costs = []
        for spare in range(limit + 1):
            spares = [0, 0]
            spares[position] = spare
            costs.append(b.plan_separate(DIE, proc, tiers, spares).per_tier[position + 1].cost_per_unit)
        assert best[position] == int(np.argmin(costs)) == limit, "spares keep paying for themselves up to the safe limit"


def test_hybrid_never_costs_more_than_separate_dies_and_saves_where_defective_top_dies_are_plentiful():
    for d0 in (0.05, 0.5, 1.0, 4.0):
        for gap in (18, 6, 3):
            tiers = tiers_with_gap(gap)
            proc = b.Process(d0)
            spares = b.best_safe_spares(DIE, proc, tiers)
            separate = b.plan_separate(DIE, proc, tiers, spares).cost_per_unit
            assert b.plan_hybrid(DIE, proc, tiers, spares).cost_per_unit <= separate + 1e-15
    tiers = tiers_with_gap(3)
    proc = b.Process(4.0)
    spares = b.best_safe_spares(DIE, proc, tiers)
    hybrid = b.plan_hybrid(DIE, proc, tiers, spares)
    assert hybrid.cost_per_unit < 0.995 * b.plan_separate(DIE, proc, tiers, spares).cost_per_unit
    assert hybrid.surplus >= 0.0


def test_hybrid_only_sells_dies_that_fall_short_of_the_top_count_below_the_top_tier():
    """The by-products available to lower tiers are exactly the dies with fewer good blocks than the top tier needs."""
    proc = b.Process(4.0)
    tiers = tiers_with_gap(3)
    pmf = b.good_block_pmf(DIE, proc)
    n_dies = b.dies_per_wafer(DIE.area_mm2)
    not_top_capable = n_dies * float(pmf[tiers[-1].blocks : tiers[0].blocks].sum())
    top_per_wafer = n_dies * float(pmf[tiers[0].blocks :].sum())
    hybrid = b.plan_hybrid(DIE, proc, tiers, b.best_safe_spares(DIE, proc, tiers))
    by_products = hybrid.top_wafers * not_top_capable
    assert hybrid.top_wafers == pytest.approx(tiers[0].share / top_per_wafer)
    served = by_products - hybrid.surplus
    assert 0.0 < served <= 1.0 - tiers[0].share + 1e-12


def test_silicon_premium_changes_sign_with_the_spacing_of_the_tiers():
    wide, narrow = tiers_with_gap(18), tiers_with_gap(6)
    for d0 in (0.1, 1.0, 4.0):
        assert b.silicon_premium(DIE, b.Process(d0), wide) < -0.1
    assert b.silicon_premium(DIE, b.Process(0.1), narrow) < 0.0 < b.silicon_premium(DIE, b.Process(4.0), narrow)
    crossing = b.breakeven_defect_density(DIE, narrow)
    assert crossing is not None and 0.5 < crossing < 1.2
    assert abs(b.silicon_premium(DIE, b.Process(crossing), narrow)) < 1e-6
    assert b.breakeven_defect_density(DIE, wide) is None


def test_nre_arithmetic_and_break_even_volume():
    assert b.total_cost_per_unit(5.0, 100.0, 50.0) == pytest.approx(7.0)
    assert b.breakeven_volume(1000.0, 0.020, 0.015) == pytest.approx(200_000.0)
    assert b.breakeven_volume(1000.0, 0.015, 0.020) == math.inf
    assert b.breakeven_volume(1000.0, 0.015, 0.015) == math.inf
    with pytest.raises(ValueError):
        b.total_cost_per_unit(1.0, 10.0, 0.0)
    with pytest.raises(ValueError):
        b.total_cost_per_unit(1.0, -1.0, 10.0)
    volume = b.breakeven_volume(5000.0, 0.02, 0.016)
    assert b.total_cost_per_unit(0.016, 5000.0, volume) == pytest.approx(0.02)


def test_invalid_parameters_are_refused():
    with pytest.raises(ValueError):
        b.Die(0.0, 4.0, 10)
    with pytest.raises(ValueError):
        b.Die(10.0, 4.0, 0)
    with pytest.raises(ValueError):
        b.Process(-0.1)
    with pytest.raises(ValueError):
        b.Process(1.0, alpha=0.0)
    with pytest.raises(ValueError, match="sum to 1"):
        b.plan_binning(DIE, b.Process(1.0), (b.Tier("a", 132, 0.5), b.Tier("b", 96, 0.4)))
    with pytest.raises(ValueError, match="strictly decreasing"):
        b.plan_binning(DIE, b.Process(1.0), (b.Tier("a", 96, 0.5), b.Tier("b", 96, 0.5)))
    with pytest.raises(ValueError, match="more blocks than the die"):
        b.plan_binning(DIE, b.Process(1.0), (b.Tier("a", 200, 0.5), b.Tier("b", 96, 0.5)))
    with pytest.raises(ValueError):
        b.secure_cost(DIE, b.Process(1.0), TIERS, design="imaginary")
    with pytest.raises(ValueError):
        b.dies_per_wafer(0.0)


def test_a_die_too_large_for_the_wafer_is_infeasible():
    huge = b.Die(60_000.0, 4.0, 144)
    assert b.dies_per_wafer(huge.area_mm2) == 0
    with pytest.raises(b.InfeasibleError):
        b.plan_binning(huge, b.Process(0.1), TIERS)
    with pytest.raises(b.InfeasibleError):
        b.plan_separate(huge, b.Process(0.1), TIERS, 0)
