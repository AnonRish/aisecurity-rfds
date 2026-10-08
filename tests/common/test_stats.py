import math

import numpy as np
import pytest
from scipy import stats as sps

from aisecurity_rfds.common.stats import (
    clopper_pearson,
    hypergeom_miss_probability,
    monte_carlo_probability,
    poisson_miss_probability,
    samples_needed_hypergeom,
    samples_needed_poisson,
    wilson_interval,
)


@pytest.mark.parametrize(
    "total,fake,sampled",
    [(100, 5, 20), (1000, 10, 300), (50, 1, 25), (10_000, 100, 500), (20, 15, 10), (30, 0, 12)],
)
def test_hypergeom_matches_scipy(total, fake, sampled):
    expected = sps.hypergeom(M=total, n=fake, N=sampled).pmf(0)
    assert hypergeom_miss_probability(total, fake, sampled) == pytest.approx(expected, rel=1e-9, abs=1e-15)


def test_hypergeom_edge_cases():
    assert hypergeom_miss_probability(10, 3, 0) == 1.0
    assert hypergeom_miss_probability(10, 3, 8) == 0.0  # pigeonhole: cannot avoid every fake
    with pytest.raises(ValueError):
        hypergeom_miss_probability(10, 11, 1)


def test_poisson_is_a_conservative_bound_and_tight_for_small_sampling_fractions():
    # Sampling without replacement can only help the verifier: P_miss = prod(1 - K/(N-i)) <= (1-K/N)^n
    # <= exp(-n K / N), so exp(-n*f) is a valid UPPER bound on the miss probability, not just an estimate.
    for total, fake, sampled in [(1_000_000, 100, 50_000), (1000, 10, 300), (500, 5, 400), (50, 2, 10)]:
        exact = hypergeom_miss_probability(total, fake, sampled)
        assert exact <= poisson_miss_probability(sampled, fake / total) + 1e-15
    # ...and it is tight when only a tiny fraction of the packets is sampled.
    total, fake, sampled = 10_000_000, 100, 10_000
    exact = hypergeom_miss_probability(total, fake, sampled)
    assert exact == pytest.approx(poisson_miss_probability(sampled, fake / total), rel=0.01)


def test_rfd018_required_verification_number():
    # RFD 018: 99.9% confidence needs N_VERIFIED * F_FAKE > -ln(0.001) ~ 7; F_FAKE = 1e-9 -> ~7e9 packets.
    assert samples_needed_poisson(1e-9, 0.999) == pytest.approx(6.9078e9, rel=1e-4)


def test_samples_needed_hypergeom_is_minimal_and_sufficient():
    for total, fake in [(1000, 3), (5000, 50), (200, 1)]:
        for conf in (0.9, 0.99, 0.999):
            s = samples_needed_hypergeom(total, fake, conf)
            assert hypergeom_miss_probability(total, fake, s) <= 1 - conf
            if s > 0:
                assert hypergeom_miss_probability(total, fake, s - 1) > 1 - conf


def test_wilson_and_clopper_pearson_known_values():
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 10), rel=1e-6)
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.21
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_interval_coverage_in_simulation():
    rng = np.random.default_rng(99)
    p, n, reps, covered = 0.3, 200, 400, 0
    for _ in range(reps):
        k = int(rng.binomial(n, p))
        lo, hi = wilson_interval(k, n)
        covered += lo <= p <= hi
    assert covered / reps > 0.92  # nominal 95%


def test_monte_carlo_probability_contains_truth():
    est = monte_carlo_probability(lambda r: r.random() < 0.3, 4000, np.random.default_rng(5))
    assert est.contains(0.3) and est.trials == 4000
    assert math.isclose(est.p, 0.3, abs_tol=0.03)
