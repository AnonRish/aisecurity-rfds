import numpy as np

from aisecurity_rfds.common.puf import FuzzyExtractor, SimulatedPuf, bit_error_rate
from aisecurity_rfds.common.rng import derive_seed, make_rng


def test_uniqueness_between_devices_is_near_half():
    refs = [SimulatedPuf(i).reference_response() for i in range(12)]
    dists = [bit_error_rate(refs[i], refs[j]) for i in range(12) for j in range(i + 1, 12)]
    assert 0.45 < float(np.mean(dists)) < 0.55


def test_reliability_matches_theory():
    # sigma=0.1 -> theoretical flip probability arccos(1/sqrt(1+sigma^2))/pi ~ 3.2%
    puf = SimulatedPuf(1, n_cells=20000, noise_sigma=0.1)
    rng = make_rng(7)
    ber = bit_error_rate(puf.measure(rng=rng), puf.reference_response())
    assert 0.025 < ber < 0.040


def test_environment_degrades_reliability_monotonically():
    puf = SimulatedPuf(2, n_cells=20000)
    rng = make_rng(3)
    ref = puf.reference_response()
    nominal = bit_error_rate(puf.measure(rng=rng), ref)
    hot = bit_error_rate(puf.measure(temperature_c=100.0, rng=rng), ref)
    old_hot = bit_error_rate(puf.measure(temperature_c=100.0, age_years=10.0, rng=rng), ref)
    assert nominal < hot < old_hot


def enrolled(device=5, **kw):
    fe = FuzzyExtractor(**kw)
    puf = SimulatedPuf(device, n_cells=fe.n_cells_required)
    key, helper = fe.enroll(puf.measure(rng=make_rng(1)), rng=make_rng(2))
    return fe, puf, key, helper


def test_fuzzy_extractor_reproduces_under_noise():
    fe, puf, key, helper = enrolled()
    rng = make_rng(10)
    ok = 0
    for _ in range(100):
        k2, good = fe.reproduce(puf.measure(rng=rng), helper)
        ok += good and k2 == key
    assert ok >= 99


def test_fuzzy_extractor_survives_hot_aged_operation():
    fe, puf, key, helper = enrolled()
    rng = make_rng(11)
    ok = sum(
        fe.reproduce(puf.measure(temperature_c=85.0, age_years=5.0, rng=rng), helper)[1]
        for _ in range(100)
    )
    assert ok >= 90


def test_other_devices_cannot_reproduce_the_key():
    fe, _, key, helper = enrolled(device=5)
    rng = make_rng(12)
    for other in range(100, 140):
        other_puf = SimulatedPuf(other, n_cells=fe.n_cells_required)
        k2, good = fe.reproduce(other_puf.measure(rng=rng), helper)
        assert not good and k2 != key


def test_heavy_intrusion_changes_the_key_but_light_intrusion_is_tolerated():
    # Documents an inherent property: error correction also forgives *small* physical changes.
    fe, puf, key, helper = enrolled()
    rng = make_rng(13)
    light = SimulatedPuf(5, n_cells=fe.n_cells_required)
    light.perturb(0.02, strength=1.0, rng=make_rng(20))
    assert sum(fe.reproduce(light.measure(rng=rng), helper)[1] for _ in range(50)) >= 45

    heavy = SimulatedPuf(5, n_cells=fe.n_cells_required)
    heavy.perturb(1.0, strength=3.0, rng=make_rng(21))
    for _ in range(50):
        k2, good = fe.reproduce(heavy.measure(rng=rng), helper)
        assert not good and k2 != key


def test_context_separation_and_determinism():
    fe = FuzzyExtractor()
    puf = SimulatedPuf(9, n_cells=fe.n_cells_required)
    raw = puf.measure(rng=make_rng(1))
    k_a, h_a = fe.enroll(raw, rng=make_rng(5), context=b"firmware")
    k_b, _ = fe.enroll(raw, rng=make_rng(5), context=b"model")
    k_a2, _ = fe.enroll(raw, rng=make_rng(5), context=b"firmware")
    assert k_a != k_b and k_a == k_a2
    assert h_a.offset != k_a and len(k_a) == 32


def test_input_validation():
    import pytest

    with pytest.raises(ValueError):
        FuzzyExtractor(repetition=14)
    fe = FuzzyExtractor()
    with pytest.raises(ValueError):
        fe.enroll(np.zeros(10, dtype=np.uint8))
    assert derive_seed("a", 1) == derive_seed("a", 1) != derive_seed("a", 2)
