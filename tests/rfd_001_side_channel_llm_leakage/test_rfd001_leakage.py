import math

import numpy as np
import pytest

from aisecurity_rfds.rfd_001_side_channel_llm_leakage import leakage as lk
from aisecurity_rfds.rfd_001_side_channel_llm_leakage.leakage import Channel, ToyLayer, hw16, mac_leakage


def channel(rate_hz: float = 1e9, gain: float = 1.0, noise_op: float = 0.0, noise_out: float = 0.0) -> Channel:
    return Channel("test", gain=gain, noise_op=noise_op, noise_out=noise_out, rate_hz=rate_hz)


def test_hamming_weight_of_known_values():
    assert [int(hw16(v)) for v in (0, 1, 255, 16384, 32767)] == [0, 1, 8, 1, 15]
    assert int(hw16(-1)) == 16 and int(hw16(-32768)) == 1


def test_hamming_weight_wraps_modulo_65536():
    for v in (-5, 7, 12345):
        assert int(hw16(v)) == int(hw16(v + 65536)) == int(hw16(v - 65536))


def test_hamming_weight_of_an_array_matches_a_bit_count():
    values = np.arange(-300, 300)
    assert hw16(values).tolist() == [bin(int(v) & 0xFFFF).count("1") for v in values]


def test_an_int8_product_always_fits_in_sixteen_bits():
    extremes = np.array([-128, 127])
    products = np.outer(extremes, extremes)
    assert products.max() == 16384 and products.min() == -16256
    assert -32768 <= products.min() and products.max() <= 32767


def test_mac_leakage_on_a_worked_example():
    # 2*4 = 8 (one bit set) and 3*-1 = -3 = 0xFFFD (fifteen bits set)
    assert mac_leakage(np.array([[2, 3]]), np.array([4, -1])).tolist() == [[1.0, 15.0]]


def test_mac_leakage_is_zero_for_a_zero_activation_and_keeps_the_shape():
    out = mac_leakage(np.arange(-17, 18).reshape(5, 7), np.zeros(7, dtype=int))
    assert out.shape == (5, 7) and not out.any()


def test_a_toy_layer_is_reproducible_from_its_seed_and_differs_between_seeds():
    a, b, c = ToyLayer.make(0), ToyLayer.make(0), ToyLayer.make(1)
    assert np.array_equal(a.weights, b.weights) and np.array_equal(a.embeddings, b.embeddings)
    assert not np.array_equal(a.weights, c.weights) and not np.array_equal(a.embeddings, c.embeddings)


def test_the_default_layer_has_the_size_the_readme_states_and_int8_values():
    layer = ToyLayer.make()
    assert (layer.vocab, layer.ops, layer.weights.shape) == (64, 288, (12, 24))
    for array in (layer.weights, layer.embeddings):
        assert array.min() >= -128 and array.max() <= 127


def test_a_template_is_the_leakage_of_that_tokens_embedding():
    layer = ToyLayer.make(3, vocab=8, rows=3, width=4)
    expected = mac_leakage(layer.weights, layer.embeddings[5]).ravel()
    assert np.array_equal(layer.template(5), expected)
    assert np.array_equal(layer.templates()[5], expected) and layer.templates().shape == (8, 12)
    assert not np.array_equal(layer.template(5), layer.template(6))


def test_the_window_is_the_number_of_operations_averaged_into_a_sample():
    assert channel(1e9).window == 1
    assert channel(5e4).window == 20_000
    assert channel(50).window == 20_000_000
    assert channel(4e9).window == 1  # faster than the operations: still one operation per sample


def test_per_operation_noise_averages_down_with_the_window_and_output_noise_does_not():
    ch = channel(1e9 / 100, noise_op=6.0, noise_out=0.3)
    assert ch.window == 100
    assert ch.noise == pytest.approx(math.sqrt(6.0**2 / 100 + 0.3**2))
    assert channel(1e9, noise_op=6.0).noise == pytest.approx(6.0)
    assert channel(50, noise_out=0.01).noise == pytest.approx(0.01)


def test_samples_per_token_is_rate_times_the_slot():
    assert channel(1000).samples_per_slot == pytest.approx(20.0)
    assert channel(50).samples_per_slot == pytest.approx(1.0)
    assert channel(10).samples_per_slot == pytest.approx(0.2)


def test_view_takes_block_means_and_pads_the_last_window_with_zeros():
    ch = channel(5e8)  # window of 2 operations
    assert np.allclose(ch.view(np.array([1.0, 3.0, 5.0])), [2.0, 2.5])
    assert np.allclose(channel(5e8, gain=3.0).view(np.array([1.0, 3.0, 5.0])), [6.0, 7.5])


def test_a_window_longer_than_the_signal_dilutes_it_by_signal_length_over_window():
    ch = channel(1e9 / 1000, gain=2.0)  # one window of 1000 operations
    out = ch.view(np.full(10, 4.0))
    assert out.shape == (1,) and out[0] == pytest.approx(2.0 * 40.0 / 1000)


def test_view_handles_a_matrix_of_signals_row_by_row():
    signals = np.array([[1.0, 3.0, 5.0, 7.0], [2.0, 2.0, 2.0, 2.0]])
    assert np.allclose(channel(5e8).view(signals), [[2.0, 6.0], [2.0, 2.0]])
    assert np.allclose(channel(1e8).view(signals), [[16.0 / 10], [8.0 / 10]])  # one window of ten operations


def test_observe_adds_noise_with_the_stated_standard_deviation(rng):
    ch = channel(1e9 / 16, noise_op=2.0, noise_out=0.5)
    out = ch.observe(np.zeros((20000, 16)), rng)
    assert out.shape == (20000, 1)
    assert out.std() == pytest.approx(ch.noise, rel=0.03) and abs(out.mean()) < 0.02


def test_repeating_a_run_shrinks_the_noise_by_the_square_root(rng):
    ch = channel(1e9, noise_op=3.0)
    single, averaged = ch.observe(np.zeros((20000, 1)), rng), ch.observe(np.zeros((20000, 1)), rng, repeats=9)
    assert averaged.std() == pytest.approx(single.std() / 3, rel=0.05)


def test_other_tenants_add_noise_in_quadrature_and_one_tenant_changes_nothing():
    ch = channel(50, noise_out=0.01)
    assert ch.with_tenants(1, 0.5) == ch
    assert ch.with_tenants(5, 0.1).noise == pytest.approx(math.sqrt(0.01**2 + 4 * 0.1**2))
    assert ch.with_tenants(5, 0.1).rate_hz == 50 and "x5" in ch.with_tenants(5, 0.1).name


@pytest.mark.parametrize(
    "kwargs",
    [{"gain": 0.0}, {"gain": -1.0}, {"rate_hz": 0.0}, {"rate_hz": -5.0}, {"noise_op": -0.1}, {"noise_out": -0.1}],
)
def test_a_channel_with_a_nonsensical_parameter_is_refused(kwargs):
    with pytest.raises(ValueError):
        channel(**kwargs)


def test_a_sensor_cannot_have_fewer_than_one_tenant():
    loud = channel(50, noise_out=1.0)  # loud enough that a negative variance is not what rejects zero
    for tenants in (0, -3):
        with pytest.raises(ValueError, match="at least one tenant"):
            loud.with_tenants(tenants, 0.1)


def test_the_assumed_channels_are_ordered_from_fast_to_slow_with_distinct_names():
    rates = [c.rate_hz for c in lk.CHANNELS]
    assert rates == sorted(rates, reverse=True)
    assert len({c.name for c in lk.CHANNELS}) == len(lk.CHANNELS) == 5
    assert lk.SCOPE.window == lk.EM_PROBE.window == 1 and lk.THERMAL.samples_per_slot < 1 <= lk.SOFTWARE.samples_per_slot
