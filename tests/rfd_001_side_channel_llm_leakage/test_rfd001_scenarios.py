from dataclasses import replace

import numpy as np
import pytest

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.rfd_001_side_channel_llm_leakage import leakage as lk
from aisecurity_rfds.rfd_001_side_channel_llm_leakage import routing as rt
from aisecurity_rfds.rfd_001_side_channel_llm_leakage import scenarios as sc
from aisecurity_rfds.rfd_001_side_channel_llm_leakage.leakage import Channel, ToyLayer

LAYER = ToyLayer.make()
ROUTER = rt.MoeRouter.make()
QUIET = Channel("noiseless", gain=1.0, noise_op=0.0, noise_out=0.0, rate_hz=1e9)


def sensor_like(noise: float) -> Channel:
    return Channel("s", gain=1.0, noise_op=0.0, noise_out=noise, rate_hz=50.0)


def test_a_fast_channel_names_tokens_and_a_slow_one_is_at_chance():
    assert sc.token_accuracy(LAYER, lk.SCOPE, 1500, make_rng(50)).p > 0.7
    for slow in (lk.ACOUSTIC, lk.SOFTWARE, lk.THERMAL):
        assert sc.token_accuracy(LAYER, slow, 3000, make_rng(51)).p < 0.06


def test_a_slow_window_shrinks_the_difference_between_tokens_to_sum_over_window():
    a, b = LAYER.template(3), LAYER.template(9)
    view = lk.ACOUSTIC.view(LAYER.templates())
    assert abs(view[3, 0] - view[9, 0]) == pytest.approx(lk.ACOUSTIC.gain * abs(a.sum() - b.sum()) / lk.ACOUSTIC.window)
    assert abs(view[3, 0] - view[9, 0]) < 1e-3 < abs(lk.SCOPE.view(a) - lk.SCOPE.view(b)).max()


def test_the_best_effect_at_a_hand_computed_point():
    layer = ToyLayer(weights=np.array([[1]]), embeddings=np.array([[0], [3]]))  # leakage 0 and HW(3) = 2
    ch = Channel("c", gain=1.0, noise_op=1.0, noise_out=0.0, rate_hz=1e9)
    # mean 1, population variance 1; delta 1; pooled sd sqrt(1 + 1/2)
    assert sc.best_effect(layer, ch, 0) == pytest.approx(1 / np.sqrt(1.5))
    assert sc.best_effect(layer, ch, 1) == pytest.approx(1 / np.sqrt(1.5))
    assert sc.median_best_effect(layer, ch) == pytest.approx(1 / np.sqrt(1.5))


def test_the_best_effect_is_large_for_a_fast_channel_and_tiny_for_a_slow_one():
    assert 0.2 < sc.median_best_effect(LAYER, lk.SCOPE) < 1.0
    assert sc.median_best_effect(LAYER, lk.EM_PROBE) > sc.median_best_effect(LAYER, lk.ACOUSTIC) * 20
    assert sc.median_best_effect(LAYER, lk.SOFTWARE) < sc.median_best_effect(LAYER, lk.ACOUSTIC)


def test_tvla_flags_a_leaking_channel_more_as_traces_accumulate():
    few = sc.tvla_flagged(LAYER, lk.SCOPE, 100, make_rng(52))[0]
    more = sc.tvla_flagged(LAYER, lk.SCOPE, 1000, make_rng(53))[0]
    most = sc.tvla_flagged(LAYER, lk.SCOPE, 5000, make_rng(54))[0]
    assert few < more < most <= LAYER.ops and most > 150


def test_tvla_flags_nothing_when_both_groups_are_random_or_the_channel_is_too_slow():
    assert sc.tvla_flagged(LAYER, lk.SCOPE, 5000, make_rng(55), fixed=None)[0] == 0
    assert sc.tvla_flagged(LAYER, lk.SOFTWARE, 5000, make_rng(56))[0] == 0
    flagged, top = sc.tvla_flagged(LAYER, lk.SCOPE, 5000, make_rng(57), fixed=None)
    assert flagged == 0 and 0 < top < 4.5


def test_the_dense_attacker_is_perfect_without_noise_and_at_chance_under_overwhelming_noise():
    assert sc.dense_category_accuracy(LAYER, QUIET, 4, 300, make_rng(58)).p == 1.0
    drowned = replace(QUIET, noise_out=1e6)
    assert 0.18 <= sc.dense_category_accuracy(LAYER, drowned, 8, 1500, make_rng(59)).p <= 0.32


def test_the_dense_attacker_gains_from_a_longer_prompt():
    marginal = replace(lk.SCOPE, noise_op=30.0)  # one token is named correctly less than a fifth of the time
    short = sc.dense_category_accuracy(LAYER, marginal, 1, 800, make_rng(60)).p
    long = sc.dense_category_accuracy(LAYER, marginal, 16, 800, make_rng(61)).p
    assert long > short + 0.15 and long > 0.6


def test_the_dense_attacker_needs_the_vocabulary_to_divide_into_topics():
    with pytest.raises(ValueError):
        sc.dense_category_accuracy(ToyLayer.make(vocab=10), QUIET, 4, 10, make_rng(1))


def test_the_token_search_stops_at_the_first_power_of_two_that_reaches_the_target():
    assert sc.tokens_for_category(ROUTER, sensor_like(1e-9), 0.6, 600, 0) == 1  # one exact token gives 0.66
    found = sc.tokens_for_category(ROUTER, lk.SOFTWARE, 0.9, 600, 0)
    assert found is not None and found & (found - 1) == 0 and found > 1


class _Reading:
    """Stands in for an estimate: the search reads only ``p``."""

    def __init__(self, p):
        self.p = p


def _search_with_stub(monkeypatch, which, accuracy_at, limit):
    """Run a token search whose accuracy is ``accuracy_at(tokens)``; return (result, token counts it looked at)."""
    probed = []

    def stub(subject, channel, tokens, trials, rng):
        probed.append(tokens)
        return _Reading(accuracy_at(tokens))

    if which == "routing":
        monkeypatch.setattr(sc, "category_accuracy", stub)
        found = sc.tokens_for_category(ROUTER, QUIET, 0.9, 10, 0, limit=limit)
    else:
        monkeypatch.setattr(sc, "dense_category_accuracy", stub)
        found = sc.tokens_for_dense_category(LAYER, QUIET, 0.9, 10, 0, limit=limit)
    return found, probed


@pytest.mark.parametrize("which", ["routing", "dense"])
def test_the_token_search_looks_only_at_powers_of_two_and_stops_at_the_limit(monkeypatch, which):
    found, probed = _search_with_stub(monkeypatch, which, lambda tokens: 0.0, limit=100)
    assert found is None
    assert probed == [1, 2, 4, 8, 16, 32, 64]


@pytest.mark.parametrize("which", ["routing", "dense"])
def test_the_token_search_returns_the_first_power_of_two_that_reaches_the_target(monkeypatch, which):
    found, probed = _search_with_stub(monkeypatch, which, lambda tokens: 0.95 if tokens >= 5 else 0.1, limit=1024)
    assert found == 8
    assert probed == [1, 2, 4, 8]  # stops at the first success; 5, 6 and 7 are never tried
    found, probed = _search_with_stub(monkeypatch, which, lambda tokens: 0.9 if tokens >= 4 else 0.0, limit=1024)
    assert found == 4  # reaching the target exactly counts


def test_the_token_search_gives_up_when_the_target_is_out_of_reach():
    flat = rt.MoeRouter(sets=ROUTER.sets, probs=ROUTER.probs, levels=np.zeros(len(ROUTER.sets)))
    assert sc.tokens_for_category(flat, lk.SOFTWARE, 0.9, 300, 0, limit=16) is None
    assert sc.tokens_for_dense_category(LAYER, lk.SOFTWARE, 0.9, 300, 0, limit=16) is None


def test_the_dense_token_search_is_quick_on_a_fast_channel():
    assert sc.tokens_for_dense_category(LAYER, lk.EM_PROBE, 0.9, 400, 0) <= 2


def test_two_channels_that_differ_only_in_name_get_the_same_answer():
    twin = replace(lk.SOFTWARE, name="another name")
    assert sc.tokens_for_category(ROUTER, twin, 0.9, 400, 3) == sc.tokens_for_category(ROUTER, lk.SOFTWARE, 0.9, 400, 3)


def test_other_tenants_slow_the_routing_attack_and_a_faster_exposed_rate_speeds_it_up():
    alone = sc.tokens_for_category(ROUTER, lk.SOFTWARE, 0.9, 400, 0, limit=256)
    crowded = sc.tokens_for_category(ROUTER, lk.SOFTWARE.with_tenants(64, sc.OTHER_TENANT_NOISE), 0.9, 400, 0, limit=256)
    fast = sc.tokens_for_category(ROUTER, replace(lk.SOFTWARE, rate_hz=1000.0), 0.9, 400, 0, limit=256)
    assert alone is not None and fast is not None and fast < alone
    assert crowded is None or crowded > alone


def test_a_faster_exposed_rate_does_not_let_a_slow_sensor_name_tokens():
    fast_api = replace(lk.SOFTWARE, rate_hz=1000.0)
    assert sc.token_accuracy(LAYER, fast_api, 3000, make_rng(62)).p < 0.05


def test_the_three_refusal_modes():
    hidden = [4, 8, 15, 16, 23, 42]
    pre, post, none = (sc.respond(m, hidden) for m in ("pre-check", "post-hoc filter", "no refusal"))
    assert pre.shown == post.shown == sc.REFUSAL_TEXT and none.shown != sc.REFUSAL_TEXT
    assert pre.decoded == () and post.decoded == none.decoded == tuple(hidden)
    with pytest.raises(ValueError):
        sc.respond("hope", hidden)


def test_a_noiseless_observer_reads_a_whole_hidden_response_unless_nothing_was_decoded():
    hidden = list(range(20))
    assert sc.recovered_fraction(LAYER, QUIET, sc.respond("post-hoc filter", hidden), hidden, make_rng(63)) == 1.0
    assert sc.recovered_fraction(LAYER, QUIET, sc.respond("pre-check", hidden), hidden, make_rng(64)) == 0.0


def test_recovery_is_scored_against_the_whole_hidden_response():
    hidden = list(range(10))
    partial = sc.Transcript("x", tuple(hidden[:5]))
    assert sc.recovered_fraction(LAYER, QUIET, partial, hidden, make_rng(65)) == 0.5


def test_an_empty_hidden_response_is_refused():
    with pytest.raises(ValueError):
        sc.recovered_fraction(LAYER, QUIET, sc.respond("post-hoc filter", []), [], make_rng(66))


def test_hiding_the_text_after_decoding_does_not_hide_the_tokens_from_the_attacker():
    assert sc.hidden_recovery(LAYER, lk.EM_PROBE, "post-hoc filter", 10, 32, make_rng(67)) > 0.85
    assert sc.hidden_recovery(LAYER, lk.EM_PROBE, "pre-check", 10, 32, make_rng(68)) == 0.0
    assert sc.hidden_recovery(LAYER, lk.SOFTWARE, "post-hoc filter", 20, 32, make_rng(69)) < 0.06


def test_the_recovery_of_a_hidden_response_matches_the_token_accuracy_of_the_channel():
    accuracy = sc.token_accuracy(LAYER, lk.SCOPE, 3000, make_rng(70)).p
    recovered = sc.hidden_recovery(LAYER, lk.SCOPE, "post-hoc filter", 20, 32, make_rng(71))
    assert abs(recovered - accuracy) < 0.06


def test_the_seed_key_of_a_channel_ignores_its_name_and_sees_every_parameter():
    base = lk.SOFTWARE
    assert sc._physics(replace(base, name="another name")) == sc._physics(base)
    for change in ({"gain": 2.0}, {"noise_op": 1.0}, {"noise_out": 0.5}, {"rate_hz": 100.0}):
        assert sc._physics(replace(base, **change)) != sc._physics(base), change


def test_a_noiseless_observer_recovers_every_hidden_response_exactly():
    assert sc.hidden_recovery(LAYER, QUIET, "post-hoc filter", 5, 8, make_rng(72)) == 1.0
    assert sc.hidden_recovery(LAYER, QUIET, "no refusal", 5, 8, make_rng(73)) == 1.0
    assert sc.hidden_recovery(LAYER, QUIET, "pre-check", 5, 8, make_rng(74)) == 0.0
