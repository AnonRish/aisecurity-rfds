"""The bandwidth analysis: constants checked by execution, closed forms checked by arithmetic, plans by brute force.

Everything here is a property of the model in ``training.py`` under stated assumptions. Tests that pin a headline
number say so in their name.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit import inference, podsize
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit import primitives as prim
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit.training import (
    FLOPS_PER_PARAM_TOKEN,
    PIPELINE_VECTORS_PER_TOKEN,
    YEAR_S,
    External,
    Hardware,
    Layout,
    Model,
    Run,
    breakeven_dp_bps,
    breakeven_pp_bps,
    compute_time,
    cost_multiplier,
    defeat_bandwidth_bps,
    dp_ceiling_flops,
    ideal_chips,
    min_stages,
    plan,
    pp_ceiling_flops,
    ring_factor,
    simulate_ring_sync_seconds,
    step_time,
    throughput_flops,
    utilisation,
)

HW = Hardware()
BIG = Run(Model.dense(1e12), 2e13)  # 1T parameters, 20T tokens: an assumed frontier-scale run
SMALL = Run(Model.dense(1e10), 2e11)  # fits one 64-chip pod, so pure data parallelism is possible


# ---- the constants, by execution ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 2, 3, 5, 8])
def test_ring_allreduce_is_exact_and_each_worker_sends_two_p_minus_one_over_p_of_the_vector(p: int):
    rng = make_rng(p)
    vectors = [rng.standard_normal(p * 7) for _ in range(p)]
    results, sent = prim.ring_allreduce(vectors)
    expected = np.sum(vectors, axis=0)
    assert all(np.allclose(r, expected) for r in results)
    nbytes = vectors[0].astype(np.float64).nbytes
    assert sent == [int(round(float(ring_factor(p)) * nbytes))] * p


def test_ring_allreduce_rejects_ragged_or_indivisible_input():
    with pytest.raises(ValueError):
        prim.ring_allreduce([np.zeros(6), np.zeros(5)])
    with pytest.raises(ValueError):
        prim.ring_allreduce([np.zeros(7), np.zeros(7)])
    with pytest.raises(ValueError):
        prim.ring_allreduce([])


@pytest.mark.parametrize("widths,tokens", [([32, 64, 48, 16], 7), ([8, 8], 1), ([100, 30, 30, 30, 5], 13)])
def test_counted_training_flops_are_six_per_parameter_per_token(widths: list[int], tokens: int):
    flops, params = prim.train_step_flops(widths, tokens, make_rng(3))
    assert flops == FLOPS_PER_PARAM_TOKEN * params * tokens


def test_a_pipeline_boundary_carries_one_activation_and_one_gradient_vector_per_token():
    d, tokens = 256, 33
    nbytes = prim.pipeline_boundary_bytes(d, tokens, make_rng(4))
    assert nbytes == PIPELINE_VECTORS_PER_TOKEN * d * 2 * tokens  # float16 is two bytes


def test_a_decoder_block_has_twelve_d_squared_parameters_and_the_dense_model_obeys_it():
    for d in (64, 1024, 12288):
        assert prim.transformer_block_params(d) == 12 * d * d
    m = Model.dense(1e12, aspect=128.0)
    assert 12 * m.n_layers * m.d_model**2 == pytest.approx(1e12, rel=1e-12)
    assert m.d_model == pytest.approx(128.0 * m.n_layers)


# ---- closed forms ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("replicas,compression,local_steps", [(2, 1.0, 1), (10, 1.0, 1), (100, 10.0, 1), (100, 1.0, 50), (400, 4.0, 8)])
def test_the_dp_break_even_cap_makes_communication_exactly_as_long_as_compute(replicas, compression, local_steps):
    b = breakeven_dp_bps(SMALL, HW, replicas, compression, local_steps)
    kw = dict(compression=compression, local_steps=local_steps, overlap=False)
    t_c = compute_time(SMALL, HW, 64, 1, replicas)
    for factor, expected in ((1.0, 2.0), (2.0, 1.5), (0.5, 3.0)):
        t = step_time(SMALL, HW, 64, External(b * factor), 1, replicas, **kw)
        assert float(t / t_c) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("stages", [2, 8, 64])
def test_the_pipeline_break_even_cap_makes_communication_exactly_as_long_as_compute(stages: int):
    b = breakeven_pp_bps(BIG, HW, stages)
    t = step_time(BIG, HW, 64, External(b), stages, 1, overlap=False)
    assert float(t / compute_time(BIG, HW, 64, stages, 1)) == pytest.approx(2.0, rel=1e-9)


def test_dp_utilisation_depends_on_neither_model_size_nor_pod_size():
    rng = make_rng(11)
    b = 5e8
    values = []
    for _ in range(12):
        n = float(10 ** rng.uniform(9, 12.5))
        s = int(rng.choice([1, 8, 64, 512]))
        run = Run(Model.dense(n), 20 * n)
        u = utilisation(run, HW, s, External(b), Layout(1, 50), overlap=False)
        values.append(u)
        assert u == pytest.approx(1.0 / (1.0 + breakeven_dp_bps(run, HW, 50) / b), rel=1e-9)
    assert max(values) - min(values) < 1e-9


def test_a_per_pod_cap_needs_pod_size_times_more_bandwidth_for_the_same_step_time():
    s = 64
    per_chip = step_time(BIG, HW, s, External(1e9, "per_chip"), 4, 38)
    per_pod = step_time(BIG, HW, s, External(1e9 * s, "per_pod"), 4, 38)
    assert float(per_chip) == pytest.approx(float(per_pod), rel=1e-12)
    with pytest.raises(ValueError):
        External(1e9, "per_rack").pod_bps(s)


def test_dp_ceiling_is_what_data_parallelism_approaches_as_pods_are_added():
    b = 1e9
    ceiling = dp_ceiling_flops(SMALL, 64, b)
    t = step_time(SMALL, HW, 64, External(b), 1, 1e7)
    assert float(throughput_flops(SMALL, t)) == pytest.approx(ceiling, rel=1e-5)
    t_few = step_time(SMALL, HW, 64, External(b), 1, 4)
    assert float(throughput_flops(SMALL, t_few)) == pytest.approx(4 * 64 * HW.effective_flops)  # compute-bound


def test_one_pipeline_cannot_exceed_its_boundary_ceiling_however_many_stages_it_has():
    b = 1e9
    ceiling = pp_ceiling_flops(BIG, 64, b)
    t = step_time(BIG, HW, 64, External(b), 1_000_000, 1)
    assert float(throughput_flops(BIG, t)) == pytest.approx(ceiling, rel=1e-5)


def test_step_time_never_gets_longer_when_the_cap_rises_or_communication_shrinks():
    for overlap in (True, False):
        prev = None
        for b in (1e7, 1e8, 1e9, 1e10, 1e11):
            t = float(step_time(BIG, HW, 64, External(b), 4, 38, overlap=overlap))
            assert prev is None or t <= prev + 1e-12
            prev = t
        prev = None
        for c in (1, 2, 8, 64, 512):
            t = float(step_time(BIG, HW, 64, External(1e8), 4, 38, compression=c, overlap=overlap))
            assert prev is None or t <= prev + 1e-12
            prev = t
        prev = None
        for h in (1, 2, 8, 64, 512):
            t = float(step_time(BIG, HW, 64, External(1e8), 4, 38, local_steps=h, overlap=overlap))
            assert prev is None or t <= prev + 1e-12
            prev = t
    assert float(step_time(BIG, HW, 64, External(1e8), 4, 38, overlap=True)) <= float(
        step_time(BIG, HW, 64, External(1e8), 4, 38, overlap=False)
    )


def test_compression_and_local_steps_matter_only_through_their_product():
    ref = step_time(BIG, HW, 64, External(5e8), 4, 38, compression=10.0, local_steps=1)
    for c, h in ((1.0, 10), (2.0, 5), (5.0, 2)):
        assert float(step_time(BIG, HW, 64, External(5e8), 4, 38, compression=c, local_steps=h)) == pytest.approx(
            float(ref), rel=1e-12
        )


@pytest.mark.parametrize("replicas,vector_bytes,pod_bps", [(2, 2e10, 64_000_000_000), (10, 2e10, 64_000_000_000), (100, 2e12, 64_000_000_000), (7, 5e9, 1_000_000_000)])
def test_a_ring_all_reduce_through_token_buckets_takes_the_closed_form_time(replicas, vector_bytes, pod_bps):
    closed = float(ring_factor(replicas)) * vector_bytes / pod_bps
    assert simulate_ring_sync_seconds(vector_bytes, replicas, pod_bps) == pytest.approx(closed, rel=2e-3)


def test_pipeline_and_data_parallel_traffic_vanish_in_the_degenerate_layouts():
    pure_dp = step_time(SMALL, HW, 64, External(1e9), 1, 1)  # one pod: nothing to communicate
    assert float(pure_dp) == pytest.approx(float(compute_time(SMALL, HW, 64, 1, 1)))
    one_pipe = step_time(BIG, HW, 64, External(1e9), 4, 1)  # one replica: no gradient exchange
    assert float(one_pipe) >= float(compute_time(BIG, HW, 64, 4, 1))


# ---- what the attacker has to buy -------------------------------------------------------------------------


@pytest.mark.parametrize("cap_bps", [2e8, 1e8])
def test_the_plan_search_equals_brute_force_over_the_same_grid(cap_bps: float):
    hw = Hardware(mem_bytes=20e9)
    run = Run(Model.dense(2e10), 4e11, batch_tokens=2e6)
    pod, deadline, ext, compression = 8, 30 * 24 * 3600.0, External(cap_bps), 4.0
    assert min_stages(run, hw, pod) == 2
    needed = run.flops / deadline
    m = run.model
    best = None
    for k in range(2, 7):
        for r in range(1, 401):
            t_c = FLOPS_PER_PARAM_TOKEN * m.n_params * run.batch_tokens / (k * r * pod * hw.effective_flops)
            pipe = (run.batch_tokens / r) * 2 * m.d_model * run.act_bytes
            dp = 0.0 if r == 1 else 2 * (r - 1) / r * (m.n_params / k) * run.grad_bytes / compression
            t = max(t_c, (pipe + dp) / (pod * ext.per_chip_bps))
            if FLOPS_PER_PARAM_TOKEN * m.n_params * run.batch_tokens / t >= needed * (1 - 1e-12):
                chips = k * r * pod
                if best is None or chips < best[0]:
                    best = (chips, k, r)
                break
    found = plan(run, hw, pod, ext, deadline, compression=compression, k_candidates=list(range(2, 7)), r_max=400)
    assert best is not None and found is not None
    assert (found.chips, found.layout.stages, found.layout.replicas) == best
    # a tighter grid cannot do better, and a search that is not allowed to span pods finds nothing
    assert plan(run, hw, pod, ext, deadline, compression=compression, k_candidates=[1], r_max=400) is None


def test_pure_data_parallelism_becomes_feasible_exactly_at_the_closed_form_ceiling():
    deadline = 6 * 3600.0  # short enough that one pod cannot do it alone
    needed = SMALL.flops / deadline
    assert needed > 64 * HW.effective_flops
    r_min = math.ceil(needed / (64 * HW.effective_flops))
    for c, h in ((1.0, 1), (4.0, 3)):
        asymptotic = needed / dp_ceiling_flops(SMALL, 64, 1.0, c, h)  # cap at which R -> infinity just suffices
        flip = asymptotic * (r_min - 1) / r_min  # with R = r_min pods the ceiling is R / (R - 1) higher
        kw = dict(compression=c, local_steps=h, k_candidates=[1], r_max=4096)
        above = plan(SMALL, HW, 64, External(flip * 1.001), deadline, **kw)
        below = plan(SMALL, HW, 64, External(flip * 0.999), deadline, **kw)
        assert above is not None and above.layout.replicas == r_min
        assert above.time_to_train_s <= deadline * (1 + 1e-9)
        # barely feasible means communication-bound: utilisation is the needed rate over the rate R pods could supply
        assert above.utilisation == pytest.approx(needed / (r_min * 64 * HW.effective_flops), rel=2e-3)
        assert below is None


def test_the_cost_of_evading_never_falls_as_the_cap_falls():
    last = 0.0
    for b in (1e11, 1e10, 1e9, 3e8, 1e8, 3e7, 1e7):
        m = cost_multiplier(BIG, HW, 64, External(b), YEAR_S)
        assert m >= last - 1e-9
        last = m
    assert last > 1.5


def test_more_compression_or_local_steps_never_helps_the_defender():
    last = math.inf
    for cl in (1, 4, 16, 64, 256, 1024):
        m = cost_multiplier(BIG, HW, 64, External(2e7), YEAR_S, compression=float(cl))
        assert m <= last + 1e-9
        last = m


def test_overlapping_communication_with_compute_never_needs_more_chips():
    for b in (3e7, 1e8, 1e9):
        with_overlap = plan(BIG, HW, 64, External(b), YEAR_S, overlap=True)
        without = plan(BIG, HW, 64, External(b), YEAR_S, overlap=False)
        assert with_overlap is not None
        assert without is None or with_overlap.chips <= without.chips


def test_the_defeat_bandwidth_is_a_threshold_of_the_cost_multiplier():
    for c in (1.0, 100.0):
        b = defeat_bandwidth_bps(BIG, HW, 64, YEAR_S, multiplier=10.0, compression=c)
        assert 1e3 < b < 1e12
        assert cost_multiplier(BIG, HW, 64, External(b * 1.05), YEAR_S, compression=c) <= 10.0
        assert cost_multiplier(BIG, HW, 64, External(b * 0.95), YEAR_S, compression=c) > 10.0


def test_the_per_pod_defeat_cap_is_the_per_chip_one_times_the_pod_size():
    chip = defeat_bandwidth_bps(BIG, HW, 64, YEAR_S, multiplier=10.0, scope="per_chip")
    pod = defeat_bandwidth_bps(BIG, HW, 64, YEAR_S, multiplier=10.0, scope="per_pod")
    assert pod == pytest.approx(64 * chip, rel=1e-6)


def test_a_model_whose_state_does_not_fit_one_pod_is_forced_across_pods():
    assert min_stages(BIG, HW, 64) == 4  # 16 TB of state against 5.12 TB per pod
    assert min_stages(SMALL, HW, 64) == 1
    p = plan(BIG, HW, 64, External(1e9), YEAR_S)
    assert p is not None and p.layout.stages >= 4
    bigger_pod = min_stages(BIG, HW, 512)
    assert bigger_pod == 1 and bigger_pod < min_stages(BIG, HW, 64)


def test_headline_scenario_at_the_rfd_example_cap_the_frontier_run_needs_almost_no_extra_chips_per_chip_scope():
    # assumption-bound result: 1T parameters, 20T tokens, 16M-token batches, perfect overlap, one year
    assert ideal_chips(BIG, HW, YEAR_S) == pytest.approx(9506, abs=2)
    p = plan(BIG, HW, 64, External(1e9, "per_chip"), YEAR_S)
    assert p is not None and p.chips / ideal_chips(BIG, HW, YEAR_S) < 1.1
    # the same 1 GB/s read as a cap on the whole pod is a different regime
    assert cost_multiplier(BIG, HW, 64, External(1e9, "per_pod"), YEAR_S) > 100.0
    assert cost_multiplier(BIG, HW, 64, External(1e9, "per_pod"), YEAR_S, compression=100.0) < 5.0


def test_the_chip_budget_check_and_the_deadline_move_together():
    short = plan(BIG, HW, 64, External(1e9), YEAR_S / 2)
    long = plan(BIG, HW, 64, External(1e9), YEAR_S)
    assert short is not None and long is not None
    assert short.chips > long.chips
    assert long.time_to_train_s <= YEAR_S * (1 + 1e-9)


# ---- pod size and time ---------------------------------------------------------------------------------------


def test_pod_compute_is_linear_in_pod_size_and_duration():
    base = podsize.pod_compute(8, HW, 1000.0)
    assert podsize.pod_compute(16, HW, 1000.0) == pytest.approx(2 * base)
    assert podsize.pod_compute(8, HW, 3000.0) == pytest.approx(3 * base)
    with pytest.raises(ValueError):
        podsize.pod_compute(0, HW)


def test_max_pod_size_is_the_largest_pod_whose_year_stays_within_the_cap():
    for cap in (1e22, 1e23, 8.2e23, 1e24, 3.3e25):
        s = podsize.max_pod_size(cap, HW)
        assert s == 0 or podsize.pod_compute(s, HW) <= cap * (1 + 1e-12)
        assert podsize.pod_compute(s + 1, HW) > cap
    assert podsize.max_pod_size(1.0, HW) == 0


def test_a_chip_four_times_faster_allows_a_pod_four_times_smaller():
    fast = Hardware(flops=4 * HW.flops)
    for cap in (1e24, 1e25, 3.3e25, 1e26):
        assert podsize.max_pod_size(cap, fast) == podsize.max_pod_size(cap, HW) // 4  # floor(floor(x) / 4) = floor(x / 4)


def test_compute_optimal_parameters_invert_six_n_d_at_twenty_tokens_per_parameter():
    for n in (1e9, 7e10, 1e12):
        c = 6 * n * (20 * n)
        assert podsize.compute_optimal_params(c) == pytest.approx(n, rel=1e-12)


def test_years_until_a_pod_reaches_a_target_follow_the_doubling_time():
    target = 1e25
    t1 = podsize.years_until_pod_reaches(target, 65, HW, doubling_years=1.0)
    assert podsize.years_until_pod_reaches(2 * target, 65, HW, doubling_years=1.0) == pytest.approx(t1 + 1.0)
    assert podsize.years_until_pod_reaches(target, 65, HW, doubling_years=2.0) == pytest.approx(2 * t1)
    assert podsize.years_until_pod_reaches(podsize.pod_compute(65, HW) / 2, 65, HW, 1.0) == 0.0
    with pytest.raises(ValueError):
        podsize.years_until_pod_reaches(target, 65, HW, doubling_years=0.0)


def test_a_sixty_five_chip_pod_under_the_default_assumptions_does_about_eight_times_ten_to_the_23_per_year():
    assert podsize.pod_compute(65, HW) == pytest.approx(8.2e23, rel=0.01)


# ---- multi-pod inference -------------------------------------------------------------------------------------


def test_training_sends_exactly_twice_the_boundary_bytes_of_serving():
    m = BIG.model
    assert inference.pipeline_bytes_per_token(m, training=True) == 2 * inference.pipeline_bytes_per_token(m)


def test_tensor_parallel_across_pods_costs_two_layers_of_ring_traffic_more_than_a_pipeline_boundary():
    m = BIG.model
    for degree in (2, 4, 16):
        ratio = inference.tensor_parallel_bytes_per_token(m, degree) / inference.pipeline_bytes_per_token(m)
        assert ratio == pytest.approx(2 * m.n_layers * float(ring_factor(degree)))
    assert inference.tensor_parallel_bytes_per_token(m, 1) == 0.0


def test_pipeline_serving_across_pods_is_compute_bound_while_tensor_parallel_serving_is_bandwidth_bound():
    m = BIG.model
    compute = inference.compute_tokens_per_s(m, 4 * 64, HW)
    pipeline = inference.boundary_tokens_per_s(inference.pipeline_bytes_per_token(m), 64, 1e9)
    tensor = inference.boundary_tokens_per_s(inference.tensor_parallel_bytes_per_token(m, 2), 64, 1e9)
    assert pipeline > 10 * compute  # the cap does not limit a pipelined server of this model
    assert tensor < compute  # but it does limit a tensor-parallel one


def test_serving_a_model_needs_no_more_pods_than_training_it():
    for n in (1e10, 1e11, 1e12):
        run = Run(Model.dense(n), 20 * n)
        assert inference.weight_pods(run.model, 64, HW, weight_bytes=1.0) <= min_stages(run, HW, 64)
    assert inference.weight_pods(BIG.model, 64, HW, weight_bytes=1.0) == 1
    assert min_stages(BIG, HW, 64) == 4
