"""The egress limiter: exact token-bucket guarantees and the two-class port (RFD 033: bandwidth limiting)."""

from __future__ import annotations

import pytest

from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit.cluster import build_fleet, establish
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit.limiter import NS, ChipPort, TokenBucket


@pytest.mark.parametrize("seed", range(8))
def test_no_window_ever_carries_more_than_burst_plus_rate_times_its_length(seed: int):
    rng = make_rng(seed)
    rate = int(rng.integers(1_000, 5_000_000))
    burst = int(rng.integers(1, 20_000))
    bucket = TokenBucket(rate, burst)
    now = 0
    log: list[tuple[int, int]] = []
    for _ in range(300):
        now += int(rng.choice([0, 1, 10, 1_000, 100_000, 5_000_000, 200_000_000]))
        want = int(rng.integers(0, 3 * burst + 1))
        got = bucket.take_up_to(want, now)
        assert 0 <= got <= want
        log.append((now, got))
    # from the start: everything granted up to time t is at most burst + rate * t
    running = 0
    for t, g in log:
        running += g
        assert running * NS <= burst * NS + rate * t
    # and for every window (i, j]: grants strictly after i up to j
    prefix = [0]
    for _, g in log:
        prefix.append(prefix[-1] + g)
    for i in range(len(log)):
        for j in range(i + 1, len(log)):
            carried = prefix[j + 1] - prefix[i + 1]
            assert carried * NS <= burst * NS + rate * (log[j][0] - log[i][0])


def test_a_flood_gets_exactly_the_burst_plus_the_rate():
    rate, burst = 10**9, 10**6
    bucket = TokenBucket(rate, burst)
    total = 0
    for step in range(0, 10_000 + 1):  # every millisecond for ten seconds
        total += bucket.take_up_to(1 << 40, step * 1_000_000)
    assert total == burst + rate * 10


def test_fractional_rates_neither_leak_nor_lose_bytes_through_rounding():
    bucket = TokenBucket(3, 1)  # 3 bytes per second, one byte of burst
    total = sum(bucket.take_up_to(1, i * 1_000_000) for i in range(100_001))  # 100 s, polled every millisecond
    assert total * NS <= 1 * NS + 3 * 100 * NS  # never more than the exact bound
    assert total >= 299  # and polling faster than a byte period wastes less than a byte overall
    coarse = TokenBucket(3, 1)
    slow = sum(coarse.take_up_to(1, i * 100_000_000) for i in range(1001))  # polled every 100 ms
    assert slow < total  # a one-byte burst cannot bank the fractions that coarse polling skips: conservative


def test_a_clock_that_goes_backwards_earns_no_credit_and_cannot_be_used_to_double_dip():
    bucket = TokenBucket(1000, 1000)
    assert bucket.take_up_to(1000, 0) == 1000
    assert bucket.take_up_to(1000, NS) == 1000  # a full second refills the bucket
    assert bucket.take_up_to(1000, NS // 2) == 0  # rolled back: nothing
    assert bucket.take_up_to(1000, NS) == 0  # returning to the old time earns nothing either
    assert bucket.take_up_to(1000, NS + NS // 2) == 500  # only genuinely new time counts


def test_idle_time_never_banks_more_than_the_burst():
    bucket = TokenBucket(10**6, 5000)
    assert bucket.take_up_to(10**9, 3600 * NS) == 5000


def test_stream_completes_when_the_rate_says_it_should():
    rate, burst, n = 10**6, 10_000, 5_000_000
    bucket = TokenBucket(rate, burst)
    done_ns = bucket.stream(n, 0)
    chunks = -(-n // burst)
    ideal = (n - burst) * NS // rate
    assert ideal <= done_ns <= ideal + chunks
    assert bucket.granted_total == n


def test_try_send_is_all_or_nothing_and_a_refusal_changes_nothing():
    bucket = TokenBucket(1000, 100)
    assert bucket.try_send(100, 0)
    assert not bucket.try_send(1, 0)
    assert bucket.take_up_to(5, 0) == 0
    assert bucket.try_send(100, NS // 10)  # 100 ms later exactly 100 bytes have accrued


def test_wait_ns_is_the_time_until_the_send_would_succeed():
    bucket = TokenBucket(1000, 100)
    bucket.take_up_to(100, 0)
    w = bucket.wait_ns(50, 0)
    assert w == 50 * NS // 1000
    assert not bucket.try_send(50, w - 1)
    assert bucket.try_send(50, w)
    with pytest.raises(ValueError):
        bucket.wait_ns(101, w)


def test_invalid_buckets_and_sizes_are_rejected():
    for rate, burst in ((0, 1), (1, 0), (-5, 10)):
        with pytest.raises(ValueError):
            TokenBucket(rate, burst)
    with pytest.raises(ValueError):
        TokenBucket(10, 10).take_up_to(-1, 0)


def test_splitting_traffic_over_many_destinations_does_not_multiply_the_external_limit():
    fleet = build_fleet(1, 2)
    port = ChipPort(fleet.chips[0], external_bps=1_000_000, burst_ns=1_000_000)  # 1 MB/s, 1 kB burst
    destinations = [bytes([i]) * 32 for i in range(1, 51)]
    total = 0
    for step in range(0, 2001):  # two seconds, polled every millisecond
        for dest in destinations:
            total += port.send(dest, 10**6, step * 1_000_000)
    bound = port.external.burst_bytes + 1_000_000 * 2
    assert total <= bound  # fifty destinations share one bucket
    assert total >= bound - 2000  # and the limiter does not starve traffic it should allow


def test_authenticated_pod_peers_get_the_pod_rate_and_everyone_else_the_external_rate():
    fleet = build_fleet(2, 3)
    a, b, outsider = fleet.chips[0], fleet.chips[1], fleet.chips[3]
    establish(a, b)
    port = ChipPort(a, external_bps=10**6, internal_bps=10**9, burst_ns=1_000_000)
    assert port.send(b.device_id, 10**9, 0) == 10**6  # one millisecond of pod bandwidth
    assert port.send(outsider.device_id, 10**9, 0) == 1000  # one millisecond of external bandwidth
    assert port.send(b.device_id, 10**9, NS) == 10**6  # refilled to the pod burst, not beyond


def test_a_member_without_a_live_link_is_treated_as_external_traffic():
    fleet = build_fleet(1, 3, seed=b"p")
    a, b, c = fleet.chips
    port = ChipPort(a, external_bps=10**6, internal_bps=10**9, burst_ns=1_000_000)
    assert port.send(b.device_id, 10**9, 0) == 1000  # same pod, but no handshake yet
    establish(a, b)
    assert port.send(b.device_id, 10**9, NS) == 10**6
    a.tick(1000)  # the link expires without re-verification
    assert port.send(b.device_id, 10**9, 2 * NS) == 1000


def test_a_destination_that_merely_names_a_pod_member_gets_no_pod_bandwidth_from_another_chip():
    fleet = build_fleet(2, 2, seed=b"q")
    a, b = fleet.chips[0], fleet.chips[1]
    c = fleet.chips[2]  # in the other pod
    establish(a, b)
    port_c = ChipPort(c, external_bps=10**6, internal_bps=10**9, burst_ns=1_000_000)
    assert port_c.send(b.device_id, 10**9, 0) == 1000  # c has no link to b, whoever b is in a's eyes
