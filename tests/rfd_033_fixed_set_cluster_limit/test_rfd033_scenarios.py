"""The attack table and the stale-window experiment (the numbers the entry point prints and the README quotes)."""

from __future__ import annotations

import math
import re

import pytest

from aisecurity_rfds.rfd_033_fixed_set_cluster_limit import __main__ as entry
from aisecurity_rfds.rfd_033_fixed_set_cluster_limit import scenarios as sc

EXPECTED = {
    "chip of another pod claims membership": "blocked",
    "manifest signed by a key that is not the manufacturer's": "blocked",
    "pod A's manifest loaded into pod B's chips": "blocked",
    "validly signed 2-chip pod offered to cap-1 chips": "blocked",
    "old manifest replayed after an update": "blocked",
    "flash image from before an update restored": "blocked",
    "flash image restored after a tamper response": "blocked",
    "burn supply starved, tamper response, flash restored": "succeeds",
    "recorded handshake replayed": "blocked",
    "chip's own hello reflected back at it": "blocked",
    "initiator identity swapped in flight": "blocked",
    "release authorisation replayed": "blocked",
    "removed chip talks to a member that has not updated": "partial",
    "flash wiped, then a manifest for another pod (flash-bound chip)": "succeeds",
    "flash wiped, then a manifest for another pod (fuse-bound chip)": "blocked",
}


def table_rows(text: str) -> list[list[str]]:
    """Body rows of every markdown table in ``text`` (a header is the row followed by a ``|---`` rule)."""
    lines = text.splitlines()
    rows = []
    for i, line in enumerate(lines):
        is_header = i + 1 < len(lines) and lines[i + 1].startswith("|-")
        if line.startswith("| ") and not is_header:
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


# ---- the attack table ------------------------------------------------------------------------------------------


def test_attack_table_outcomes_are_pinned():
    assert {v.attack: v.outcome for v in sc.all_verdicts()} == EXPECTED


def test_attack_table_is_deterministic_and_has_no_duplicate_rows():
    first, second = sc.all_verdicts(), sc.all_verdicts()
    assert first == second
    assert len({v.attack for v in first}) == len(first) == 15


def test_every_outcome_is_one_of_three_words_and_every_row_names_what_stops_it():
    for v in sc.all_verdicts():
        assert v.outcome in {"blocked", "partial", "succeeds"}
        assert v.defence.strip()


def test_the_rows_where_the_attacker_gets_something_are_the_documented_limitations():
    open_rows = {v.attack for v in sc.all_verdicts() if v.outcome != "blocked"}
    assert open_rows == {
        "burn supply starved, tamper response, flash restored",
        "removed chip talks to a member that has not updated",
        "flash wiped, then a manifest for another pod (flash-bound chip)",
    }


def test_the_same_flash_wipe_that_moves_a_flash_bound_chip_does_not_move_a_fuse_bound_one():
    assert sc.flash_wiped_then_moved_flash_bound().outcome == "succeeds"
    assert sc.flash_wiped_then_moved_fuse_bound().outcome == "blocked"


# ---- the stale window ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4, 8, 16])
def test_every_member_except_the_updated_one_still_accepts_the_removed_chip_until_gossip_runs(n: int):
    w = sc.stale_window(n, seed=1)
    assert w.accepted_before == n - 2
    assert w.exposed_per_round[0] == n - 2
    assert w.accepted_after == 0


def test_without_any_gossip_round_the_probe_still_finds_the_stale_members():
    w = sc.stale_window(8, seed=0, max_rounds=0)
    assert w.rounds_to_full_adoption == 0
    assert w.accepted_after == 6  # the probe is not hard-wired to report success
    assert w.exposed_per_round == (6,)


@pytest.mark.parametrize("n,seed", [(8, 0), (16, 3), (32, 7)])
def test_exposure_never_grows_and_ends_at_zero(n: int, seed: int):
    w = sc.stale_window(n, seed)
    e = w.exposed_per_round
    assert len(e) == w.rounds_to_full_adoption + 1
    assert all(a >= b for a, b in zip(e, e[1:]))
    assert e[-1] == 0 and e[0] == n - 2


@pytest.mark.parametrize("n", [8, 16, 32])
def test_spreading_one_manifest_takes_a_few_logarithmic_rounds(n: int):
    rounds = [sc.stale_window(n, seed).rounds_to_full_adoption for seed in range(5)]
    assert max(rounds) <= 4 * math.log2(n)
    assert min(rounds) >= 1


def test_the_experiment_is_a_function_of_the_seed_and_the_seed_matters():
    assert sc.stale_window(16, 4) == sc.stale_window(16, 4)
    traces = {sc.stale_window(16, s).exposed_per_round for s in range(8)}
    assert len(traces) > 1


def test_a_pod_too_small_to_have_two_remaining_members_is_rejected():
    for n in (0, 1, 2):
        with pytest.raises(ValueError):
            sc.stale_window(n, 0)


# ---- the entry point's own sections ----------------------------------------------------------------------------


def test_entry_point_constants_table_counts_equal_what_the_model_assumes(capsys: pytest.CaptureFixture[str]):
    entry.section_constants()
    rows = table_rows(capsys.readouterr().out)
    assert len(rows) == 7
    for _, counted, assumed, _ in rows:
        assert float(counted) == float(assumed)


def test_entry_point_limiter_section_reports_every_bound_respected(capsys: pytest.CaptureFixture[str]):
    entry.section_limiter()
    out = capsys.readouterr().out
    rows = table_rows(out)
    greedy = [r for r in rows if r[-1] in {"yes", "NO"}]
    assert len(greedy) == 3 and all(r[-1] == "yes" for r in greedy)
    for r in greedy:
        assert int(r[3]) <= int(r[4])
    ratio = float(re.search(r"highest carried/allowed ratio ([0-9.]+)", out).group(1))  # type: ignore[union-attr]
    assert 0.9 < ratio <= 1.0  # the bound is nearly tight and never exceeded
    shared = re.search(r"(\d+) bytes in 2 s against a bound of (\d+)", out)
    assert shared is not None and int(shared.group(1)) <= int(shared.group(2))
    ring = [r for r in rows if len(r) == 6 and r[-1] not in {"yes", "NO"}]
    assert len(ring) == 4 and all(float(r[5]) < 5e-3 for r in ring)


def test_entry_point_membership_section_counts_its_rows(capsys: pytest.CaptureFixture[str]):
    entry.section_membership()
    out = capsys.readouterr().out
    assert "rows: 15; blocked 12, partial 1, attacker succeeds 2" in out


def test_entry_point_key_extraction_merges_exactly_the_pods_that_lost_a_key(capsys: pytest.CaptureFixture[str]):
    entry.section_key_extraction()
    got = {r[0]: int(r[1]) for r in table_rows(capsys.readouterr().out)}
    assert got["none"] == 8 and got["one key from pod 1"] == 8 and got["all 8 keys of pod 1"] == 8
    for m in (2, 3, 4, 8):
        assert got[f"one key from each of {m} pods"] == 8 * m


def test_entry_point_gate_matrix_marks_only_key_gating_as_surviving_firmware_and_frame_attacks(
    capsys: pytest.CaptureFixture[str],
):
    entry.section_gate()
    rows = {r[0]: r[1:] for r in table_rows(capsys.readouterr().out)}
    assert rows["handshake then AEAD frames"] == ["holds", "holds", "holds", "BYPASSED"]
    assert rows["firmware limiter"] == ["holds", "BYPASSED", "BYPASSED", "holds"]
    assert rows["handshake then plain frames"] == ["holds", "BYPASSED", "BYPASSED", "BYPASSED"]


def test_entry_point_number_formatting():
    assert entry.bps(1e9) == "1 GB/s"
    assert entry.bps(7.491e7) == "74.9 MB/s"
    assert entry.bps(2.4e3) == "2.4 kB/s"
    assert entry.bps(3.0) == "3 B/s"
    assert entry.bps(math.inf) == "never"
    assert entry.mult(math.inf) == "none"
    assert entry.mult(1.0234) == "1.02x"
    assert entry.mult(2730.4) == "2730x"
    assert entry.mult(613.2) == "613x"
    assert entry.mult(8.0) == "8x"


def test_entry_point_table_helper_pads_columns_and_keeps_the_header_rule():
    out = entry._table(["a", "bb"], [[1, "x"], ["long", 22]])
    lines = out.splitlines()
    assert lines[0] == "| a    | bb |"
    assert lines[1] == "|------|----|"
    assert lines[2] == "| 1    | x  |"
    assert lines[3] == "| long | 22 |"
