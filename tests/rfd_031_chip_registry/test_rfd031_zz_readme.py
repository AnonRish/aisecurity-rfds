"""The README quotes only figures the entry point prints, copies its tables verbatim, and cites tests that exist.

Named ``zz`` so that it runs last: it executes the whole entry point once (about nine seconds).
"""

from __future__ import annotations

import contextlib
import io
import re
import tomllib
from pathlib import Path

import pytest

from aisecurity_rfds.rfd_031_chip_registry import __main__ as entry

README = Path(entry.__file__).with_name("README.md").read_text(encoding="utf-8")
TEST_DIR = Path(__file__).parent
ROOT = TEST_DIR.parents[1]


@pytest.fixture(scope="module")
def output() -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        entry.main()
    return buf.getvalue()


def section(title: str, nxt: str) -> str:
    start = README.index(f"## {title}")
    return README[start : README.index(f"## {nxt}")]


def without_fences(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def table_rows(text: str) -> list[tuple[str, ...]]:
    """The cells of every markdown table row in ``text``, ignoring padding, header rules and the header itself."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|"):
            cells = tuple(c.strip() for c in line.strip("|").split("|"))
            if not all(re.fullmatch(r"-+", c) for c in cells):
                rows.append(cells)
    return rows


def test_every_figure_in_backticks_in_the_findings_is_printed_by_the_entry_point(output: str):
    spans = re.findall(r"`([^`\n]+)`", without_fences(section("Findings", "Open questions")))
    figures = [s for s in spans if s[0].isdigit()]
    assert len(figures) > 100, "the findings should quote many figures"
    missing = sorted({s for s in figures if s not in output})
    assert not missing, f"README quotes figures the entry point does not print: {missing}"


def test_every_table_row_in_a_readme_code_block_is_a_line_of_the_output(output: str):
    printed = {line.rstrip() for line in output.splitlines()}
    blocks = re.findall(r"```(?:\w*)\n(.*?)```", README, flags=re.S)
    # blocks inside numbered findings are indented; compare after removing the indentation
    rows = [line.strip() for block in blocks for line in block.splitlines() if line.strip().startswith("|")]
    assert len(rows) >= 60, "expected the copied tables"
    assert [r for r in rows if r not in printed] == []


def test_the_attack_table_has_every_attack_the_program_runs_and_only_figures_it_prints(output: str):
    printed = []
    collecting = False  # inside a table whose header is "Attack | Defence under test | Outcome"
    for line in output.splitlines():
        if not line.startswith("|"):
            collecting = False
            continue
        cells = tuple(c.strip() for c in line.strip("|").split("|"))
        if cells[0] == "Attack":
            collecting = True
        elif collecting and not re.fullmatch(r"-+", cells[0]):
            printed.append(cells)
    assert len(printed) == 40 == len(set(printed))
    block = section("Attacks tested", "Related RFDs")
    rows = table_rows(block)[1:]  # the first row is the header
    assert [r for r in printed if r not in rows] == [], "an attack the program runs is missing from the README table"
    assert rows[:40] == printed, "the first 40 rows must be the program's attacks, in the order it runs them"
    extras = rows[40:]
    assert 8 <= len(extras) <= 14
    for row in extras:
        assert len(row) == 3
        for cell in row:
            for span in re.findall(r"`([^`\n]+)`", cell):
                assert span in output, f"extra attack row quotes {span!r}, which the program does not print: {row}"


def test_every_test_the_readme_cites_exists():
    defined: set[str] = set()
    for f in TEST_DIR.glob("test_*.py"):
        defined.update(re.findall(r"^def (test_\w+)\(", f.read_text(encoding="utf-8"), flags=re.M))
    cited = set(re.findall(r"`(test_\w+)`", README))
    assert len(cited) >= 80
    assert sorted(cited - defined) == []


def test_every_file_the_readme_cites_exists():
    pkg = Path(entry.__file__).parent
    cited = set(re.findall(r"`(\w+\.py)`", README))
    assert {"smt.py", "identity.py", "registry.py", "witness.py", "location.py", "scenario.py", "attacks.py", "__main__.py"} <= cited
    for name in cited:
        assert (pkg / name).is_file() or (TEST_DIR / name).is_file(), name


def test_readme_counts_the_open_questions_the_way_rfd_toml_does():
    meta = tomllib.loads(Path(entry.__file__).with_name("rfd.toml").read_text(encoding="utf-8"))
    block = section("Open questions", "Not established")
    items = re.findall(r"^\d+\.\s", block, flags=re.M)
    assert len(items) == meta["open_questions_total"] == 6
    flat = " ".join(block.split())  # a phrase may wrap across lines
    verdicts = re.findall(r"\*\*(Not addressed|Partially|Answered in the model)", flat)
    assert verdicts == ["Not addressed", "Partially", "Partially", "Partially", "Partially", "Answered in the model"]
    assert meta["open_questions_addressed"] == 6 - verdicts.count("Not addressed") == 5


def test_the_open_questions_are_quoted_as_the_rfd_words_them():
    block = " ".join(section("Open questions", "Not established").split())
    for quoted in (
        "How robust are current on-chip keys (NVIDIA GPUs, security modules, PUFs) against sophisticated state actors with physical access?",
        "Should location information be included given transparency-security tradeoffs for military hardware?",
        'Can "logical location" (institutional responsibility) substitute for physical location tracking?',
        "What governance structure should manage the registry? (International body, consortium, distributed?)",
        "How to handle chip destruction, recycling, or repurposing?",
        "What happens when chips are sold or transferred between organizations?",
    ):
        assert quoted in block


def test_every_finding_names_a_test_or_an_output_section():
    findings = section("Findings", "Open questions")
    numbered = re.split(r"^(?=\d+\. \*\*)", findings, flags=re.M)[1:]
    assert len(numbered) == 15
    for item in numbered:
        assert re.search(r"`test_\w+`|§\d", item), item[:80]


def test_the_readme_counts_the_seeded_bugs_it_ships():
    spec = tomllib.loads((ROOT / "docs" / "mutations" / "rfd_031.toml").read_text(encoding="utf-8"))
    claimed = re.search(r"mutation_check\.py 31\s+# (\d+) seeded bugs", README)
    assert claimed is not None and int(claimed.group(1)) == len(spec["mutation"])


def test_the_readme_attack_counts_match_what_the_program_tallies(output: str):
    assert "17 attacks: 16 rejected by a rule, 1 raised as an alert, 0 accepted." in output
    assert (
        "23 attacks: 7 detected by an auditor, 2 refused by a witness, 6 rejected by a client,"
        " 4 fork attempts against a quorum, 4 that passed the check named." in output
    )
    status = " ".join(section("Status", "What is implemented").split())
    assert "runs 40 attacks against it (17 against the rules, 23 against an operator that cheats)" in status
