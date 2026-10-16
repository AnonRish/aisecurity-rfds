"""The README quotes only figures the entry point prints, copies its tables verbatim, and cites tests that exist.

Named ``zz`` so that it runs last: it executes the whole entry point once (about six seconds).
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

from aisecurity_rfds.rfd_037_secure_chip_binning import __main__ as entry

README = Path(entry.__file__).with_name("README.md").read_text(encoding="utf-8")
TEST_DIR = Path(__file__).parent


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


def test_every_figure_in_backticks_in_the_findings_is_printed_by_the_entry_point(output: str):
    spans = re.findall(r"`([^`\n]+)`", without_fences(section("Findings", "Open questions")))
    figures = [s for s in spans if s[0].isdigit()]
    assert len(figures) > 90, "the findings should quote many figures"
    missing = sorted({s for s in figures if s not in output})
    assert not missing, f"README quotes figures the entry point does not print: {missing}"


def test_every_table_row_in_a_readme_code_block_is_a_line_of_the_output(output: str):
    printed = {line.rstrip() for line in output.splitlines()}
    blocks = re.findall(r"```(?:\w*)\n(.*?)```", README, flags=re.S)
    # blocks inside numbered findings are indented; compare after removing the indentation
    rows = [line.strip() for block in blocks for line in block.splitlines() if line.strip().startswith("|")]
    assert len(rows) >= 30, "expected the copied tables"
    assert [r for r in rows if r not in printed] == []


def test_every_test_the_readme_cites_exists():
    defined: set[str] = set()
    for f in TEST_DIR.glob("test_*.py"):
        defined.update(re.findall(r"^def (test_\w+)\(", f.read_text(encoding="utf-8"), flags=re.M))
    cited = set(re.findall(r"`(test_\w+)`", README))
    assert len(cited) >= 40
    assert sorted(cited - defined) == []


def test_every_file_the_readme_cites_exists():
    pkg = Path(entry.__file__).parent
    cited = set(re.findall(r"`(\w+\.py)`", README))
    assert {"binning.py", "mechanisms.py", "lockout.py", "rma.py", "audit.py", "__main__.py"} <= cited
    for name in cited:
        assert (pkg / name).is_file() or (TEST_DIR / name).is_file(), name


def test_readme_counts_the_open_questions_the_way_rfd_toml_does():
    import tomllib

    meta = tomllib.loads(Path(entry.__file__).with_name("rfd.toml").read_text(encoding="utf-8"))
    block = section("Open questions", "Not established")
    items = re.findall(r"^\d+\.\s", block, flags=re.M)
    assert len(items) == meta["open_questions_total"] == 5
    flat = " ".join(block.split())  # a phrase may wrap across lines
    addressed = re.findall(r"\*\*(Not addressed|Partially|Answered in the model)", flat)
    assert len(addressed) == 5
    assert meta["open_questions_addressed"] == 5 - addressed.count("Not addressed") == 4


def test_the_open_questions_are_quoted_as_the_rfd_words_them():
    block = section("Open questions", "Not established")
    for quoted in (
        "What's the cost premium for separate dies vs. binning?",
        "Can verification authorities realistically inspect chip layouts?",
        "How to handle chips that are legitimately sold at one tier but later RMA'd to another?",
        "Are there legitimate use cases for software-limited features that governance should accommodate?",
        "How to handle feature unlocking that's legitimately sold (like Intel On Demand)?",
    ):
        assert quoted in " ".join(block.split())


def test_every_finding_names_a_test_or_an_output_section():
    findings = section("Findings", "Open questions")
    numbered = re.split(r"^(?=\d+\. \*\*)", findings, flags=re.M)[1:]
    assert len(numbered) == 13
    for item in numbered:
        assert re.search(r"`test_\w+`|§\d", item), item[:80]
