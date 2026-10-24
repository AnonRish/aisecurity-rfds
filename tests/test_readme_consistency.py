"""A README may quote only what its package prints, cite only tests and files that exist, and count its open
questions the way ``rfd.toml`` does.

Packages that ship their own ``test_*zz_readme.py`` are checked there (those files also check details specific to
the package); every other package is checked here. A package's entry point is run once and its output cached.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import re
import tomllib
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "aisecurity_rfds"
TESTS = ROOT / "tests"

ALL_PACKAGES = sorted(d.name for d in SRC.glob("rfd_[0-9][0-9][0-9]_*") if (d / "README.md").is_file())

# READMEs written before this file existed. Their figures were checked by hand, not by a test, and they do not yet
# follow the conventions below. They are to be migrated, one at a time, by removing them from this set; it must not grow.
LEGACY_UNCHECKED = {
    "rfd_032_offline_licensing",
    "rfd_034_license_replay_prevention",
    "rfd_035_perf_counter_integrity",
}
GENERIC = [
    n for n in ALL_PACKAGES if n not in LEGACY_UNCHECKED and not any((TESTS / n).glob("test_*zz_readme.py"))
]


@lru_cache(maxsize=None)
def readme(name: str) -> str:
    return (SRC / name / "README.md").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def printed(name: str) -> str:
    entry = importlib.import_module(f"aisecurity_rfds.{name}.__main__")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        entry.main()
    return buf.getvalue()


def section(name: str, title: str, nxt: str) -> str:
    text = readme(name)
    start = text.index(f"## {title}")
    return text[start : text.index(f"## {nxt}")]


def without_fences(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def defined_tests(name: str) -> set[str]:
    found: set[str] = set()
    for f in (TESTS / name).glob("test_*.py"):
        found.update(re.findall(r"^def (test_\w+)\(", f.read_text(encoding="utf-8"), flags=re.M))
    return found


def test_the_legacy_set_can_only_shrink():
    assert len(LEGACY_UNCHECKED) <= 3
    for name in LEGACY_UNCHECKED:
        assert name in ALL_PACKAGES, f"{name} is listed as legacy but is not a package"
        assert not any((TESTS / name).glob("test_*zz_readme.py")), f"{name} has its own README test: drop it from the set"


generic = pytest.mark.parametrize("name", GENERIC)


@pytest.mark.parametrize("name", ALL_PACKAGES)
def test_no_package_test_is_skipped_or_expected_to_fail(name):
    for f in (TESTS / name).glob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"pytest\.mark\.(skip|xfail)|pytest\.skip\(|importorskip", text), f.name


@generic
def test_every_figure_in_backticks_in_the_findings_is_printed(name):
    spans = re.findall(r"`([^`\n]+)`", without_fences(section(name, "Findings", "Open questions")))
    figures = [s for s in spans if s[0].isdigit()]
    assert len(figures) >= 10, "the findings should quote figures from the output"
    missing = sorted({s for s in figures if s not in printed(name)})
    assert not missing, f"the README quotes figures the entry point does not print: {missing}"


@generic
def test_every_table_row_in_a_readme_code_block_is_a_line_of_the_output(name):
    lines = {line.rstrip() for line in printed(name).splitlines()}
    blocks = re.findall(r"```(?:\w*)\n(.*?)```", readme(name), flags=re.S)
    rows = [ln.strip() for block in blocks for ln in block.splitlines() if ln.strip().startswith("|")]
    assert rows, "copy at least one table from the output into the README"
    assert [r for r in rows if r not in lines] == []


@generic
def test_every_test_and_file_the_readme_cites_exists(name):
    prose = without_fences(readme(name))
    cited = set(re.findall(r"`(test_\w+)`", prose))
    assert len(cited) >= 8, "the findings should name the tests that back them"
    assert sorted(cited - defined_tests(name)) == []
    for fname in set(re.findall(r"`(\w+\.py)`", prose)):
        homes = (SRC / name, TESTS / name, ROOT / "scripts")
        assert any((home / fname).is_file() for home in homes), f"{fname} is cited but does not exist"


@generic
def test_every_section_number_the_readme_cites_is_a_numbered_section_of_the_output(name):
    cited = {int(n) for n in re.findall(r"§(\d+)", readme(name))}
    assert cited, "the findings should cite the output sections they come from"
    for n in cited:
        assert re.search(rf"^{n}\. ", printed(name), flags=re.M), f"§{n} is cited but the output has no section {n}"


@generic
def test_every_finding_names_a_test_or_an_output_section(name):
    findings = section(name, "Findings", "Open questions")
    numbered = re.split(r"^(?=\d+\. \*\*)", findings, flags=re.M)[1:]
    assert len(numbered) >= 5
    for item in numbered:
        assert re.search(r"`test_\w+`|§\d", item), item[:80]


@generic
def test_the_open_questions_are_counted_and_quoted_the_way_rfd_toml_has_them(name):
    meta = tomllib.loads((SRC / name / "rfd.toml").read_text(encoding="utf-8"))
    block = section(name, "Open questions", "Not established")
    flat = " ".join(block.split())
    total = meta["open_questions_total"]
    assert len(re.findall(r"^\d+\.\s", block, flags=re.M)) == total
    verdicts = re.findall(r"\*\*(Not addressed|Partially|Answered in the model)", flat)
    assert len(verdicts) == total
    assert meta["open_questions_addressed"] == total - verdicts.count("Not addressed")
    if "open_questions" in meta:
        assert len(meta["open_questions"]) == total
        for quoted in meta["open_questions"]:
            assert " ".join(quoted.split()) in flat, f"open question not quoted verbatim: {quoted}"


@generic
def test_the_readme_counts_the_seeded_bugs_it_ships(name):
    number = int(name.split("_")[1])
    spec = tomllib.loads((ROOT / "docs" / "mutations" / f"rfd_{number:03d}.toml").read_text(encoding="utf-8"))
    claimed = re.search(rf"mutation_check\.py {number}\s+# (\d+) seeded bugs", readme(name))
    assert claimed is not None and int(claimed.group(1)) == len(spec["mutation"])


@generic
def test_the_reproduce_block_names_this_packages_commands(name):
    block = section(name, "Reproduce", "Findings")
    assert f"pytest tests/{name}" in block
    assert f"python -m aisecurity_rfds.{name}" in block


@generic
def test_the_attack_table_quotes_only_figures_the_program_prints(name):
    block = section(name, "Attacks tested", "Related RFDs")
    rows = [ln for ln in block.splitlines() if ln.strip().startswith("|")]
    assert len(rows) >= 5, "a header, a rule and at least three attacks"
    for span in re.findall(r"`([^`\n]+)`", block):
        if span[0].isdigit():
            assert span in printed(name), f"the attack table quotes {span!r}, which the program does not print"
