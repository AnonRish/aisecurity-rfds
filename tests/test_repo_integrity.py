"""Mechanical enforcement of docs/CONVENTIONS.md for all 42 RFD packages.

Run one package with e.g. ``pytest tests/test_repo_integrity.py -k 005``.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "aisecurity_rfds"
TESTS = ROOT / "tests"
INDEX = json.loads((ROOT / "docs" / "rfd_index.json").read_text())
BY_NUMBER = {e["number"]: e for e in INDEX}

KINDS = {"reference-implementation", "simulation", "analysis"}
MIN_TESTS = 15
ALLOWED_THIRD_PARTY = {"numpy", "scipy", "cryptography"}
ALLOWED_TEST_ONLY = {"pytest", "hypothesis"}

README_SECTIONS = [
    "status",
    "what is implemented",
    "reproduce",
    "findings",
    "open questions",
    "not established",
    "attacks tested",
    "related rfds",
]

OVERCLAIM = re.compile(
    r"\b(solved|solves|production[- ]ready|unbreakable|provably secure|military[- ]grade|100\s?% (?:secure|safe))\b",
    re.IGNORECASE,
)
NEGATION = re.compile(r"(\bnot\b|\bno\b|\bnever\b|\bneither\b|\bcannot\b|n't)", re.IGNORECASE)


def pkg_dir(number: int) -> Path | None:
    matches = sorted(SRC.glob(f"rfd_{number:03d}_*"))
    return matches[0] if matches else None


REQUIRE_ALL = os.environ.get("AISEC_REQUIRE_ALL") == "1"


def skip_if_not_started(number: int) -> None:
    """RFDs without a package yet are reported as skipped (and listed as 'not started' in
    docs/STATUS.md). Set AISEC_REQUIRE_ALL=1 to turn every missing package into a failure."""
    if pkg_dir(number) is None and not REQUIRE_ALL:
        pytest.skip(f"RFD {number:03d}: not started (see docs/STATUS.md); AISEC_REQUIRE_ALL=1 makes this fail")


def test_index_is_complete_and_unique():
    assert [e["number"] for e in INDEX] == list(range(1, 43))
    assert len({e["slug"] for e in INDEX}) == 42
    for e in INDEX:
        assert re.fullmatch(r"[a-z0-9_]+", e["slug"]), e
        assert e["state"] in {"idea", "discussing"}


def test_no_stray_rfd_packages():
    expected = {f"rfd_{e['number']:03d}_{e['slug']}" for e in INDEX}
    found = {p.name for p in SRC.glob("rfd_*") if p.is_dir()}
    assert found <= expected, f"unexpected package dirs: {sorted(found - expected)}"


def _module_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _full_imports(path: Path) -> set[str]:
    names: set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def _count_tests(test_dir: Path) -> int:
    n = 0
    for f in test_dir.rglob("test_*.py"):
        for node in ast.walk(ast.parse(f.read_text(), filename=str(f))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                n += 1
    return n


@pytest.mark.parametrize("number", range(1, 43), ids=lambda n: f"rfd-{n:03d}")
def test_rfd_entry_point_runs_deterministically(number: int):
    """`python -m aisecurity_rfds.rfd_NNN_<slug>` exits 0, prints findings, and is reproducible."""
    import subprocess

    skip_if_not_started(number)
    entry = BY_NUMBER[number]
    module = f"aisecurity_rfds.rfd_{number:03d}_{entry['slug']}"
    outputs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-m", module],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=ROOT,
            env={**__import__("os").environ, "PYTHONHASHSEED": "0"},
        )
        assert proc.returncode == 0, f"{module} exited {proc.returncode}: {proc.stderr[-800:]}"
        assert len(proc.stdout.strip()) >= 200, f"{module} printed too little to be a findings report"
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1], f"{module} output is not deterministic between runs"


@pytest.mark.parametrize("number", range(1, 43), ids=lambda n: f"rfd-{n:03d}")
def test_rfd_package_meets_conventions(number: int):
    skip_if_not_started(number)
    entry = BY_NUMBER[number]
    d = pkg_dir(number)
    assert d is not None, f"missing package src/aisecurity_rfds/rfd_{number:03d}_*"
    assert d.name == f"rfd_{number:03d}_{entry['slug']}", f"directory must be rfd_{number:03d}_{entry['slug']}"

    for required in ("__init__.py", "__main__.py", "README.md", "rfd.toml"):
        assert (d / required).is_file(), f"{d.name}/{required} missing"

    # ---- rfd.toml -------------------------------------------------------------------------
    meta = tomllib.loads((d / "rfd.toml").read_text())
    assert meta.get("number") == number
    assert meta.get("slug") == entry["slug"]
    assert meta.get("title") == entry["title"]
    assert meta.get("author") == entry["author"]
    assert meta.get("upstream_state") == entry["state"]
    assert meta.get("kind") in KINDS, f"kind must be one of {sorted(KINDS)}"
    summary = meta.get("summary")
    assert isinstance(summary, str) and len(summary.strip()) >= 30, "summary must be a real sentence"
    not_est = meta.get("not_established")
    assert isinstance(not_est, list) and len(not_est) >= 2
    assert all(isinstance(s, str) and len(s.strip()) >= 30 for s in not_est), "not_established entries must be specific"
    total, addressed = meta.get("open_questions_total"), meta.get("open_questions_addressed")
    assert isinstance(total, int) and isinstance(addressed, int) and 0 <= addressed <= total and total >= 1

    # ---- README ---------------------------------------------------------------------------
    readme = (d / "README.md").read_text()
    first = readme.lstrip().splitlines()[0]
    assert re.match(rf"#\s+RFD\s+0*{number}\b", first), f"README must start with '# RFD {number:03d} - ...': {first!r}"
    headings = [h.lower() for h in re.findall(r"^##\s+(.+)$", readme, flags=re.MULTILINE)]
    for section in README_SECTIONS:
        assert any(section in h for h in headings), f"README missing a '## ...{section}...' section"
    assert len(readme.split()) >= 400, "README is too thin to be an honest write-up"

    # ---- overclaim lint -------------------------------------------------------------------
    texts = {"README.md": readme, "rfd.toml": (d / "rfd.toml").read_text()}
    init_doc = ast.get_docstring(ast.parse((d / "__init__.py").read_text())) or ""
    texts["__init__ docstring"] = init_doc
    assert init_doc.strip(), "__init__.py needs a module docstring"
    for name, text in texts.items():
        for line in text.splitlines():
            if OVERCLAIM.search(line) and not NEGATION.search(line):
                pytest.fail(f"overclaiming language in {name}: {line.strip()!r}")

    # ---- imports --------------------------------------------------------------------------
    stdlib = set(sys.stdlib_module_names)
    for f in d.rglob("*.py"):
        for full in _full_imports(f):
            if full.startswith("aisecurity_rfds."):
                assert full == "aisecurity_rfds.common" or full.startswith("aisecurity_rfds.common."), (
                    f"{f.name} imports {full}; packages may only import aisecurity_rfds.common"
                )
        for root in _module_roots(f):
            if root == "aisecurity_rfds":
                continue
            assert root in stdlib or root in ALLOWED_THIRD_PARTY, f"{f.name}: disallowed import {root!r}"

    # ---- seeded bugs (checked statically here; run by scripts/mutation_check.py) -----------
    assert (ROOT / "docs" / "mutations" / f"rfd_{number:03d}.toml").is_file(), (
        f"docs/mutations/rfd_{number:03d}.toml missing: list the seeded bugs that scripts/mutation_check.py must see caught"
    )

    # ---- tests ----------------------------------------------------------------------------
    test_dir = TESTS / d.name
    assert test_dir.is_dir(), f"tests/{d.name}/ missing"
    assert _count_tests(test_dir) >= MIN_TESTS, f"need at least {MIN_TESTS} test functions in tests/{d.name}/"
    for f in test_dir.rglob("*.py"):
        for root in _module_roots(f):
            if root == "aisecurity_rfds":
                continue
            assert root in stdlib or root in ALLOWED_THIRD_PARTY or root in ALLOWED_TEST_ONLY or root == "conftest", (
                f"tests/{d.name}/{f.name}: disallowed import {root!r}"
            )
        for full in _full_imports(f):
            if full.startswith("aisecurity_rfds."):
                other = full.split(".")[1]
                assert other == "common" or other == d.name, f"{f.name} imports foreign package {full}"
