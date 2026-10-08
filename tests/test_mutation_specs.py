"""Static checks on docs/mutations/*.toml (the seeded-bug lists used by scripts/mutation_check.py).

Running the mutants is slow, so CI does not; but a mutation whose pattern has drifted out of the source would silently
stop guarding anything, so the cheap part is checked here: every pattern still occurs in the file it names.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "aisecurity_rfds"
SPECS = sorted((ROOT / "docs" / "mutations").glob("rfd_*.toml"))


def _package(number: int) -> Path | None:
    matches = sorted(SRC.glob(f"rfd_{number:03d}_*"))
    return matches[0] if matches else None


def test_there_is_at_least_one_mutation_file():
    assert SPECS, "docs/mutations/ is empty"


@pytest.mark.parametrize("spec", SPECS, ids=lambda p: p.stem)
def test_mutation_spec_is_well_formed_and_still_matches_the_source(spec: Path):
    number = int(re.fullmatch(r"rfd_(\d{3})", spec.stem).group(1))  # type: ignore[union-attr]
    pkg = _package(number)
    assert pkg is not None, f"{spec.name}: no package for RFD {number:03d}"
    data = tomllib.loads(spec.read_text(encoding="utf-8"))
    mutations = data.get("mutation", [])
    assert len(mutations) >= 3, f"{spec.name}: list at least three seeded bugs"
    names = [m["name"] for m in mutations]
    assert len(names) == len(set(names)), f"{spec.name}: duplicate mutation names"
    for m in mutations:
        assert set(m) == {"name", "file", "pattern", "replace"}, f"{spec.name}: unexpected keys in {m['name']!r}"
        assert m["pattern"] != m["replace"], f"{spec.name}: {m['name']!r} changes nothing"
        target = pkg / m["file"]
        assert target.is_file(), f"{spec.name}: {m['file']} does not exist in {pkg.name}"
        assert m["pattern"] in target.read_text(encoding="utf-8"), (
            f"{spec.name}: pattern for {m['name']!r} no longer occurs in {m['file']}"
        )
