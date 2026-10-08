#!/usr/bin/env python3
"""Seeded-bug ("mutation") checks: do a package's tests notice when its code is broken?

For every mutation listed in ``docs/mutations/rfd_NNN.toml`` this script copies ``src/`` to a temporary directory,
applies ONE small textual edit to the copy, runs that package's tests against the copy and reports whether any test
failed. A mutation that no test notices ("SURVIVED") means a claim in the README is not actually guarded.

    python scripts/mutation_check.py 35            # one RFD (number)
    python scripts/mutation_check.py --all         # every RFD that has a mutation file
    python scripts/mutation_check.py 35 --list     # show the mutations without running them

Two controls run first for every package: the unmodified copy must pass (otherwise the harness itself is broken),
and a copy whose ``__init__.py`` raises on import must fail (otherwise the tests are not importing the copy, and a
"caught" verdict would mean nothing). The exit status is non-zero if a control or any mutation misbehaves.

The edit is applied to the first occurrence of ``pattern`` only, and ``pattern`` must occur in the file.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MUT_DIR = ROOT / "docs" / "mutations"


@dataclass(frozen=True)
class Mutation:
    name: str
    file: str
    pattern: str
    replace: str


def package_dir(number: int) -> Path:
    matches = sorted((SRC / "aisecurity_rfds").glob(f"rfd_{number:03d}_*"))
    if not matches:
        raise SystemExit(f"no package for RFD {number:03d}")
    return matches[0]


def load_mutations(number: int) -> list[Mutation]:
    path = MUT_DIR / f"rfd_{number:03d}.toml"
    if not path.is_file():
        raise SystemExit(f"{path.relative_to(ROOT)} does not exist")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return [Mutation(m["name"], m["file"], m["pattern"], m["replace"]) for m in data.get("mutation", [])]


def apply_mutation(tmp_src: Path, number: int, m: Mutation) -> None:
    target = tmp_src / "aisecurity_rfds" / package_dir(number).name / m.file
    text = target.read_text(encoding="utf-8")
    if m.pattern not in text:
        raise SystemExit(f"[{m.name}] pattern not found in {m.file}: {m.pattern!r}")
    if m.pattern == m.replace:
        raise SystemExit(f"[{m.name}] replacement equals the pattern")
    target.write_text(text.replace(m.pattern, m.replace, 1), encoding="utf-8")


def run_tests(number: int, tmp_src: Path, timeout: int) -> tuple[bool, str]:
    """Return (all tests passed, first failing test id or a short note)."""
    test_dir = ROOT / "tests" / package_dir(number).name
    cmd = [
        sys.executable, "-m", "pytest", str(test_dir), "-x", "-q", "-p", "no:cacheprovider",
        "-o", f"pythonpath={tmp_src}", "-o", "addopts=--import-mode=importlib", "-rf",
    ]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s (counts as noticed)"
    if proc.returncode == 0:
        return True, ""
    for line in proc.stdout.splitlines():
        if line.startswith("FAILED") or line.startswith("ERROR"):
            return False, line.split(" - ")[0].replace("FAILED ", "").replace("ERROR ", "").strip()
    return False, f"pytest exit code {proc.returncode}"


def fresh_copy(parent: Path) -> Path:
    dst = parent / "src"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    return dst


def check_package(number: int, timeout: int) -> tuple[int, int, int]:
    """Returns (caught, survived, harness_errors)."""
    mutations = load_mutations(number)
    caught = survived = errors = 0
    with tempfile.TemporaryDirectory(prefix=f"mut{number:03d}_") as tmp:
        parent = Path(tmp)
        # control 1: unmodified copy passes
        ok, note = run_tests(number, fresh_copy(parent), timeout)
        print(f"  control (unmodified copy passes): {'ok' if ok else 'FAILED ' + note}")
        errors += 0 if ok else 1
        # control 2: a copy that cannot be imported fails, so a "caught" verdict really means the copy was imported
        tmp_src = fresh_copy(parent)
        init = tmp_src / "aisecurity_rfds" / package_dir(number).name / "__init__.py"
        init.write_text('raise ImportError("canary: the mutated copy is the one being imported")\n' + init.read_text(encoding="utf-8"))
        ok, note = run_tests(number, tmp_src, timeout)
        print(f"  control (canary import error is noticed): {'ok' if not ok else 'FAILED, tests import some other copy'}")
        errors += 0 if not ok else 1
        for m in mutations:
            tmp_src = fresh_copy(parent)
            apply_mutation(tmp_src, number, m)
            passed, note = run_tests(number, tmp_src, timeout)
            if passed:
                survived += 1
                print(f"  SURVIVED  {m.name}  ({m.file})")
            else:
                caught += 1
                print(f"  caught    {m.name}  ({m.file}) by {note}")
    return caught, survived, errors


def numbers_with_mutations() -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in MUT_DIR.glob("rfd_*.toml"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("number", nargs="?", type=int, help="RFD number, e.g. 35")
    ap.add_argument("--all", action="store_true", help="run every RFD that has a mutation file")
    ap.add_argument("--list", action="store_true", help="list mutations without running tests")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per test run (default 600)")
    args = ap.parse_args()
    numbers = numbers_with_mutations() if args.all else ([args.number] if args.number else [])
    if not numbers:
        ap.error("give an RFD number or --all")
    exit_code = 0
    total_caught = total_survived = 0
    for n in numbers:
        mutations = load_mutations(n)
        print(f"RFD {n:03d}: {len(mutations)} mutation(s)")
        if args.list:
            for m in mutations:
                print(f"  - {m.name}: {m.file}: {m.pattern!r} -> {m.replace!r}")
            continue
        caught, survived, errors = check_package(n, args.timeout)
        total_caught += caught
        total_survived += survived
        print(f"  => {caught} caught, {survived} survived, {errors} harness error(s)")
        if survived or errors:
            exit_code = 1
    if not args.list:
        print(f"TOTAL: {total_caught} caught, {total_survived} survived")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
