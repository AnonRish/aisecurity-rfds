"""Plain-text report helpers for the entry points: aligned markdown tables and fixed number formats.

Every entry point prints the numbers its README quotes, and a test compares the two, so the format must be
stable: nothing here depends on the clock, the locale or set ordering.
"""

from __future__ import annotations

from collections.abc import Sequence


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A markdown table with every column padded to its widest cell (``|---|`` rule under the header)."""
    cells = [[str(c) for c in row] for row in rows]
    for row in cells:
        if len(row) != len(headers):
            raise ValueError(f"row has {len(row)} cells but there are {len(headers)} headers: {row}")
    widths = [max([len(h), *(len(r[i]) for r in cells)]) for i, h in enumerate(headers)]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in cells]
    return "\n".join([line, rule, *body])


def sci(x: float, digits: int = 1) -> str:
    """Scientific notation with a fixed mantissa width and no zero-padded exponent: ``3.2e-05`` becomes ``3.2e-5``."""
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def pct(x: float, digits: int = 1) -> str:
    """A fraction as a percentage with a fixed number of decimals."""
    return f"{100 * x:.{digits}f}%"
