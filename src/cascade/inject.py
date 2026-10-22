"""Realistic wrong answers to place in the conversation history.

Each error type is a mistake a model or analyst actually makes with filings:

- ``scale``: thousands vs millions (x1000 or /1000);
- ``sign``: the sign flipped (a decrease reported as an increase);
- ``percent``: a ratio written as a percent, or the reverse (x100 or /100);
- ``period``: the value from another column of the same table row (the wrong year);
- ``line_item``: the value from another row of the same table column (the wrong line item);
- ``digits``: two adjacent digits swapped.

``period`` and ``line_item`` need the gold value to appear in the table; otherwise they return None
and the run records the type as not applicable for that turn.
"""

from __future__ import annotations

import random
import re

from .data import Conversation
from .match import match, to_number

ERROR_TYPES = ("scale", "sign", "percent", "period", "line_item", "digits")


def _cell_number(cell: str) -> float | None:
    cleaned = re.sub(r"[^0-9.\-%()]", "", cell)
    return to_number(cleaned) if cleaned else None


def _find_cell(table: list[list[str]], value: float) -> tuple[int, int] | None:
    for r, row in enumerate(table):
        for c, cell in enumerate(row[1:], start=1):
            v = _cell_number(cell)
            if v is not None and match(v, value):
                return r, c
    return None


def _swap_digits(value: float, rng: random.Random) -> float | None:
    s = f"{value:.6g}"
    # Only swap among the first three significant digits: a swap further right changes the value
    # by less than the 0.5% matching tolerance, so it would not count as an error.
    digit_pos = [i for i, ch in enumerate(s) if ch.isdigit() and not (ch == "0" and all(x in "0.-" for x in s[:i]))][:3]
    idx = [i for i in digit_pos if i + 1 in digit_pos and s[i] != s[i + 1]]
    if not idx:
        return None
    i = rng.choice(idx)
    out = s[:i] + s[i + 1] + s[i] + s[i + 2 :]
    try:
        return float(out)
    except ValueError:
        return None


def corrupt(value: float | str, kind: str, conv: Conversation, rng: random.Random) -> float | str | None:
    """Return a wrong value of the given kind, or None if the kind does not apply."""
    if isinstance(value, str):  # yes/no answers: only a flip applies
        return ("no" if value == "yes" else "yes") if kind == "sign" else None
    if kind == "scale":
        out = value * 1000 if abs(value) < 1000 or rng.random() < 0.5 else value / 1000
    elif kind == "sign":
        out = -value if value != 0 else None
    elif kind == "percent":
        out = value * 100 if abs(value) < 5 else value / 100
    elif kind in ("period", "line_item"):
        pos = _find_cell(conv.table, value)
        if pos is None:
            return None
        r, c = pos
        if kind == "period":
            cands = [_cell_number(x) for j, x in enumerate(conv.table[r][1:], start=1) if j != c]
        else:
            cands = [_cell_number(row[c]) for i, row in enumerate(conv.table) if i != r and c < len(row)]
        cands = [x for x in cands if x is not None and not match(x, value)]
        out = rng.choice(cands) if cands else None
    elif kind == "digits":
        out = _swap_digits(value, rng)
    else:
        raise ValueError(f"unknown error type {kind}")
    if out is None or match(out, value):
        return None
    return out


def fmt(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value)}"
    return f"{value:.6g}"
