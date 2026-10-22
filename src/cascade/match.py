"""Answer extraction and matching.

Two matchers are reported:
- ``strict``: the same number, to a relative tolerance of 0.5% (absolute 1e-6 near zero), or the same
  number after rounding to 5 decimal places (ConvFinQA stores gold answers rounded to 5 decimals,
  so 0.000514 must match a gold of 0.00051);
- ``lenient``: also accepts a percentage written as a percent (14.2) when the gold is a ratio (0.142),
  and the reverse. This is common in LLM evaluations of FinQA; it is reported as a sensitivity check.

The primary metric is ``strict``. This is fixed in PREREGISTRATION.md.
"""

from __future__ import annotations

import math
import re

# Signed scientific notation (6.4748e+06) is read whole. Until 20 Oct 2026 the exponent was dropped (6.4748e+06 read as
# 6.4748); the frozen result rows keep the values parsed then, and the parser sensitivity report (CHANGELOG R53)
# re-reads every stored reply with this rule.
_EXP = r"(?:[eE][+-]?\d+)?"
_NUM_RE = re.compile(rf"-?\(?\$?\d[\d,]*(?:\.\d+)?{_EXP}\)?%?|-?\$?\.\d+{_EXP}%?")
_ANSWER_RE = re.compile(r"answer\s*[:=]\s*(.+)", re.IGNORECASE)

REL_TOL = 0.005
ABS_TOL = 1e-6


def to_number(text: str) -> float | None:
    t = text.strip().replace(",", "").replace("$", "")
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    pct = t.endswith("%")
    t = t.rstrip("%")
    try:
        v = float(t)
    except ValueError:
        return None
    if neg:
        v = -v
    return v / 100.0 if pct else v


def extract_answer(text: str) -> float | str | None:
    """Take the value after the last 'Answer:' line; otherwise the last number in the text."""
    if text is None:
        return None
    lines = _ANSWER_RE.findall(text)
    target = lines[-1] if lines else text
    low = target.strip().lower()
    if low.startswith(("yes", "no")) and lines:
        return "yes" if low.startswith("yes") else "no"
    nums = _NUM_RE.findall(target)
    if not nums:
        return None
    # When an Answer: line exists, take its first number; otherwise the last number of the reply.
    raw = nums[0] if lines else nums[-1]
    return to_number(raw)


GOLD_DECIMALS = 5


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL) or round(a, GOLD_DECIMALS) == round(b, GOLD_DECIMALS)


def match(pred: float | str | None, gold: float | str, mode: str = "strict") -> bool:
    if pred is None:
        return False
    if isinstance(gold, str) or isinstance(pred, str):
        return str(pred).lower() == str(gold).lower()
    if _close(pred, gold):
        return True
    if mode == "lenient":
        return _close(pred / 100.0, gold) or _close(pred * 100.0, gold)
    return False
