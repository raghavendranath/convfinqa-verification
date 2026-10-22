"""The revised program breaker, ``program_v2`` (PREREGISTRATION change 4; CHANGELOG R37, R38).

A prospectively specified exploratory extension, separate from the pre-registered H4 experiment. The frozen
``program`` breaker (``breakers.check``) is not changed. Rules fixed in PREREGISTRATION change 4 Part A
(``paper/CHANGE4.md`` section 2):

- the model resolves the question (asked, unit, format), cites every input as a table cell, a quote from the
  report text or a quote from the question, and writes a DSL program from the cited numbers and constants, or
  writes ``Verdict: unable``;
- every citation is validated against its source, and every non-constant literal in the program must equal a
  cited number; numbers are compared as written (``$``, commas and ``%`` ignored, parentheses negative) with the
  strict matcher; labels and quotes are compared after lower-casing, removing punctuation and collapsing
  whitespace;
- outcome ``agree`` (valid program, value matches the checked answer strictly), ``disagree`` (trips) or
  ``unable`` (any failed check), with one reason code.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .data import Conversation
from .dsl import TABLE_OPS, ProgramError, execute, parse
from .match import match

REASONS = (
    "truncated",
    "model_unable",
    "no_program",
    "cell_not_found",
    "cell_value_mismatch",
    "text_quote_not_found",
    "number_not_in_quote",
    "question_quote_not_found",
    "uncited_literal",
    "program_error",
)
SOURCES = ("table", "text", "question")

# Optional list marker before a line: a step number ("4. "), a bullet, bold markers (code fix, 17 Oct 2026, R39).
_PREFIX = r"^\s*(?:\d+[.)]\s*)?(?:[-*•]\s*)?\**\s*"
_INPUT = re.compile(_PREFIX + r"input\s*:\s*(table|text|question)\s*:\s*(.+?)\s*=\s*([^=]+?)\s*$", re.IGNORECASE | re.MULTILINE)
_FINAL = re.compile(_PREFIX + r"(program\s*:\s*(.+?)|verdict\s*:\s*unable\b.*?)\s*$", re.IGNORECASE | re.MULTILINE)
_RESOLVED = re.compile(_PREFIX + r"(asked|unit|format)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUMBER = re.compile(r"\(?\s?-?\$?\s?(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s?\)?%?")
_QUOTES = "\"'“”‘’`"


@dataclass
class CheckV2:
    breaker: str
    model: str | None
    tripped: bool
    usable: bool  # False when the outcome is unable
    value: float | str | None = None
    raw: str = ""
    outcome: str = "unable"  # agree | disagree | unable
    reason: str | None = None  # one of REASONS when unable
    cites: list[dict] = field(default_factory=list)
    resolved: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def norm(text: str) -> str:
    """Lower-case, remove punctuation, collapse whitespace."""
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def as_written(token: str) -> float | None:
    """A number as written: $, commas, % and spaces ignored; parentheses mean negative."""
    t = token.strip().replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def numbers_in(text: str) -> list[float]:
    return [v for v in (as_written(m) for m in _NUMBER.findall(text)) if v is not None]


def _has(nums: list[float], x: float) -> bool:
    return any(match(n, x, mode="strict") for n in nums)


def parse_reply(text: str) -> tuple[list[dict], str | None, bool, dict]:
    """Citations, the final program (None if absent), whether the final line declares unable, and the resolution."""
    cites = []
    for m in _INPUT.finditer(text):
        source, body, num = m.group(1).lower(), m.group(2).strip(), m.group(3).strip()
        found = numbers_in(num)
        cites.append({"source": source, "body": body, "number": found[0] if found else None})
    finals = list(_FINAL.finditer(text))
    program, unable = None, False
    if finals:
        last = finals[-1]
        if last.group(2) is not None:
            program = last.group(2).strip(" `*")
        else:
            unable = True
    resolved = {m.group(1).lower(): m.group(2) for m in _RESOLVED.finditer(text)}
    return cites, program, unable, resolved


def _check_table(body: str, number: float, table: list[list[str]]) -> str | None:
    parts = [p.strip() for p in body.split("|")]
    if len(parts) < 2 or not table:
        return "cell_not_found"
    header = [norm(c) for c in table[0]]
    cells = []
    for i in range(1, len(parts)):  # labels may contain "|": try every split into row and column
        row_l, col_l = norm(" | ".join(parts[:i])), norm(" | ".join(parts[i:]))
        cols = [j for j, h in enumerate(header) if h == col_l]
        for row in table:
            if row and norm(row[0]) == row_l:
                cells += [row[j] for j in cols if j < len(row)]
    if not cells:
        return "cell_not_found"
    return None if any(_has(numbers_in(c), number) for c in cells) else "cell_value_mismatch"


def _check_quote(body: str, number: float, source_text: str, not_found: str) -> str | None:
    quote = body.strip().strip(_QUOTES).strip()
    if not quote or norm(quote) not in norm(source_text):
        return not_found
    return None if _has(numbers_in(quote), number) else "number_not_in_quote"


def validate(cites: list[dict], conv: Conversation, k: int) -> str | None:
    """First failing citation's reason, or None when every citation is valid."""
    text = " ".join(conv.pre_text) + " " + " ".join(conv.post_text)
    for c in cites:
        if c["number"] is None:
            reason = {"table": "cell_value_mismatch", "text": "number_not_in_quote", "question": "number_not_in_quote"}[c["source"]]
        elif c["source"] == "table":
            reason = _check_table(c["body"], c["number"], conv.table)
        elif c["source"] == "text":
            reason = _check_quote(c["body"], c["number"], text, "text_quote_not_found")
        else:
            reason = _check_quote(c["body"], c["number"], conv.turns[k].question, "question_quote_not_found")
        c["ok"], c["reason"] = reason is None, reason
        if reason is not None:
            return reason
    return None


def uncited(program: str, cites: list[dict]) -> bool:
    """True when the program uses a table op or a literal that equals no cited number."""
    cited = [c["number"] for c in cites if c["number"] is not None]
    for step in parse(program):
        if step.op in TABLE_OPS:
            return True
        for tok in step.args:
            t = tok.strip()
            if t.startswith("#") or t.startswith("const_") or t == "none":
                continue
            v = as_written(t)
            if v is None or not _has(cited, v):
                return True
    return False


def score(reply_text: str, finish_reason: str | None, conv: Conversation, k: int, answer: float | str, model: str) -> CheckV2:
    """Parse, validate, execute and compare one program_v2 reply."""
    cites, program, unable, resolved = parse_reply(reply_text)

    def out(outcome: str, reason: str | None = None, value=None) -> CheckV2:
        return CheckV2("program_v2", model, outcome == "disagree", outcome != "unable", value, reply_text, outcome, reason, cites, resolved)

    if finish_reason == "length":
        return out("unable", "truncated")
    if unable:
        return out("unable", "model_unable")
    if program is None:
        return out("unable", "no_program")
    reason = validate(cites, conv, k)
    if reason is not None:
        return out("unable", reason)
    try:
        if uncited(program, cites):
            return out("unable", "uncited_literal")
        value = execute(program, conv.table).answer
    except (ProgramError, ArithmeticError, ValueError, IndexError, OverflowError):
        return out("unable", "program_error")
    return out("disagree" if not match(value, answer, mode="strict") else "agree", value=value)
