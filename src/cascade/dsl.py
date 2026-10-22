"""Executor for the FinQA / ConvFinQA program language.

A program is a comma-separated list of steps such as
``subtract(206588, 181001), divide(#0, 181001)``. ``#n`` refers to the result of
step ``n`` in the same program. Constants are written ``const_100``, ``const_m1``
and so on. Table operations take a row name and ``none``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_STEP_RE = re.compile(r"\s*([a-z_]+)\(([^()]*)\)\s*")

ARITH_OPS = {"add", "subtract", "multiply", "divide", "exp", "greater"}
TABLE_OPS = {"table_max", "table_min", "table_sum", "table_average"}


class ProgramError(ValueError):
    pass


@dataclass(frozen=True)
class Step:
    op: str
    args: tuple[str, str]


@dataclass(frozen=True)
class Execution:
    steps: tuple[Step, ...]
    results: tuple[float | str, ...]
    literals: tuple[float, ...]  # numeric literals written in the program (not constants or #refs)

    @property
    def answer(self) -> float | str:
        return self.results[-1]


def parse(program: str) -> tuple[Step, ...]:
    program = program.strip()
    if not program:
        raise ProgramError("empty program")
    steps: list[Step] = []
    pos = 0
    while pos < len(program):
        m = _STEP_RE.match(program, pos)
        if not m:
            # A bare literal (ConvFinQA uses e.g. "206588" for lookup turns).
            if "(" not in program and "," not in program:
                return (Step("identity", (program, "none")),)
            raise ProgramError(f"cannot parse at {pos}: {program!r}")
        op, raw = m.group(1), m.group(2)
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 2:
            raise ProgramError(f"step {op} needs 2 args: {raw!r}")
        steps.append(Step(op, (parts[0], parts[1])))
        pos = m.end()
        if pos < len(program):
            if program[pos] != ",":
                raise ProgramError(f"expected ',' at {pos}: {program!r}")
            pos += 1
    return tuple(steps)


def parse_number(token: str) -> float:
    t = token.strip().replace(",", "").replace("$", "")
    if t.startswith("const_"):
        c = t[len("const_") :]
        if c == "m1":
            return -1.0
        try:
            return float(c)
        except ValueError as e:  # e.g. const_neg_3929 written by a model: not valid program syntax
            raise ProgramError(f"bad constant: {token!r}") from e
    pct = t.endswith("%")
    if pct:
        t = t[:-1]
    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]
    try:
        v = float(t)
    except ValueError as e:
        raise ProgramError(f"not a number: {token!r}") from e
    return v / 100.0 if pct else v


def _is_literal(token: str) -> bool:
    t = token.strip()
    if t.startswith("#") or t.startswith("const_") or t == "none":
        return False
    try:
        parse_number(t)
        return True
    except ProgramError:
        return False


def _table_row(table: list[list[str]] | None, name: str) -> list[float]:
    if not table:
        raise ProgramError("table op without a table")
    key = name.strip().lower()
    for row in table:
        if row and row[0].strip().lower() == key:
            vals = []
            for cell in row[1:]:
                try:
                    vals.append(parse_number(re.sub(r"[^0-9.\-%()]", "", cell) or "x"))
                except ProgramError:
                    continue
            if not vals:
                raise ProgramError(f"row {name!r} has no numbers")
            return vals
    raise ProgramError(f"row not found: {name!r}")


def execute(program: str, table: list[list[str]] | None = None) -> Execution:
    steps = parse(program)
    results: list[float | str] = []
    literals: list[float] = []

    def val(tok: str) -> float:
        tok = tok.strip()
        if tok.startswith("#"):
            i = int(tok[1:])
            if i >= len(results):
                raise ProgramError(f"bad reference {tok}")
            r = results[i]
            if isinstance(r, str):
                raise ProgramError("cannot do arithmetic on yes/no")
            return r
        if _is_literal(tok):
            literals.append(parse_number(tok))
        return parse_number(tok)

    for s in steps:
        a, b = s.args
        if s.op == "identity":
            results.append(val(a))
        elif s.op == "add":
            results.append(val(a) + val(b))
        elif s.op == "subtract":
            results.append(val(a) - val(b))
        elif s.op == "multiply":
            results.append(val(a) * val(b))
        elif s.op == "divide":
            d = val(b)
            if d == 0:
                raise ProgramError("division by zero")
            results.append(val(a) / d)
        elif s.op == "exp":
            results.append(val(a) ** val(b))
        elif s.op == "greater":
            results.append("yes" if val(a) > val(b) else "no")
        elif s.op in TABLE_OPS:
            row = _table_row(table, a)
            if s.op == "table_max":
                results.append(max(row))
            elif s.op == "table_min":
                results.append(min(row))
            elif s.op == "table_sum":
                results.append(sum(row))
            else:
                results.append(sum(row) / len(row))
        else:
            raise ProgramError(f"unknown op {s.op}")
    return Execution(tuple(steps), tuple(results), tuple(literals))
