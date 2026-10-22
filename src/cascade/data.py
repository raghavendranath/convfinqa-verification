"""Loading ConvFinQA (Chen et al., EMNLP 2022) from the original release.

Expected input: ``data/raw/convfinqa/{train,dev}.json`` from
https://github.com/czyssrs/ConvFinQA (``data.zip``). The test split's answers are private,
so the study uses ``dev`` as the evaluation set and ``train`` only for fitting the cascade model.

Each record has ``pre_text``, ``post_text``, ``table`` and ``annotation`` with parallel lists
``dialogue_break`` (questions), ``turn_program`` (one self-contained program per turn) and
``exe_ans_list`` (gold answers).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .dsl import ProgramError, execute
from .match import match


@dataclass
class Turn:
    question: str
    program: str
    answer: float | str


@dataclass
class Conversation:
    id: str
    pre_text: list[str]
    table: list[list[str]]
    post_text: list[str]
    turns: list[Turn]
    issues: list[str] = field(default_factory=list)

    def context(self) -> str:
        rows = [" | ".join(c.strip() for c in r) for r in self.table]
        return "\n".join(
            [
                "Text before the table:",
                " ".join(self.pre_text).strip(),
                "",
                "Table:",
                *rows,
                "",
                "Text after the table:",
                " ".join(self.post_text).strip(),
            ]
        )


def _as_answer(x: object) -> float | str:
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower()
    if s in {"yes", "no"}:
        return s
    return float(s)


def from_record(rec: dict) -> Conversation:
    ann = rec["annotation"]
    qs, progs, ans = ann["dialogue_break"], ann["turn_program"], ann["exe_ans_list"]
    if not (len(qs) == len(progs) == len(ans)):
        raise ValueError(f"{rec.get('id')}: annotation lists differ in length")
    conv = Conversation(
        id=rec["id"],
        pre_text=list(rec.get("pre_text", [])),
        table=[list(map(str, r)) for r in rec.get("table", [])],
        post_text=list(rec.get("post_text", [])),
        turns=[Turn(q, p, _as_answer(a)) for q, p, a in zip(qs, progs, ans, strict=True)],
    )
    # Check that each gold program reproduces its gold answer; record mismatches rather than drop silently.
    for i, t in enumerate(conv.turns):
        try:
            got = execute(t.program, conv.table).answer
            if not match(got, t.answer):
                conv.issues.append(f"turn {i}: program gives {got}, gold {t.answer}")
        except ProgramError as e:
            conv.issues.append(f"turn {i}: {e}")
    return conv


def load(path: str | Path) -> list[Conversation]:
    with open(path) as f:
        records = json.load(f)
    return [from_record(r) for r in records]
