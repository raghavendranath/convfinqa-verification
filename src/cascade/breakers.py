"""Turn-level circuit breakers.

A breaker inspects one answer before it enters the conversation history and either passes it
or trips. When it trips, the run applies one of two actions (both are reported):

- ``halt``: stop the conversation and ask the user (later turns count as abstained, not wrong);
- ``repair``: replace the answer with the breaker's own value when it has one (program breakers),
  otherwise halt.

Breakers:
- ``self``: the answering model is asked whether the answer is correct;
- ``cross``: a different model is asked the same question;
- ``program``: a model writes a program over the report, which is executed in Python and
  compared with the answer (the model checks numbers, Python does the arithmetic).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .data import Conversation
from .dsl import ProgramError, execute
from .llm import Client
from .match import match
from .program_v2 import score as score_v2
from .prompts import program_messages, program_v2_messages, verify_messages

_VERDICT = re.compile(r"verdict\s*:\s*(correct|incorrect)", re.IGNORECASE)
_PROGRAM = re.compile(r"program\s*:\s*(.+)", re.IGNORECASE)


@dataclass
class Check:
    breaker: str
    model: str | None
    tripped: bool
    usable: bool  # False when the breaker gave no parseable verdict/program
    value: float | str | None = None  # program breakers: the recomputed value
    raw: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def check(
    kind: str,
    client: Client,
    model: str | None,
    conv: Conversation,
    k: int,
    history: list[float | str],
    answer: float | str,
    max_tokens: int = 8192,
) -> Check:
    if kind == "none":
        return Check("none", None, tripped=False, usable=True)
    assert model is not None
    if kind in ("self", "cross"):
        reply = client.chat(model, verify_messages(conv, k, history, answer), max_tokens=max_tokens)
        found = _VERDICT.findall(reply.text)
        if not found:
            return Check(kind, model, tripped=False, usable=False, raw=reply.text)
        return Check(kind, model, tripped=found[-1].lower() == "incorrect", usable=True, raw=reply.text)
    if kind == "program":
        reply = client.chat(model, program_messages(conv, k, history), max_tokens=max_tokens)
        found = _PROGRAM.findall(reply.text)
        if not found:
            return Check(kind, model, tripped=False, usable=False, raw=reply.text)
        prog = found[-1].strip().strip("`").strip()
        try:
            value = execute(prog, conv.table).answer
        except (ProgramError, ArithmeticError, ValueError, IndexError):  # any program that cannot run: unusable
            return Check(kind, model, tripped=False, usable=False, raw=reply.text)
        return Check(kind, model, tripped=not match(value, answer, mode="lenient"), usable=True, value=value, raw=reply.text)
    if kind == "program_v2":  # revised program breaker, PREREGISTRATION change 4 (exploratory extension)
        reply = client.chat(model, program_v2_messages(conv, k, history), max_tokens=max_tokens)
        return score_v2(reply.text, reply.finish_reason, conv, k, answer, model)
    raise ValueError(f"unknown breaker {kind}")
