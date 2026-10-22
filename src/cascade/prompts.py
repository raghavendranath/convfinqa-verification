"""Prompts. Frozen before the test run (see PREREGISTRATION.md); any change bumps PROMPT_VERSION."""

from __future__ import annotations

from .data import Conversation
from .inject import fmt

PROMPT_VERSION = "v2"

# v2 (smoke run, 7 Oct 2026): v1's example "0.142" led to 3-decimal answers that miss the 0.5% tolerance
# (0.019 for 0.01887). v2 asks for unrounded values and states the ConvFinQA conventions: ratios and
# percentage changes as decimals unless the question asks to multiply by 100 or convert to a percentage,
# and amounts in the report's own units unless the question asks for another unit.
SYSTEM_ANSWER = (
    "You are a financial analyst answering a sequence of questions about one company report. "
    "Use only the report below. Questions may refer to earlier questions and answers in the conversation. "
    "Answer only the latest question. Think briefly, then end with one final line of the form 'Answer: <number>' "
    "(a plain number, no units or % sign), or 'Answer: yes' / 'Answer: no' for yes/no questions.\n"
    "Number format:\n"
    "- Do not round: give at least 5 significant digits when the value is not a whole number (e.g. 0.14218, 1.4241).\n"
    "- Give ratios and percentage changes as decimals (e.g. 0.14218, not 14.218), even if the question says "
    "'percent' or 'percentage', unless the question explicitly asks to multiply by 100 or to convert the value "
    "to a percentage.\n"
    "- Give amounts in the units the report uses (e.g. 173.2 when the table is in millions), "
    "unless the question asks for another unit.\n\n"
    "REPORT\n{context}"
)

# v2-guard (exploratory, added after freezing; PREREGISTRATION.md change 2, CHANGELOG.md R31): v2 with this
# paragraph after the number-format rules and before the report. Wording fixed before any run; not revised.
GUARD_PARAGRAPH = (
    "Earlier answers in this conversation may contain mistakes. When the latest question needs an earlier value, "
    "recompute it from the report instead of reusing the earlier answer. Decide units and number format from the "
    "latest question alone, not from the format of earlier answers."
)
_REPORT = "REPORT\n{context}"
assert SYSTEM_ANSWER.endswith("\n\n" + _REPORT)
SYSTEM_ANSWER_GUARD = SYSTEM_ANSWER[: -len(_REPORT)] + GUARD_PARAGRAPH + "\n\n" + _REPORT

# Answer prompt versions; v2 is the pre-registered one (PROMPT_VERSION, the default).
SYSTEM_PROMPTS = {"v2": SYSTEM_ANSWER, "v2-guard": SYSTEM_ANSWER_GUARD}


def answer_messages(conv: Conversation, k: int, history: list[float | str], prompt: str = PROMPT_VERSION) -> list[dict]:
    """Messages asking turn k, with the given answers (len == k) shown for turns 0..k-1, under answer prompt ``prompt``."""
    assert len(history) == k
    msgs = [{"role": "system", "content": SYSTEM_PROMPTS[prompt].format(context=conv.context())}]
    for j in range(k):
        msgs.append({"role": "user", "content": conv.turns[j].question})
        msgs.append({"role": "assistant", "content": f"Answer: {fmt(history[j])}"})
    msgs.append({"role": "user", "content": conv.turns[k].question})
    return msgs


VERIFY_PROMPT = (
    "Here is a company report, a conversation about it, and a proposed answer to the latest question.\n\n"
    "REPORT\n{context}\n\nCONVERSATION SO FAR\n{history}\n\nLATEST QUESTION\n{question}\n\n"
    "PROPOSED ANSWER\n{answer}\n\n"
    "Check the proposed answer against the report. Is it correct? "
    "Think briefly, then end with one final line: 'Verdict: correct' or 'Verdict: incorrect'."
)

PROGRAM_PROMPT = (
    "Here is a company report, a conversation about it, and the latest question.\n\n"
    "REPORT\n{context}\n\nCONVERSATION SO FAR\n{history}\n\nLATEST QUESTION\n{question}\n\n"
    "Write a program that computes the answer from numbers in the REPORT (not from earlier answers). "
    "Allowed operations: add(a, b), subtract(a, b), multiply(a, b), divide(a, b), exp(a, b), greater(a, b). "
    "Steps are separated by commas; #0, #1 ... refer to earlier steps' results; constants are written const_100, const_1000. "
    "For a plain lookup, write just the number. Example: subtract(206588, 181001), divide(#0, 181001)\n"
    "End with one final line: 'Program: <program>'."
)


def _history_text(conv: Conversation, k: int, history: list[float | str]) -> str:
    lines = []
    for j in range(k):
        lines.append(f"Q{j + 1}: {conv.turns[j].question}")
        lines.append(f"A{j + 1}: {fmt(history[j])}")
    return "\n".join(lines) if lines else "(none)"


def verify_messages(conv: Conversation, k: int, history: list[float | str], answer: float | str) -> list[dict]:
    content = VERIFY_PROMPT.format(
        context=conv.context(), history=_history_text(conv, k, history), question=conv.turns[k].question, answer=fmt(answer)
    )
    return [{"role": "user", "content": content}]


def program_messages(conv: Conversation, k: int, history: list[float | str]) -> list[dict]:
    content = PROGRAM_PROMPT.format(context=conv.context(), history=_history_text(conv, k, history), question=conv.turns[k].question)
    return [{"role": "user", "content": content}]


# program_v2 (PREREGISTRATION change 4, CHANGELOG R37, R38): a revised program breaker, a prospectively specified
# exploratory extension. Developed on the 20 dev smoke conversations only (at most 5 versions, each kept and
# listed); the final version is frozen in PREREGISTRATION change 4 Part B before the final evaluation.
_PV2_DEV1 = (
    "Here is a company report, a conversation about it, and the latest question.\n\n"
    "REPORT\n{context}\n\nCONVERSATION SO FAR\n{history}\n\nLATEST QUESTION\n{question}\n\n"
    "Recompute the answer to the latest question from the REPORT (not from earlier answers), in four steps.\n\n"
    "1. Resolve the question. Write three lines:\n"
    "Asked: <what the latest question asks for>\n"
    "Unit: <the unit of the answer>\n"
    "Format: <decimal ratio, percent number, amount in the report's units, count, or yes/no>\n"
    "Format rules: give ratios and percentage changes as decimals (e.g. 0.14218, not 14.218), even if the question "
    "says 'percent' or 'percentage', unless the question explicitly asks to multiply by 100 or to convert the value "
    "to a percentage. Give amounts in the units the report uses, unless the question asks for another unit.\n\n"
    "2. Cite every number you will use, one per line, in one of these forms:\n"
    "Input: table: <row label> | <column label> = <number>\n"
    'Input: text: "<exact words from the report text, including the number>" = <number>\n'
    'Input: question: "<exact words from the latest question, including the number>" = <number>\n'
    "Copy row labels, column labels and quotes exactly as they appear, and write each number as it appears there. "
    "Do not cite earlier answers.\n\n"
    "3. Write a program that uses only the cited numbers and constants. "
    "Allowed operations: add(a, b), subtract(a, b), multiply(a, b), divide(a, b), exp(a, b), greater(a, b). "
    "Steps are separated by commas; #0, #1 ... refer to earlier steps' results; constants are written const_100, "
    "const_1000. For a plain lookup, write just the number. The result must already be in the format from step 1 "
    "(multiply by const_100 only when the format is a percent number).\n\n"
    "4. End with one final line: 'Program: <program>'. If the REPORT does not contain the numbers needed, end "
    "instead with the line 'Verdict: unable'."
)

# pv2-dev2 (17 Oct 2026, after the pv2-dev1 dev smoke; results/change4_dev/pv2-dev1 and pv2-dev1-parserfix): the
# models converted numbers when citing them (0.18 for "18 percent"), shortened or paraphrased quotes, quoted
# earlier questions, took the column label from the wrong cell, and wrote nested or "#0 =" programs. dev2 states
# each of these rules explicitly. Steps 1 and 4 are unchanged in substance.
_PV2_DEV2 = (
    "Here is a company report, a conversation about it, and the latest question.\n\n"
    "REPORT\n{context}\n\nCONVERSATION SO FAR\n{history}\n\nLATEST QUESTION\n{question}\n\n"
    "Recompute the answer to the latest question from the REPORT (not from earlier answers), in four steps.\n\n"
    "1. Resolve the question. Write three lines:\n"
    "Asked: <what the latest question asks for>\n"
    "Unit: <the unit of the answer>\n"
    "Format: <decimal ratio, percent number, amount in the report's units, count, or yes/no>\n"
    "Format rules: give ratios and percentage changes as decimals (e.g. 0.14218, not 14.218), even if the question "
    "says 'percent' or 'percentage', unless the question explicitly asks to multiply by 100 or to convert the value "
    "to a percentage. Give amounts in the units the report uses, unless the question asks for another unit.\n\n"
    "2. Cite every number you will use, one per line, in one of these forms:\n"
    "Input: table: <row label> | <column label> = <number>\n"
    'Input: text: "<exact words from the report text, including the number>" = <number>\n'
    'Input: question: "<exact words from the latest question, including the number>" = <number>\n'
    "Rules for citations:\n"
    "- Row label: the first cell of the table row. Column label: the cell in the table's first row above the number "
    "(write - if that cell is empty). Copy both exactly.\n"
    "- Quotes: copy a short, continuous passage (a few words around the number) exactly as it appears; no '...'. "
    "Only the latest question may be quoted. Earlier questions and answers cannot be cited: recompute any earlier "
    "value from the report.\n"
    "- Write each number exactly as it appears in its source, before any conversion: 18 for '18 percent', 173.2 for "
    "'$ 173.2 million', 40 for '40 cents', -45 for '( 45 )'.\n\n"
    "3. Write a program in this exact syntax: one operation per step, steps separated by commas, each step "
    "op(a, b) where a and b are cited numbers, constants (const_100, const_1000, const_1000000) or #n, the result of "
    "step n (counting from 0). Allowed operations: add, subtract, multiply, divide, exp, greater. No nested "
    "operations and no '#0 ='. Example: subtract(206588, 181001), divide(#0, 181001). Do unit and percent "
    "conversions in the program, e.g. divide(18, const_100) or multiply(173.2, const_1000000). For a plain lookup, "
    "write just the cited number. The result must already be in the format from step 1.\n\n"
    "4. End with one final line that starts with 'Program:' followed by the program. If the REPORT does not contain "
    "the numbers needed, end instead with the line 'Verdict: unable'."
)
# pv2-dev3 (17 Oct 2026, after the pv2-dev2 dev smoke; results/change4_dev/pv2-dev2): Qwen3 32B's remaining unable
# checks were mostly invalid quotes (paraphrased or shortened with "...", table rows quoted as text, the latest
# question quoted without a number to carry an earlier answer). dev3 changes only the quote rules. Stop rule fixed
# before dev3 was run (CHANGELOG R40): freeze whichever of dev2 and dev3 has the lower mean unable rate over both
# models on the dev smoke; catch and false-alarm rates on dev are not used.
_PV2_DEV3 = (
    "Here is a company report, a conversation about it, and the latest question.\n\n"
    "REPORT\n{context}\n\nCONVERSATION SO FAR\n{history}\n\nLATEST QUESTION\n{question}\n\n"
    "Recompute the answer to the latest question from the REPORT (not from earlier answers), in four steps.\n\n"
    "1. Resolve the question. Write three lines:\n"
    "Asked: <what the latest question asks for>\n"
    "Unit: <the unit of the answer>\n"
    "Format: <decimal ratio, percent number, amount in the report's units, count, or yes/no>\n"
    "Format rules: give ratios and percentage changes as decimals (e.g. 0.14218, not 14.218), even if the question "
    "says 'percent' or 'percentage', unless the question explicitly asks to multiply by 100 or to convert the value "
    "to a percentage. Give amounts in the units the report uses, unless the question asks for another unit.\n\n"
    "2. Cite every number you will use, one per line, in one of these forms:\n"
    "Input: table: <row label> | <column label> = <number>\n"
    'Input: text: "<exact words from the report text, including the number>" = <number>\n'
    'Input: question: "<exact words from the latest question, including the number>" = <number>\n'
    "Rules for citations:\n"
    "- Row label: the first cell of the table row. Column label: the cell in the table's first row above the number "
    "(write - if that cell is empty). Copy both exactly.\n"
    "- Numbers in the table are cited as table cells, never as text.\n"
    "- Quotes: copy a short passage from one sentence, word for word, with nothing left out or changed (no '...'); "
    "the passage must contain the number. Only the latest question may be quoted, and only when it contains the "
    "number itself. Earlier questions and answers cannot be cited: recompute any earlier value from the report.\n"
    "- Write each number exactly as it appears in its source, before any conversion: 18 for '18 percent', 173.2 for "
    "'$ 173.2 million', 40 for '40 cents', -45 for '( 45 )'.\n\n"
    "3. Write a program in this exact syntax: one operation per step, steps separated by commas, each step "
    "op(a, b) where a and b are cited numbers, constants (const_100, const_1000, const_1000000) or #n, the result of "
    "step n (counting from 0). Allowed operations: add, subtract, multiply, divide, exp, greater. No nested "
    "operations and no '#0 ='. Example: subtract(206588, 181001), divide(#0, 181001). Do unit and percent "
    "conversions in the program, e.g. divide(18, const_100) or multiply(173.2, const_1000000). For a plain lookup, "
    "write just the cited number. The result must already be in the format from step 1.\n\n"
    "4. End with one final line that starts with 'Program:' followed by the program. If the REPORT does not contain "
    "the numbers needed, end instead with the line 'Verdict: unable'."
)
PROGRAM_V2_PROMPTS = {"pv2-dev1": _PV2_DEV1, "pv2-dev2": _PV2_DEV2, "pv2-dev3": _PV2_DEV3}
# FROZEN 18 Oct 2026 as PREREGISTRATION change 4 Part B (stop rule of CHANGELOG R40; R41). Do not edit.
PROGRAM_V2_VERSION = "pv2-dev3"
PROGRAM_V2_SHA256 = "6a2249340a2ac19d4a24a16ca81bbd580aba4a0ce819ecb0ee15f4aa05852adf"
PROGRAM_V2_PROMPT = PROGRAM_V2_PROMPTS[PROGRAM_V2_VERSION]


def program_v2_messages(conv: Conversation, k: int, history: list[float | str]) -> list[dict]:
    content = PROGRAM_V2_PROMPT.format(context=conv.context(), history=_history_text(conv, k, history), question=conv.turns[k].question)
    return [{"role": "user", "content": content}]
