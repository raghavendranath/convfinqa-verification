"""program_v2 (PREREGISTRATION change 4): citation parsing, validation, reason codes, verdicts, report logic."""

import hashlib
from pathlib import Path

import pytest

from cascade.change4 import classify_false_alarms, classify_recall, filing, model_call, paired, paired_diff
from cascade.data import Conversation, Turn
from cascade.program_v2 import as_written, norm, numbers_in, parse_reply, score

CONV = Conversation(
    "Single_ABC/2015/page_10.pdf-1",
    ["Net sales rose to $1,234.5 million in 2015, an increase of 12.5% over the prior year."],
    [["", "2015", "2014"], ["net sales", "$ 1,234.5", "$ 1,097.3"], ["operating income ( loss )", "( 45 )", "30"]],
    ["The company employed 9,800 people."],
    [Turn("what were net sales in 2015?", "1234.5", 1234.5), Turn("what was the change from 2014 to 2015, if it fell by 5?", "", 137.2)],
)


def reply(inputs: str, final: str) -> str:
    return f"Asked: the change\nUnit: millions\nFormat: amount in the report's units\n{inputs}\nProgram text here.\n{final}"


def test_helpers():
    assert norm("Operating income ( loss ):") == "operating income loss"
    assert numbers_in("( 45 )") == [-45.0] and numbers_in("$ ( 1,203.4 )") == [-1203.4] and numbers_in("12.5% of 9") == [12.5, 9.0]
    assert as_written("$ 1,234.5") == 1234.5 and as_written("( 45 )") == -45 and as_written("12.5%") == 12.5
    cites, program, unable, resolved = parse_reply(reply("Input: table: net sales | 2015 = 1,234.5", "Program: 1234.5"))
    assert cites == [{"source": "table", "body": "net sales | 2015", "number": 1234.5}]
    assert program == "1234.5" and not unable and resolved["unit"] == "millions"
    assert parse_reply("Program: add(1, 2)\nVerdict: unable")[2] is True  # the last final line wins
    numbered = parse_reply("1. Asked: x\n2. Input: table: net sales | 2015 = 1,234.5\n4. Program: subtract(1234.5, 1)  ")
    assert numbered[0][0]["number"] == 1234.5 and numbered[1] == "subtract(1234.5, 1)" and numbered[3]["asked"] == "x"
    assert parse_reply("**Program:** `divide(1, 2)`")[1] == "divide(1, 2)"


def test_agree_and_disagree_with_table_citations():
    inputs = "Input: table: net sales | 2015 = 1,234.5\nInput: table: net sales | 2014 = 1,097.3"
    text = reply(inputs, "Program: subtract(1234.5, 1097.3)")
    ok = score(text, "stop", CONV, 1, 137.2, "m")
    assert ok.outcome == "agree" and not ok.tripped and ok.usable and ok.reason is None
    bad = score(text, "stop", CONV, 1, 1372.0, "m")
    assert bad.outcome == "disagree" and bad.tripped and bad.usable


def test_text_and_question_citations_are_supported():
    t = reply('Input: text: "an increase of 12.5% over the prior year" = 12.5', "Program: divide(12.5, const_100)")
    assert score(t, "stop", CONV, 1, 0.125, "m").outcome == "agree"
    q = reply('Input: question: "if it fell by 5" = 5\nInput: table: net sales | 2015 = 1,234.5', "Program: subtract(1234.5, 5)")
    assert score(q, "stop", CONV, 1, 1229.5, "m").outcome == "agree"
    neg = reply("Input: table: Operating income (loss) | 2015 = (45)", "Program: -45")
    assert score(neg, "stop", CONV, 0, -45.0, "m").outcome == "agree"


@pytest.mark.parametrize(
    ("inputs", "final", "finish", "reason"),
    [
        ("Input: table: net sales | 2015 = 1,234.5", "Program: 1234.5", "length", "truncated"),
        ("", "Verdict: unable", "stop", "model_unable"),
        ("Input: table: net sales | 2015 = 1,234.5", "no final line", "stop", "no_program"),
        ("Input: table: revenue | 2015 = 1,234.5", "Program: 1234.5", "stop", "cell_not_found"),
        ("Input: table: net sales | 2015 = 1,300", "Program: 1300", "stop", "cell_value_mismatch"),
        ('Input: text: "sales were 1,234.5 million" = 1234.5', "Program: 1234.5", "stop", "text_quote_not_found"),
        ('Input: text: "The company employed 9,800 people" = 9,900', "Program: 9900", "stop", "number_not_in_quote"),
        ('Input: question: "if it rose by 5" = 5', "Program: 5", "stop", "question_quote_not_found"),
        ("Input: table: net sales | 2015 = 1,234.5", "Program: subtract(1234.5, 1097.3)", "stop", "uncited_literal"),
        ("Input: table: net sales | 2015 = 1,234.5", "Program: table_sum(net sales, none)", "stop", "uncited_literal"),
        ("Input: table: net sales | 2015 = 1,234.5", "Program: divide(1234.5, const_0)", "stop", "program_error"),
    ],
)
def test_unable_reasons(inputs, final, finish, reason):
    c = score(reply(inputs, final), finish, CONV, 1, 137.2, "m")
    assert c.outcome == "unable" and c.reason == reason and not c.tripped and not c.usable


def test_classifications_and_calls():
    assert classify_recall({"ci_low": 0.01, "ci_high": 0.2}) == "improved"
    assert classify_recall({"ci_low": -0.2, "ci_high": -0.01}) == "worse"
    assert classify_recall({"ci_low": -0.01, "ci_high": 0.2}) == "inconclusive"
    assert classify_false_alarms({"ci_low": -0.1, "ci_high": 0.019}, 0.02) == "non-inferior"
    assert classify_false_alarms({"ci_low": -0.1, "ci_high": 0.021}, 0.02) == "inconclusive"  # crosses the margin
    assert classify_false_alarms({"ci_low": 0.03, "ci_high": 0.08}, 0.02) == "materially higher"
    assert model_call("improved", "non-inferior") == "improved"
    assert model_call("improved", "inconclusive") == "not shown"
    assert filing("Single_JKHY/2009/page_28.pdf-3") == "JKHY/2009" and filing("Double_AAL/2014/page_59.pdf") == "AAL/2014"


def test_paired_diff_uses_strict_rescore_and_keeps_unable_in_denominator():
    def row(key, conv, wrong, tripped, usable=True, value=None, answer=0.25):
        return {
            "key": key,
            "conv": conv,
            "label": "free",
            "is_wrong": wrong,
            "tripped": tripped,
            "usable": usable,
            "value": value,
            "answer": answer,
        }

    frozen = [
        row("a", "c1", True, False, value=25.0),  # lenient: no trip; strict: trips (percent form)
        row("b", "c2", True, False, value=0.25),  # no trip either way
        row("c", "c3", False, False, value=0.25),
    ]
    revised = [row("a", "c1", True, True), row("b", "c2", True, False, usable=False), row("c", "c3", False, False)]
    pairs = paired(frozen, revised)
    strict = paired_diff(pairs, True, "strict")
    lenient = paired_diff(pairs, True, "lenient")
    assert strict["n"] == 2 and strict["comparator"] == 0.5 and strict["revised"] == 0.5 and strict["diff"] == 0
    assert lenient["comparator"] == 0 and lenient["diff"] == 0.5
    flagged = paired_diff(pairs, True, "strict", revised_trip=lambda r: bool(r["tripped"]) or not r["usable"])
    assert flagged["revised"] == 1.0


def test_locked_sample_hash_matches_preregistration():
    text = Path("configs/change4_sample.txt").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == "597f495dc8454de850498c1de16efe1ca08cf481832c576cf0f7d84ed38177dd"
    assert len(text.split()) == 600


def test_frozen_prompt_unchanged():
    from cascade.prompts import PROGRAM_V2_PROMPT, PROGRAM_V2_PROMPTS, PROGRAM_V2_SHA256, PROGRAM_V2_VERSION

    assert PROGRAM_V2_VERSION == "pv2-dev3" and PROGRAM_V2_PROMPT is PROGRAM_V2_PROMPTS["pv2-dev3"]
    assert hashlib.sha256(PROGRAM_V2_PROMPT.encode()).hexdigest() == PROGRAM_V2_SHA256
    assert PROGRAM_V2_SHA256 in Path("PREREGISTRATION.md").read_text()


def test_posthoc_dash_column_only_inside_the_relaxed_check():
    """Post-hoc (R45): "-" as column label is valid when the number is in the cited row, only while relaxed."""
    from cascade import program_v2
    from cascade.change4_posthoc import dash_column, rescore

    frozen = program_v2._check_table
    dash = reply("Input: table: net sales | - = 1,234.5", "Program: 1234.5")
    wrong_value = reply("Input: table: net sales | - = 1,300", "Program: 1300")
    # Frozen: "-" matches only an empty first-row cell, here the row-label column, so the value does not match.
    assert score(dash, "stop", CONV, 0, 1234.5, "m").reason == "cell_value_mismatch"
    with dash_column():
        assert score(dash, "stop", CONV, 0, 1234.5, "m").outcome == "agree"
        assert score(dash, "stop", CONV, 0, 1300.0, "m").outcome == "disagree"
        assert score(wrong_value, "stop", CONV, 0, 1300.0, "m").reason == "cell_value_mismatch"
        assert score(reply("Input: table: revenue | - = 1,234.5", "Program: 1234.5"), "stop", CONV, 0, 1.0, "m").reason == "cell_not_found"
        truncated = {"reason": "truncated", "outcome": "unable", "tripped": False, "usable": False, "value": None, "raw": dash}
        assert rescore(truncated, CONV)["reason"] == "truncated"
    assert program_v2._check_table is frozen
    assert score(dash, "stop", CONV, 0, 1234.5, "m").reason == "cell_value_mismatch"
