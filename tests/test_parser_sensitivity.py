from cascade.parser_sensitivity import correct_answer_row, correct_breaker_row, numeric_diff


def test_correct_answer_row_rereads_exponents():
    r = {"pred": 6.4748, "gold": 6474791.52, "raw": "Answer: 6.4748e+06", "correct": False, "correct_lenient": False}
    new = correct_answer_row(r)
    assert new["pred"] == 6474800.0 and new["correct"] and not r["correct"]  # the stored row is not changed
    assert correct_answer_row({**r, "pred": 6474800.0, "raw": "Answer: 6474800"}) is None
    inj = correct_answer_row({**r, "wrong": 6474800.0})
    assert inj["copied_wrong"]


def test_correct_breaker_row_rescores_program_checks_only():
    free = {("c", 3): {"pred": 6474800.0, "correct": True}}
    base = {
        "label": "free",
        "conv": "c",
        "turn": 3,
        "answer": 6.4748,
        "is_wrong": True,
        "tripped": True,
        "usable": True,
        "value": 6474791.5,
    }
    new, how = correct_breaker_row(base, "program", free)
    assert not new["is_wrong"] and not new["tripped"] and how == "re-scored"
    new, _ = correct_breaker_row({**base, "outcome": "disagree"}, "program_v2", free)
    assert new["outcome"] == "agree" and not new["tripped"]
    new, how = correct_breaker_row(base, "self", free)
    assert new["tripped"] and "kept" in how
    assert correct_breaker_row({**base, "label": "gold"}, "program", free) is None
    assert correct_breaker_row({**base, "turn": 1}, "program", free) is None


def test_numeric_diff():
    a = {"x": 1.0, "rows": [{"group": "g", "v": 0.5}], "flag": True}
    b = {"x": 1.0, "rows": [{"group": "g", "v": 0.6}], "flag": False}
    assert numeric_diff(a, b) == [("/rows[g]/v", 0.5, 0.6)]
    assert numeric_diff({"x": float("nan")}, {"x": float("nan")}) == []
