from cascade.breaker_posthoc import outcomes, repair, rescore, tripped_strict


def _row(conv, turn, label, wrong, answer, tripped, usable=True, value=None):
    return {
        "conv": conv,
        "turn": turn,
        "label": label,
        "is_wrong": wrong,
        "answer": answer,
        "tripped": tripped,
        "usable": usable,
        "value": value,
    }


def test_percent_error_not_caught_leniently_but_caught_strictly():
    # Gold 60.94; planted percent error 0.6094; the program recomputes 60.94. Lenient sees a match (not tripped).
    r = _row("c", 0, "inject:percent", True, 0.6094, tripped=False, value=60.94)
    assert tripped_strict(r)
    assert not tripped_strict(_row("c", 0, "gold", False, 60.94, tripped=False, value=60.94))
    assert not tripped_strict(_row("c", 0, "gold", False, 60.94, tripped=False, usable=False))
    res = {x["subset"]: x for x in rescore([r])}
    assert res["inject:wrong"]["lenient"]["value"] == 0 and res["inject:wrong"]["strict"]["value"] == 1
    assert res["inject:percent"]["strict"]["n"] == 1


def test_outcomes_and_repair():
    gold = {("a", 0): 10.0, ("a", 1): 5.0, ("b", 0): 2.0}
    rows = [
        _row("a", 0, "inject:scale", True, 10000.0, tripped=True, value=10.0),  # repaired right
        _row("a", 1, "inject:sign", True, -5.0, tripped=True, value=7.0),  # replaced by another wrong
        _row("a", 0, "gold", False, 10.0, tripped=True, value=11.0),  # corrupted
        _row("b", 0, "gold", False, 2.0, tripped=False, value=2.0),  # kept
        _row("b", 0, "free", True, 3.0, tripped=False, usable=False),  # unable
    ]
    o = outcomes(rows)
    assert o["inject:wrong"] == {"agree": 0, "disagree": 2, "unable": 0}
    assert o["free:wrong"]["unable"] == 1
    rp = repair(rows, gold)
    assert rp["inject:wrong"]["repaired_right"] == 1 and rp["inject:wrong"]["replaced_wrong"] == 1
    assert rp["gold:right"]["corrupted"] == 1 and rp["gold:right"]["kept"] == 1
    assert rp["free:wrong"]["unable"] == 1


def test_relation_and_oracle_format_rule():
    from cascade.breaker_posthoc import natural_kind, relation, tripped_oracle

    assert relation(0.25, 0.25) == "equal"
    assert relation(25.0, 0.25) == "percent_form"
    assert relation(173.2, 173200.0) == "scale"
    assert relation(-5.0, 5.0) == "sign"
    assert relation(7.0, 5.0) == "other"
    assert relation("yes", 5.0) == "nonnumeric"
    gold = {("c", 0): 0.25}
    # Right answer in gold's form, program in percent form: strict trips, the oracle does not.
    right = _row("c", 0, "gold", False, 0.25, tripped=False, value=25.0)
    assert tripped_strict(right) and not tripped_oracle(right, gold)
    # Planted percent error (answer 25, gold 0.25), program recomputes 0.25: the oracle trips.
    planted = _row("c", 0, "inject:percent", True, 25.0, tripped=False, value=0.25)
    assert tripped_oracle(planted, gold)
    # Substantively wrong answer in gold's form, program value is its percent form: strict trips, the oracle does not.
    wrong = _row("c", 0, "free", True, 0.3, tripped=False, value=30.0)
    assert tripped_strict(wrong) and not tripped_oracle(wrong, gold)
    # Program in the same wrong form as the answer: no trip.
    assert not tripped_oracle(_row("c", 0, "free", True, 25.0, tripped=False, value=25.0), gold)
    assert not tripped_oracle(_row("c", 0, "free", True, 25.0, tripped=False, usable=False), gold)
    assert natural_kind(_row("c", 0, "free", True, 25.0, tripped=False), gold) == "free:wrong, format only"
    assert natural_kind(_row("c", 0, "free", True, 0.3, tripped=False), gold) == "free:wrong, substantive"
    assert natural_kind(right, gold) is None
