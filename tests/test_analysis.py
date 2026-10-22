import random

import numpy as np

from cascade.analysis import _predict, cluster_bootstrap, fit_transmission, mechanism, mechanism_table


def test_graph_audit_scores():
    from cascade.analysis import AUDIT_REF, AUDIT_VALUE, graph_audit_scores

    def row(conv, turn, found, value, ref="-"):
        return {"conv": conv, "turn": turn, "parents_found": found, AUDIT_VALUE: value, AUDIT_REF: ref}

    rows = [
        row("a", 0, "", "-"),
        row("a", 1, "0", "0"),  # true positive
        row("a", 2, "0 1", "1"),  # one true positive, one false positive (0)
        row("b", 0, "", "-"),
        row("b", 1, "", "0"),  # missed edge
        row("b", 2, "", "-", "1"),  # reference-only link (V/2008 kind)
    ]
    s = graph_audit_scores(rows)
    assert s["complete"] and s["edges_found"] == 3 and s["edges_true"] == 3
    assert s["precision"] == 2 / 3 and s["recall"] == 2 / 3
    assert s["reference_only_edges"] == 1 and s["reference_only_edges_found_by_graph"] == 0
    rows[1][AUDIT_VALUE] = ""
    assert graph_audit_scores(rows) == {"complete": False, "unfilled": 1, "first_unfilled": ["a turn 1"]}


def test_mechanism_rules():
    base = {"gold": 318.0, "copied_wrong": False, "finish_reason": "stop"}
    assert mechanism({**base, "pred": None, "finish_reason": "length"}) == "loop"
    assert mechanism({**base, "pred": None}) == "no_answer"
    assert mechanism({**base, "pred": 324000.0, "copied_wrong": True}) == "copied"  # copied wins over convention
    assert mechanism({**base, "pred": 318000.0}) == "convention"  # LMT/2006 in the smoke run
    assert mechanism({**base, "gold": 0.61, "pred": 61.0}) == "convention"  # percent written as 61
    assert mechanism({**base, "gold": -34.3, "pred": -0.343}) == "convention"
    assert mechanism({**base, "pred": 0.0}) == "other"  # 0 is not "318 / 10^9"
    assert mechanism({**base, "pred": -318.0}) == "other"  # sign is not a convention
    assert mechanism({**base, "pred": 101.67, "gold": 91.66667}) == "other"


def test_mechanism_table_counts_only_newly_wrong():
    rows = [
        {"clean_correct": True, "correct": False, "depends": False, "pred": 318000.0, "gold": 318.0, "copied_wrong": False},
        {"clean_correct": True, "correct": False, "depends": True, "pred": 5.0, "gold": 7.0, "copied_wrong": True},
        {"clean_correct": False, "correct": False, "depends": True, "pred": 1.0, "gold": 7.0, "copied_wrong": False},
        {"clean_correct": True, "correct": True, "depends": True, "pred": 7.0, "gold": 7.0, "copied_wrong": False},
    ]
    t = {m["turns"]: m for m in mechanism_table(rows)}
    assert t["independent"]["newly_wrong"] == 1 and t["independent"]["convention"] == 1
    assert t["depends"]["newly_wrong"] == 1 and t["depends"]["copied"] == 1
    assert t["all"]["newly_wrong"] == 2


def test_bootstrap_point_estimate():
    m, lo, hi = cluster_bootstrap({"a": [1, 0], "b": [1, 1], "c": [0]})
    assert m == 3 / 5 and lo <= m <= hi


def test_transmission_fit_recovers_parameters():
    """Simulate the model with known a and s; the fit should land close."""
    rng = random.Random(0)
    a_true, s_true, rows = 0.4, 0.05, []
    for i in range(4000):
        clean_ok = rng.random() > 0.2
        paths = rng.choice([0, 0, 1, 1, 2, 3])
        p = _predict(a_true, s_true, np.array([0.0 if clean_ok else 1.0]), np.array([paths]))[0]
        rows.append({"conv": f"c{i}", "clean_correct": clean_ok, "paths": paths, "correct": rng.random() > p})
    fit = fit_transmission(rows)
    assert abs(fit["a_transmission"] - a_true) < 0.1
    assert fit["s_spillover"] < 0.15
    # When errors really follow the graph, the path model beats the graph-blind model (H2 comparison).
    assert fit["brier_diff_ci_high"] < 0


def test_graph_blind_model_not_beaten_when_graph_is_irrelevant():
    """If extra error does not depend on paths, the path model should not beat the graph-blind model."""
    rng = random.Random(1)
    rows = []
    for i in range(4000):
        clean_ok = rng.random() > 0.2
        p = 1 - (1 - (0.0 if clean_ok else 1.0)) * (1 - 0.15)
        rows.append({"conv": f"c{i}", "clean_correct": clean_ok, "paths": rng.choice([0, 1, 2, 3]), "correct": rng.random() > p})
    fit = fit_transmission(rows)
    assert abs(fit["graph_blind_c"] - 0.15) < 0.05
    assert fit["brier_diff_ci_low"] < 0 < fit["brier_diff_ci_high"] or fit["brier_diff"] > 0


def test_labels_changed():
    from cascade.analysis import AUDIT_REF, AUDIT_VALUE, labels_changed

    def row(turn, value, ref="-"):
        return {"conv": "a", "turn": str(turn), AUDIT_VALUE: value, AUDIT_REF: ref}

    audit = [row(0, "-"), row(1, "0"), row(2, "0 1")]
    draft = [row(0, "-"), row(1, "0"), row(2, "1 0", "-")]  # same set, other order: not a change
    assert labels_changed(audit, draft) == []
    draft[1] = row(1, "-", "0")
    assert labels_changed(audit, draft) == ["a turn 1"]


def test_label_agreement():
    from cascade.analysis import AUDIT_REF, AUDIT_VALUE, label_agreement

    def sheet(vals):
        return [{"conv": c, "turn": str(t), AUDIT_VALUE: v, AUDIT_REF: "-"} for (c, t), v in vals.items()]

    a = sheet({("x", 0): "-", ("x", 1): "0", ("x", 2): "0 1", ("y", 0): "-", ("y", 1): "0", ("z", 0): "-", ("z", 1): "0"})
    b = sheet({("x", 0): "-", ("x", 1): "0", ("x", 2): "1", ("y", 0): "-", ("y", 1): "0", ("z", 0): "-", ("z", 1): "-"})
    ag = label_agreement(a, b, exclude={"z"}, n_boot=50)
    assert ag["conversations"] == 2 and ag["turns"] == 3
    v = ag["value"]
    assert (v["same_turns"], v["edges"], v["both"], v["a_only"], v["b_only"]) == (2, 4, 3, 1, 0)
    assert np.isclose(v["kappa"], 0.0)  # po = 0.75, pe = 0.75
    assert ag["reference"]["same_turns"] == 3
