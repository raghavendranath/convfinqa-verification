import math

from cascade.analysis import AUDIT_REF, AUDIT_VALUE
from cascade.data import Conversation, Turn
from cascade.deps import Graph
from cascade.match import match
from cascade.posthoc_spread import (
    HISTORY_GROUPS,
    audit_graph,
    graph_sensitivity,
    history_group,
    loo_path_vs_blind,
    own_history,
    relabel,
    signature,
    signature_table,
    substituted_answer,
)


def _conv(cid: str = "c") -> Conversation:
    # Turn 0 looks up 100, turn 1 looks up 80, turn 2 is their difference, turn 3 divides it by turn 1.
    turns = [
        Turn("q0", "100", 100.0),
        Turn("q1", "80", 80.0),
        Turn("q2", "subtract(100, 80)", 20.0),
        Turn("q3", "subtract(100, 80), divide(#0, 80)", 0.25),
    ]
    return Conversation(cid, [], [["", "2019"], ["sales", "100"]], [], turns)


def test_substituted_answer_literal_and_intermediate():
    assert substituted_answer("subtract(100, 80)", None, 100.0, 1000.0) == 920.0
    # Turn 2's answer (20) is an intermediate of turn 3's program; replacing it carries through.
    assert substituted_answer("subtract(100, 80), divide(#0, 80)", None, 20.0, 40.0) == 0.5
    assert substituted_answer("subtract(100, 80)", None, 7.0, 70.0) is None  # the value does not occur
    assert substituted_answer("divide(5, 0)", None, 5.0, 6.0) is None


def test_substitution_sites_come_from_the_original_trace():
    # Original trace 20, 30, 31. Only step 0 is a site; the modified step 1 (10 + 10 = 20) must not be replaced again.
    assert substituted_answer("add(10, 10), add(#0, 10), add(#1, const_1)", None, 20.0, 10.0) == 21.0
    # A site the original trace has must not be missed because an earlier replacement changed its value.
    assert substituted_answer("add(5, 15), multiply(#0, const_1), add(#1, 1)", None, 20.0, 7.0) == 8.0
    # Nearby distinct figures are not the same input: 2022 is not a site for 2031 (it is under the 0.5% matcher).
    assert substituted_answer("add(2022, const_9), add(#0, const_10)", None, 2031.0, 231.0) == 241.0
    assert substituted_answer("add(2022, 5)", None, 2031.0, 231.0) is None
    assert substituted_answer("add(2022, 5)", None, 2031.0, 231.0, same=match) == 236.0


def _row(**kw) -> dict:
    base = {
        "conv": "c",
        "j": 0,
        "turn": 2,
        "type": "digits",
        "wrong": 120.0,
        "gold": 20.0,
        "pred": 40.0,
        "correct": False,
        "clean_correct": True,
        "copied_wrong": False,
        "finish_reason": "stop",
        "depends": True,
        "paths": 1,
        "key": "k",
    }
    return {**base, **kw}


def test_signature_reuse_before_power_of_ten():
    conv = _conv()
    assert signature(_row(), conv)["signature"] == "planted_input_reused"  # 120 - 80 = 40
    # A scaled intermediate used in arithmetic, (20 * 1000) / 80 = 250 = gold 0.25 * 1000: reuse and power of ten.
    r = _row(j=2, turn=3, type="scale", wrong=20000.0, gold=0.25, pred=250.0)
    sg = signature(r, conv)
    assert sg["signature"] == "planted_input_reused" and sg["reuse_and_power_of_ten"]
    # A power-of-ten answer that is not the reused value, offset by the planted factor.
    r = _row(j=0, turn=1, type="scale", wrong=100000.0, gold=80.0, pred=80000.0, depends=False, paths=0)
    sg = signature(r, conv)
    assert sg["signature"] == "power_of_ten" and sg["power_of_ten_same_factor"]
    assert signature(_row(pred=None, finish_reason="length"), conv)["signature"] == "loop"
    assert signature(_row(pred=120.0, copied_wrong=True), conv)["signature"] == "equals_planted"
    assert signature(_row(pred=33.0), conv)["signature"] == "other"


def test_signature_table_counts():
    conv = _conv()
    rows = [_row(), _row(pred=33.0, key="k2"), _row(correct=True, key="k3")]
    t = {(x["type"], x["turns"]): x for x in signature_table(rows, {"c": conv})}
    assert t[("all", "all")]["newly_wrong"] == 2
    assert t[("all", "depends")]["planted_input_reused"] == 1 and t[("all", "depends")]["other"] == 1


def test_history_group():
    conv = _conv()
    assert history_group(conv, 3, set()) == HISTORY_GROUPS[0]
    assert history_group(conv, 3, {1}) == HISTORY_GROUPS[1]  # turn 3 uses turn 1 (80)
    g = Graph(4, frozenset({(0, 2)}))
    assert history_group(conv, 3, {0}, g) == HISTORY_GROUPS[2]


def test_own_history_pairs_free_with_clean():
    conv = _conv()
    clean = [{"conv": "c", "turn": k, "correct": True} for k in range(4)]
    free = [
        {"conv": "c", "turn": 0, "correct": True},
        {"conv": "c", "turn": 1, "correct": False},
        {"conv": "c", "turn": 2, "correct": False},
        {"conv": "c", "turn": 3, "correct": False},
    ]
    out = own_history(clean, free, {"c": conv})
    assert out[HISTORY_GROUPS[0]]["n"] == 1 and out[HISTORY_GROUPS[0]]["extra_error"] == 1.0
    assert out[HISTORY_GROUPS[1]]["n"] == 2 and out[HISTORY_GROUPS[1]]["extra_error"] == 1.0


def test_audit_graph_and_relabel():
    sheet = [
        {"conv": "c", "turn": "0", AUDIT_VALUE: "-", AUDIT_REF: "-"},
        {"conv": "c", "turn": "2", AUDIT_VALUE: "0 1", AUDIT_REF: "-"},
        {"conv": "c", "turn": "3", AUDIT_VALUE: "1", AUDIT_REF: "2"},
    ]
    g = audit_graph(sheet, {"c": 4})["c"]
    assert g.direct == frozenset({(0, 2), (1, 2), (1, 3)})
    rows = relabel([_row(j=0, turn=3), _row(j=1, turn=3)], {"c": g})
    assert [r["depends"] for r in rows] == [False, True] and rows[1]["paths"] == 1


def test_graph_sensitivity_and_loo_run():
    rows = []
    for i in range(8):
        for k in (2, 3):
            for j in (0, 1):
                rows.append(_row(conv=f"c{i}", j=j, turn=k, key=f"{i}|{j}|{k}", correct=(i + j + k) % 3 != 0, clean_correct=i % 4 != 0))
    graphs = {f"c{i}": Graph(4, frozenset({(0, 2), (1, 2), (1, 3)})) for i in range(8)}
    out = graph_sensitivity(rows, {"graph": graphs, "primary audit": graphs})
    assert out["graph"]["dependent"]["n"] + out["graph"]["independent"]["n"] == len(rows)
    assert out["primary audit"]["moved_to_independent"] == 0
    pv = loo_path_vs_blind(relabel(rows, graphs))
    assert pv["conversations"] == 8 and math.isfinite(pv["brier_diff"])


def test_source_span_keeps_signs():
    from cascade.change4_posthoc import sign_check, source_span

    src = "Revenue was 45 million. Net loss was (1,234)."
    assert source_span("Revenue was -45 million.", src) == "Revenue was 45 million."
    assert source_span("not in the text", src) is None
    conv = Conversation("c", [src], [], [], [Turn("q", "45", 45.0)])
    row = {"key": "k", "turn": 0, "cites": [{"source": "text", "body": '"Revenue was -45 million."', "number": -45.0, "ok": True}]}
    assert len(sign_check(row, conv)) == 1  # passes the frozen check, fails the sign-preserving one
    row["cites"][0]["number"] = 45.0
    assert sign_check(row, conv) == []
