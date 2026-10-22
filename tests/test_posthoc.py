import random

import numpy as np

from cascade.analysis import cascade_table, fit_transmission, half
from cascade.posthoc import (
    a_difference,
    bootstrap_transmission,
    convention_by_type,
    dep_minus_indep,
    link_group,
    link_groups,
    posthoc_section,
)


def _synthetic(seed: int, a: float = 0.15, s: float = 0.05, n_conv: int = 120) -> list[dict]:
    """Rows drawn from the path-transmission model with known a and s."""
    rng = random.Random(seed)
    rows = []
    for i in range(n_conv):
        conv = f"c{i}"
        for turn in range(6):
            m = rng.choice([0, 0, 1, 2, 3])
            clean_ok = rng.random() > 0.2
            e = 0.0 if clean_ok else 1.0
            p_ok = (1 - e) * (1 - a) ** m * ((1 - s) if m == 0 else 1.0)
            rows.append(
                {
                    "conv": conv,
                    "turn": turn,
                    "type": rng.choice(["scale", "sign"]),
                    "paths": m,
                    "depends": m > 0,
                    "direct": m > 0 and rng.random() < 0.5,
                    "correct": rng.random() < p_ok,
                    "clean_correct": clean_ok,
                    "pred": 1.0,
                    "gold": 1.0,
                    "copied_wrong": False,
                    "finish_reason": "stop",
                }
            )
    return rows


def test_bootstrap_point_equals_fit_transmission():
    for seed in (0, 1):
        rows = _synthetic(seed)
        fit = fit_transmission(rows)
        bt = bootstrap_transmission(rows, n_boot=50)
        assert bt["a"] == fit["a_transmission"] and bt["s"] == fit["s_spillover"]
        assert bt["a_ci_low"] <= bt["a"] <= bt["a_ci_high"]
        assert len(bt["a_boot"]) == 50


def test_bootstrap_ignores_test_half_and_missing_ids():
    rows = _synthetic(0)
    ids = sorted({r["conv"] for r in rows}) + ["zz_missing"]
    a1 = bootstrap_transmission(rows, n_boot=20)
    a2 = bootstrap_transmission(rows, [c for c in ids if half(c) == "fit" or c == "zz_missing"], n_boot=20)
    assert a1["a"] == a2["a"]


def test_a_difference_paired():
    rows = _synthetic(0)
    d = a_difference(rows, rows, n_boot=30)
    assert d["diff"] == 0 and d["ci_low"] == 0 and d["ci_high"] == 0  # same rows, same resamples


def test_dep_minus_indep_matches_cascade_table():
    rows = _synthetic(2)
    tab = {t["turns"]: t["extra_error"] for t in cascade_table(rows) if t["type"] == "all"}
    d = dep_minus_indep(rows, n_boot=200)
    assert np.isclose(d["diff"], tab["depends"] - tab["independent"], rtol=0, atol=1e-12)
    assert d["ci_low"] <= d["diff"] <= d["ci_high"]


def test_link_group_classification():
    assert link_group({"direct": True, "depends": True}) == "direct"
    assert link_group({"direct": False, "depends": True}) == "indirect_only"
    assert link_group({"direct": False, "depends": False}) == "independent"
    rows = _synthetic(3)
    g = link_groups(rows, n_boot=100)
    assert g["direct"]["n"] == sum(r["direct"] for r in rows)
    assert g["indirect_only"]["n"] == sum(r["depends"] and not r["direct"] for r in rows)
    assert g["independent"]["n"] == sum(not r["depends"] for r in rows)


def test_direct_link_model_recovers_rates():
    rng = random.Random(5)
    true_c = {"direct": 0.4, "indirect_only": 0.1, "independent": 0.02}
    rows = []
    for i in range(400):
        for turn in range(6):
            g = rng.choice(list(true_c))
            rows.append(
                {
                    "conv": f"c{i}",
                    "turn": turn,
                    "type": "sign",
                    "paths": 0 if g == "independent" else 1,
                    "depends": g != "independent",
                    "direct": g == "direct",
                    "clean_correct": True,
                    "correct": rng.random() > true_c[g],
                }
            )
    dl = link_groups(rows, n_boot=50)["direct_link_model"]
    for g, c in true_c.items():
        # e_k is shrunk toward base = 0 here, so P = c_g exactly.
        assert abs(dl["c"][g] - c) < 0.04
    assert dl["brier_diff"] < 0 and dl["brier_diff_ci_high"] < 0


def test_convention_by_type_counts():
    base = {"conv": "a", "type": "scale", "depends": False, "gold": 2.0, "copied_wrong": False, "finish_reason": "stop"}
    rows = [
        {**base, "pred": 200.0, "correct": False, "clean_correct": True},  # convention
        {**base, "pred": 3.0, "correct": False, "clean_correct": True},  # other
        {**base, "pred": 3.0, "correct": False, "clean_correct": False},  # wrong before too: not newly wrong
        {**base, "pred": 2.0, "correct": True, "clean_correct": True},
        {**base, "type": "sign", "depends": True, "pred": None, "correct": False, "clean_correct": True},  # no_answer
    ]
    out = {(c["type"], c["turns"]): c for c in convention_by_type(rows)}
    sc = out[("scale", "independent")]
    assert (sc["rows"], sc["newly_wrong"], sc["convention"], sc["other"]) == (4, 2, 1, 1)
    assert sc["convention_rate"] == 0.25
    assert out[("sign", "depends")]["no_answer"] == 1
    assert ("scale", "depends") not in out
    assert out[("all", "independent")]["rows"] == 4


def test_posthoc_section_reads_files(tmp_path):
    import json

    rows = _synthetic(4, n_conv=40)
    for stage in ("clean", "inject"):
        (tmp_path / stage).mkdir()
    clean = [{"conv": r["conv"], "turn": r["turn"], "correct": r["clean_correct"]} for r in rows]
    inject = [{k: v for k, v in r.items() if k != "clean_correct"} for r in rows]
    for stage, rs in (("clean", clean), ("inject", inject)):
        for slug in ("org__m1", "org__m2"):
            (tmp_path / stage / f"{slug}.jsonl").write_text("\n".join(json.dumps(r) for r in rs))
    summary: dict = {}
    cfg = {"models": ["org/m1", "org/missing"], "posthoc": {"size_pair": ["org/m1", "org/missing"]}}
    lines = posthoc_section(cfg, tmp_path, summary)
    assert lines[0].startswith("## Post-hoc analyses (added 14 Oct 2026")
    assert list(summary["posthoc"]["models"]) == ["org/m1"]
    assert "a_difference" not in summary["posthoc"]
    cfg = {"models": ["org/m1", "org/m2"], "posthoc": {"size_pair": ["org/m1", "org/m2"]}}
    lines = posthoc_section(cfg, tmp_path, summary)
    ad = summary["posthoc"]["a_difference"]
    assert ad["model_a"] == "org/m1" and ad["diff"] == 0
    assert any(line.startswith("a of org/m1 minus a of org/m2") for line in lines)
    assert not any("—" in line for line in lines)
    json.dumps(summary)  # serialisable
