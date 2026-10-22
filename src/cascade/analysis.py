"""Analysis: cascade size, the path-transmission model, and breaker scores.

Path-transmission model (the paper's central equation). For an injected error at turn j, later
turn k is wrong with probability

    P(wrong_k) = 1 - (1 - e_k) * (1 - a) ** m_jk * (1 - s) ** [m_jk == 0]

where e_k is turn k's error rate with a clean history (measured in ``clean``), m_jk is the number
of dependency paths from j to k in the gold-program graph, a is the per-path transmission rate,
and s is spillover to turns that do not depend on j. a and s are fitted per model by maximum
likelihood on the fitting half of conversations and tested on the held-out half.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .match import REL_TOL


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def half(conv_id: str) -> str:
    """Deterministic 50/50 split by conversation (fit vs held-out)."""
    return "fit" if int(hashlib.sha256(conv_id.encode()).hexdigest(), 16) % 2 == 0 else "test"


def cluster_bootstrap(values_by_cluster: dict[str, list[float]], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% interval, resampling conversations (turns in one conversation are not independent)."""
    keys = list(values_by_cluster)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    sums = np.array([sum(values_by_cluster[k]) for k in keys], dtype=float)
    counts = np.array([len(values_by_cluster[k]) for k in keys], dtype=float)
    point = sums.sum() / counts.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boots = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


AUDIT_VALUE, AUDIT_REF = "value_parents (fill in)", "reference_parents (fill in)"


def _turns(cell: str) -> set[int]:
    cell = cell.strip()
    return set() if cell in ("", "-") else {int(x) for x in cell.replace(",", " ").split()}


def graph_audit_scores(rows: list[dict]) -> dict:
    """Precision and recall of the value-dependency graph against the hand audit.

    ``value_parents``: earlier turns whose answer (the number) turn k uses; this is what the graph should find.
    ``reference_parents``: earlier turns that turn k refers to without using their number (e.g. "this average
    volume for JCB"); outside the graph's definition, counted separately. "-" means none. Edges are (j, k) pairs;
    intervals resample conversations.
    """
    missing = [f"{r['conv']} turn {r['turn']}" for r in rows if not r[AUDIT_VALUE].strip()]
    if missing:
        return {"complete": False, "unfilled": len(missing), "first_unfilled": missing[:5]}
    prec: dict[str, list[float]] = defaultdict(list)
    rec: dict[str, list[float]] = defaultdict(list)
    ref_only = ref_only_found = 0
    for r in rows:
        found, true, ref = _turns(r["parents_found"]), _turns(r[AUDIT_VALUE]), _turns(r[AUDIT_REF])
        prec[r["conv"]] += [float(j in true) for j in found]
        rec[r["conv"]] += [float(j in found) for j in true]
        ref_only += len(ref - true)
        ref_only_found += len((ref - true) & found)
    p, p_lo, p_hi = cluster_bootstrap({c: v for c, v in prec.items() if v})
    rc, r_lo, r_hi = cluster_bootstrap({c: v for c, v in rec.items() if v})
    return {
        "complete": True,
        "conversations": len({r["conv"] for r in rows}),
        "turns": len(rows),
        "edges_found": int(sum(len(v) for v in prec.values())),
        "edges_true": int(sum(len(v) for v in rec.values())),
        "precision": p,
        "precision_ci": [p_lo, p_hi],
        "recall": rc,
        "recall_ci": [r_lo, r_hi],
        "reference_only_edges": ref_only,
        "reference_only_edges_found_by_graph": ref_only_found,
    }


def labels_changed(audit: list[dict], draft: list[dict]) -> list[str]:
    """Turns whose value or reference labels differ between the audit sheet and a draft of it."""
    d = {(r["conv"], r["turn"]): r for r in draft}
    if set(d) != {(r["conv"], r["turn"]) for r in audit}:
        raise ValueError("sheets cover different turns")
    return [
        f"{r['conv']} turn {r['turn']}"
        for r in audit
        if any(_turns(r[c]) != _turns(d[(r["conv"], r["turn"])][c]) for c in (AUDIT_VALUE, AUDIT_REF))
    ]


def label_agreement(a: list[dict], b: list[dict], exclude: set[str] = frozenset(), n_boot: int = 2000, seed: int = 0) -> dict:
    """Agreement between two audit sheets of the same turns (added after freezing, CHANGELOG R33).

    Per column (value parents, reference parents): the share of non-first turns with the same parent set, and
    Cohen's kappa over every candidate edge (j, k) with j < k, with a conversation bootstrap interval for kappa.
    ``exclude``: conversations left out (e.g. ones a labeller saw as worked examples).
    """
    bmap = {(r["conv"], r["turn"]): r for r in b}
    if set(bmap) != {(r["conv"], r["turn"]) for r in a}:
        raise ValueError("sheets cover different turns")
    rows = [r for r in a if r["conv"] not in exclude and int(r["turn"]) > 0]
    convs = sorted({r["conv"] for r in rows})
    out: dict = {"conversations": len(convs), "turns": len(rows)}
    for name, col in (("value", AUDIT_VALUE), ("reference", AUDIT_REF)):
        same = 0
        cells = np.zeros((len(convs), 4))  # per conversation: both, a only, b only, neither
        ci = {c: i for i, c in enumerate(convs)}
        for r in rows:
            x, y = _turns(r[col]), _turns(bmap[(r["conv"], r["turn"])][col])
            same += x == y
            for j in range(int(r["turn"])):
                p, q = j in x, j in y
                cells[ci[r["conv"]], (0 if q else 1) if p else (2 if q else 3)] += 1

        def kappa(t: np.ndarray) -> float:
            n = t.sum()
            po = (t[0] + t[3]) / n
            pe = ((t[0] + t[1]) * (t[0] + t[2]) + (t[3] + t[2]) * (t[3] + t[1])) / n**2
            return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")

        tot = cells.sum(0)
        rng = np.random.default_rng(seed)
        boots = np.array([kappa(cells[rng.integers(0, len(convs), len(convs))].sum(0)) for _ in range(n_boot)])
        lo, hi = np.percentile(boots[~np.isnan(boots)], [2.5, 97.5]) if (~np.isnan(boots)).any() else (np.nan, np.nan)
        out[name] = {
            "same_turns": int(same),
            "same_share": same / len(rows),
            "edges": int(tot.sum()),
            "both": int(tot[0]),
            "a_only": int(tot[1]),
            "b_only": int(tot[2]),
            "kappa": kappa(tot),
            "kappa_ci": [float(lo), float(hi)],
        }
    return out


def reply_health(rows: list[dict]) -> dict:
    """Replies with no parseable answer, and replies cut off by max_tokens (``finish_reason == "length"``)."""
    n = len(rows)
    return {
        "n": n,
        "unparsed": sum(r["pred"] is None for r in rows),
        "truncated": sum(r.get("finish_reason") == "length" for r in rows),
    }


def noise_floor(clean: list[dict], repeat: list[dict]) -> dict:
    """Run-to-run noise: the same clean-history questions asked twice (``clean`` and ``clean_repeat``).

    At temperature 0 the seed should not matter; it only makes the second request distinct so it is not
    served from the cache. Any turn whose outcome differs shows nondeterminism of the hosted model, and the
    flip rate bounds how much of an extra-error estimate could be noise.
    """
    first = {(r["conv"], r["turn"]): r["correct"] for r in clean}
    flips: dict[str, list[float]] = defaultdict(list)
    right_to_wrong = wrong_to_right = 0
    for r in repeat:
        key = (r["conv"], r["turn"])
        if key not in first:
            continue
        flips[r["conv"]].append(float(first[key] != r["correct"]))
        right_to_wrong += first[key] and not r["correct"]
        wrong_to_right += (not first[key]) and r["correct"]
    rate, lo, hi = cluster_bootstrap(flips)
    n = sum(len(v) for v in flips.values())
    paired = [(r["conv"], r["turn"]) for r in repeat if (r["conv"], r["turn"]) in first]
    acc_repeat = sum(r["correct"] for r in repeat if (r["conv"], r["turn"]) in first) / max(1, n)
    acc_first = sum(first[k] for k in paired) / max(1, n)
    return {
        "n": n,
        "flip_rate": rate,
        "ci_low": lo,
        "ci_high": hi,
        "right_to_wrong": int(right_to_wrong),
        "wrong_to_right": int(wrong_to_right),
        "accuracy_clean": acc_first,
        "accuracy_repeat": acc_repeat,
    }


def cascade_rows(clean: list[dict], inject: list[dict]) -> list[dict]:
    """Attach the clean-history outcome of the same (conversation, turn) to each injected row."""
    base = {(r["conv"], r["turn"]): r["correct"] for r in clean}
    out = []
    for r in inject:
        key = (r["conv"], r["turn"])
        if key not in base:
            continue
        out.append({**r, "clean_correct": base[key]})
    return out


def cascade_table(rows: list[dict]) -> list[dict]:
    """Extra error caused by the injection, for dependent and non-dependent later turns, by error type."""
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        dep = "depends" if r["depends"] else "independent"
        delta = float(not r["correct"]) - float(not r["clean_correct"])
        for t in (r["type"], "all"):
            groups[(t, dep)][r["conv"]].append(delta)
    table = []
    for (t, dep), by_conv in sorted(groups.items()):
        m, lo, hi = cluster_bootstrap(by_conv)
        table.append({"type": t, "turns": dep, "extra_error": m, "ci_low": lo, "ci_high": hi, "n": sum(map(len, by_conv.values()))})
    return table


MECHANISMS = ("copied", "convention", "loop", "no_answer", "other")
_RESCALE = (100.0, 1000.0, 1e6, 1e9)


def mechanism(r: dict) -> str:
    """Why a turn that was right with the clean history is wrong after the injection. The first rule that
    applies wins, in this order (fixed in PREREGISTRATION.md):

    - ``loop``: cut off by max_tokens (``finish_reason == "length"``) with no answer;
    - ``no_answer``: finished but gave no parseable answer;
    - ``copied``: the answer equals the planted wrong value (``copied_wrong``, strict match);
    - ``convention``: the answer is the gold value rescaled by 100, 1,000, 10^6 or 10^9, or divided by one of
      them (a percent or unit convention carried over, the number itself right);
    - ``other``: anything else.
    """
    pred, gold = r["pred"], r["gold"]
    if pred is None:
        return "loop" if r.get("finish_reason") == "length" else "no_answer"
    if r.get("copied_wrong"):
        return "copied"
    if isinstance(pred, (int, float)) and isinstance(gold, (int, float)) and gold != 0 and pred != 0:
        # Relative tolerance only: an absolute tolerance would call 0 a rescaling of gold / 10^9.
        for f in _RESCALE:
            if math.isclose(pred, gold * f, rel_tol=REL_TOL) or math.isclose(pred, gold / f, rel_tol=REL_TOL):
                return "convention"
    return "other"


def mechanism_table(rows: list[dict]) -> list[dict]:
    """Counts of each mechanism among turns that became wrong (right with the clean history), by dependence."""
    out = []
    for dep in ("depends", "independent", "all"):
        new = [r for r in rows if r["clean_correct"] and not r["correct"] and (dep == "all" or r["depends"] == (dep == "depends"))]
        counts = {m: 0 for m in MECHANISMS}
        for r in new:
            counts[mechanism(r)] += 1
        out.append({"turns": dep, "newly_wrong": len(new), **counts})
    return out


def _predict(a: float, s: float, e: np.ndarray, m: np.ndarray) -> np.ndarray:
    p_ok = (1 - e) * (1 - a) ** m * np.where(m == 0, 1 - s, 1.0)
    return np.clip(1 - p_ok, 1e-6, 1 - 1e-6)


def fit_transmission(rows: list[dict]) -> dict:
    """Grid-search MLE for (a, s) on the fit half; report held-out log-loss, Brier and calibration."""
    data = defaultdict(list)
    test_convs = []
    for r in rows:
        # e_k: the turn's clean-history outcome (0/1 for one model), shrunk toward the model's mean
        # (lam = 0.5, fixed in PREREGISTRATION.md) so one clean outcome does not force P to 0 or 1.
        data[half(r["conv"])].append((float(not r["clean_correct"]), r["paths"], float(not r["correct"])))
        if half(r["conv"]) == "test":
            test_convs.append(r["conv"])
    if not data["fit"] or not data["test"]:
        return {"error": "not enough rows"}
    fit = np.array(data["fit"])
    test = np.array(data["test"])
    base = fit[:, 0].mean()
    lam = 0.5  # shrinkage of per-turn clean error toward the mean
    grid = np.linspace(0, 1, 201)

    def ll(arr: np.ndarray, a: float, s: float) -> float:
        e = lam * arr[:, 0] + (1 - lam) * base
        p = _predict(a, s, e, arr[:, 1])
        y = arr[:, 2]
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    best = max(((ll(fit, a, s), a, s) for a in grid for s in grid[:101]), key=lambda x: x[0])
    _, a, s = best
    e_t = lam * test[:, 0] + (1 - lam) * base
    p_t = _predict(a, s, e_t, test[:, 1])
    y_t = test[:, 2]
    # H2 comparison: a graph-blind model, P = 1 - (1 - e_k)(1 - c), one extra-error rate c for every later
    # turn whatever its paths. Same e_k, same fit half. Paired bootstrap over held-out conversations.
    c = max(grid, key=lambda c: ll(np.column_stack([fit[:, 0], np.ones(len(fit)), fit[:, 2]]), c, 0.0))
    p_b = _predict(c, 0.0, e_t, np.ones(len(test)))
    diff_by_conv: dict[str, list[float]] = defaultdict(list)
    for conv, d in zip(test_convs, (p_t - y_t) ** 2 - (p_b - y_t) ** 2, strict=True):
        diff_by_conv[conv].append(float(d))
    d_mean, d_lo, d_hi = cluster_bootstrap(diff_by_conv)
    # Calibration by number of paths: predicted vs observed error rate on held-out turns.
    calib = []
    for m in sorted(set(test[:, 1].astype(int))):
        mask = test[:, 1] == m
        calib.append({"paths": int(m), "n": int(mask.sum()), "predicted": float(p_t[mask].mean()), "observed": float(y_t[mask].mean())})
    return {
        "a_transmission": float(a),
        "s_spillover": float(s),
        "heldout_logloss": float(-np.mean(y_t * np.log(p_t) + (1 - y_t) * np.log(1 - p_t))),
        "heldout_brier": float(np.mean((p_t - y_t) ** 2)),
        "heldout_n": int(len(y_t)),
        "graph_blind_c": float(c),
        "graph_blind_heldout_brier": float(np.mean((p_b - y_t) ** 2)),
        # Path model minus graph-blind model; below 0 means the graph helps (H2). Mean per turn.
        "brier_diff": d_mean,
        "brier_diff_ci_low": d_lo,
        "brier_diff_ci_high": d_hi,
        "calibration_by_paths": calib,
    }


def breaker_scores(rows: list[dict]) -> list[dict]:
    """Catch rate on wrong answers and false-alarm rate on right answers, per label group."""
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    unusable: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        label = r["label"].split(":")[0]  # inject / gold / free
        kind = "wrong" if r["is_wrong"] else "right"
        groups[f"{label}:{kind}"][r["conv"]].append(float(r["tripped"]))
        unusable[f"{label}:{kind}"].append(float(not r["usable"]))
    out = []
    for g, by_conv in sorted(groups.items()):
        m, lo, hi = cluster_bootstrap(by_conv)
        out.append(
            {
                "group": g,
                "metric": "catch_rate" if g.endswith("wrong") else "false_alarm_rate",
                "value": m,
                "ci_low": lo,
                "ci_high": hi,
                "n": sum(map(len, by_conv.values())),
                "unusable_rate": float(np.mean(unusable[g])),
            }
        )
    return out
