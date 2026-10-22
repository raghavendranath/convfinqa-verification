"""Post-hoc analyses of the history design, the graph and the answer signatures (added 20 Oct 2026 after the main
results; not pre-registered; CHANGELOG R50, PREREGISTRATION change 6). No model calls.

1. Own-history comparison. The ``inject`` stage keeps every earlier answer gold except the planted one, so it
   measures the effect of one wrong item in an otherwise gold history, not errors passed on through the model's
   own intervening answers. The ``free`` stage is a sequential rollout (each turn sees the model's own earlier
   answers). Pairing each free turn with the same turn asked with the gold history (``clean``) gives the extra
   error of the model's own history, grouped by whether that history holds a wrong answer on a turn the graph
   links to this one, only on unlinked turns, or none. Observational: which turns have a wrong history is not
   assigned, so it is supporting evidence, not a causal estimate.
2. Graph sensitivity on the audited conversations. The 50 audited conversations are all in the evaluation set.
   Dependence and path counts are rebuilt from each audit's value-parent labels and the cascade split and the path
   model are recomputed on those conversations, next to the same quantities from the heuristic graph. The path
   model is compared with the graph-blind model by leave-one-conversation-out Brier (too few conversations for the
   fixed half split).
3. Answer signatures. The pre-registered mechanism rule records the form of a newly wrong answer, not its cause.
   This adds one signature it lacks: the answer equals turn k's gold program re-executed with turn j's gold value
   replaced by the planted value at the sites where it appears in the original gold computation (as a literal or an
   intermediate result), i.e. the planted input used in otherwise correct arithmetic. Sites are fixed from the
   unmodified trace with a near-exact value match (CHANGELOG R53 corrected the first version, which tested the
   already-modified intermediates with the 0.5% answer matcher). A scaled input gives a scaled result, so this overlaps the
   power-of-ten signature; overlaps are counted. For power-of-ten answers after a scale or percent injection, it
   also counts how many are offset by the same factor as the planted value.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from . import analysis
from .analysis import _predict, cluster_bootstrap, mechanism
from .data import Conversation
from .deps import Graph, build
from .dsl import ProgramError, _is_literal, _table_row, parse, parse_number
from .match import REL_TOL, match
from .posthoc import GRID, LAM, S_GRID, paired_mean_diff

HEADING = "## Post-hoc analyses of the history design, graph and answer signatures (added 20 Oct 2026; not pre-registered)"
HISTORY_GROUPS = ("history all right", "wrong answer on a linked turn", "wrong answers only on unlinked turns")
SIGNATURES = ("loop", "no_answer", "equals_planted", "planted_input_reused", "power_of_ten", "other")
LABELLINGS = ("graph", "primary audit", "second labeller")
PATH_BINS = ("0", "1", "2", "3+")


def _extra(r: dict) -> float:
    return float(not r["correct"]) - float(not r["clean_correct"])


# 1. Own-history comparison -------------------------------------------------------------------------------------


def history_group(conv: Conversation, k: int, wrong_before: set[int], graph: Graph | None = None) -> str:
    if not wrong_before:
        return HISTORY_GROUPS[0]
    g = graph or build(conv)
    linked = {i for i in wrong_before if k in g.descendants(i)}
    return HISTORY_GROUPS[1] if linked else HISTORY_GROUPS[2]


def own_history(clean: list[dict], free: list[dict], convs: dict[str, Conversation]) -> dict:
    """Free-stage error minus clean-stage error for the same turn, by what the model's own history holds."""
    base = {(r["conv"], r["turn"]): r["correct"] for r in clean}
    free_ok = {(r["conv"], r["turn"]): r["correct"] for r in free}
    graphs = {c: build(conv) for c, conv in convs.items()}
    by_group: dict[str, dict[str, list[float]]] = {g: defaultdict(list) for g in HISTORY_GROUPS}
    rates: dict[str, list[tuple[float, float]]] = {g: [] for g in HISTORY_GROUPS}
    for r in free:
        key = (r["conv"], r["turn"])
        if r["turn"] == 0 or key not in base or r["conv"] not in convs:
            continue
        wrong_before = {i for i in range(r["turn"]) if not free_ok.get((r["conv"], i), False)}
        g = history_group(convs[r["conv"]], r["turn"], wrong_before, graphs[r["conv"]])
        by_group[g][r["conv"]].append(float(not r["correct"]) - float(not base[key]))
        rates[g].append((float(not r["correct"]), float(not base[key])))
    out: dict = {}
    for g in HISTORY_GROUPS:
        m, lo, hi = cluster_bootstrap(by_group[g])
        n = len(rates[g])
        out[g] = {
            "n": n,
            "conversations": len(by_group[g]),
            "free_error": float(np.mean([x for x, _ in rates[g]])) if n else float("nan"),
            "clean_error": float(np.mean([y for _, y in rates[g]])) if n else float("nan"),
            "extra_error": m,
            "ci_low": lo,
            "ci_high": hi,
        }
    d = paired_mean_diff(by_group[HISTORY_GROUPS[1]], by_group[HISTORY_GROUPS[0]])
    out["linked_minus_all_right"] = {"diff": d["diff"], "ci_low": d["ci_low"], "ci_high": d["ci_high"]}
    return out


# 2. Graph sensitivity on the audited conversations ---------------------------------------------------------------


def audit_graph(audit_rows: list[dict], n_turns: dict[str, int]) -> dict[str, Graph]:
    """One graph per audited conversation from the value-parent labels."""
    edges: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for r in audit_rows:
        for j in analysis._turns(r[analysis.AUDIT_VALUE]):
            edges[r["conv"]].add((j, int(r["turn"])))
    return {c: Graph(n_turns[c], frozenset(edges.get(c, set()))) for c in {r["conv"] for r in audit_rows}}


def relabel(rows: list[dict], graphs: dict[str, Graph]) -> list[dict]:
    """Inject rows of the audited conversations with ``depends`` and ``paths`` from ``graphs``."""
    out = []
    for r in rows:
        g = graphs.get(r["conv"])
        if g is None:
            continue
        out.append({**r, "depends": r["turn"] in g.descendants(r["j"]), "paths": g.path_count(r["j"], r["turn"])})
    return out


def _fit(cw: np.ndarray, m: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Grid MLE of (a, s) and the base clean error, on rows; same grid, shrinkage and tie order as the fit."""
    keys = sorted(set(zip(cw.tolist(), m.tolist(), strict=True)))
    kpos = {k: i for i, k in enumerate(keys)}
    idx = np.array([kpos[k] for k in zip(cw.tolist(), m.tolist(), strict=True)])
    n1 = np.bincount(idx, weights=y, minlength=len(keys))
    n0 = np.bincount(idx, weights=1 - y, minlength=len(keys))
    kcw = np.array([k[0] for k in keys])
    km = np.array([k[1] for k in keys])
    base = float(cw.mean())
    e = LAM * kcw + (1 - LAM) * base
    p = _predict(GRID[:, None, None], S_GRID[None, :, None], e[None, None, :], km[None, None, :])
    ll = (n1 * np.log(p) + n0 * np.log(1 - p)).sum(-1)
    i, j = np.unravel_index(int(np.argmax(ll)), ll.shape)
    return float(GRID[i]), float(S_GRID[j]), base


def loo_path_vs_blind(rows: list[dict]) -> dict:
    """Leave-one-conversation-out held-out Brier of the path model minus the graph-blind model.

    For each conversation, a, s, the graph-blind c and the base clean error are fitted on the other conversations
    only and used to predict that conversation's rows. ``a_all`` and ``s_all`` are a separate fit on every row, for
    description only; no prediction uses them."""
    convs = sorted({r["conv"] for r in rows})
    cw = np.array([float(not r["clean_correct"]) for r in rows])
    m = np.array([float(r["paths"]) for r in rows])
    y = np.array([float(not r["correct"]) for r in rows])
    cid = np.array([r["conv"] for r in rows])
    diff_by_conv: dict[str, list[float]] = {}
    for c in convs:
        tr, te = cid != c, cid == c
        a, s, base = _fit(cw[tr], m[tr], y[tr])
        cb, _, _ = _fit(cw[tr], np.ones(int(tr.sum())), y[tr])  # graph-blind: every turn one "path", s unused
        e = LAM * cw[te] + (1 - LAM) * base
        p_path = _predict(a, s, e, m[te])
        p_blind = _predict(cb, 0.0, e, np.ones(int(te.sum())))
        diff_by_conv[c] = list((p_path - y[te]) ** 2 - (p_blind - y[te]) ** 2)
    d, lo, hi = cluster_bootstrap(diff_by_conv)
    a, s, _ = _fit(cw, m, y)
    return {"a_all": a, "s_all": s, "brier_diff": d, "ci_low": lo, "ci_high": hi, "conversations": len(convs)}


def _path_bin(m: int) -> str:
    return "3+" if m >= 3 else str(m)


def graph_sensitivity(rows: list[dict], graphs_by_labelling: dict[str, dict[str, Graph]]) -> dict:
    """The cascade split, observed error by path count and the path model, per labelling, on the same rows."""
    out: dict = {}
    graph_dep = {r["key"]: bool(r["depends"]) for r in relabel(rows, graphs_by_labelling["graph"])}
    for name, graphs in graphs_by_labelling.items():
        rs = relabel(rows, graphs)
        dep = [r for r in rs if r["depends"]]
        ind = [r for r in rs if not r["depends"]]
        res: dict = {}
        for label, sub in (("dependent", dep), ("independent", ind)):
            by: dict[str, list[float]] = defaultdict(list)
            for r in sub:
                by[r["conv"]].append(_extra(r))
            mm, lo, hi = cluster_bootstrap(by)
            res[label] = {"extra_error": mm, "ci_low": lo, "ci_high": hi, "n": len(sub)}

        def by_conv(sub: list[dict]) -> dict[str, list[float]]:
            d: dict[str, list[float]] = defaultdict(list)
            for r in sub:
                d[r["conv"]].append(_extra(r))
            return d

        dd = paired_mean_diff(by_conv(dep), by_conv(ind))
        res["dep_minus_indep"] = {"diff": dd["diff"], "ci_low": dd["ci_low"], "ci_high": dd["ci_high"]}
        res["moved_to_independent"] = sum(graph_dep[r["key"]] and not r["depends"] for r in rs)
        res["moved_to_dependent"] = sum(not graph_dep[r["key"]] and r["depends"] for r in rs)
        res["observed_by_paths"] = {
            b: {
                "n": len(sub := [r for r in rs if _path_bin(int(r["paths"])) == b]),
                "error": float(np.mean([not r["correct"] for r in sub])) if sub else float("nan"),
                "extra_error": float(np.mean([_extra(r) for r in sub])) if sub else float("nan"),
            }
            for b in PATH_BINS
        }
        res["path_vs_blind"] = loo_path_vs_blind(rs)
        out[name] = res
    return out


# 3. Answer signatures ------------------------------------------------------------------------------------------


def same_value(a: float, b: float) -> bool:
    """Value identity for a substitution site: equal after rounding to 5 decimals (the precision of ConvFinQA's gold
    answers) or within 1e-6 relative. Tighter than the 0.5% answer matcher, which would treat nearby distinct
    figures (2022 and 2031) as the same input."""
    return round(a, 5) == round(b, 5) or math.isclose(a, b, rel_tol=1e-6)


def _run(steps, table, literal_sites: set[tuple[int, int]], step_sites: set[int], new: float) -> list[float | str]:
    """Execute parsed steps, replacing the literal arguments in ``literal_sites`` and the results of the steps in
    ``step_sites`` by ``new``."""
    results: list[float | str] = []

    def val(n: int, i: int, tok: str) -> float:
        tok = tok.strip()
        if tok.startswith("#"):
            ref = int(tok[1:])
            if ref >= len(results) or isinstance(results[ref], str):
                raise ProgramError("bad reference")
            return float(results[ref])
        return new if (n, i) in literal_sites else parse_number(tok)

    for n, s in enumerate(steps):
        a, b = s.args
        if s.op == "identity":
            r: float | str = val(n, 0, a)
        elif s.op == "add":
            r = val(n, 0, a) + val(n, 1, b)
        elif s.op == "subtract":
            r = val(n, 0, a) - val(n, 1, b)
        elif s.op == "multiply":
            r = val(n, 0, a) * val(n, 1, b)
        elif s.op == "divide":
            d = val(n, 1, b)
            if d == 0:
                raise ZeroDivisionError
            r = val(n, 0, a) / d
        elif s.op == "exp":
            r = val(n, 0, a) ** val(n, 1, b)
        elif s.op == "greater":
            r = "yes" if val(n, 0, a) > val(n, 1, b) else "no"
        elif s.op.startswith("table_"):
            row = _table_row(table, a)
            r = {"table_max": max, "table_min": min, "table_sum": sum}.get(s.op, lambda x: sum(x) / len(x))(row)
        else:
            raise ProgramError(f"unknown op {s.op}")
        results.append(new if n in step_sites else r)
    return results


def substituted_answer(program: str, table: list[list[str]] | None, old: float, new: float, same=same_value) -> float | None:
    """Turn k's gold program re-executed with turn j's gold value ``old`` replaced by the planted value ``new``.

    The substitution sites come from the original gold computation: every literal argument, and every intermediate
    result (not the final one) of the unmodified program, that is the same value as ``old``. The program is then run
    again with those sites set to ``new``; a modified intermediate that happens to equal ``old`` is not substituted
    again. None when ``old`` occurs nowhere, the result is yes/no, or the program cannot run."""
    try:
        steps = parse(program)
        original = _run(steps, table, set(), set(), new)
    except (ProgramError, OverflowError, ZeroDivisionError):
        return None
    literal_sites = {
        (n, i) for n, s in enumerate(steps) for i, tok in enumerate(s.args) if _is_literal(tok) and same(parse_number(tok), old)
    }
    step_sites = {n for n, r in enumerate(original[:-1]) if not isinstance(r, str) and same(r, old)}
    if not literal_sites and not step_sites:
        return None
    try:
        result = _run(steps, table, literal_sites, step_sites, new)[-1]
    except (ProgramError, OverflowError, ZeroDivisionError):
        return None
    return None if isinstance(result, str) else result


def _power_of_ten(pred, gold) -> float | None:
    """The factor f (100, 1,000, 10^6, 10^9 or its inverse) with pred = gold * f, else None (pre-registered test)."""
    if not (isinstance(pred, (int, float)) and isinstance(gold, (int, float))) or gold == 0 or pred == 0:
        return None
    for f in analysis._RESCALE:
        for g in (f, 1 / f):
            if math.isclose(pred, gold * g, rel_tol=REL_TOL):
                return g
    return None


def signature(r: dict, conv: Conversation) -> dict:
    """Signature of a newly wrong answer: the pre-registered rule with ``planted_input_reused`` inserted before
    ``power_of_ten`` (the pre-registered ``convention``). Also flags the overlaps."""
    pre = mechanism(r)
    gold_j = conv.turns[r["j"]].answer

    def reuse(same) -> bool:
        if not (isinstance(gold_j, float) and isinstance(r["wrong"], (int, float))):
            return False
        reused = substituted_answer(conv.turns[r["turn"]].program, conv.table, gold_j, float(r["wrong"]), same)
        return reused is not None and isinstance(r["pred"], (int, float)) and match(r["pred"], reused) and not match(reused, r["gold"])

    is_reuse = reuse(same_value)
    # Sensitivity: substitution sites found with the 0.5% answer matcher, as the dependency graph does.
    is_reuse_graph_matcher = reuse(match)
    factor = _power_of_ten(r["pred"], r["gold"])
    planted_factor = _power_of_ten(r["wrong"], gold_j) if isinstance(gold_j, float) and isinstance(r["wrong"], (int, float)) else None
    if pre in ("loop", "no_answer"):
        sig = pre
    elif pre == "copied":
        sig = "equals_planted"
    elif is_reuse:
        sig = "planted_input_reused"
    elif pre == "convention":
        sig = "power_of_ten"
    else:
        sig = "other"
    return {
        "signature": sig,
        "reuse_graph_matcher": bool(is_reuse_graph_matcher and pre not in ("loop", "no_answer", "copied")),
        "reuse_and_power_of_ten": bool(is_reuse and pre == "convention"),
        "reuse_and_equals_planted": bool(is_reuse and pre == "copied"),
        "power_of_ten_same_factor": bool(
            sig == "power_of_ten" and factor is not None and planted_factor is not None and math.isclose(factor, planted_factor)
        ),
    }


def signature_table(rows: list[dict], convs: dict[str, Conversation]) -> list[dict]:
    """Newly wrong turns by signature, for dependent, independent and all later turns, and per error type."""
    out = []
    types = ["all", *sorted({r["type"] for r in rows})]
    for t in types:
        for dep in ("depends", "independent", "all"):
            new = [
                r
                for r in rows
                if r["clean_correct"]
                and not r["correct"]
                and (t == "all" or r["type"] == t)
                and (dep == "all" or bool(r["depends"]) == (dep == "depends"))
            ]
            counts = {s: 0 for s in SIGNATURES}
            flags = (
                "reuse_and_power_of_ten",
                "reuse_and_equals_planted",
                "power_of_ten_same_factor",
                "power_of_ten_scale_or_percent",
                "reuse_graph_matcher",
            )
            extra = dict.fromkeys(flags, 0)
            for r in new:
                sg = signature(r, convs[r["conv"]])
                counts[sg["signature"]] += 1
                for k in ("reuse_and_power_of_ten", "reuse_and_equals_planted", "power_of_ten_same_factor", "reuse_graph_matcher"):
                    extra[k] += sg[k]
                extra["power_of_ten_scale_or_percent"] += sg["signature"] == "power_of_ten" and r["type"] in ("scale", "percent")
            out.append({"type": t, "turns": dep, "newly_wrong": len(new), **counts, **{k: int(v) for k, v in extra.items()}})
    return out


# Report --------------------------------------------------------------------------------------------------------


def _ci(lo: float, hi: float, fmt: str = "+.3f") -> str:
    return f"[{lo:{fmt}}, {hi:{fmt}}]"


def posthoc_spread_section(cfg: dict, out, summary: dict, convs: dict[str, Conversation], audits: dict[str, list[dict]]) -> list[str]:
    """Compute the three analyses for every model with results, fill summary["posthoc_spread"], return markdown.

    ``audits``: {"primary audit": rows, "second labeller": rows} from the audit sheets (all 50 conversations).
    """
    from .posthoc import _load_rows, _slug

    audited = sorted({r["conv"] for rows in audits.values() for r in rows} & set(convs))
    n_turns = {c: len(convs[c].turns) for c in audited}
    graphs_by_labelling = {"graph": {c: build(convs[c]) for c in audited}}
    for name, rows in audits.items():
        graphs_by_labelling[name] = audit_graph([r for r in rows if r["conv"] in n_turns], n_turns)

    ps: dict = {"audited_conversations": len(audited), "models": {}}
    for model in cfg["models"]:
        rows = _load_rows(out, model)
        fp = out / "free" / f"{_slug(model)}.jsonl"
        if rows is None or not fp.exists():
            continue
        clean = analysis.read_jsonl(out / "clean" / f"{_slug(model)}.jsonl")
        audited_set = set(audited)
        ps["models"][model] = {
            "own_history": own_history(clean, analysis.read_jsonl(fp), convs),
            "graph_sensitivity": graph_sensitivity([r for r in rows if r["conv"] in audited_set], graphs_by_labelling),
            "signatures": signature_table(rows, convs),
        }
    summary["posthoc_spread"] = ps

    lines = [
        HEADING,
        "",
        "Chosen after seeing the results (CHANGELOG R50). No model calls. Intervals are 95% conversation cluster "
        "bootstraps (2,000 resamples, seed 0).",
        "",
        "### Own history: free-stage error minus clean-stage error for the same turn",
        "",
        "Every later turn of the free stage (the model's own earlier answers as history), paired with the same turn "
        "asked with the gold history. Grouped by the model's own earlier answers: all right; at least one wrong on a "
        "turn the graph links to this one; wrong only on unlinked turns. Observational (which turns have a wrong "
        "history is not assigned).",
        "",
        "| Model | History | n | Conversations | Free error | Clean error | Extra error | 95% CI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, r in ps["models"].items():
        oh = r["own_history"]
        for g in HISTORY_GROUPS:
            x = oh[g]
            lines.append(
                f"| {model} | {g} | {x['n']} | {x['conversations']} | {x['free_error']:.3f} | {x['clean_error']:.3f} | "
                f"{x['extra_error']:+.3f} | {_ci(x['ci_low'], x['ci_high'])} |"
            )
    lines += ["", "Linked wrong history minus all-right history (paired conversation bootstrap):", ""]
    lines += ["| Model | Difference | 95% CI |", "|---|---|---|"]
    for model, r in ps["models"].items():
        d = r["own_history"]["linked_minus_all_right"]
        lines.append(f"| {model} | {d['diff']:+.3f} | {_ci(d['ci_low'], d['ci_high'])} |")
    lines += [
        "",
        f"### Graph sensitivity on the {ps['audited_conversations']} audited conversations",
        "",
        "Dependence and path counts rebuilt from each audit's value-parent labels; the same inject rows each time. "
        "Moved: rows that change group relative to the graph. Path model minus graph-blind model by "
        "leave-one-conversation-out Brier (below 0 favours the path model): for each conversation, a, s, the "
        "graph-blind c and the base clean error are fitted on the other conversations and predict the held-out "
        "conversation's rows. The a and s columns are a separate fit on all audited rows, for description only; "
        "no prediction uses them.",
        "",
        "| Model | Labels | Dependent | n | Independent | n | Dependent minus independent | Moved to independent "
        "| Moved to dependent | a (all rows) | s (all rows) | Path minus graph-blind (LOO Brier) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, r in ps["models"].items():
        for name in LABELLINGS:
            g = r["graph_sensitivity"].get(name)
            if g is None:
                continue
            pv = g["path_vs_blind"]
            lines.append(
                f"| {model} | {name} | {g['dependent']['extra_error']:+.3f} {_ci(g['dependent']['ci_low'], g['dependent']['ci_high'])} | "
                f"{g['dependent']['n']} | {g['independent']['extra_error']:+.3f} "
                f"{_ci(g['independent']['ci_low'], g['independent']['ci_high'])} | {g['independent']['n']} | "
                f"{g['dep_minus_indep']['diff']:+.3f} {_ci(g['dep_minus_indep']['ci_low'], g['dep_minus_indep']['ci_high'])} | "
                f"{g['moved_to_independent']} | {g['moved_to_dependent']} | {pv['a_all']:.3f} | {pv['s_all']:.3f} | "
                f"{pv['brier_diff']:+.4f} {_ci(pv['ci_low'], pv['ci_high'], '+.4f')} |"
            )
    lines += [
        "",
        "Observed error (extra error) by number of paths from the planted turn, audited conversations:",
        "",
        "| Model | Labels | " + " | ".join(f"{b} paths (n)" for b in PATH_BINS) + " |",
        "|---|---|" + "---|" * len(PATH_BINS),
    ]
    for model, r in ps["models"].items():
        for name in LABELLINGS:
            g = r["graph_sensitivity"].get(name)
            if g is None:
                continue
            cells = []
            for b in PATH_BINS:
                o = g["observed_by_paths"][b]
                cells.append(f"{o['error']:.3f} ({o['extra_error']:+.3f}; {o['n']})" if o["n"] else "- (0)")
            lines.append(f"| {model} | {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### Answer signatures of newly wrong turns",
        "",
        "The pre-registered rule with one signature added before power_of_ten (the pre-registered convention): "
        "planted_input_reused, the answer equals turn k's gold program re-executed with the planted value at every site "
        "of the original gold computation (literal, or intermediate result other than the final one) that equals turn "
        "j's gold value (same after rounding to 5 decimals, or within 1e-6 relative; CHANGELOG R53). Reuse and power of "
        "ten: reused answers that are also a power-of-ten offset of gold (counted as reuse here, as convention by the "
        "pre-registered rule). Same factor: power-of-ten answers after a scale or percent injection offset by the planted "
        "factor. Reuse, graph matcher: the reuse count if substitution sites were found with the 0.5% answer matcher, "
        "as the dependency graph does (sensitivity).",
        "",
        "| Model | Error type | Later turns | Newly wrong | " + " | ".join(SIGNATURES) + " | Reuse and power of ten "
        "| Reuse and equals planted | Power of ten after scale or percent | Same factor | Reuse, graph matcher |",
        "|---|---|---|---|" + "---|" * len(SIGNATURES) + "---|---|---|---|---|",
    ]
    for model, r in ps["models"].items():
        for s in r["signatures"]:
            lines.append(
                f"| {model} | {s['type']} | {s['turns']} | {s['newly_wrong']} | "
                + " | ".join(str(s[k]) for k in SIGNATURES)
                + f" | {s['reuse_and_power_of_ten']} | {s['reuse_and_equals_planted']} | "
                f"{s['power_of_ten_scale_or_percent']} | {s['power_of_ten_same_factor']} | {s['reuse_graph_matcher']} |"
            )
    lines.append("")
    return lines
