"""Post-hoc analyses, added 14 Oct 2026 after the main results. Not pre-registered.

1. Dependent minus independent extra error, with a paired conversation bootstrap (H2a).
2. A bootstrap interval for the transmission rate a, and for the difference in a between two models (H3).
3. Extra error by direct link, indirect-only paths and independent turns, and a held-out "direct-link model"
   against the graph-blind model (follows from the H2 calibration).
4. Convention spillover by error type (H5 exploratory).

The pre-registered code in ``analysis.py`` is not changed. The transmission bootstrap replicates the grid of
``analysis.fit_transmission`` (a in linspace(0, 1, 201), s in its first 101 values, lam = 0.5, base = mean clean
error of the fit-half rows, first maximum in a-major order) so that its point estimate equals that fit.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from . import analysis
from .analysis import _predict, cluster_bootstrap, half, mechanism

HEADING = "## Post-hoc analyses (added 14 Oct 2026 after the main results; not pre-registered)"
LAM = 0.5
GRID = np.linspace(0, 1, 201)
S_GRID = GRID[:101]
N_BOOT = 2000


def _extra(r: dict) -> float:
    return float(not r["correct"]) - float(not r["clean_correct"])


def _by_conv(rows: list[dict]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        out[r["conv"]].append(_extra(r))
    return out


def paired_mean_diff(x_by_conv: dict[str, list[float]], y_by_conv: dict[str, list[float]], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Pooled mean of x minus pooled mean of y, with a paired conversation bootstrap.

    Each replicate resamples the sorted union of conversations once, and both pooled means are recomputed from
    the same draws. A conversation with no values in one group adds nothing to that group's mean.
    """
    keys = sorted(set(x_by_conv) | set(y_by_conv))
    if not keys:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "x": float("nan"), "y": float("nan")}

    def arrays(d: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([sum(d.get(k, [])) for k in keys], dtype=float),
            np.array([len(d.get(k, [])) for k in keys], dtype=float),
        )

    xs, xc = arrays(x_by_conv)
    ys, yc = arrays(y_by_conv)
    x_point, y_point = xs.sum() / xc.sum(), ys.sum() / yc.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = xs[idx].sum(1) / xc[idx].sum(1) - ys[idx].sum(1) / yc[idx].sum(1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"diff": float(x_point - y_point), "ci_low": float(lo), "ci_high": float(hi), "x": float(x_point), "y": float(y_point)}


def dep_minus_indep(rows: list[dict], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Extra error of dependent turns minus that of independent turns (pooled means, paired bootstrap)."""
    d = paired_mean_diff(_by_conv([r for r in rows if r["depends"]]), _by_conv([r for r in rows if not r["depends"]]), n_boot, seed)
    return {
        "dependent": d["x"],
        "independent": d["y"],
        "diff": d["diff"],
        "ci_low": d["ci_low"],
        "ci_high": d["ci_high"],
        "n_dependent": sum(bool(r["depends"]) for r in rows),
        "n_independent": sum(not r["depends"] for r in rows),
    }


def _cells(rows: list[dict], conv_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse rows into cells (clean_wrong, paths): counts of y = 1 and y = 0 per conversation.

    Returns cw (K,), m (K,), n1 (C, K), n0 (C, K) with conversations in the order of ``conv_ids``.
    """
    keys = sorted({(float(not r["clean_correct"]), int(r["paths"])) for r in rows})
    kpos = {k: i for i, k in enumerate(keys)}
    cpos = {c: i for i, c in enumerate(conv_ids)}
    n1 = np.zeros((len(conv_ids), len(keys)))
    n0 = np.zeros((len(conv_ids), len(keys)))
    for r in rows:
        c = cpos.get(r["conv"])
        if c is None:
            continue
        k = kpos[(float(not r["clean_correct"]), int(r["paths"]))]
        if r["correct"]:
            n0[c, k] += 1
        else:
            n1[c, k] += 1
    cw = np.array([k[0] for k in keys], dtype=float)
    m = np.array([k[1] for k in keys], dtype=float)
    return cw, m, n1, n0


def _fit_cells(cw: np.ndarray, m: np.ndarray, n1: np.ndarray, n0: np.ndarray) -> tuple[float, float]:
    """Grid MLE of (a, s) from cell counts; same grid, shrinkage and tie order as fit_transmission."""
    tot = n1 + n0
    base = float((cw * tot).sum() / tot.sum())
    e = LAM * cw + (1 - LAM) * base
    p = _predict(GRID[:, None, None], S_GRID[None, :, None], e[None, None, :], m[None, None, :])
    ll = (n1 * np.log(p) + n0 * np.log(1 - p)).sum(-1)
    i, j = np.unravel_index(int(np.argmax(ll)), ll.shape)  # first maximum, a-major then s
    return float(GRID[i]), float(S_GRID[j])


def bootstrap_transmission(rows: list[dict], conv_ids: list[str] | None = None, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Point estimate and conversation bootstrap of (a, s), refitting on the fit half each time.

    ``conv_ids`` is the sorted list of conversations to resample (pass the same list for two models to share
    resamples); test-half ids in it are dropped. A conversation with no rows for this model adds nothing.
    """
    fit_rows = [r for r in rows if half(r["conv"]) == "fit"]
    if conv_ids is None:
        conv_ids = sorted({r["conv"] for r in fit_rows})
    conv_ids = [c for c in conv_ids if half(c) == "fit"]
    cw, m, n1, n0 = _cells(fit_rows, conv_ids)
    a, s = _fit_cells(cw, m, n1.sum(0), n0.sum(0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(conv_ids), size=(n_boot, len(conv_ids)))
    a_boot = np.empty(n_boot)
    s_boot = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(idx[b], minlength=len(conv_ids)).astype(float)
        a_boot[b], s_boot[b] = _fit_cells(cw, m, w @ n1, w @ n0)
    a_lo, a_hi = np.percentile(a_boot, [2.5, 97.5])
    s_lo, s_hi = np.percentile(s_boot, [2.5, 97.5])
    return {
        "a": a,
        "a_ci_low": float(a_lo),
        "a_ci_high": float(a_hi),
        "s": s,
        "s_ci_low": float(s_lo),
        "s_ci_high": float(s_hi),
        "n_conversations": len(conv_ids),
        "n_boot": n_boot,
        "a_boot": a_boot,
        "s_boot": s_boot,
    }


def _public(bt: dict) -> dict:
    return {k: v for k, v in bt.items() if k not in ("a_boot", "s_boot")}


def a_difference(rows_a: list[dict], rows_b: list[dict], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """a of model A minus a of model B, paired: both refits use the same resampled conversations."""
    ids = sorted({r["conv"] for r in rows_a + rows_b if half(r["conv"]) == "fit"})
    ba = bootstrap_transmission(rows_a, ids, n_boot, seed)
    bb = bootstrap_transmission(rows_b, ids, n_boot, seed)
    d = ba["a_boot"] - bb["a_boot"]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {
        "diff": ba["a"] - bb["a"],
        "ci_low": float(lo),
        "ci_high": float(hi),
        "share_a_lower_or_equal": float(np.mean(d <= 0)),
        "bootstrap_a": _public(ba),
        "bootstrap_b": _public(bb),
    }


GROUPS = ("direct", "indirect_only", "independent")


def link_group(r: dict) -> str:
    if r["direct"]:
        return "direct"
    return "indirect_only" if r["depends"] else "independent"


def _direct_link_model(rows: list[dict]) -> dict:
    """Held-out comparison: one extra-error rate per link group against one for all turns (graph-blind)."""
    fit = [r for r in rows if half(r["conv"]) == "fit"]
    test = [r for r in rows if half(r["conv"]) == "test"]
    if not fit or not test:
        return {"error": "not enough rows"}

    def arr(rs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        return np.array([float(not r["clean_correct"]) for r in rs]), np.array([float(not r["correct"]) for r in rs])

    cw_f, y_f = arr(fit)
    base = cw_f.mean()

    def mle(cw: np.ndarray, y: np.ndarray) -> float:
        if not len(y):
            return 0.0
        e = LAM * cw + (1 - LAM) * base
        p = _predict(GRID[:, None], 0.0, e[None, :], np.ones((1, len(e))))
        ll = (y * np.log(p) + (1 - y) * np.log(1 - p)).sum(1)
        return float(GRID[int(np.argmax(ll))])

    g_f = np.array([link_group(r) for r in fit])
    c_g = {g: mle(cw_f[g_f == g], y_f[g_f == g]) for g in GROUPS}
    c_blind = mle(cw_f, y_f)
    cw_t, y_t = arr(test)
    e_t = LAM * cw_t + (1 - LAM) * base
    ones = np.ones(len(test))
    p_link = _predict(np.array([c_g[link_group(r)] for r in test]), 0.0, e_t, ones)
    p_blind = _predict(c_blind, 0.0, e_t, ones)
    diff_by_conv: dict[str, list[float]] = defaultdict(list)
    for r, d in zip(test, (p_link - y_t) ** 2 - (p_blind - y_t) ** 2, strict=True):
        diff_by_conv[r["conv"]].append(float(d))
    d_mean, d_lo, d_hi = cluster_bootstrap(diff_by_conv)
    return {
        "c": c_g,
        "graph_blind_c": c_blind,
        "heldout_brier_direct_link": float(np.mean((p_link - y_t) ** 2)),
        "heldout_brier_graph_blind": float(np.mean((p_blind - y_t) ** 2)),
        "heldout_n": len(test),
        # Direct-link model minus graph-blind model; below 0 favours the direct-link model.
        "brier_diff": d_mean,
        "brier_diff_ci_low": d_lo,
        "brier_diff_ci_high": d_hi,
    }


def link_groups(rows: list[dict], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Extra error for direct-link, indirect-only and independent turns, and the direct-link model."""
    by_group = {g: [r for r in rows if link_group(r) == g] for g in GROUPS}
    out: dict = {}
    for g, rs in by_group.items():
        m, lo, hi = cluster_bootstrap(_by_conv(rs), n_boot, seed)
        out[g] = {"extra_error": m, "ci_low": lo, "ci_high": hi, "n": len(rs)}
    d = paired_mean_diff(_by_conv(by_group["direct"]), _by_conv(by_group["indirect_only"]), n_boot, seed)
    out["direct_minus_indirect"] = {"diff": d["diff"], "ci_low": d["ci_low"], "ci_high": d["ci_high"]}
    out["direct_link_model"] = _direct_link_model(rows)
    return out


def convention_by_type(rows: list[dict]) -> list[dict]:
    """Per error type (then all types) and dependence: rows, newly wrong turns by mechanism, convention / rows."""
    out = []
    for t in [*sorted({r["type"] for r in rows}), "all"]:
        for dep in ("depends", "independent"):
            rs = [r for r in rows if (t == "all" or r["type"] == t) and bool(r["depends"]) == (dep == "depends")]
            if not rs:
                continue
            new = [r for r in rs if r["clean_correct"] and not r["correct"]]
            counts = {k: 0 for k in analysis.MECHANISMS}
            for r in new:
                counts[mechanism(r)] += 1
            out.append(
                {
                    "type": t,
                    "turns": dep,
                    "rows": len(rs),
                    "newly_wrong": len(new),
                    **counts,
                    "convention_rate": counts["convention"] / len(rs),
                }
            )
    return out


def _slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def _load_rows(out: Path, model: str) -> list[dict] | None:
    cp, ip = out / "clean" / f"{_slug(model)}.jsonl", out / "inject" / f"{_slug(model)}.jsonl"
    if not (cp.exists() and ip.exists()):
        return None
    return analysis.cascade_rows(analysis.read_jsonl(cp), analysis.read_jsonl(ip))


def _ci(lo: float, hi: float, fmt: str = "+.3f") -> str:
    return f"[{lo:{fmt}}, {hi:{fmt}}]"


def posthoc_section(cfg: dict, out: Path, summary: dict) -> list[str]:
    """Compute the post-hoc analyses for every model with results, fill summary["posthoc"], return markdown."""
    rows_by_model = {m: rs for m in cfg["models"] if (rs := _load_rows(out, m)) is not None}
    ph: dict = {"models": {}}
    for model, rows in rows_by_model.items():
        ph["models"][model] = {
            "dep_minus_indep": dep_minus_indep(rows),
            "transmission_bootstrap": _public(bootstrap_transmission(rows)),
            "link_groups": link_groups(rows),
            "convention_by_type": convention_by_type(rows),
        }
    pair = (cfg.get("posthoc") or {}).get("size_pair")
    if pair and len(pair) == 2 and all(p in rows_by_model for p in pair):
        ph["a_difference"] = {"model_a": pair[0], "model_b": pair[1], **a_difference(rows_by_model[pair[0]], rows_by_model[pair[1]])}
    summary["posthoc"] = ph

    lines = [
        HEADING,
        "",
        "Analyses chosen after seeing the main results. Intervals are 95% conversation cluster bootstraps "
        "(2,000 resamples, seed 0). Extra error per row is error with the injection minus error with the clean history.",
        "",
        "### Dependent minus independent extra error (paired bootstrap)",
        "",
        "| Model | Dependent | Independent | Dependent minus independent | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for model, r in ph["models"].items():
        d = r["dep_minus_indep"]
        lines.append(f"| {model} | {d['dependent']:+.3f} | {d['independent']:+.3f} | {d['diff']:+.3f} | {_ci(d['ci_low'], d['ci_high'])} |")
    lines += [
        "",
        "### Interval on the transmission rate a (fit half resampled and refitted on the same grid)",
        "",
        "| Model | a | 95% CI | s | 95% CI | Fit-half conversations |",
        "|---|---|---|---|---|---|",
    ]
    for model, r in ph["models"].items():
        t = r["transmission_bootstrap"]
        lines.append(
            f"| {model} | {t['a']:.3f} | {_ci(t['a_ci_low'], t['a_ci_high'], '.3f')} | {t['s']:.3f} | "
            f"{_ci(t['s_ci_low'], t['s_ci_high'], '.3f')} | {t['n_conversations']} |"
        )
    if "a_difference" in ph:
        ad = ph["a_difference"]
        lines += [
            "",
            f"a of {ad['model_a']} minus a of {ad['model_b']} (same resampled conversations for both): "
            f"{ad['diff']:+.3f} {_ci(ad['ci_low'], ad['ci_high'])}; share of resamples with the difference at or "
            f"below 0: {ad['share_a_lower_or_equal']:.3f}. On the shared resamples, a = {ad['bootstrap_a']['a']:.3f} "
            f"{_ci(ad['bootstrap_a']['a_ci_low'], ad['bootstrap_a']['a_ci_high'], '.3f')} and "
            f"{ad['bootstrap_b']['a']:.3f} {_ci(ad['bootstrap_b']['a_ci_low'], ad['bootstrap_b']['a_ci_high'], '.3f')}.",
        ]
    lines += [
        "",
        "### Direct link, indirect-only paths and independent turns",
        "",
        "| Model | Direct link | n | Indirect only | n | Independent | n | Direct minus indirect only |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, r in ph["models"].items():
        g = r["link_groups"]
        cells = " | ".join(f"{g[k]['extra_error']:+.3f} {_ci(g[k]['ci_low'], g[k]['ci_high'])} | {g[k]['n']}" for k in GROUPS)
        dd = g["direct_minus_indirect"]
        lines.append(f"| {model} | {cells} | {dd['diff']:+.3f} {_ci(dd['ci_low'], dd['ci_high'])} |")
    lines += [
        "",
        "Indirect only means the turn depends on the injected turn only through other turns (no direct link).",
        "",
        "Direct-link model: P = 1 - (1 - e_k)(1 - c_g), one rate c_g per group (direct, indirect only, independent), "
        "fitted on the fit half by grid MLE with the same e_k as the transmission model. Compared on the held-out "
        "half with the graph-blind model (below 0 favours the direct-link model).",
        "",
        "| Model | c direct | c indirect only | c independent | Brier direct-link | Brier graph-blind | Difference | 95% CI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, r in ph["models"].items():
        dl = r["link_groups"]["direct_link_model"]
        if "error" in dl:
            continue
        c = dl["c"]
        lines.append(
            f"| {model} | {c['direct']:.3f} | {c['indirect_only']:.3f} | {c['independent']:.3f} | "
            f"{dl['heldout_brier_direct_link']:.4f} | {dl['heldout_brier_graph_blind']:.4f} | {dl['brier_diff']:+.4f} | "
            f"{_ci(dl['brier_diff_ci_low'], dl['brier_diff_ci_high'], '+.4f')} |"
        )
    lines += ["", "### Convention spillover by error type (newly wrong turns by mechanism)", ""]
    for model, r in ph["models"].items():
        lines += [
            f"#### {model}",
            "",
            "| Error type | Later turns | Rows | Newly wrong | " + " | ".join(analysis.MECHANISMS) + " | Convention / rows |",
            "|---|---|---|---|" + "---|" * len(analysis.MECHANISMS) + "---|",
        ]
        for c in r["convention_by_type"]:
            lines.append(
                f"| {c['type']} | {c['turns']} | {c['rows']} | {c['newly_wrong']} | "
                + " | ".join(str(c[k]) for k in analysis.MECHANISMS)
                + f" | {c['convention_rate']:.3f} |"
            )
        lines.append("")
    return lines
