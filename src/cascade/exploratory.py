"""Report sections for the exploratory runs added after freezing (PREREGISTRATION.md change 2, CHANGELOG.md R31).

Called from ``cascade report``. Each function returns markdown lines and fills ``summary``. These runs are not
tests of H1 to H5; every one that was run is reported.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from . import analysis
from .run import _slug


def _breaker_table(scores: list[dict]) -> list[str]:
    lines = ["| Group | Metric | Value | 95% CI | n | Unusable |", "|---|---|---|---|---|---|"]
    for sc in scores:
        ci = f"[{sc['ci_low']:.3f}, {sc['ci_high']:.3f}]"
        lines.append(f"| {sc['group']} | {sc['metric']} | {sc['value']:.3f} | {ci} | {sc['n']} | {sc['unusable_rate']:.3f} |")
    return lines


def exploratory_breakers_section(cfg: dict, out: Path, summary: dict) -> list[str]:
    """Scores of ``cfg["exploratory_breakers"]``, in the same table as the pre-registered breakers."""
    found = []
    for b in cfg.get("exploratory_breakers") or []:
        bp = Path(out) / "breakers" / f"{b['name']}.jsonl"
        if bp.exists():
            found.append((b, analysis.breaker_scores(analysis.read_jsonl(bp))))
    if not found:
        return []
    lines = [
        "## Exploratory breakers (added after freezing; PREREGISTRATION change 2)",
        "",
        "Not tests of H4. Same prompts, parsing, unusable-check rule and scores as the pre-registered breakers.",
        "",
    ]
    store = summary.setdefault("exploratory_breakers", {})
    for b, scores in found:
        store[b["name"]] = scores
        lines += [f"### Breaker: {b['name']} ({b['kind']}, answerer {b['answerer']}, checker {b['checker']})", ""]
        lines += _breaker_table(scores)
        lines.append("")
    return lines


def _read(path: Path) -> list[dict] | None:
    return analysis.read_jsonl(path) if path.exists() else None


def _by_turn(rows: list[dict]) -> dict:
    return {(r["conv"], r["turn"]): r for r in rows}


def _err(r: dict) -> float:
    """Extra error of one injected row: wrong now minus wrong with its own run's clean history."""
    return float(not r["correct"]) - float(not r["clean_correct"])


def _paired(guard: dict, base: dict, value) -> dict:
    """Mean of paired differences (guarded minus v2) over keys in both runs, clustered by conversation."""
    keys = sorted(set(guard) & set(base))
    diffs: dict[str, list[float]] = defaultdict(list)
    g_vals: dict[str, list[float]] = defaultdict(list)
    b_vals: dict[str, list[float]] = defaultdict(list)
    for k in keys:
        conv = guard[k]["conv"]
        g, b = value(guard[k]), value(base[k])
        diffs[conv].append(g - b)
        g_vals[conv].append(g)
        b_vals[conv].append(b)
    d, lo, hi = analysis.cluster_bootstrap(diffs)
    n = len(keys)
    return {
        "n_paired": n,
        "n_conversations": len(diffs),
        "n_guarded": len(guard),
        "n_v2": len(base),
        "guarded": sum(map(sum, g_vals.values())) / n if n else float("nan"),
        "v2": sum(map(sum, b_vals.values())) / n if n else float("nan"),
        "diff": d,
        "ci_low": lo,
        "ci_high": hi,
    }


def guard_comparison(guard_dir: Path, base_dir: Path, model: str) -> dict:
    """Guarded minus v2 for one model: clean accuracy, dependent and independent extra error, free accuracy."""
    slug = _slug(model)
    res: dict = {}
    clean_g, clean_b = _read(guard_dir / "clean" / f"{slug}.jsonl"), _read(base_dir / "clean" / f"{slug}.jsonl")
    if clean_g is not None and clean_b is not None:
        res["clean_accuracy"] = _paired(_by_turn(clean_g), _by_turn(clean_b), lambda r: float(r["correct"]))
        inj_g, inj_b = _read(guard_dir / "inject" / f"{slug}.jsonl"), _read(base_dir / "inject" / f"{slug}.jsonl")
        if inj_g is not None and inj_b is not None:
            rows_g, rows_b = analysis.cascade_rows(clean_g, inj_g), analysis.cascade_rows(clean_b, inj_b)
            for name, dep in (("dependent_extra_error", True), ("independent_extra_error", False)):
                g = {r["key"]: r for r in rows_g if bool(r["depends"]) == dep}
                b = {r["key"]: r for r in rows_b if bool(r["depends"]) == dep}
                res[name] = _paired(g, b, _err)
    free_g, free_b = _read(guard_dir / "free" / f"{slug}.jsonl"), _read(base_dir / "free" / f"{slug}.jsonl")
    if free_g is not None and free_b is not None:
        res["free_accuracy"] = _paired(_by_turn(free_g), _by_turn(free_b), lambda r: float(r["correct"]))
    return res


_LABELS = {
    "clean_accuracy": "Clean accuracy",
    "dependent_extra_error": "Dependent extra error",
    "independent_extra_error": "Independent extra error",
    "free_accuracy": "Free (own-history) accuracy",
}


def guard_comparison_section(cfg: dict, out: Path, summary: dict) -> list[str]:
    """Per model in ``cfg["models"]``: guarded run (``out``) minus the v2 run in ``cfg["compare_to"]``."""
    base = cfg.get("compare_to")
    if not base:
        return []
    out, base = Path(out), Path(base)
    lines = [
        "## Guarded prompt minus v2 (exploratory, added after freezing)",
        "",
        f"Guarded run (prompt {cfg.get('prompt', 'v2')}): `{out}`; v2 run: `{base}`. Each value is a mean of paired row "
        "differences (same conversation and turn; for injected rows also the same injection), with a conversation "
        "cluster bootstrap 95% interval (2,000 resamples, seed 0). Extra error per injected row is wrong after the "
        "injection minus wrong with the clean history, each run against its own clean answers. Only rows present in "
        "both runs are paired. A lower clean accuracy is a cost of the prompt.",
        "",
    ]
    store = summary.setdefault("guard_comparison", {})
    for model in cfg["models"]:
        res = guard_comparison(out, base, model)
        if not res:
            continue
        store[model] = res
        lines += [f"### {model}", ""]
        lines += [
            "| Measure | Guarded | v2 | Guarded minus v2 | 95% CI | n paired | Conversations | Rows guarded / v2 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for key, label in _LABELS.items():
            if key not in res:
                continue
            m = res[key]
            if not m["n_paired"]:
                lines.append(f"| {label} | | | | | 0 | 0 | {m['n_guarded']} / {m['n_v2']} |")
                continue
            lines.append(
                f"| {label} | {m['guarded']:.3f} | {m['v2']:.3f} | {m['diff']:+.3f} | [{m['ci_low']:+.3f}, {m['ci_high']:+.3f}] "
                f"| {m['n_paired']} | {m['n_conversations']} | {m['n_guarded']} / {m['n_v2']} |"
            )
        lines.append("")
    return lines
