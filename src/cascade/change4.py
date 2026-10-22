"""Report section for PREREGISTRATION change 4: the revised program breaker against the frozen one.

A prospectively specified exploratory extension, separate from the pre-registered H4 experiment (CHANGELOG
R37, R38). Analysis fixed in PREREGISTRATION change 4 Part A (``paper/CHANGE4.md`` section 5), written before
any final-evaluation output existed:

- pairs of rows with the same key (same item, answer and history) from the frozen ``program`` breaker and
  ``program_v2``, per answering model;
- primary pair: natural-error recall (own wrong answers) and natural false-alarm rate (own right answers);
- primary comparator: the frozen breaker rescored with the strict matcher; secondary: as originally scored;
- paired row differences (revised minus comparator), conversation-level cluster bootstrap, two-sided 95%,
  2,000 resamples, seed 0;
- recall: improved (lower bound > 0), worse (upper bound < 0), else inconclusive; false alarms with margin
  delta: non-inferior (upper bound < delta), materially higher (lower bound > delta), else inconclusive;
- improved for a model only if recall is improved and false alarms non-inferior against the primary comparator;
  across models only if every model is improved.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from . import analysis
from .analysis import cluster_bootstrap
from .breaker_posthoc import group, natural_kind, repair, tripped_strict
from .match import match
from .program_v2 import REASONS, SOURCES

HEADING = "## Revised program breaker (PREREGISTRATION change 4; prospectively specified exploratory extension)"
N_BOOT, SEED = 2000, 0


def filing(conv: str) -> str:
    """Company and year of a conversation id, e.g. 'JKHY/2009' (sensitivity clustering)."""
    return "/".join(re.sub(r"-\d+$", "", conv.split("_", 1)[1]).split("/")[:2])


def paired(frozen: list[dict], revised: list[dict]) -> list[tuple[dict, dict]]:
    by_key = {r["key"]: r for r in frozen}
    return [(by_key[r["key"]], r) for r in revised if r["key"] in by_key]


def _natural(r: dict, wrong: bool) -> bool:
    return r["label"] == "free" and bool(r["is_wrong"]) == wrong


COMPARATORS = {"strict": tripped_strict, "lenient": lambda r: bool(r["tripped"])}


def paired_diff(pairs: list[tuple[dict, dict]], wrong: bool, comparator: str, revised_trip=None, cluster=None) -> dict:
    """Mean paired difference (revised minus comparator) on natural wrong (recall) or right (false alarms) rows."""
    revised_trip = revised_trip or (lambda r: bool(r["tripped"]))
    cluster = cluster or (lambda r: r["conv"])
    comp = COMPARATORS[comparator]
    by: dict[str, list[float]] = defaultdict(list)
    rev, base = [], []
    for f, r in pairs:
        if _natural(r, wrong):
            a, b = float(revised_trip(r)), float(comp(f))
            by[cluster(r)].append(a - b)
            rev.append(a)
            base.append(b)
    m, lo, hi = cluster_bootstrap(by, n_boot=N_BOOT, seed=SEED)
    n = len(rev)
    return {
        "diff": m,
        "ci_low": lo,
        "ci_high": hi,
        "n": n,
        "conversations": len(by),
        "revised": sum(rev) / n if n else float("nan"),
        "comparator": sum(base) / n if n else float("nan"),
    }


def classify_recall(d: dict) -> str:
    if d["ci_low"] > 0:
        return "improved"
    if d["ci_high"] < 0:
        return "worse"
    return "inconclusive"


def classify_false_alarms(d: dict, delta: float) -> str:
    if d["ci_high"] < delta:
        return "non-inferior"
    if d["ci_low"] > delta:
        return "materially higher"
    return "inconclusive"


def model_call(recall: str, false_alarms: str) -> str:
    return "improved" if recall == "improved" and false_alarms == "non-inferior" else "not shown"


def _rate(rows: list[dict], fn) -> dict:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by[r["conv"]].append(float(fn(r)))
    m, lo, hi = cluster_bootstrap(by, n_boot=N_BOOT, seed=SEED)
    return {"value": m, "ci_low": lo, "ci_high": hi, "n": len(rows)}


def _ci(d: dict) -> str:
    return f"{d['value']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"


def _dci(d: dict) -> str:
    return f"{d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


def change4_section(cfg: dict, out: Path, summary: dict, gold: dict) -> list[str]:
    spec = cfg.get("change4") or {}
    delta = float(spec.get("margin", 0.02))
    found = []
    for p in spec.get("pairs", []):
        fp, rp = Path(out) / "breakers" / f"{p['frozen']}.jsonl", Path(out) / "breakers" / f"{p['revised']}.jsonl"
        if fp.exists() and rp.exists():
            found.append((p, analysis.read_jsonl(fp), analysis.read_jsonl(rp)))
    if not found:
        return []
    store: dict = {}
    lines = [
        HEADING,
        "",
        "Not part of the original confirmatory study and not a test of H4. Paired rows: the same item, answer and "
        "history checked by the frozen program breaker and by program_v2. Differences are revised minus comparator, "
        f"with a conversation-level cluster bootstrap 95% interval ({N_BOOT:,} resamples, seed {SEED}). Primary "
        "comparator: the frozen breaker rescored with the strict matcher; secondary: as originally scored. Unable "
        f"counts as no flag and stays in every denominator. Margin for false alarms: {delta:+.2f}.",
        "",
        "### Primary comparison",
        "",
        "| Answerer | Outcome | Comparator | program_v2 | Comparator | Difference | n | Conversations | Classification |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    calls = {}
    for p, frozen, revised in found:
        pairs = paired(frozen, revised)
        res = {}
        for comp in ("strict", "lenient"):
            rec, fa = paired_diff(pairs, True, comp), paired_diff(pairs, False, comp)
            res[comp] = {
                "recall": {**rec, "class": classify_recall(rec)},
                "false_alarms": {**fa, "class": classify_false_alarms(fa, delta)},
            }
            for name, d in (("natural recall", res[comp]["recall"]), ("natural false alarms", res[comp]["false_alarms"])):
                label = "strict (primary)" if comp == "strict" else "as run (secondary)"
                lines.append(
                    f"| {p['answerer']} | {name} | {label} | {d['revised']:.3f} | {d['comparator']:.3f} | {_dci(d)} | "
                    f"{d['n']} | {d['conversations']} | {d['class']} |"
                )
        calls[p["answerer"]] = model_call(res["strict"]["recall"]["class"], res["strict"]["false_alarms"]["class"])
        store[p["revised"]] = {
            "answerer": p["answerer"],
            "frozen": p["frozen"],
            "comparisons": res,
            "n_paired": len(pairs),
            "n_revised": len(revised),
            "n_frozen": len(frozen),
        }
    across = "improved for every answerer" if calls and all(c == "improved" for c in calls.values()) else "not shown for every answerer"
    lines += ["", "Calls against the primary comparator (improved only if recall is improved and false alarms are non-inferior):", ""]
    lines += [f"- {a}: {c}" for a, c in calls.items()] + [f"- Across answerers: {across}", ""]
    summary["change4"] = {"pairs": store, "calls": calls, "across": across, "margin": delta}

    lines += [
        "### Secondary outcomes (descriptive)",
        "",
        "Trip rate by group. Substantive: natural wrong answers that are also wrong under the lenient matcher.",
        "",
        "| Answerer | Group | n | program_v2 | Frozen, strict | Frozen, as run | program_v2 unable | Frozen unusable |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p, frozen, revised in found:
        pairs = paired(frozen, revised)
        groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        for f, r in pairs:
            groups[group(r)].append((f, r))
            if (k := natural_kind(r, gold)) is not None:
                groups[k].append((f, r))
        for g in ("inject:wrong", "gold:right", "free:wrong", "free:right", "free:wrong, substantive"):
            ps = groups.get(g, [])
            if not ps:
                continue
            fs, rs = [f for f, _ in ps], [r for _, r in ps]
            lines.append(
                f"| {p['answerer']} | {g} | {len(ps)} | {_ci(_rate(rs, lambda r: r['tripped']))} | "
                f"{_ci(_rate(fs, tripped_strict))} | {_ci(_rate(fs, lambda r: r['tripped']))} | "
                f"{_rate(rs, lambda r: not r['usable'])['value']:.3f} | {_rate(fs, lambda r: not r['usable'])['value']:.3f} |"
            )
    lines += [
        "",
        "Planted catch by error type (program_v2 / frozen strict / frozen as run):",
        "",
        "| Answerer | Type | n | program_v2 | Frozen, strict | Frozen, as run |",
        "|---|---|---|---|---|---|",
    ]
    for p, frozen, revised in found:
        by_type: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        for f, r in paired(frozen, revised):
            if r["label"].startswith("inject:"):
                by_type[r["label"].split(":")[1]].append((f, r))
        for t, ps in sorted(by_type.items()):
            fs, rs = [f for f, _ in ps], [r for _, r in ps]
            lines.append(
                f"| {p['answerer']} | {t} | {len(ps)} | {_rate(rs, lambda r: r['tripped'])['value']:.3f} | "
                f"{_rate(fs, tripped_strict)['value']:.3f} | {_rate(fs, lambda r: r['tripped'])['value']:.3f} |"
            )

    lines += [
        "",
        "Three outcomes and unable reasons for program_v2, by group:",
        "",
        "| Answerer | Group | Agree | Disagree | Unable | " + " | ".join(REASONS) + " |",
        "|---|---|---|---|---|" + "---|" * len(REASONS),
    ]
    for p, _, revised in found:
        by_g: dict[str, Counter] = defaultdict(Counter)
        for r in revised:
            by_g[group(r)][r.get("outcome", "unable")] += 1
            if r.get("reason"):
                by_g[group(r)][r["reason"]] += 1
        for g in ("inject:wrong", "gold:right", "free:wrong", "free:right"):
            if g in by_g:
                c = by_g[g]
                lines.append(
                    f"| {p['answerer']} | {g} | {c['agree']} | {c['disagree']} | {c['unable']} | "
                    + " | ".join(str(c[x]) for x in REASONS)
                    + " |"
                )

    lines += [
        "",
        "Citations by source (program_v2, all rows):",
        "",
        "| Answerer | Citations | " + " | ".join(SOURCES) + " | Invalid |",
        "|---|---|" + "---|" * len(SOURCES) + "---|",
    ]
    for p, _, revised in found:
        c = Counter(x["source"] for r in revised for x in r.get("cites") or [])
        bad = sum(1 for r in revised for x in r.get("cites") or [] if x.get("ok") is False)
        total = sum(c.values())
        lines.append(f"| {p['answerer']} | {total} | " + " | ".join(str(c[s]) for s in SOURCES) + f" | {bad} |")

    lines += [
        "",
        "Offline repair (if a tripped check replaced the answer with the program value; strict match with gold):",
        "",
        "| Answerer | Breaker | Group | Kept | Wrong repaired to right | Wrong replaced by another wrong | Right corrupted "
        "| Format only | Unable |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p, frozen, revised in found:
        for name, rows in (("program_v2", revised), ("frozen, as run", frozen)):
            for g, c in repair(rows, gold).items():
                if g.startswith("free"):
                    lines.append(
                        f"| {p['answerer']} | {name} | {g} | "
                        + " | ".join(str(c[k]) for k in ("kept", "repaired_right", "replaced_wrong", "corrupted", "format_only", "unable"))
                        + " |"
                    )

    lines += [
        "",
        "### Sensitivity analyses (no call)",
        "",
        "| Answerer | Variant | Natural recall difference | Natural false-alarm difference |",
        "|---|---|---|---|",
    ]
    variants = {
        "unable counted as a flag": {"revised_trip": lambda r: bool(r["tripped"]) or not r["usable"]},
        "program_v2 scored leniently": {
            "revised_trip": lambda r: (
                bool(r["usable"]) and r.get("value") is not None and not match(r["value"], r["answer"], mode="lenient")
            )
        },
        "clustered by filing": {"cluster": lambda r: filing(r["conv"])},
    }
    for p, frozen, revised in found:
        pairs = paired(frozen, revised)
        for name, kw in variants.items():
            rec, fa = paired_diff(pairs, True, "strict", **kw), paired_diff(pairs, False, "strict", **kw)
            lines.append(f"| {p['answerer']} | {name} | {_dci(rec)} | {_dci(fa)} |")
    lines.append("")
    return lines
