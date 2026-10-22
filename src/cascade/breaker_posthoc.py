"""Post-hoc breaker analyses (added 17 Oct 2026 after the main results; not pre-registered; CHANGELOG R32).

1. Strict rescoring of the program breaker. The pre-registered program breaker trips when the recomputed value
   does not match the checked answer under the *lenient* matcher, which treats x100 and /100 as equal. A planted
   ``percent`` error is exactly gold x100 or /100 (``inject.corrupt``), so a program that recomputes the right
   value cannot trip on it. Rescoring with the strict matcher (the study's primary metric) needs no model calls:
   the recomputed value is stored in every row. The pre-registered scores are kept as the primary result.
2. Catch rate on planted errors by error type, for every breaker (as run).
3. Three outcomes per check: agree, disagree, unable to verify (no verdict, or a program that cannot run). The
   pre-registered rule (unable counts as not tripped) is unchanged; this only shows the third outcome.
4. Program repair, evaluated without running it: if a tripped check replaced the answer with the program's value,
   would the result be right (strict match with gold)? Counts of wrong answers repaired, wrong answers replaced by
   another wrong value, and right answers corrupted.
5. Scoring audit, number format (added 17 Oct 2026, after the strict rescoring). The answer prompt fixes the number
   format (ratios as decimals unless the question asks for x100); the program prompt does not, so a program may
   return a percentage in percent form. Classifies how each program value differs from the checked answer, splits
   natural errors into format-only (right under the lenient matcher) and substantive, and adds an oracle bound that
   knows the required form from gold's order of magnitude: a percent-form difference trips only when the program
   value is in that form and the checked answer is not. It uses gold, so it is a bound for a format-aware rule,
   not a breaker.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from . import analysis
from .analysis import cluster_bootstrap
from .match import match

HEADING = "## Post-hoc breaker analyses (added 17 Oct 2026 after the main results; not pre-registered)"
GROUPS = ("inject:wrong", "gold:right", "free:wrong", "free:right")
OUTCOMES = ("agree", "disagree", "unable")
REPAIR = ("kept", "repaired_right", "replaced_wrong", "corrupted", "format_only", "unable")
RELATIONS = ("equal", "percent_form", "scale", "sign", "other", "nonnumeric")
PERCENT_WORDS = re.compile(r"percent|%", re.IGNORECASE)
NATURAL = ("free:wrong, substantive", "free:wrong, format only")


def group(r: dict) -> str:
    return r["label"].split(":")[0] + (":wrong" if r["is_wrong"] else ":right")


def outcome(r: dict) -> str:
    if not r["usable"]:
        return "unable"
    return "disagree" if r["tripped"] else "agree"


def tripped_strict(r: dict) -> bool:
    """Program breaker trip rule with the strict matcher; an unusable check stays not tripped."""
    return bool(r["usable"]) and not match(r["value"], r["answer"], mode="strict")


def relation(value, answer) -> str:
    """How a program value differs from the checked answer: equal (strict), percent_form (x100 or /100), scale
    (x or / 1,000 or 1,000,000), sign, other; nonnumeric when either side is not a number."""
    if value is None:
        return "nonnumeric"
    if match(value, answer, mode="strict"):
        return "equal"
    if isinstance(value, str) or isinstance(answer, str):
        return "nonnumeric"
    if match(value, answer, mode="lenient"):
        return "percent_form"
    if any(match(value, answer * f, mode="strict") for f in (1e3, 1e-3, 1e6, 1e-6)):
        return "scale"
    if match(value, -answer, mode="strict"):
        return "sign"
    return "other"


def _in_form(x: float, g) -> bool:
    """x is in gold's form: within a factor of 10 of gold (of two values 100 times apart, at most one is)."""
    return not isinstance(g, str) and g != 0 and x != 0 and 0.1 < abs(x / g) < 10


def tripped_oracle(r: dict, gold: dict) -> bool:
    """Oracle format-aware rule. It knows only which form the question requires, taken from gold's order of
    magnitude: a percent-form difference trips when the program value is in that form and the answer is not, and
    does not trip otherwise. Every other difference trips; an unusable check stays not tripped."""
    if not r["usable"]:
        return False
    rel = relation(r["value"], r["answer"])
    if rel == "equal":
        return False
    if rel == "percent_form":
        g = gold[(r["conv"], r["turn"])]
        return _in_form(r["value"], g) and not _in_form(r["answer"], g)
    return True


def natural_kind(r: dict, gold: dict) -> str | None:
    """For a natural wrong answer: format only (right under the lenient matcher) or substantive."""
    if not (r["label"].startswith("free") and r["is_wrong"]):
        return None
    return NATURAL[1] if match(r["answer"], gold[(r["conv"], r["turn"])], mode="lenient") else NATURAL[0]


def _rate(rows: list[dict], fn) -> dict:
    by_conv: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_conv[r["conv"]].append(float(fn(r)))
    m, lo, hi = cluster_bootstrap(by_conv)
    return {"value": m, "ci_low": lo, "ci_high": hi, "n": len(rows)}


def _subsets(rows: list[dict]) -> dict[str, list[dict]]:
    """The four groups, then one subset per planted error type."""
    out: dict[str, list[dict]] = {g: [] for g in GROUPS}
    types: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out.setdefault(group(r), []).append(r)
        if r["label"].startswith("inject:"):
            types[r["label"]].append(r)
    return {**{g: rs for g, rs in out.items() if rs}, **dict(sorted(types.items()))}


def rescore(rows: list[dict]) -> list[dict]:
    """Trip rate as run (lenient) and rescored strict, per group and per planted error type."""
    return [
        {"subset": name, "lenient": _rate(rs, lambda r: r["tripped"]), "strict": _rate(rs, tripped_strict)}
        for name, rs in _subsets(rows).items()
    ]


def catch_by_type(rows: list[dict]) -> list[dict]:
    """Catch rate (as run) on planted errors, per error type."""
    return [{"subset": n, **_rate(rs, lambda r: r["tripped"])} for n, rs in _subsets(rows).items() if n.startswith("inject:")]


def outcomes(rows: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        out[group(r)][outcome(r)] += 1
    return {g: {o: out[g][o] for o in OUTCOMES} for g in GROUPS if g in out}


def repair(rows: list[dict], gold: dict[tuple[str, int], float | str]) -> dict[str, dict[str, int]]:
    """What replacing a tripped answer with the program's value would do (original lenient trip rule).

    ``kept``: not tripped, answer kept; ``repaired_right``: a wrong answer replaced by the gold value;
    ``replaced_wrong``: a wrong answer replaced by another wrong value; ``corrupted``: a right answer replaced by a
    wrong value; ``format_only``: a right answer replaced by a value that also matches gold strictly; ``unable``:
    no usable program (nothing to repair with).
    """
    out: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        g = group(r)
        if not r["usable"]:
            out[g]["unable"] += 1
        elif not r["tripped"]:
            out[g]["kept"] += 1
        else:
            right = match(r["value"], gold[(r["conv"], r["turn"])], mode="strict")
            if r["is_wrong"]:
                out[g]["repaired_right" if right else "replaced_wrong"] += 1
            else:
                out[g]["format_only" if right else "corrupted"] += 1
    return {g: {k: out[g][k] for k in REPAIR} for g in GROUPS if g in out}


def _audit_subsets(rows: list[dict], gold: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[group(r)].append(r)
        if (k := natural_kind(r, gold)) is not None:
            out[k].append(r)
        if r["label"] == "inject:percent":
            out["inject:percent"].append(r)
    order = (*GROUPS, *NATURAL, "inject:percent")
    return {k: out[k] for k in order if out.get(k)}


def three_rules(rows: list[dict], gold: dict) -> list[dict]:
    """Program trip rate under the lenient rule (as run), strict, and the oracle format-aware bound."""
    return [
        {
            "subset": name,
            "lenient": _rate(rs, lambda r: r["tripped"]),
            "strict": _rate(rs, tripped_strict),
            "oracle": _rate(rs, lambda r: tripped_oracle(r, gold)),
        }
        for name, rs in _audit_subsets(rows, gold).items()
    ]


def relation_counts(rows: list[dict], questions: dict) -> dict[str, dict[str, int]]:
    """Usable program checks per group by relation; percent_form_pct_q counts percent-form cases whose question
    contains 'percent' or '%'."""
    out: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if not r["usable"]:
            continue
        rel = relation(r["value"], r["answer"])
        out[group(r)][rel] += 1
        if rel == "percent_form" and PERCENT_WORDS.search(questions.get((r["conv"], r["turn"]), "")):
            out[group(r)]["percent_form_pct_q"] += 1
    return {g: {k: out[g][k] for k in (*RELATIONS, "percent_form_pct_q")} for g in GROUPS if g in out}


def percent_plants(rows: list[dict], gold: dict) -> dict[str, int]:
    """Planted percent errors by direction (``inject.corrupt``: x100 when |gold| < 5, else /100)."""
    c = Counter()
    for r in rows:
        if r["label"] == "inject:percent":
            g = gold[(r["conv"], r["turn"])]
            c["x100" if not isinstance(g, str) and abs(g) < 5 else "div100"] += 1
    return {"x100": c["x100"], "div100": c["div100"]}


def natural_split(rows: list[dict], gold: dict) -> dict[str, dict]:
    """Catch rate (as run) on natural errors, format only against substantive."""
    subs = _audit_subsets(rows, gold)
    return {k: _rate(subs[k], lambda r: r["tripped"]) for k in NATURAL if k in subs}


def _ci(d: dict) -> str:
    return f"{d['value']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"


def breaker_posthoc_section(
    cfg: dict, out: Path, summary: dict, gold: dict[tuple[str, int], float | str], questions: dict[tuple[str, int], str] | None = None
) -> list[str]:
    """All breakers in ``breakers`` and ``exploratory_breakers`` that have results; fills summary["posthoc_breakers"]."""
    found = []
    for b in list(cfg.get("breakers") or []) + list(cfg.get("exploratory_breakers") or []):
        bp = Path(out) / "breakers" / f"{b['name']}.jsonl"
        if bp.exists():
            found.append((b, analysis.read_jsonl(bp)))
    if not found:
        return []
    store: dict = {}
    for b, rows in found:
        store[b["name"]] = {"catch_by_type": catch_by_type(rows), "outcomes": outcomes(rows)}
        if b["kind"] == "program":
            store[b["name"]]["strict_rescore"] = rescore(rows)
            store[b["name"]]["repair"] = repair(rows, gold)
            store[b["name"]]["format_audit"] = {
                "relations": relation_counts(rows, questions or {}),
                "rules": three_rules(rows, gold),
                "percent_plants": percent_plants(rows, gold),
            }
        store[b["name"]]["natural_split"] = natural_split(rows, gold)
    summary["posthoc_breakers"] = store

    lines = [
        HEADING,
        "",
        "Chosen after seeing the main results. The pre-registered breaker scores above stay the primary result. "
        "Intervals are 95% conversation cluster bootstraps (2,000 resamples, seed 0).",
        "",
        "### Catch rate on planted errors by error type (as run)",
        "",
        "| Breaker | "
        + " | ".join(
            ("all" if r["subset"] == "inject:wrong" else r["subset"].split(":")[1]) for r in store[found[0][0]["name"]]["catch_by_type"]
        )
        + " |",
        "|---|" + "---|" * len(store[found[0][0]["name"]]["catch_by_type"]),
    ]
    for b, _ in found:
        lines.append(f"| {b['name']} | " + " | ".join(_ci(r) for r in store[b["name"]]["catch_by_type"]) + " |")
    lines += [
        "",
        "### Program breaker rescored with the strict matcher",
        "",
        "As run, the program breaker compares its value with the checked answer under the lenient matcher, which "
        "treats x100 and /100 as equal; a planted percent error is exactly gold x100 or /100, so a correct program "
        "cannot trip on it. Strict: the same stored values compared with the strict matcher (the study's primary "
        "metric). No model calls. Neither rule uses units, which the program output does not carry.",
        "",
    ]
    for b, _ in found:
        if b["kind"] != "program":
            continue
        lines += [
            f"#### {b['name']} (answerer {b['answerer']})",
            "",
            "| Checked | n | Trip rate, lenient (as run) | Trip rate, strict |",
            "|---|---|---|---|",
        ]
        for r in store[b["name"]]["strict_rescore"]:
            lines.append(f"| {r['subset']} | {r['lenient']['n']} | {_ci(r['lenient'])} | {_ci(r['strict'])} |")
        lines.append("")
    lines += [
        "### Three outcomes per check: agree, disagree, unable to verify",
        "",
        "Unable: no parseable verdict, or a program that cannot run. The pre-registered rule counts it as not tripped.",
        "",
        "| Breaker | Checked | Agree | Disagree | Unable |",
        "|---|---|---|---|---|",
    ]
    for b, _ in found:
        for g, c in store[b["name"]]["outcomes"].items():
            lines.append(f"| {b['name']} | {g} | {c['agree']} | {c['disagree']} | {c['unable']} |")
    lines += [
        "",
        "### Program repair, evaluated without running it",
        "",
        "If every tripped check (as run) replaced the answer with the program's value: is the result right (strict "
        "match with gold)? Kept: not tripped. Format only: tripped, but the program's value also matches gold.",
        "",
        "| Breaker | Checked | Kept | Wrong repaired to right | Wrong replaced by another wrong | Right corrupted | Format only | Unable |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for b, _ in found:
        if b["kind"] != "program":
            continue
        for g, c in store[b["name"]]["repair"].items():
            lines.append(f"| {b['name']} | {g} | " + " | ".join(str(c[k]) for k in REPAIR) + " |")
    lines += [
        "",
        "### Scoring audit: number format (added 17 Oct 2026, after the strict rescoring)",
        "",
        "The answer prompt fixes the number format (ratios and percentage changes as decimals unless the question asks "
        "to multiply by 100); the program prompt states no format, so a program may return a percentage in percent "
        "form. Relation: how each usable program value differs from the checked answer. Percent form: x100 or /100 "
        "(trips under strict, not under lenient). Scale: x or / 1,000 or 1,000,000.",
        "",
        "Planted percent errors by direction: "
        + "; ".join(
            f"{b['name']}: {store[b['name']]['format_audit']['percent_plants']['x100']} x100 on a gold below 5 (a ratio "
            f"written as a percent), {store[b['name']]['format_audit']['percent_plants']['div100']} /100 on a larger gold "
            "(an amount 100 times too small)"
            for b, _ in found
            if b["kind"] == "program"
        )
        + ". The lenient matcher accepts both directions.",
        "",
        "| Breaker | Checked | Equal | Percent form | of which question says percent | Scale | Sign | Other | Not numeric |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for b, _ in found:
        if b["kind"] != "program":
            continue
        for g, c in store[b["name"]]["format_audit"]["relations"].items():
            cells = [c["equal"], c["percent_form"], c["percent_form_pct_q"], c["scale"], c["sign"], c["other"], c["nonnumeric"]]
            lines.append(f"| {b['name']} | {g} | " + " | ".join(str(x) for x in cells) + " |")
    lines += [
        "",
        "Trip rate under three rules. Lenient: as run (pre-registered). Strict: any difference trips. Oracle format: "
        "knows only which form the question requires, taken from gold's order of magnitude; a percent-form difference "
        "trips when the program value is in that form and the checked answer is not. It uses gold, so it is a bound "
        "for a format-aware rule, not a breaker. Natural errors are split into format only (right under "
        "the lenient matcher) and substantive.",
        "",
    ]
    for b, _ in found:
        if b["kind"] != "program":
            continue
        lines += [
            f"#### {b['name']}",
            "",
            "| Checked | n | Lenient (as run) | Strict | Oracle format |",
            "|---|---|---|---|---|",
        ]
        for r in store[b["name"]]["format_audit"]["rules"]:
            lines.append(f"| {r['subset']} | {r['lenient']['n']} | {_ci(r['lenient'])} | {_ci(r['strict'])} | {_ci(r['oracle'])} |")
        lines.append("")
    lines += [
        "Catch rate (as run) on natural errors, substantive against format only:",
        "",
        "| Breaker | Substantive | n | Format only | n |",
        "|---|---|---|---|---|",
    ]
    for b, _ in found:
        ns = store[b["name"]]["natural_split"]
        if all(k in ns for k in NATURAL):
            s, f = ns[NATURAL[0]], ns[NATURAL[1]]
            lines.append(f"| {b['name']} | {_ci(s)} | {s['n']} | {_ci(f)} | {f['n']} |")
    lines.append("")
    return lines
