"""Answer-parser sensitivity (added 20 Oct 2026; CHANGELOG R53, PREREGISTRATION change 7). No model calls.

Until 20 Oct 2026 ``match.extract_answer`` dropped the exponent of an answer in scientific notation ("Answer:
6.4748e+06" was read as 6.4748). The frozen result rows keep the values parsed then and stay the reported results.
This analysis re-reads every stored answer reply with the corrected parser, writes corrected copies of the result
rows under ``out``, re-runs ``cascade report`` on each copy, and lists every number in summary.json that changes.

What a stored row cannot tell us is left as it was, and counted:
- a later free-stage turn whose history showed the misparsed value was answered with that history; its reply
  cannot be re-run without a model call;
- a self or cross check of a natural answer judged the misparsed number; its verdict is kept. Program checks are
  re-scored from their stored value with their frozen trip rule (lenient for ``program``, strict for
  ``program_v2``).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import yaml

from .match import extract_answer, match

STAGES = ("clean", "clean_repeat", "inject", "free")


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
    return a == b


def _slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def correct_answer_row(r: dict) -> dict | None:
    """The row re-read with the current parser, or None when the parsed value does not change."""
    pred = extract_answer(r.get("raw"))
    if _same(pred, r["pred"]):
        return None
    new = {**r, "pred": pred, "correct": match(pred, r["gold"]), "correct_lenient": match(pred, r["gold"], mode="lenient")}
    if "wrong" in r:  # inject rows
        new["copied_wrong"] = match(pred, r["wrong"]) if pred is not None else False
    return new


def correct_breaker_row(r: dict, kind: str, free: dict) -> tuple[dict, str] | None:
    """A natural (free) breaker item re-scored with the corrected answer, and how; None when its answer is unchanged."""
    if r["label"] != "free":
        return None
    fr = free.get((r["conv"], r["turn"]))
    if fr is None:
        return None
    new = {**r, "answer": fr["pred"], "is_wrong": not fr["correct"]}
    if kind == "program":
        new["tripped"] = bool(r["usable"]) and not match(r["value"], fr["pred"], mode="lenient")
        how = "re-scored"
    elif kind == "program_v2":
        if r["usable"]:
            new["outcome"] = "agree" if match(r["value"], fr["pred"], mode="strict") else "disagree"
            new["tripped"] = new["outcome"] == "disagree"
        how = "re-scored"
    else:
        how = "verdict kept (it judged the misparsed number)"
    return new, how


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def corrected_copy(cfg: dict, dst: Path) -> dict:
    """Write corrected copies of one config's result rows to ``dst``; return what changed."""
    src = Path(cfg["out"])
    changes: dict = {"answer_rows": 0, "answers": [], "breakers": [], "later_turns_with_misparsed_history": 0}
    fixed_free: dict[str, dict] = {}
    for stage in STAGES:
        for f in sorted((src / stage).glob("*.jsonl")) if (src / stage).exists() else []:
            rows, free = [], {}
            for r in _read(f):
                changes["answer_rows"] += 1
                new = correct_answer_row(r)
                if new is not None:
                    changes["answers"].append(
                        {
                            "file": f"{src.name}/{stage}/{f.name}",
                            "conv": r["conv"],
                            "turn": r["turn"],
                            "key": r.get("key", ""),
                            "saved": r["pred"],
                            "corrected": new["pred"],
                            "gold": r["gold"],
                            "correct_saved": r["correct"],
                            "correct_corrected": new["correct"],
                        }
                    )
                    if stage == "free":
                        free[(r["conv"], r["turn"])] = new
                        changes["later_turns_with_misparsed_history"] += r["n_turns"] - 1 - r["turn"]
                rows.append(new or r)
            if stage == "free":
                fixed_free[f.stem] = free
            _write(dst / stage / f.name, rows)
    kinds = {b["name"]: (b["kind"], b["answerer"]) for b in cfg.get("breakers", []) + cfg.get("exploratory_breakers", [])}
    for f in sorted((src / "breakers").glob("*.jsonl")) if (src / "breakers").exists() else []:
        kind, answerer = kinds.get(f.stem, (None, None))
        free = fixed_free.get(_slug(answerer), {}) if answerer else {}
        rows = []
        for r in _read(f):
            out = correct_breaker_row(r, kind, free) if free else None
            if out is not None:
                new, how = out
                changes["breakers"].append(
                    {
                        "breaker": f.stem,
                        "conv": r["conv"],
                        "turn": r["turn"],
                        "how": how,
                        "is_wrong": f"{r['is_wrong']} -> {new['is_wrong']}",
                        "tripped": f"{r['tripped']} -> {new['tripped']}",
                    }
                )
                r = new
            rows.append(r)
        _write(dst / "breakers" / f.name, rows)
    if (src / "manifest.json").exists():
        shutil.copy(src / "manifest.json", dst / "manifest.json")
    return changes


def numeric_diff(a, b, path: str = "") -> list[tuple[str, float, float]]:
    """Numeric leaves that differ between two summary.json trees (paths joined with "/")."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in a if k in b for d in numeric_diff(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            label = next(
                (str(x[k]) for k in ("group", "type", "turns", "paths", "breaker", "name") if isinstance(x, dict) and k in x), str(i)
            )
            if isinstance(x, dict) and "type" in x and "turns" in x:
                label = f"{x['type']}|{x['turns']}"
            out += numeric_diff(x, y, f"{path}[{label}]")
        return out
    both_nan = isinstance(a, float) and isinstance(b, float) and a != a and b != b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and a != b and not both_nan:
        return [(path, float(a), float(b))]
    return []


def parser_sensitivity_section(cfg: dict, out: Path, summary: dict, report: Callable[[str], None]) -> list[str]:
    """Build corrected copies for every config in ``cfg["parser_sensitivity"]``, report each, and compare."""
    summary["parser_sensitivity"] = {}
    lines = [
        "## Answer-parser sensitivity (added 20 Oct 2026; not pre-registered; CHANGELOG R53)",
        "",
        "Every stored answer reply re-read with the corrected parser (scientific notation read whole). Corrected copies of the "
        "result rows are under this folder and each is reported with `cascade report` (its own TABLES.md and summary.json). "
        "The frozen result rows are unchanged and remain the reported results. No model calls.",
        "",
    ]
    for item in cfg["parser_sensitivity"]:
        sub = yaml.safe_load(Path(item["config"]).read_text())
        src = Path(sub["out"])
        dst = out / src.name
        if dst.exists():
            shutil.rmtree(dst)
        changes = corrected_copy(sub, dst)
        sub["out"] = str(dst)
        if "compare_to" in sub:
            sub["compare_to"] = str(out / Path(sub["compare_to"]).name)
        sub_cfg = dst / "config.yaml"
        sub_cfg.write_text(yaml.safe_dump(sub, sort_keys=False))
        report(str(sub_cfg))
        diff = numeric_diff(json.loads((src / "summary.json").read_text()), json.loads((dst / "summary.json").read_text()))
        summary["parser_sensitivity"][src.name] = {
            **changes,
            "summary_changes": [{"path": p, "frozen": a, "corrected": b} for p, a, b in diff],
        }
        lines += [
            f"### {src.name} (`{item['config']}`)",
            "",
            f"Answer rows re-read: {changes['answer_rows']}; parsed value changed: {len(changes['answers'])}.",
            "",
        ]
        lines += [
            "| Result file | Key | Saved answer | Corrected answer | Gold | Correct, saved | Correct, corrected |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in changes["answers"]:
            lines.append(
                f"| {c['file']} | {c['key'] or c['conv'] + '|' + str(c['turn'])} | {c['saved']} | {c['corrected']} | {c['gold']} | "
                f"{c['correct_saved']} | {c['correct_corrected']} |"
            )
        if not changes["answers"]:
            lines.append("| (none) | | | | | | |")
        lines += [
            "",
            f"Later free-stage turns whose history showed a misparsed value (not re-run; their stored replies are kept): "
            f"{changes['later_turns_with_misparsed_history']}.",
            "",
        ]
        if changes["breakers"]:
            lines += [
                "Natural breaker items whose checked answer changes:",
                "",
                "| Breaker | Conversation | Turn | Rescoring | Wrong | Tripped |",
                "|---|---|---|---|---|---|",
            ]
            for b in changes["breakers"]:
                lines.append(f"| {b['breaker']} | {b['conv']} | {b['turn']} | {b['how']} | {b['is_wrong']} | {b['tripped']} |")
            lines.append("")
        lines += [f"Numbers in summary.json that change: {len(diff)}.", ""]
        if diff:
            lines += ["| summary.json path | Frozen | Corrected | Change |", "|---|---|---|---|"]
            for p, a, b in diff:
                lines.append(f"| `{p}` | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
            lines.append("")
    return lines
