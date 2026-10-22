"""Post-hoc analysis of the change 4 final evaluation: column label "-" accepted (CHANGELOG R45).

Not pre-registered: added 18 Oct 2026 after the final results (CHANGELOG R44) and makes no call. The frozen
prompt asks for the cell in the table's first row above the number as the column label, and "-" only when that
cell is empty. Many ConvFinQA tables have no header row (their first row is data), and the models then wrote
"-". This analysis re-checks the stored program_v2 replies with one rule relaxed:

- a table citation whose column label is "-" (empty after normalising) is valid when its row label matches a row
  and the cited number matches a cell of that row (any column).

Everything else is as frozen: same replies (no new calls), parser, other checks, verdicts and analysis. Rows
truncated at max_tokens keep their verdict (the finish reason is not stored, and no program was accepted). The
frozen code is not changed: the rule is swapped in only while the relaxed re-check runs, and the stored verdicts
are first reproduced with the frozen rules as a check.
"""

from __future__ import annotations

import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from . import analysis, program_v2
from .breaker_posthoc import group
from .change4 import _dci, paired, paired_diff
from .data import Conversation

HEADING = '### Post-hoc: column label "-" accepted (not pre-registered; no call)'
GROUPS = ("inject:wrong", "gold:right", "free:wrong", "free:right")


@contextmanager
def dash_column():
    """Accept "-" as the column label when the row matches and the number is in that row."""
    frozen = program_v2._check_table

    def relaxed(body: str, number: float, table: list[list[str]]) -> str | None:
        reason = frozen(body, number, table)
        parts = [p.strip() for p in body.split("|")]
        if reason not in ("cell_not_found", "cell_value_mismatch") or len(parts) < 2 or program_v2.norm(parts[-1]):
            return reason
        row_l = program_v2.norm(" | ".join(parts[:-1]))
        cells = [c for row in table if row and program_v2.norm(row[0]) == row_l for c in row[1:]]
        if not cells:
            return reason
        return None if any(program_v2._has(program_v2.numbers_in(c), number) for c in cells) else "cell_value_mismatch"

    program_v2._check_table = relaxed
    try:
        yield
    finally:
        program_v2._check_table = frozen


def rescore(r: dict, conv: Conversation) -> dict:
    """The stored reply scored again under the current rules; truncated rows are kept as stored."""
    if r["reason"] == "truncated":
        return {"outcome": r["outcome"], "reason": r["reason"], "tripped": r["tripped"], "usable": r["usable"], "value": r["value"]}
    c = program_v2.score(r["raw"], None, conv, r["turn"], r["answer"], r["model"])
    return {"outcome": c.outcome, "reason": c.reason, "tripped": c.tripped, "usable": c.usable, "value": c.value}


def source_span(quote: str, source: str) -> str | None:
    """The stretch of ``source`` that the frozen quote check matched: the quote's normalised text (``program_v2.norm``)
    found in the source with any punctuation allowed between characters, so signs, parentheses, $ and % around the
    matched words are kept. None when the quote is not in the source."""
    q = program_v2.norm(quote.strip().strip(program_v2._QUOTES))
    if not q:
        return None
    pattern = r"[^\w\s]*" + "".join(r"[^\w]*\s+[^\w]*" if c == " " else re.escape(c) + r"[^\w\s]*" for c in q)
    m = re.search(pattern, source, flags=re.IGNORECASE)
    return m.group(0) if m else None


def sign_check(r: dict, conv: Conversation) -> list[dict]:
    """Accepted text and question citations of one stored program_v2 row whose cited number is not in the matched
    source span as written there (a sign or a digit the quote changed). Post-hoc (CHANGELOG R53)."""
    text = " ".join(conv.pre_text) + " " + " ".join(conv.post_text)
    bad = []
    for c in r.get("cites", []):
        if not c.get("ok") or c["source"] not in ("text", "question"):
            continue
        source = text if c["source"] == "text" else conv.turns[r["turn"]].question
        span = source_span(c["body"], source)
        if span is None or not program_v2._has(program_v2.numbers_in(span), c["number"]):
            bad.append({"key": r["key"], "source": c["source"], "body": c["body"], "number": c["number"], "span": span})
    return bad


def _same(a: dict, r: dict) -> bool:
    return a["outcome"] == r["outcome"] and a["reason"] == r["reason"]


def change4_posthoc_section(cfg: dict, out: Path, summary: dict, convs: dict[str, Conversation]) -> list[str]:
    found = []
    for p in (cfg.get("change4") or {}).get("pairs", []):
        fp, rp = Path(out) / "breakers" / f"{p['frozen']}.jsonl", Path(out) / "breakers" / f"{p['revised']}.jsonl"
        if fp.exists() and rp.exists():
            found.append((p, analysis.read_jsonl(fp), analysis.read_jsonl(rp)))
    if not found:
        return []
    lines = [
        HEADING,
        "",
        "Added after the final results; not pre-registered and not part of change 4's analysis. The stored program_v2 "
        'replies are checked again with one rule relaxed: a table citation with column label "-" is valid when its row '
        'label matches a row and the cited number is in that row (the frozen prompt allows "-" only for an empty '
        "first-row cell, and many ConvFinQA tables have no header row). No new model calls; everything else as frozen. "
        "Truncated rows keep their verdict.",
        "",
        "| Answerer | Rows | Stored verdicts reproduced (frozen rules) | Rows changed | Unable to agree | Unable to disagree "
        "| Unable, other reason |",
        "|---|---|---|---|---|---|---|",
    ]
    store: dict = {}
    rescored: dict[str, dict[str, dict]] = {}
    for p, _, revised in found:
        same = sum(_same(rescore(r, convs[r["conv"]]), r) for r in revised)
        with dash_column():
            new = {r["key"]: rescore(r, convs[r["conv"]]) for r in revised}
        rescored[p["revised"]] = new
        moves = Counter((r["outcome"], new[r["key"]]["outcome"]) for r in revised if not _same(new[r["key"]], r))
        changed = sum(moves.values())
        store[p["revised"]] = {"rows": len(revised), "reproduced": same, "changed": changed}
        store[p["revised"]]["moves"] = {f"{a}->{b}": n for (a, b), n in moves.items()}
        lines.append(
            f"| {p['answerer']} | {len(revised)} | {same} | {changed} | {moves[('unable', 'agree')]} | "
            f"{moves[('unable', 'disagree')]} | {moves[('unable', 'unable')]} |"
        )

    lines += [
        "",
        "Trip and unable rates by group, as run and post-hoc:",
        "",
        "| Answerer | Group | n | Trip rate, as run | Trip rate, post-hoc | Unable, as run | Unable, post-hoc |",
        "|---|---|---|---|---|---|---|",
    ]
    for p, _, revised in found:
        new = rescored[p["revised"]]
        for g in GROUPS:
            rows = [r for r in revised if group(r) == g]
            if not rows:
                continue
            n = len(rows)
            rate = {
                "trip": sum(r["tripped"] for r in rows) / n,
                "trip_post": sum(new[r["key"]]["tripped"] for r in rows) / n,
                "unable": sum(not r["usable"] for r in rows) / n,
                "unable_post": sum(not new[r["key"]]["usable"] for r in rows) / n,
            }
            store[p["revised"]].setdefault("groups", {})[g] = {"n": n, **rate}
            lines.append(
                f"| {p['answerer']} | {g} | {n} | {rate['trip']:.3f} | {rate['trip_post']:.3f} | "
                f"{rate['unable']:.3f} | {rate['unable_post']:.3f} |"
            )

    lines += [
        "",
        "Primary pair, program_v2 minus the frozen breaker (paired, conversation-level cluster bootstrap as in change 4):",
        "",
        "| Answerer | Outcome | Comparator | As run | Post-hoc |",
        "|---|---|---|---|---|",
    ]
    for p, frozen, revised in found:
        new = rescored[p["revised"]]
        pairs = paired(frozen, revised)
        for wrong, name in ((True, "natural recall"), (False, "natural false alarms")):
            for comp, label in (("strict", "strict (primary)"), ("lenient", "as run (secondary)")):
                before = paired_diff(pairs, wrong, comp)
                after = paired_diff(pairs, wrong, comp, revised_trip=lambda r, new=new: bool(new[r["key"]]["tripped"]))
                store[p["revised"]].setdefault("diffs", {})[f"{name}|{comp}"] = after
                lines.append(f"| {p['answerer']} | {name} | {label} | {_dci(before)} | {_dci(after)} |")
    lines += [
        "",
        "### Post-hoc: sign-preserving check of quoted numbers (added 20 Oct 2026; not pre-registered; no call)",
        "",
        "The frozen quote check compares the quote with the source after removing punctuation, and then looks for the "
        "cited number in the quote, so a quote that changed 45 to -45 would pass. This re-check finds the source span the "
        "quote matched (punctuation kept) and requires the cited number, with its sign, in that span. It covers every "
        "text and question citation the frozen check accepted; a citation that fails it would have made its check "
        "unusable. CHANGELOG R53.",
        "",
        "| Answerer | Accepted text and question citations | Cited number not in the matched source span |",
        "|---|---|---|",
    ]
    for p, _, revised in found:
        conv_rows = [(r, convs[r["conv"]]) for r in revised]
        accepted = sum(1 for r, _ in conv_rows for c in r.get("cites", []) if c.get("ok") and c["source"] in ("text", "question"))
        bad = [b for r, conv in conv_rows for b in sign_check(r, conv)]
        store[p["revised"]]["sign_check"] = {"accepted": accepted, "failing": len(bad), "examples": bad[:20]}
        lines.append(f"| {p['answerer']} | {accepted} | {len(bad)} |")
    lines.append("")
    summary["change4_posthoc"] = store
    return lines
