"""Command line: ``uv run cascade <command>``.

fetch          download ConvFinQA (data.zip) into data/raw/convfinqa
check-data     load the splits, verify every gold program, print dependency-graph statistics
audit-graph    write a CSV of 50 dev conversations for hand-checking the dependency graph
plan           print how many model calls each stage needs (no API calls)
run STAGE      run clean | clean_repeat | inject | free | breakers  (needs OPENROUTER_API_KEY)
report         write results/TABLES.md and results/summary.json from the run files
spend          save the account's OpenRouter spend for the study's models (needs OPENROUTER_MANAGEMENT_KEY)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import statistics
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

from . import analysis, breaker_posthoc, parser_sensitivity, posthoc, posthoc_spread, spend
from .change4 import change4_section
from .change4_posthoc import change4_posthoc_section
from .data import Conversation, load
from .deps import build
from .exploratory import exploratory_breakers_section, guard_comparison_section
from .llm import Cache, Client
from .prompts import PROGRAM_V2_PROMPT, PROGRAM_V2_SHA256, PROGRAM_V2_VERSION, SYSTEM_PROMPTS
from .run import Settings, _slug, injection_plan, stage_breakers, stage_clean, stage_free, stage_inject

AUDIT_CSV = Path("results/audit/graph_audit.csv")
# How the audit labels were made (CHANGELOG R24, R25); printed with the scores so they are never read as a pure hand audit.
AUDIT_PROVENANCE = (
    "Labels: 47 of 50 conversations drafted by GPT 6.1 Sol (high reasoning) and reviewed by the author; "
    "3 conversations (11 non-first turns) labelled by the author alone, 4 of those turns then corrected to the "
    "guide's rules. Not an independent hand audit: the final labels equal the GPT draft on every turn."
)
# Second, human labeller (PREREGISTRATION change 3, CHANGELOG R33, R34). They saw the author's labels for the first
# three conversations of the sheet as worked examples; the headline uses all 50, a sensitivity row leaves these 3 out.
SECOND_AUDIT_CSV = Path("results/audit/graph_audit_human_audited.csv")
SECOND_AUDIT_EXCLUDE = {"Single_HII/2017/page_124.pdf-1", "Single_GS/2017/page_143.pdf-1", "Single_LKQ/2016/page_26.pdf-2"}
SECOND_AUDIT_PROVENANCE = (
    "Labelled by hand by a second person, after the main run, following `results/audit/GUIDE.md` in `labeler.html`: "
    "no AI assistance, no access to the GPT draft or the primary labels; background outside finance. All 50 conversations "
    "are scored. For 3 of them the sheet holds the author's own labels, shown to the labeller as worked examples (also "
    "made without the GPT draft; 4 of their turns differ from the primary audit, CHANGELOG R24); the 47-conversation "
    "rows leave those 3 out."
)
# The unreviewed GPT draft (CHANGELOG R23, R24).
LLM_AUDIT_CSV = Path("results/audit/graph_audit_gpt-6.1-sol-high.csv")
LLM_AUDIT_NAME = "GPT 6.1 Sol (high reasoning)"


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


DATA_URL = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"
RAW = Path("data/raw/convfinqa")


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _select(convs: list[Conversation], cfg: dict) -> list[Conversation]:
    """Conversations used: clean gold programs, at least min_turns turns, optional fixed-seed sample."""
    keep = [c for c in convs if not c.issues and len(c.turns) >= cfg.get("min_turns", 3)]
    if cfg.get("ids_file"):  # a locked sample (PREREGISTRATION change 4): exactly these ids, file checked by hash
        text = Path(cfg["ids_file"]).read_text()
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest != cfg.get("ids_sha256"):
            sys.exit(f"{cfg['ids_file']} has SHA-256 {digest}, not the locked {cfg.get('ids_sha256')}")
        ids = set(text.split())
        keep = [c for c in keep if c.id in ids]
        if len(keep) != len(ids):
            sys.exit(f"{len(ids) - len(keep)} ids in {cfg['ids_file']} are not eligible conversations")
        return sorted(keep, key=lambda c: c.id)
    n = cfg.get("sample")
    if n and n < len(keep):
        keep = random.Random(cfg.get("seed", 0)).sample(keep, n)
        keep.sort(key=lambda c: c.id)
    return keep


def cmd_fetch(_: argparse.Namespace) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"downloading {DATA_URL}")
    blob = urllib.request.urlopen(DATA_URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            if name.endswith(".json"):
                target = RAW / Path(name).name
                target.write_bytes(z.read(name))
                print(f"  wrote {target}")


def cmd_check(_: argparse.Namespace) -> None:
    for split in ("train", "dev"):
        path = RAW / f"{split}.json"
        if not path.exists():
            print(f"{path} missing: run `cascade fetch`")
            continue
        convs = load(path)
        bad = [c for c in convs if c.issues]
        lens = [len(c.turns) for c in convs]
        graphs = [build(c) for c in convs if not c.issues]
        edges = [len(g.direct) for g in graphs]
        chained = sum(1 for g in graphs if any(g.descendants(j) for j in range(g.n)))
        print(f"{split}: {len(convs)} conversations, turns mean {statistics.mean(lens):.2f} (max {max(lens)})")
        print(f"  gold-program issues: {len(bad)} conversations ({len(bad) / len(convs):.1%}); excluded from the study")
        print(f"  dependency edges per conversation: mean {statistics.mean(edges):.2f}; with any chain: {chained}/{len(graphs)}")
        for c in bad[:5]:
            print(f"    e.g. {c.id}: {c.issues[0]}")


def cmd_audit(_: argparse.Namespace) -> None:
    convs = [c for c in load(RAW / "dev.json") if not c.issues and len(c.turns) >= 3]
    sample = random.Random(7).sample(convs, min(50, len(convs)))
    out = AUDIT_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        hand = (analysis.AUDIT_VALUE, analysis.AUDIT_REF, "parents_true (fill in)", "note")  # incl. the old sheet's column
        filled = [r for r in _read_csv(out) if any((r.get(c) or "").strip() for c in hand)]
        if filled:
            sys.exit(f"{out} already has {len(filled)} hand-filled rows; not overwriting")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conv", "turn", "question", "program", "gold", "parents_found", analysis.AUDIT_VALUE, analysis.AUDIT_REF, "note"])
        for c in sample:
            g = build(c)
            for k, t in enumerate(c.turns):
                w.writerow([c.id, k, t.question, t.program, t.answer, " ".join(map(str, g.parents(k))), "", "", ""])
    print(f"wrote {out}: fill both columns for every turn ('-' for none); see results/audit/GUIDE.md")


def cmd_plan(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    convs = _select(load(RAW / f"{cfg['split']}.json"), cfg)
    s = _settings(cfg)
    turns = sum(len(c.turns) for c in convs)
    plan = injection_plan(convs, s)
    by_id = {c.id: c for c in convs}
    inject_calls = sum(len(by_id[p["conv"]].turns) - p["j"] - 1 for p in plan)
    n_models = len(cfg["models"])
    print(f"conversations: {len(convs)}, turns: {turns}, injections: {len(plan)}")
    print(f"calls per model: clean {turns}, clean_repeat {turns}, free {turns}, inject {inject_calls}")
    print(f"calls for {n_models} models: {n_models * (3 * turns + inject_calls)}")
    n_b, n_x = len(cfg.get("breakers") or []), len(cfg.get("exploratory_breakers") or [])
    print(f"answer prompt: {s.prompt}")
    print(f"breaker calls: {n_b} breakers + {n_x} exploratory breakers = {n_b + n_x} x about {len(plan) + turns * 2} checks")


def _settings(cfg: dict) -> Settings:
    return Settings(
        models=cfg["models"],
        error_types=cfg["error_types"],
        types_per_turn=cfg.get("types_per_turn", 2),
        seed=cfg.get("seed", 0),
        workers=cfg.get("workers", 8),
        out=Path(cfg.get("out", "results")),
        max_tokens=cfg.get("max_tokens", 8192),
        repeat_seed=cfg.get("repeat_seed", 1),
        prompt=_prompt(cfg),
    )


def _prompt(cfg: dict) -> str:
    prompt = cfg.get("prompt", "v2")
    if prompt not in SYSTEM_PROMPTS:
        sys.exit(f"unknown prompt {prompt!r}; known: {', '.join(SYSTEM_PROMPTS)}")
    return prompt


def _all_breakers(cfg: dict) -> list[dict]:
    """Pre-registered breakers, then the exploratory ones added after freezing (PREREGISTRATION change 2)."""
    return list(cfg.get("breakers") or []) + list(cfg.get("exploratory_breakers") or [])


def cmd_run(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    convs = _select(load(RAW / f"{cfg['split']}.json"), cfg)
    s = _settings(cfg)
    if args.models:
        s.models = args.models
    client = Client(
        Cache(cfg.get("cache", "cache/replies.sqlite")),
        providers=cfg.get("providers") or {},
        offline=args.offline,
        request_extra=cfg.get("request_extra") or {},
    )
    frozen = cfg.get("program_v2_version")  # PREREGISTRATION change 4 Part B: the final run uses the frozen prompt only
    if frozen and (frozen != PROGRAM_V2_VERSION or hashlib.sha256(PROGRAM_V2_PROMPT.encode()).hexdigest() != PROGRAM_V2_SHA256):
        sys.exit(f"program_v2 prompt is {PROGRAM_V2_VERSION}, not the frozen {frozen} (or its text changed)")
    s.out.mkdir(parents=True, exist_ok=True)
    (s.out / "manifest.json").write_text(json.dumps({"config": cfg, "n_conversations": len(convs), "ids": [c.id for c in convs]}, indent=1))
    if args.stage == "clean":
        stage_clean(convs, client, s)
    elif args.stage == "clean_repeat":
        stage_clean(convs, client, s, repeat=True)
    elif args.stage == "inject":
        stage_inject(convs, client, s)
    elif args.stage == "free":
        stage_free(convs, client, s)
    elif args.stage == "breakers":
        stage_breakers(convs, client, s, _all_breakers(cfg))
    else:
        sys.exit(f"unknown stage {args.stage}")


def cmd_report(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    out = Path(cfg.get("out", "results"))
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"models": {}, "breakers": {}}
    lines = ["# Results tables", "", "Generated by `cascade report`; do not edit by hand.", ""]
    if "parser_sensitivity" in cfg:  # corrected-parser copies of other configs' rows, each reported (CHANGELOG R53)
        lines += parser_sensitivity.parser_sensitivity_section(cfg, out, summary, lambda c: cmd_report(argparse.Namespace(config=c)))
        (out / "TABLES.md").write_text("\n".join(lines))
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"wrote {out / 'TABLES.md'} and {out / 'summary.json'}")
        return
    if "spend" in cfg:  # spend config: account activity only, no run files (CHANGELOG R47)
        lines += spend.spend_section(cfg, out, summary)
        (out / "TABLES.md").write_text("\n".join(lines))
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"wrote {out / 'TABLES.md'} and {out / 'summary.json'}")
        return
    for model in cfg["models"]:
        cp, ip = out / "clean" / f"{_slug(model)}.jsonl", out / "inject" / f"{_slug(model)}.jsonl"
        if not (cp.exists() and ip.exists()):
            continue
        clean, inject = analysis.read_jsonl(cp), analysis.read_jsonl(ip)
        rows = analysis.cascade_rows(clean, inject)
        acc = sum(r["correct"] for r in clean) / len(clean)
        table = analysis.cascade_table(rows)
        fit = analysis.fit_transmission(rows)
        copied = sum(r["copied_wrong"] for r in rows if r["depends"]) / max(1, sum(r["depends"] for r in rows))
        stage_rows = {"clean": clean, "inject": inject}
        for st in ("clean_repeat", "free"):
            sp = out / st / f"{_slug(model)}.jsonl"
            if sp.exists():
                stage_rows[st] = analysis.read_jsonl(sp)
        health = {st: analysis.reply_health(rs) for st, rs in stage_rows.items()}
        noise = analysis.noise_floor(clean, stage_rows["clean_repeat"]) if "clean_repeat" in stage_rows else None
        mech = analysis.mechanism_table(rows)
        summary["models"][model] = {
            "clean_accuracy": acc,
            "cascade": table,
            "transmission_fit": fit,
            "copied_wrong_rate_dependent": copied,
            "reply_health": health,
            "noise_floor": noise,
            "mechanisms": mech,
        }
        lines += [f"## {model}", "", f"Clean-history accuracy: {acc:.3f} ({len(clean)} turns)", ""]
        lines += ["| Stage | Replies | No parseable answer | Truncated (max_tokens) |", "|---|---|---|---|"]
        for st, h in health.items():
            lines.append(f"| {st} | {h['n']} | {h['unparsed']} | {h['truncated']} |")
        lines.append("")
        if noise:
            lines += [
                f"Run-to-run noise floor (clean asked twice, n = {noise['n']} turns): flip rate {noise['flip_rate']:.3f} "
                f"[{noise['ci_low']:.3f}, {noise['ci_high']:.3f}]; right to wrong {noise['right_to_wrong']}, "
                f"wrong to right {noise['wrong_to_right']}; accuracy {noise['accuracy_clean']:.3f} then {noise['accuracy_repeat']:.3f}",
                "",
            ]
        lines += ["| Error type | Later turns | Extra error | 95% CI | n |", "|---|---|---|---|---|"]
        for t in table:
            lines.append(f"| {t['type']} | {t['turns']} | {t['extra_error']:+.3f} | [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] | {t['n']} |")
        lines += [
            "",
            "Why turns became wrong after the injection (right with the clean history; exploratory, rules in PREREGISTRATION.md):",
            "",
            "| Later turns | Newly wrong | " + " | ".join(analysis.MECHANISMS) + " |",
            "|---|---|" + "---|" * len(analysis.MECHANISMS),
        ]
        for m in mech:
            lines.append(f"| {m['turns']} | {m['newly_wrong']} | " + " | ".join(str(m[k]) for k in analysis.MECHANISMS) + " |")
        if "a_transmission" in fit:
            lines += [
                "",
                f"Transmission model: a = {fit['a_transmission']:.3f}, s = {fit['s_spillover']:.3f}; "
                f"held-out Brier {fit['heldout_brier']:.4f} (n = {fit['heldout_n']})",
                "",
                f"Graph-blind model (H2 comparison): c = {fit['graph_blind_c']:.3f}; held-out Brier "
                f"{fit['graph_blind_heldout_brier']:.4f}. Path minus graph-blind: {fit['brier_diff']:+.4f} "
                f"[{fit['brier_diff_ci_low']:+.4f}, {fit['brier_diff_ci_high']:+.4f}] (below 0 favours the graph)",
                "",
                "| Paths j→k | n | Predicted error | Observed error |",
                "|---|---|---|---|",
            ]
            for c in fit["calibration_by_paths"]:
                lines.append(f"| {c['paths']} | {c['n']} | {c['predicted']:.3f} | {c['observed']:.3f} |")
        lines.append("")
    for b in cfg.get("breakers", []):
        bp = out / "breakers" / f"{b['name']}.jsonl"
        if not bp.exists():
            continue
        scores = analysis.breaker_scores(analysis.read_jsonl(bp))
        summary["breakers"][b["name"]] = scores
        lines += [f"## Breaker: {b['name']} ({b['kind']}, checker {b['checker']})", ""]
        lines += ["| Group | Metric | Value | 95% CI | n | Unusable |", "|---|---|---|---|---|---|"]
        for sc in scores:
            ci = f"[{sc['ci_low']:.3f}, {sc['ci_high']:.3f}]"
            lines.append(f"| {sc['group']} | {sc['metric']} | {sc['value']:.3f} | {ci} | {sc['n']} | {sc['unusable_rate']:.3f} |")
        lines.append("")
    # Post-hoc analyses, added 14 Oct 2026 after the main results; not pre-registered (CHANGELOG R30).
    if "posthoc" in cfg:
        lines += posthoc.posthoc_section(cfg, out, summary)
    if "posthoc" in cfg:
        # Post-hoc breaker analyses, added 17 Oct 2026; not pre-registered (CHANGELOG R32).
        split_convs = load(RAW / f"{cfg['split']}.json")
        gold = {(c.id, k): t.answer for c in split_convs for k, t in enumerate(c.turns)}
        questions = {(c.id, k): t.question for c in split_convs for k, t in enumerate(c.turns)}
        lines += breaker_posthoc.breaker_posthoc_section(cfg, out, summary, gold, questions)
        # Post-hoc: own-history comparison, graph sensitivity on the audits, answer signatures (20 Oct 2026; R50).
        audits = {name: _read_csv(p) for name, p in (("primary audit", AUDIT_CSV), ("second labeller", SECOND_AUDIT_CSV)) if p.exists()}
        lines += posthoc_spread.posthoc_spread_section(cfg, out, summary, {c.id: c for c in split_convs}, audits)
    # PREREGISTRATION change 4 (CHANGELOG R37, R38): revised program breaker against the frozen one.
    if "change4" in cfg:
        split_convs = load(RAW / f"{cfg['split']}.json")
        gold4 = {(c.id, k): t.answer for c in split_convs for k, t in enumerate(c.turns)}
        lines += change4_section(cfg, out, summary, gold4)
        # Post-hoc, added 18 Oct 2026 after the final results; not pre-registered (CHANGELOG R45).
        lines += change4_posthoc_section(cfg, out, summary, {c.id: c for c in split_convs})
    # Added after freezing (PREREGISTRATION change 2, CHANGELOG R31): exploratory breakers and the guarded prompt.
    lines += exploratory_breakers_section(cfg, out, summary)
    lines += guard_comparison_section(cfg, out, summary)
    if AUDIT_CSV.exists() and "compare_to" not in cfg:
        rows = _read_csv(AUDIT_CSV)
        audit = analysis.graph_audit_scores(rows) if rows and analysis.AUDIT_VALUE in rows[0] else {"complete": False}
        summary["graph_audit"] = audit
        lines += [f"## Dependency graph audit (`{AUDIT_CSV}`)", "", AUDIT_PROVENANCE, ""]
        if audit["complete"]:
            lines += [
                f"{audit['conversations']} conversations, {audit['turns']} turns. Value-dependency edges: "
                f"{audit['edges_found']} found by the graph, {audit['edges_true']} in the audit labels.",
                "",
                f"Precision {audit['precision']:.3f} [{audit['precision_ci'][0]:.3f}, {audit['precision_ci'][1]:.3f}]; "
                f"recall {audit['recall']:.3f} [{audit['recall_ci'][0]:.3f}, {audit['recall_ci'][1]:.3f}].",
                "",
                f"Reference-only links (turn refers to an earlier turn without using its number): "
                f"{audit['reference_only_edges']}, of which the graph has {audit['reference_only_edges_found_by_graph']}.",
                "",
            ]
        else:
            lines += [f"Not complete: {audit.get('unfilled', 'all')} turns not yet filled in.", ""]
        if SECOND_AUDIT_CSV.exists() and audit["complete"]:
            second_rows = _read_csv(SECOND_AUDIT_CSV)
            if analysis.graph_audit_scores(second_rows)["complete"]:
                # Headline: all 50 conversations (author's decision, CHANGELOG R34). Sensitivity: without the 3
                # conversations whose labels the second labeller was shown as worked examples.
                scopes = {}
                for scope, excl in (("all 50", set()), ("47, worked examples left out", SECOND_AUDIT_EXCLUDE)):
                    scopes[scope] = {
                        "primary": analysis.graph_audit_scores([r for r in rows if r["conv"] not in excl]),
                        "second": analysis.graph_audit_scores([r for r in second_rows if r["conv"] not in excl]),
                        "agreement": analysis.label_agreement(rows, second_rows, exclude=excl),
                    }
                summary["graph_audit_second"] = scopes
                lines += [
                    f"### Second labeller, human, independent of the GPT draft (`{SECOND_AUDIT_CSV}`; added after freezing)",
                    "",
                    SECOND_AUDIT_PROVENANCE,
                    "",
                    "| Conversations | Labels | Value edges | Precision of the graph | Recall of the graph "
                    "| Reference-only links (graph has) |",
                    "|---|---|---|---|---|---|",
                ]
                for scope, d in scopes.items():
                    for name in ("primary", "second"):
                        sc = d[name]
                        lines.append(
                            f"| {scope} | {'Primary audit' if name == 'primary' else 'Second labeller'} | {sc['edges_true']} | "
                            f"{sc['precision']:.3f} [{sc['precision_ci'][0]:.3f}, {sc['precision_ci'][1]:.3f}] | "
                            f"{sc['recall']:.3f} [{sc['recall_ci'][0]:.3f}, {sc['recall_ci'][1]:.3f}] | "
                            f"{sc['reference_only_edges']} ({sc['reference_only_edges_found_by_graph']}) |"
                        )
                lines += [
                    "",
                    "| Conversations | Non-first turns | Column | Identical turns | Edge kappa | 95% CI | Both "
                    "| Primary only | Second only |",
                ]
                lines.append("|---|---|---|---|---|---|---|---|---|")
                for scope, d in scopes.items():
                    ag = d["agreement"]
                    for col in ("value", "reference"):
                        v = ag[col]
                        lines.append(
                            f"| {scope} | {ag['turns']} | {col} parents | {v['same_turns']} | {v['kappa']:.3f} | "
                            f"[{v['kappa_ci'][0]:.3f}, {v['kappa_ci'][1]:.3f}] | {v['both']} | {v['a_only']} | {v['b_only']} |"
                        )
                lines.append("")
        if LLM_AUDIT_CSV.exists():
            llm_rows = _read_csv(LLM_AUDIT_CSV)
            llm = analysis.graph_audit_scores(llm_rows)
            summary["graph_audit_llm"] = {"labeller": LLM_AUDIT_NAME, **llm}
            lines += [f"### Unreviewed GPT draft: {LLM_AUDIT_NAME} (`{LLM_AUDIT_CSV}`)", ""]
            if llm["complete"]:
                lines += [
                    f"Against the graph: precision {llm['precision']:.3f} [{llm['precision_ci'][0]:.3f}, "
                    f"{llm['precision_ci'][1]:.3f}]; recall {llm['recall']:.3f} [{llm['recall_ci'][0]:.3f}, "
                    f"{llm['recall_ci'][1]:.3f}]; {llm['edges_true']} value edges; reference-only links "
                    f"{llm['reference_only_edges']}.",
                    "",
                ]
            if audit["complete"] and llm["complete"]:
                # Not an independent agreement: most audit labels started from this draft (R24).
                changed = analysis.labels_changed(rows, llm_rows)
                summary["graph_audit_llm"]["turns_differing_from_audit"] = changed
                lines += [f"Turns where the audit labels differ from the draft: {len(changed)} of {len(rows)}.", ""]
    (out / "TABLES.md").write_text("\n".join(lines))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out / 'TABLES.md'} and {out / 'summary.json'}")


def cmd_spend(args: argparse.Namespace) -> None:
    spend.fetch(_load_config(args.config))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cascade")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)
    sub.add_parser("check-data").set_defaults(fn=cmd_check)
    sub.add_parser("audit-graph").set_defaults(fn=cmd_audit)
    for name, fn in (("plan", cmd_plan), ("report", cmd_report), ("spend", cmd_spend)):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default="configs/main.yaml")
        sp.set_defaults(fn=fn)
    r = sub.add_parser("run")
    r.add_argument("stage", choices=["clean", "clean_repeat", "inject", "free", "breakers"])
    r.add_argument("--config", default="configs/main.yaml")
    r.add_argument("--models", nargs="*")
    r.add_argument("--offline", action="store_true", help="use cached replies only")
    r.set_defaults(fn=cmd_run)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
