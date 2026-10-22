"""Exploratory runs added after freezing (PREREGISTRATION.md change 2): guarded prompt, exploratory breakers, report."""

import argparse
import hashlib
import json
from pathlib import Path

import pytest
from test_run_offline import FakeClient

from cascade import cli, exploratory
from cascade.prompts import GUARD_PARAGRAPH, PROMPT_VERSION, SYSTEM_ANSWER, SYSTEM_PROMPTS, answer_messages
from cascade.run import Settings, Sink, stage_clean

# SHA-256 of the v2 system prompt before v2-guard was added (14 Oct 2026); cached v2 replies depend on it.
V2_SHA256 = "1c5f8608dea183c63a4e3a257175609dc264abe3d2e48489af59a9eea050258b"
GUARD_TEXT = (
    "Earlier answers in this conversation may contain mistakes. When the latest question needs an earlier value, "
    "recompute it from the report instead of reusing the earlier answer. Decide units and number format from the "
    "latest question alone, not from the format of earlier answers."
)


def test_v2_prompt_unchanged(convs):
    assert PROMPT_VERSION == "v2" and SYSTEM_PROMPTS["v2"] is SYSTEM_ANSWER
    assert hashlib.sha256(SYSTEM_ANSWER.encode()).hexdigest() == V2_SHA256
    c = convs[0]
    assert answer_messages(c, 1, [c.turns[0].answer]) == answer_messages(c, 1, [c.turns[0].answer], "v2")


def test_guard_prompt(convs):
    guard = SYSTEM_PROMPTS["v2-guard"]
    assert GUARD_PARAGRAPH == GUARD_TEXT and guard != SYSTEM_ANSWER
    # After the number-format rules, before the report; otherwise identical to v2.
    assert guard == SYSTEM_ANSWER.replace("\n\nREPORT\n{context}", "\n\n" + GUARD_TEXT + "\n\nREPORT\n{context}")
    assert guard.index("unless the question asks for another unit.") < guard.index(GUARD_TEXT) < guard.index("REPORT\n")
    c = convs[0]
    msgs = answer_messages(c, 1, [c.turns[0].answer], "v2-guard")
    assert GUARD_TEXT in msgs[0]["content"] and msgs[1:] == answer_messages(c, 1, [c.turns[0].answer])[1:]


def test_sink_refuses_mixing_prompt_versions(tmp_path: Path):
    p = tmp_path / "clean" / "m.jsonl"
    sink = Sink(p, 8192)
    sink.write({"key": "c|0"})
    sink.close()
    assert json.loads(p.read_text())["prompt_version"] == "v2"
    with pytest.raises(RuntimeError, match="move the old results folder aside"):
        Sink(p, 8192, "v2-guard")
    assert Sink(p, 8192, "v2").done == {"c|0"}


def test_stage_clean_guard_stamps_rows(convs, tmp_path: Path):
    convs = [c for c in convs if not c.issues]
    systems = []

    class Recorder(FakeClient):
        def chat(self, model, messages, max_tokens=0, temperature=0.0, seed=0):
            systems.append(messages[0]["content"])
            return super().chat(model, messages, max_tokens, temperature, seed)

    s = Settings(models=["fake/m"], error_types=["scale"], types_per_turn=1, seed=0, workers=2, out=tmp_path, prompt="v2-guard")
    stage_clean(convs, Recorder(convs), s)
    rows = [json.loads(x) for x in (tmp_path / "clean" / "fake__m.jsonl").read_text().splitlines()]
    assert rows and {r["prompt_version"] for r in rows} == {"v2-guard"}
    assert systems and all(GUARD_TEXT in sp for sp in systems)


def test_unknown_prompt_rejected():
    with pytest.raises(SystemExit, match="unknown prompt"):
        cli._prompt({"prompt": "v3"})
    assert cli._prompt({}) == "v2"


def test_breakers_stage_runs_exploratory_breakers(convs, tmp_path: Path, monkeypatch):
    """`cascade run breakers` runs `breakers` + `exploratory_breakers` through cmd_run."""
    convs = [c for c in convs if not c.issues]
    cfg = {
        "split": "dev",
        "min_turns": 1,
        "seed": 0,
        "workers": 2,
        "out": str(tmp_path / "out"),
        "cache": str(tmp_path / "c.sqlite"),
        "models": ["fake/m"],
        "error_types": ["scale", "sign"],
        "breakers": [{"name": "pre", "kind": "self", "answerer": "fake/m", "checker": "fake/m"}],
        "exploratory_breakers": [{"name": "expl", "kind": "program", "answerer": "fake/m", "checker": "fake/m"}],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(json.dumps(cfg))  # JSON is valid YAML
    monkeypatch.setattr(cli, "load", lambda _: convs)
    monkeypatch.setattr(cli, "Client", lambda *a, **k: FakeClient(convs))
    cli.cmd_run(argparse.Namespace(config=str(path), stage="breakers", models=None, offline=True))
    for name in ("pre", "expl"):
        rows = [json.loads(x) for x in (tmp_path / "out" / "breakers" / f"{name}.jsonl").read_text().splitlines()]
        assert rows and {r["prompt_version"] for r in rows} == {"v2"}
    summary: dict = {}
    lines = exploratory.exploratory_breakers_section(cfg, Path(cfg["out"]), summary)
    assert lines[0] == "## Exploratory breakers (added after freezing; PREREGISTRATION change 2)"
    assert list(summary["exploratory_breakers"]) == ["expl"]
    assert "—" not in "\n".join(lines)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_guard_comparison_paired_differences(tmp_path: Path):
    g, b = tmp_path / "guard", tmp_path / "main"
    m = "fake/m"

    def clean(conv, turn, ok):
        return {"conv": conv, "turn": turn, "correct": ok, "key": f"{conv}|{turn}"}

    def inj(conv, j, turn, dep, ok, kind="scale"):
        return {"conv": conv, "turn": turn, "j": j, "depends": dep, "correct": ok, "key": f"{conv}|{j}|{kind}|{turn}"}

    # Clean: B|1 goes from wrong to right (+1 of 4 paired); C|0 exists only in the guarded run (unpaired).
    _write(b / "clean" / "fake__m.jsonl", [clean("A", 0, True), clean("A", 1, True), clean("B", 0, True), clean("B", 1, False)])
    guard_clean = [clean("A", 0, True), clean("A", 1, True), clean("B", 0, True), clean("B", 1, True), clean("C", 0, True)]
    _write(g / "clean" / "fake__m.jsonl", guard_clean)
    # Dependent: A|0->1 extra error 1 (v2) vs 0 (guard): -1; B|0->1 v2 clean wrong, still wrong: 0; guard clean right,
    # now wrong: 1; diff +1. Mean of (-1, +1) = 0 over 2 rows; v2 mean 0.5, guard mean 0.5.
    # A second dependent row in A, A|0->2 (guard only): unpaired.
    _write(b / "inject" / "fake__m.jsonl", [inj("A", 0, 1, True, False), inj("B", 0, 1, True, False), inj("A", 0, 1, False, True, "sign")])
    _write(
        g / "inject" / "fake__m.jsonl",
        [inj("A", 0, 1, True, True), inj("B", 0, 1, True, False), inj("A", 0, 2, True, False), inj("A", 0, 1, False, False, "sign")],
    )
    # Free: guard loses A|1.
    _write(b / "free" / "fake__m.jsonl", [clean("A", 0, True), clean("A", 1, True)])
    _write(g / "free" / "fake__m.jsonl", [clean("A", 0, True), clean("A", 1, False)])
    res = exploratory.guard_comparison(g, b, m)
    ca = res["clean_accuracy"]
    assert ca["n_paired"] == 4 and ca["n_guarded"] == 5 and ca["n_v2"] == 4
    assert ca["diff"] == pytest.approx(0.25) and ca["guarded"] == 1.0 and ca["v2"] == 0.75
    dep = res["dependent_extra_error"]
    assert dep["n_paired"] == 2 and dep["diff"] == pytest.approx(0.0) and dep["guarded"] == 0.5 and dep["v2"] == 0.5
    ind = res["independent_extra_error"]  # v2: right, clean right -> 0; guard: wrong, clean right -> 1
    assert ind["n_paired"] == 1 and ind["diff"] == pytest.approx(1.0)
    fr = res["free_accuracy"]
    assert fr["n_paired"] == 2 and fr["diff"] == pytest.approx(-0.5) and fr["n_conversations"] == 1

    summary: dict = {}
    cfg = {"models": [m, "other/missing"], "compare_to": str(b), "prompt": "v2-guard"}
    lines = exploratory.guard_comparison_section(cfg, g, summary)
    text = "\n".join(lines)
    assert lines[0] == "## Guarded prompt minus v2 (exploratory, added after freezing)"
    assert list(summary["guard_comparison"]) == [m] and "—" not in text
    assert "| Clean accuracy | 1.000 | 0.750 | +0.250 |" in text
    assert exploratory.guard_comparison_section({"models": [m]}, g, {}) == []


def test_guard_comparison_skips_missing_stages(tmp_path: Path):
    g, b = tmp_path / "guard", tmp_path / "main"
    _write(b / "clean" / "fake__m.jsonl", [{"conv": "A", "turn": 0, "correct": True}])
    _write(g / "clean" / "fake__m.jsonl", [{"conv": "A", "turn": 0, "correct": False}])
    res = exploratory.guard_comparison(g, b, "fake/m")
    assert set(res) == {"clean_accuracy"} and res["clean_accuracy"]["diff"] == -1.0
