"""End-to-end run of every stage with a fake model (no network)."""

import json
from pathlib import Path

from cascade import analysis
from cascade.llm import Reply
from cascade.run import Settings, injection_plan, stage_breakers, stage_clean, stage_free, stage_inject


class FakeClient:
    """Answers with the gold value unless the history holds a wrong number, which it copies forward."""

    def __init__(self, convs):
        self.by_q = {t.question: (c, i) for c in convs for i, t in enumerate(c.turns)}
        self.calls = 0

    def chat(self, model, messages, max_tokens=0, temperature=0.0, seed=0):
        self.calls += 1
        last = messages[-1]["content"]
        if "Verdict:" in last:
            return Reply("Verdict: incorrect" if "25587" not in last else "Verdict: correct", model, 1, 1, False)
        if "Program:" in last:
            return Reply("Program: subtract(206588, 181001)", model, 1, 1, False)
        conv, k = self.by_q[last]
        gold = conv.turns[k].answer
        shown = [m["content"] for m in messages if m["role"] == "assistant"]
        wrong_shown = any(s != f"Answer: {gold}" and "Answer: 2" in s for s in shown) and k >= 2
        val = gold if not wrong_shown or isinstance(gold, str) else gold * 2
        return Reply(f"Answer: {val}", model, 10, 2, False)


def test_all_stages(convs, tmp_path: Path):
    convs = [c for c in convs if not c.issues]
    s = Settings(models=["fake/m"], error_types=["scale", "sign"], types_per_turn=2, seed=0, workers=2, out=tmp_path)
    client = FakeClient(convs)
    stage_clean(convs, client, s)
    stage_inject(convs, client, s)
    stage_free(convs, client, s)
    stage_breakers(convs, client, s, [{"name": "b", "kind": "self", "answerer": "fake/m", "checker": "fake/m"}])
    clean = analysis.read_jsonl(tmp_path / "clean" / "fake__m.jsonl")
    inject = analysis.read_jsonl(tmp_path / "inject" / "fake__m.jsonl")
    assert len(clean) == 7
    plan = injection_plan(convs, s)
    assert len(inject) == sum(len(next(c for c in convs if c.id == p["conv"]).turns) - p["j"] - 1 for p in plan)
    rows = analysis.cascade_rows(clean, inject)
    assert any(r["depends"] for r in rows) and all("paths" in r for r in rows)
    table = analysis.cascade_table(rows)
    assert any(t["type"] == "all" for t in table)
    scores = analysis.breaker_scores(analysis.read_jsonl(tmp_path / "breakers" / "b.jsonl"))
    assert {sc["metric"] for sc in scores} >= {"catch_rate", "false_alarm_rate"}
    # Resume: a second run makes no new calls.
    before = client.calls
    stage_clean(convs, client, s)
    stage_inject(convs, client, s)
    assert client.calls == before
    json.dumps(analysis.fit_transmission(rows))


def test_clean_repeat_and_noise_floor(convs, tmp_path: Path):
    """clean_repeat asks the clean questions again under repeat_seed; noise_floor counts flipped outcomes."""
    convs = [c for c in convs if not c.issues]
    s = Settings(models=["fake/m"], error_types=["scale"], types_per_turn=1, seed=0, workers=2, out=tmp_path)
    seeds = []

    class SeedClient(FakeClient):
        def chat(self, model, messages, max_tokens=0, temperature=0.0, seed=0):
            seeds.append(seed)
            return super().chat(model, messages, max_tokens, temperature, seed)

    client = SeedClient(convs)
    stage_clean(convs, client, s)
    stage_clean(convs, client, s, repeat=True)
    clean = analysis.read_jsonl(tmp_path / "clean" / "fake__m.jsonl")
    repeat = analysis.read_jsonl(tmp_path / "clean_repeat" / "fake__m.jsonl")
    assert sorted(set(seeds)) == [0, 1] and {r["seed"] for r in repeat} == {1}
    nf = analysis.noise_floor(clean, repeat)
    assert nf["n"] == len(clean) and nf["flip_rate"] == 0.0
    repeat[0]["correct"] = not repeat[0]["correct"]
    nf = analysis.noise_floor(clean, repeat)
    assert nf["flip_rate"] == 1 / len(clean) and nf["right_to_wrong"] + nf["wrong_to_right"] == 1


def test_cache_shared_by_two_processes(tmp_path: Path):
    """Two Cache objects on one file (as two terminals would have) can both write and read."""
    from cascade.llm import Cache

    a, b = Cache(tmp_path / "c.sqlite"), Cache(tmp_path / "c.sqlite")
    a.put("k1", {"x": 1}, {"choices": [1]})
    b.put("k2", {"x": 2}, {"choices": [2]})
    assert b.get("k1") == {"choices": [1]} and a.get("k2") == {"choices": [2]}


def test_bad_model_id_fails_fast(tmp_path: Path, monkeypatch):
    """A 4xx other than 429 (e.g. wrong model ID) raises at once instead of retrying silently."""
    import pytest
    import requests

    from cascade.llm import Cache, Client, FatalAPIError

    class R:
        status_code = 400
        text = '{"error": "not a valid model ID"}'

    calls = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1) or R())
    with pytest.raises(FatalAPIError, match="not a valid model ID"):
        Client(Cache(tmp_path / "c.sqlite")).chat("bad/model", [{"role": "user", "content": "hi"}])
    assert len(calls) == 1


def test_rate_limit_in_body_is_retried(tmp_path: Path, monkeypatch):
    """An upstream 429 sent as HTTP 200 with the error in the body is retried, not fatal (R43)."""
    import requests

    from cascade import llm
    from cascade.llm import Cache, Client

    bodies = [
        {"error": {"message": "Provider returned error", "code": 429, "metadata": {"error_type": "rate_limit_exceeded"}}},
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}},
    ]

    class R:
        status_code = 200
        headers: dict = {}

        def __init__(self):
            self.body = bodies.pop(0)

        def json(self):
            return self.body

    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(requests, "post", lambda *a, **k: R())
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    reply = Client(Cache(tmp_path / "c.sqlite")).chat("d/m", [{"role": "user", "content": "hi"}])
    assert reply.text == "ok" and not bodies


def test_request_extra_and_finish_reason(tmp_path: Path, monkeypatch):
    """Per-model request fields are sent (and so enter the cache key); finish_reason is kept."""
    import requests

    from cascade.llm import Cache, Client

    sent = []

    class R:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": ""}, "finish_reason": "length"}], "usage": {}}

    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(requests, "post", lambda *a, **k: sent.append(k["json"]) or R())
    extra = {"q/m": {"reasoning": {"max_tokens": 1024}}}
    reply = Client(Cache(tmp_path / "c.sqlite"), request_extra=extra).chat("q/m", [{"role": "user", "content": "hi"}])
    assert sent[0]["reasoning"] == {"max_tokens": 1024}
    assert reply.finish_reason == "length" and reply.text == ""
    Client(Cache(tmp_path / "c.sqlite")).chat("q/m", [{"role": "user", "content": "hi"}])
    assert len(sent) == 2  # without the extra field the request differs, so it is not a cache hit


def test_sink_refuses_rows_from_other_settings(tmp_path: Path):
    """Rows written under another prompt version or max_tokens must not be resumed as done."""
    import pytest

    from cascade.prompts import PROMPT_VERSION
    from cascade.run import Sink

    p = tmp_path / "clean" / "m.jsonl"
    p.parent.mkdir()
    for row in ({"key": "c|0", "prompt_version": "v0", "max_tokens": 8192}, {"key": "c|0", "prompt_version": PROMPT_VERSION}):
        p.write_text(json.dumps(row) + "\n")
        with pytest.raises(RuntimeError, match="move the old results folder aside"):
            Sink(p, 8192)
    p.write_text(json.dumps({"key": "c|0", "prompt_version": PROMPT_VERSION, "max_tokens": 8192}) + "\n")
    assert Sink(p, 8192).done == {"c|0"}


def test_rate_limit_retries_then_succeeds(tmp_path: Path, monkeypatch):
    """HTTP 429 is retried (honouring Retry-After) rather than failing the stage."""
    import requests

    from cascade import llm

    class R:
        def __init__(self, code):
            self.status_code, self.text, self.headers = code, "rate-limited upstream", {"Retry-After": "7"}

        def json(self):
            return {"choices": [{"message": {"content": "Answer: 1"}, "finish_reason": "stop"}], "usage": {}}

    codes, waits = [429, 429, 200], []
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(requests, "post", lambda *a, **k: R(codes.pop(0)))
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    reply = llm.Client(llm.Cache(tmp_path / "c.sqlite")).chat("q/m", [{"role": "user", "content": "hi"}])
    assert reply.text == "Answer: 1" and waits == [7.0, 7.0]


def test_cut_off_body_is_retried(tmp_path: Path, monkeypatch):
    """HTTP 200 whose body is only keep-alive blank lines (upstream cut off) is retried, not fatal."""
    import requests

    from cascade import llm

    class R:
        status_code, headers = 200, {}

        def __init__(self, ok):
            self.ok, self.text = ok, "ok" if ok else "\n" * 98

        def json(self):
            if not self.ok:
                raise requests.exceptions.JSONDecodeError("Expecting value", self.text, 98)
            return {"choices": [{"message": {"content": "Answer: 2"}, "finish_reason": "stop"}], "usage": {}}

    oks, waits = [False, True], []
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(requests, "post", lambda *a, **k: R(oks.pop(0)))
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    reply = llm.Client(llm.Cache(tmp_path / "c.sqlite")).chat("q/m", [{"role": "user", "content": "hi"}])
    assert reply.text == "Answer: 2" and len(waits) == 1
