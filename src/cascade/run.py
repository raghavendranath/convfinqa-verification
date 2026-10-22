"""Experiment stages. Each writes one JSON line per model call to ``results/<stage>/<model>.jsonl``.

- ``clean``: every turn k >= 1 answered with the gold history (the no-error reference).
- ``inject``: for each turn j, one wrong answer per sampled error type replaces the gold answer at j;
  every later turn k > j is answered with that history. Comparing with ``clean`` gives the cascade.
- ``free``: the model's own answers form the history (the natural cascade, no injection).
- ``breakers``: each breaker checks (a) the injected wrong answers, (b) the gold answers (false alarms)
  and (c) the model's own wrong and right answers from ``free`` (natural errors).

Stages are resumable: completed rows are skipped by key.
"""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .breakers import check
from .data import Conversation
from .deps import build
from .inject import corrupt
from .llm import Client
from .match import extract_answer, match
from .prompts import PROGRAM_V2_VERSION, PROMPT_VERSION, answer_messages


@dataclass
class Settings:
    models: list[str]
    error_types: list[str]
    types_per_turn: int
    seed: int
    workers: int
    out: Path
    max_tokens: int = 8192
    repeat_seed: int = 1  # seed for ``clean_repeat``, the run-to-run noise floor
    prompt: str = PROMPT_VERSION  # answer prompt version (``prompts.SYSTEM_PROMPTS``); v2-guard is exploratory


def _slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


class Sink:
    def __init__(self, path: Path, max_tokens: int, prompt_version: str = PROMPT_VERSION, extra: dict | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.done: set[str] = set()
        # Every row is stamped with the settings that change the request; resuming is by key, so rows
        # written under other settings would otherwise be skipped silently.
        self.stamp = {"prompt_version": prompt_version, "max_tokens": max_tokens, **(extra or {})}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    old = {k: row.get(k) for k in self.stamp}
                    if old != self.stamp:
                        raise RuntimeError(
                            f"{path} holds rows written with {old}, not {self.stamp}; move the old results folder aside before running"
                        )
                    self.done.add(row["key"])
        self._f = open(path, "a")  # noqa: SIM115 - kept open for the stage, closed by close()

    def write(self, row: dict) -> None:
        row.update(self.stamp)
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()
        self.done.add(row["key"])

    def close(self) -> None:
        self._f.close()


def _parallel(jobs: Iterable[tuple[str, Callable[[], dict]]], sink: Sink, workers: int) -> int:
    todo = [(k, fn) for k, fn in jobs if k not in sink.done]
    print(f"  {sink.path.parent.name}/{sink.path.name}: {len(todo)} to do ({len(sink.done)} already done)", flush=True)
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn): k for k, fn in todo}
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception:
                ex.shutdown(wait=False, cancel_futures=True)  # stop queued calls; finished rows are already saved
                print(f"  stopped at job {futs[f]} ({sink.path.name})", file=sys.stderr, flush=True)
                raise
            row["key"] = futs[f]
            sink.write(row)
            n += 1
            if n <= 3 or n % 10 == 0 or n == len(todo):
                print(f"  {sink.path.name}: {n}/{len(todo)}  (last: turn {row.get('turn')}, correct={row.get('correct')})", flush=True)
    return n


def _ask(
    client: Client, model: str, conv: Conversation, k: int, history: list, max_tokens: int, seed: int = 0, prompt: str = PROMPT_VERSION
) -> dict:
    reply = client.chat(model, answer_messages(conv, k, history, prompt), max_tokens=max_tokens, seed=seed)
    pred = extract_answer(reply.text)
    gold = conv.turns[k].answer
    return {
        "conv": conv.id,
        "turn": k,
        "n_turns": len(conv.turns),
        "pred": pred,
        "gold": gold,
        "correct": match(pred, gold),
        "correct_lenient": match(pred, gold, mode="lenient"),
        "raw": reply.text,
        "tokens_in": reply.prompt_tokens,
        "tokens_out": reply.completion_tokens,
        "finish_reason": reply.finish_reason,
        "provider": reply.provider,
    }


def injection_plan(convs: list[Conversation], s: Settings) -> list[dict]:
    """Deterministic list of (conversation, turn j, error type, wrong value)."""
    rng = random.Random(s.seed)
    plan = []
    for conv in convs:
        for j in range(len(conv.turns) - 1):
            applicable = []
            for kind in s.error_types:
                wrong = corrupt(conv.turns[j].answer, kind, conv, random.Random(f"{s.seed}-{conv.id}-{j}-{kind}"))
                if wrong is not None:
                    applicable.append((kind, wrong))
            rng.shuffle(applicable)
            for kind, wrong in applicable[: s.types_per_turn]:
                plan.append({"conv": conv.id, "j": j, "type": kind, "wrong": wrong})
    return plan


def stage_clean(convs: list[Conversation], client: Client, s: Settings, repeat: bool = False) -> None:
    """``repeat=True`` (stage ``clean_repeat``) asks the same questions again under ``s.repeat_seed``; the
    share of turns whose outcome flips is the run-to-run noise floor for the extra error."""
    stage, seed = ("clean_repeat", s.repeat_seed) if repeat else ("clean", 0)
    for model in s.models:
        sink = Sink(s.out / stage / f"{_slug(model)}.jsonl", s.max_tokens, s.prompt)
        jobs = []
        for conv in convs:
            gold = [t.answer for t in conv.turns]
            for k in range(len(conv.turns)):

                def fn(c=conv, k=k, g=gold) -> dict:
                    return {**_ask(client, model, c, k, g[:k], s.max_tokens, seed, s.prompt), "seed": seed}

                jobs.append((f"{conv.id}|{k}", fn))
        print(f"{stage} {model}: {_parallel(jobs, sink, s.workers)} new")
        sink.close()


def stage_inject(convs: list[Conversation], client: Client, s: Settings) -> None:
    by_id = {c.id: c for c in convs}
    plan = injection_plan(convs, s)
    for model in s.models:
        sink = Sink(s.out / "inject" / f"{_slug(model)}.jsonl", s.max_tokens, s.prompt)
        jobs = []
        for p in plan:
            conv = by_id[p["conv"]]
            graph = build(conv)
            hist = [t.answer for t in conv.turns]
            hist[p["j"]] = p["wrong"]
            for k in range(p["j"] + 1, len(conv.turns)):

                def fn(c=conv, k=k, h=hist, p=p, g=graph) -> dict:
                    row = _ask(client, model, c, k, h[:k], s.max_tokens, prompt=s.prompt)
                    row.update(
                        j=p["j"],
                        type=p["type"],
                        wrong=p["wrong"],
                        depends=k in g.descendants(p["j"]),
                        direct=(p["j"], k) in g.direct,
                        paths=g.path_count(p["j"], k),
                        hops=g.depth(p["j"], k),
                        copied_wrong=match(row["pred"], p["wrong"]) if row["pred"] is not None else False,
                    )
                    return row

                jobs.append((f"{p['conv']}|{p['j']}|{p['type']}|{k}", fn))
        print(f"inject {model}: {_parallel(jobs, sink, s.workers)} new")
        sink.close()


def stage_free(convs: list[Conversation], client: Client, s: Settings) -> None:
    """Sequential within a conversation (each turn needs the previous reply); parallel across conversations."""
    for model in s.models:
        sink = Sink(s.out / "free" / f"{_slug(model)}.jsonl", s.max_tokens, s.prompt)

        def run_conv(conv: Conversation) -> list[dict]:
            rows, hist = [], []
            for k in range(len(conv.turns)):
                key = f"{conv.id}|{k}"
                row = _ask(client, model, conv, k, list(hist), s.max_tokens, prompt=s.prompt)  # cached on re-run
                row["history"] = list(hist)
                row["key"] = key
                rows.append(row)
                hist.append(row["pred"] if row["pred"] is not None else "unknown")
            return rows

        todo = [c for c in convs if f"{c.id}|{len(c.turns) - 1}" not in sink.done]
        with ThreadPoolExecutor(max_workers=s.workers) as ex:
            print(f"  free/{sink.path.name}: {len(todo)} conversations to do", flush=True)
            for i, fut in enumerate(as_completed([ex.submit(run_conv, c) for c in todo]), start=1):
                try:
                    rows = fut.result()
                except Exception:
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise
                print(f"  free/{sink.path.name}: {i}/{len(todo)} conversations", flush=True)
                for row in rows:
                    if row["key"] not in sink.done:
                        sink.write(row)
        print(f"free {model}: {len(todo)} conversations")
        sink.close()


def stage_breakers(convs: list[Conversation], client: Client, s: Settings, breakers: list[dict]) -> None:
    """``breakers`` entries: {"name": ..., "kind": self|cross|program, "answerer": model, "checker": model}."""
    by_id = {c.id: c for c in convs}
    plan = injection_plan(convs, s)
    for b in breakers:
        # program_v2 rows also carry the breaker prompt version, so rows from two development versions never mix.
        extra = {"breaker_prompt": PROGRAM_V2_VERSION} if b["kind"] == "program_v2" else None
        sink = Sink(s.out / "breakers" / f"{b['name']}.jsonl", s.max_tokens, s.prompt, extra)
        jobs = []

        def add(key: str, conv: Conversation, j: int, history: list, answer, label: str, wrong: bool, b=b) -> None:
            def fn() -> dict:
                c = check(b["kind"], client, b["checker"], conv, j, history, answer, max_tokens=s.max_tokens)
                return {"conv": conv.id, "turn": j, "label": label, "is_wrong": wrong, "answer": answer, **c.to_dict()}

            jobs.append((key, fn))

        # (a) injected wrong answers and (b) gold answers, with gold history before j.
        seen_gold = set()
        for p in plan:
            conv = by_id[p["conv"]]
            gold = [t.answer for t in conv.turns]
            add(f"inj|{p['conv']}|{p['j']}|{p['type']}", conv, p["j"], gold[: p["j"]], p["wrong"], f"inject:{p['type']}", True)
            if (p["conv"], p["j"]) not in seen_gold:
                seen_gold.add((p["conv"], p["j"]))
                add(f"gold|{p['conv']}|{p['j']}", conv, p["j"], gold[: p["j"]], gold[p["j"]], "gold", False)
        # (c) the answering model's own answers from the free run.
        free_path = s.out / "free" / f"{_slug(b['answerer'])}.jsonl"
        if free_path.exists():
            for line in free_path.read_text().splitlines():
                r = json.loads(line)
                if r["pred"] is None:
                    continue
                conv = by_id.get(r["conv"])
                if conv is None:
                    continue
                add(f"free|{r['conv']}|{r['turn']}", conv, r["turn"], r["history"], r["pred"], "free", not r["correct"])
        print(f"breaker {b['name']}: {_parallel(jobs, sink, s.workers)} new")
        sink.close()
