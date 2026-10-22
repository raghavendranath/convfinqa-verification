"""OpenAI-compatible chat client (OpenRouter by default) with a permanent response cache.

Every call is cached in SQLite, keyed by the full request. Re-running a stage is free and
resumes where it stopped; the cache is released with the paper so every number can be traced
to a stored model reply. The API key is read from ``OPENROUTER_API_KEY`` and never written.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_BASE = "https://openrouter.ai/api/v1"
# Upstream rate limits (HTTP 429) can last minutes: back off 1, 2, 4 ... 60 s, about 6.5 minutes in all.
RETRIES = 10


@dataclass
class Reply:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    provider: str | None = None
    finish_reason: str | None = None  # "length" means max_tokens ran out (often inside reasoning)


class FatalAPIError(RuntimeError):
    """An API error that retrying cannot fix (wrong model ID, bad key, no credit)."""


class Cache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Several stages may run at once in separate terminals and share this file:
        # WAL mode lets readers and one writer work together, and the timeout waits out brief locks.
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=120)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, request TEXT, response TEXT, ts REAL)")
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT response FROM replies WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, request: dict, response: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO replies VALUES (?,?,?,?)", (key, json.dumps(request), json.dumps(response), time.time())
            )
            self._db.commit()


class Client:
    def __init__(
        self,
        cache: Cache,
        base_url: str | None = None,
        providers: dict[str, str] | None = None,
        offline: bool = False,
        request_extra: dict[str, dict] | None = None,
    ):
        self.cache = cache
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE)
        self.providers = providers or {}
        self.offline = offline  # True: only cached replies; a cache miss raises (used for re-scoring)
        # Per-model request fields (e.g. OpenRouter's "reasoning" settings); part of the cache key.
        self.request_extra = request_extra or {}

    def chat(self, model: str, messages: list[dict], max_tokens: int = 2048, temperature: float = 0.0, seed: int = 0) -> Reply:
        request = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed}
        request.update(self.request_extra.get(model, {}))
        if model in self.providers:
            request["provider"] = {"order": [self.providers[model]], "allow_fallbacks": False}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        hit = self.cache.get(key)
        if hit is not None:
            return _reply(hit, cached=True)
        if self.offline:
            raise RuntimeError(f"cache miss in offline mode for {model}")
        key_env = os.environ.get("OPENROUTER_API_KEY")
        if not key_env:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        last: Exception | None = None
        for attempt in range(RETRIES):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {key_env}"},
                    json=request,
                    timeout=300,
                )
            except requests.RequestException as e:  # network trouble: retry
                last = e
                print(f"  [retry {attempt + 1}/{RETRIES}] {model}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                time.sleep(min(60, 2**attempt))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                print(f"  [retry {attempt + 1}/{RETRIES}] {model}: {last}", file=sys.stderr, flush=True)
                time.sleep(_wait(r, attempt))
                continue
            if r.status_code >= 400:
                # Wrong model ID, bad key, no credit, bad request: retrying will not help, so stop now.
                raise FatalAPIError(f"{model}: HTTP {r.status_code}: {r.text[:500]}")
            try:
                data = r.json()
            except ValueError:  # body cut off upstream (often only OpenRouter's keep-alive blank lines): retry
                last = RuntimeError(f"HTTP {r.status_code} with a body that is not JSON: {r.text.strip()[:200]!r}")
                print(f"  [retry {attempt + 1}/{RETRIES}] {model}: {last}", file=sys.stderr, flush=True)
                time.sleep(min(60, 2**attempt))
                continue
            if not data.get("choices"):
                err = data.get("error")
                # OpenRouter can return an upstream rate limit or outage as HTTP 200 with the code in the body: retry those.
                code = err.get("code") if isinstance(err, dict) else None
                if code in (429, 500, 502, 503, 504):
                    last = RuntimeError(f"error {code} in body: {str(err)[:200]}")
                    print(f"  [retry {attempt + 1}/{RETRIES}] {model}: {last}", file=sys.stderr, flush=True)
                    time.sleep(_wait(r, attempt))
                    continue
                if err is not None:
                    raise FatalAPIError(f"{model}: {str(err)[:500]}")
                last = RuntimeError(f"no choices: {str(data)[:200]}")
                print(f"  [retry {attempt + 1}/{RETRIES}] {model}: {last}", file=sys.stderr, flush=True)
                time.sleep(min(60, 2**attempt))
                continue
            self.cache.put(key, request, data)
            return _reply(data, cached=False)
        raise RuntimeError(f"{model}: failed after retries: {last}")


def _wait(r: requests.Response, attempt: int) -> float:
    """Seconds to wait: the server's Retry-After when it gives one (capped at 60), else exponential backoff."""
    try:
        return min(60.0, float(r.headers.get("Retry-After", "")))
    except ValueError:
        return min(60, 2**attempt)


def _reply(data: dict, cached: bool) -> Reply:
    choice = data["choices"][0]
    msg = choice["message"]
    usage = data.get("usage") or {}
    return Reply(
        text=msg.get("content") or "",
        model=data.get("model", ""),
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        cached=cached,
        provider=data.get("provider"),
        finish_reason=choice.get("finish_reason"),
    )
