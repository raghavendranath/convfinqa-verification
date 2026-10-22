"""OpenRouter spend for the study's models, by UTC day (CHANGELOG R47).

``cascade spend`` saves the account's activity rows (``GET /api/v1/activity``) for the models in the config to
``<out>/activity.json``. That endpoint needs a management key, read from ``OPENROUTER_MANAGEMENT_KEY`` and never
written; it covers the last 30 completed UTC days, so days that have not ended are refused. Only rows for the
study's models are kept: the account also holds other work, which is left out by model. ``cascade report`` then
tables the saved rows offline.

The reply cache records the cost of each reply it keeps, but undercounts: the program breaker's planted and gold
items for a turn send identical requests, which parallel workers sent separately, and the cache keeps one reply per
request (CHANGELOG R46). The account's activity counts every call.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path

import requests

ACTIVITY_URL = "https://openrouter.ai/api/v1/activity"


def days(start: str, end: str) -> list[str]:
    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    return [(a + dt.timedelta(n)).isoformat() for n in range((b - a).days + 1)]


def fetch(cfg: dict, today: dt.date | None = None) -> Path:
    """Save the activity rows of the config's models for each day in ``cfg["spend"]``; return the file."""
    key = os.environ.get("OPENROUTER_MANAGEMENT_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_MANAGEMENT_KEY is not set (a management key: openrouter.ai/settings/management-keys)")
    span = cfg["spend"]
    dates = days(str(span["from"]), str(span["to"]))
    today = today or dt.datetime.now(dt.timezone.utc).date()
    if dates[-1] >= today.isoformat():
        raise RuntimeError(f"{dates[-1]} (UTC) has not ended; run after 00:00 UTC on the day after it")
    models = set(cfg["models"])
    rows, other = [], defaultdict(float)
    for d in dates:
        r = requests.get(ACTIVITY_URL, params={"date": d}, headers={"Authorization": f"Bearer {key}"}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"activity {d}: HTTP {r.status_code}: {r.text[:300]}")
        for row in r.json().get("data", []):
            if row.get("model") in models:
                rows.append(row)
            else:
                other[row.get("model")] += row.get("usage") or 0.0
    out = Path(cfg.get("out", "results/spend"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "activity.json"
    meta = {"source": ACTIVITY_URL, "fetched": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    path.write_text(json.dumps({**meta, "dates": dates, "models": sorted(models), "rows": rows}, indent=1))
    # Printed so the filter can be checked, not saved: the account's other work is not part of the study.
    for m, usd in sorted(other.items(), key=lambda kv: -kv[1]):
        print(f"  left out (not a study model): {m}: ${usd:.2f}")
    print(f"wrote {path}: {len(rows)} rows for {len(models)} models over {len(dates)} days")
    return path


def spend_section(cfg: dict, out: Path, summary: dict) -> list[str]:
    path = Path(out) / "activity.json"
    if not path.exists():
        return []
    saved = json.loads(path.read_text())
    rows = saved["rows"]
    by_day: dict = defaultdict(lambda: {"usage": 0.0, "byok": 0.0, "requests": 0})
    by_model: dict = defaultdict(lambda: {"usage": 0.0, "byok": 0.0, "requests": 0})
    for r in rows:
        # The API gives "YYYY-MM-DD" or "YYYY-MM-DD 00:00:00"; BYOK inference is billed by the provider, not OpenRouter.
        for agg in (by_day[(r["date"][:10], r["model"])], by_model[r["model"]]):
            agg["usage"] += r["usage"]
            agg["byok"] += r.get("byok_usage_inference") or 0.0
            agg["requests"] += r["requests"]
    credits = sum(m["usage"] for m in by_model.values())
    byok = sum(m["byok"] for m in by_model.values())
    total = credits + byok
    lines = [
        "## OpenRouter spend",
        "",
        f"From the account's activity (`{saved['source']}`, fetched {saved['fetched']}), UTC days "
        f"{saved['dates'][0]} to {saved['dates'][-1]}, study models only. Every call is counted, including calls whose "
        "reply the cache did not keep (CHANGELOG R46). USD. OpenRouter credits: billed by OpenRouter. BYOK inference: "
        "calls sent through OpenRouter on the author's own provider key, billed by that provider.",
        "",
        "| Model | Requests | OpenRouter credits | BYOK inference | Total |",
        "|---|---|---|---|---|",
    ]
    for m in sorted(by_model):
        a = by_model[m]
        lines.append(f"| {m} | {a['requests']} | {a['usage']:.2f} | {a['byok']:.2f} | {a['usage'] + a['byok']:.2f} |")
    lines += [f"| all | {sum(m['requests'] for m in by_model.values())} | {credits:.2f} | {byok:.2f} | {total:.2f} |", ""]
    lines += ["By UTC day:", "", "| Date | Model | Requests | OpenRouter credits | BYOK inference |", "|---|---|---|---|---|"]
    for (d, m), agg in sorted(by_day.items()):
        lines.append(f"| {d} | {m} | {agg['requests']} | {agg['usage']:.2f} | {agg['byok']:.2f} |")
    lines.append("")
    summary["spend"] = {
        "dates": [saved["dates"][0], saved["dates"][-1]],
        "total_usd": total,
        "openrouter_credits_usd": credits,
        "byok_inference_usd": byok,
        "by_model": dict(by_model),
        "by_day": [{"date": d, "model": m, **agg} for (d, m), agg in sorted(by_day.items())],
    }
    return lines
