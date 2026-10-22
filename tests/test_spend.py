"""cascade spend: account activity kept for the study's models only, tabled offline (CHANGELOG R47)."""

import datetime as dt
import json
from pathlib import Path

import pytest

from cascade import spend


def _row(date, model, usage, requests=10, byok=0.0):
    return {"date": f"{date} 00:00:00", "model": model, "usage": usage, "requests": requests, "byok_usage_inference": byok}


def test_fetch_keeps_study_models_and_tables_them(tmp_path: Path, monkeypatch):
    import requests

    sent = []

    class R:
        status_code = 200

        def __init__(self, date):
            self.date = date

        def json(self):
            d = self.date
            return {"data": [_row(d, "a/m", 1.25), _row(d, "a/m", 0.5, 2, byok=0.25), _row(d, "other/g", 9.0)]}

    def get(url, params, headers, timeout):
        sent.append((params["date"], headers["Authorization"]))
        return R(params["date"])

    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", "mk")
    monkeypatch.setattr(requests, "get", get)
    cfg = {"out": str(tmp_path), "models": ["a/m"], "spend": {"from": "2026-10-18", "to": "2026-10-19"}}
    path = spend.fetch(cfg, today=dt.date(2026, 10, 20))
    saved = json.loads(path.read_text())
    assert [d for d, _ in sent] == ["2026-10-18", "2026-10-19"] and all(h == "Bearer mk" for _, h in sent)
    assert {r["model"] for r in saved["rows"]} == {"a/m"} and "mk" not in path.read_text()
    summary: dict = {}
    lines = spend.spend_section(cfg, tmp_path, summary)
    assert summary["spend"]["total_usd"] == pytest.approx(4.0) and summary["spend"]["byok_inference_usd"] == pytest.approx(0.5)
    assert "| a/m | 24 | 3.50 | 0.50 | 4.00 |" in lines and "| 2026-10-19 | a/m | 12 | 1.75 | 0.25 |" in lines


def test_fetch_refuses_a_day_that_has_not_ended(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", "mk")
    cfg = {"out": str(tmp_path), "models": ["a/m"], "spend": {"from": "2026-10-18", "to": "2026-10-19"}}
    with pytest.raises(RuntimeError, match="has not ended"):
        spend.fetch(cfg, today=dt.date(2026, 10, 19))
