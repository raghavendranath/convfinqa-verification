"""Power check for PREREGISTRATION change 4 (revised program breaker), run before freezing (17 Oct 2026).

No model calls. Uses the dev-split cluster structure of the frozen program breaker's natural items (free:wrong and
free:right per conversation, results/main/breakers/program_*.jsonl) with the strict-rescored trips as the
baseline. A revised breaker is simulated by flipping baseline outcomes, so that the true paired difference takes
a chosen value. Conversations are resampled to the planned sample size. For each scenario: the share of
simulated studies whose paired conversation-bootstrap 95% interval meets the criterion
(recall: lower bound > 0; false alarms: upper bound < +0.02).

    uv run python scripts/change4_power.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cascade.breaker_posthoc import tripped_strict

SIMS, BOOT, SEED = 300, 1000, 0
# (label, share of baseline catches the revised breaker loses, share of baseline non-alarms that become alarms,
# sample sizes in conversations). Higher discordance means more changed verdicts, so wider intervals.
SCENARIOS = (("base discordance", 0.10, 0.03, (305, 440, 600, 900)), ("high discordance", 0.25, 0.08, (440, 600)))


def items(path: Path) -> dict[str, dict[str, list[int]]]:
    """Per conversation, strict-rescored trips on natural wrong and natural right answers."""
    out: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"wrong": [], "right": []})
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["label"] == "free":
            out[r["conv"]]["wrong" if r["is_wrong"] else "right"].append(int(tripped_strict(r)))
    return dict(out)


def power(per_conv: list[list[int]], delta: float, p_down: float | None, p_up: float | None, n_conv: int, rule, rng) -> float:
    base = [np.array(x) for x in per_conv]
    b = np.concatenate(base).mean()
    if p_down is not None:  # recall scenario: fix p_down, solve p_up
        p_up = (delta + b * p_down) / (1 - b)
    else:  # false-alarm scenario: fix p_up, solve p_down
        p_down = ((1 - b) * p_up - delta) / b
    if not (0 <= p_up <= 1 and 0 <= p_down <= 1):
        return float("nan")
    hits = 0
    for _ in range(SIMS):
        pick = rng.integers(0, len(base), n_conv)
        sums, counts = np.zeros(n_conv), np.zeros(n_conv)
        for i, c in enumerate(pick):
            x = base[c]
            u = rng.random(len(x))
            new = np.where(x == 1, u >= p_down, u < p_up).astype(int)
            sums[i], counts[i] = (new - x).sum(), len(x)
        keep = counts > 0
        sums, counts = sums[keep], counts[keep]
        idx = rng.integers(0, len(sums), (BOOT, len(sums)))
        boots = sums[idx].sum(1) / counts[idx].sum(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        hits += rule(lo, hi)
    return hits / SIMS


def main() -> None:
    rng = np.random.default_rng(SEED)
    for name in ("program_qwen32b", "program_deepseek"):
        conv = items(Path("results/main/breakers") / f"{name}.jsonl")
        wrong = [v["wrong"] for v in conv.values() if v["wrong"]]
        right = [v["right"] for v in conv.values() if v["right"]]
        print(
            f"\n{name}: dev 305 conversations; natural wrong {sum(map(len, wrong))} in {len(wrong)} conversations, "
            f"baseline strict recall {np.mean(np.concatenate(wrong)):.3f}; natural right {sum(map(len, right))}, "
            f"baseline strict false alarms {np.mean(np.concatenate(right)):.3f}"
        )
        for label, p_down, p_up, sizes in SCENARIOS:
            for n in sizes:
                nw, nr = round(len(wrong) * n / 305), round(len(right) * n / 305)
                rec = {d: power(wrong, d, p_down, None, nw, lambda lo, hi: lo > 0, rng) for d in (0.0, 0.05, 0.10, 0.15)}
                fa = {d: power(right, d, None, p_up, nr, lambda lo, hi: hi < 0.02, rng) for d in (0.02, 0.0, -0.05, -0.10)}
                print(f"  {label}, {n} conversations, recall improvement shown: " + ", ".join(f"{d:+.2f}: {q:.2f}" for d, q in rec.items()))
                print(f"  {label}, {n} conversations, false alarms non-inferior: " + ", ".join(f"{d:+.2f}: {q:.2f}" for d, q in fa.items()))


if __name__ == "__main__":
    main()
