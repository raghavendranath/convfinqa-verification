"""Select and lock the fresh evaluation sample for PREREGISTRATION change 4 (revised program breaker).

Run once, before any model call on these conversations (17 Oct 2026). Writes configs/change4_sample.txt (one
conversation id per line) and prints its SHA-256, which is recorded in PREREGISTRATION.md change 4.

Rule, from the ConvFinQA train split:
1. eligible: gold programs reproduce the gold answers and at least 3 turns (the study's selection rule);
2. leave out the 300 conversations of the optional out-of-split fit sample (configs/fit.yaml, seed 1), never run;
3. leave out every conversation from a filing (company and year) that appears in the dev split, so that no
   underlying annual report is shared with the conversations already inspected;
4. at most one conversation per report page (pages carry several conversations about the same table);
5. a fixed-seed sample of 600 pages, seed 2.

    uv run python scripts/change4_sample.py
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path

from cascade.data import load

RAW = Path("data/raw/convfinqa")
OUT = Path("configs/change4_sample.txt")
N, SEED, FIT_N, FIT_SEED = 600, 2, 300, 1


def page(cid: str) -> str:
    """Report page, e.g. 'JKHY/2009/page_28.pdf' from 'Single_JKHY/2009/page_28.pdf-3'."""
    return re.sub(r"-\d+$", "", cid.split("_", 1)[1])


def filing(cid: str) -> str:
    return "/".join(page(cid).split("/")[:2])


def main() -> None:
    keep = [c for c in load(RAW / "train.json") if not c.issues and len(c.turns) >= 3]
    fit = {c.id for c in random.Random(FIT_SEED).sample(keep, FIT_N)}  # same draw as cli._select for fit.yaml
    dev_filings = {filing(c.id) for c in load(RAW / "dev.json")}
    pool = [c for c in keep if c.id not in fit and filing(c.id) not in dev_filings]
    by_page: dict[str, list[str]] = defaultdict(list)
    for c in pool:
        by_page[page(c.id)].append(c.id)
    rng = random.Random(SEED)
    one_per_page = sorted(rng.choice(sorted(ids)) for _, ids in sorted(by_page.items()))
    sample = sorted(rng.sample(one_per_page, N))
    text = "\n".join(sample) + "\n"
    OUT.write_text(text)
    print(
        f"eligible {len(keep)}; after fit and dev-filing exclusions {len(pool)}; pages {len(by_page)}; "
        f"filings {len({filing(i) for i in sample})} in the sample"
    )
    print(f"wrote {OUT}: {len(sample)} conversations, sha256 {hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
