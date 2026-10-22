"""Build results/audit/labeler.html, a click-through page for labelling the dependency-graph audit.

Usage: uv run python scripts/make_labeler.py
Open the page in a browser, label every turn, press "Download CSV", and put the file back at
results/audit/graph_audit.csv. The page hides the graph's own guesses (parents_found) so they do
not steer the labels; the exported CSV keeps every original column unchanged.
"""

import csv
import json
from pathlib import Path

from cascade.data import load

SHEET = Path("results/audit/graph_audit.csv")
OUT = Path("results/audit/labeler.html")

with open(SHEET, newline="") as f:
    rows = list(csv.DictReader(f))
columns = list(rows[0].keys())
# The report each conversation is about (text before the table, the table, text after), shown on the page.
wanted = {r["conv"] for r in rows}
reports = {
    c.id: {"pre": " ".join(c.pre_text).strip(), "table": c.table, "post": " ".join(c.post_text).strip()}
    for c in load("data/raw/convfinqa/dev.json")
    if c.id in wanted
}
assert set(reports) == wanted, f"reports missing for {wanted - set(reports)}"
data = json.dumps({"columns": columns, "rows": rows, "reports": reports}).replace("</", "<\\/")
OUT.write_text(Path(__file__).with_name("labeler_template.html").read_text().replace("__DATA__", data))
print(f"wrote {OUT} ({len(rows)} turns). Open it with: open {OUT}")
