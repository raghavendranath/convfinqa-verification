"""Read-only check of preserved research artifacts; no API calls."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
failures = []
for name, expected in manifest["sha256"].items():
    path = ROOT / name
    if not path.is_file():
        failures.append(f"missing: {name}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(f"changed: {name}")
if failures:
    print("\n".join(failures))
    sys.exit(1)
print(f"Verified {len(manifest['sha256'])} preserved research artifacts against baseline {manifest['baseline_commit']}.")
