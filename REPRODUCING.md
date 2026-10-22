# Reproducing the analyses

The repository includes saved model replies and evaluation outputs. The steps below explain how to inspect the results, run offline checks, and recompute analyses without collecting new model replies.

## 1. Inspect the frozen outputs without running anything

Read `results/main/TABLES.md`, `results/exploratory_guard/TABLES.md`, `results/change4/TABLES.md` and `results/parser_sensitivity/TABLES.md`. The matching `summary.json` files provide machine-readable estimates. `paper/figures/` contains the six original figures.

`RELEASE_MANIFEST.json` records file checksums for integrity checking. Registration-document snapshots and source-version metadata are documented in [`provenance/`](provenance/README.md), including the limits of the available timestamp evidence.

## 2. Install and run offline checks

Python 3.10 or later is declared in `pyproject.toml`. The existing `uv.lock` records the development dependency resolution. A virtual environment is recommended.

```bash
python -m pip install -e '.[dev,figures]'
python scripts/verify_release.py
python -m pytest -q
```

The 105 tests use fixtures and mocked model clients. They do not require provider keys. The integrity script reads files only and makes no network calls.

## 3. Obtain the benchmark only if needed

```bash
cascade fetch
cascade check-data
```

This downloads the original ConvFinQA release. It makes no model calls. Keep `data/` untracked, retain the upstream license, and use the exact files/checksums in `ARTIFACTS.md` to identify the local benchmark version used by the study.

## 4. Recompute statistics and figures, without collecting new replies

The following commands read the saved result rows. They may be computationally intensive because they repeat cluster bootstraps. They overwrite derived report/figure files, so run them in a disposable working copy or save the existing outputs first.

```bash
cascade report --config configs/main.yaml
cascade report --config configs/exploratory_guard.yaml
cascade report --config configs/change4.yaml
cascade report --config configs/parser_sensitivity.yaml
python scripts/figures.py
```

Compare regenerated outputs against the preserved manifest and numerical tables. Numerical serialization or plotting-library differences may change file hashes even when the substantive estimates agree. The original stored replies, frozen scores and corrected-parser sensitivity must remain distinguishable.

`cascade run ...` collects or replays model requests and is not needed for the steps above. Use `cascade --help` to inspect the collection interface if needed. Collecting new replies is optional and is not required to reproduce analyses of the saved outputs.
