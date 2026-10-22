# Artifact provenance and scope

This document describes the saved experimental records, benchmark version, and file-integrity metadata accompanying the study.

Saved stage-level JSONL records contain the replies used for scoring. They are authoritative where simultaneous identical requests produced different outputs. Request-keyed SQLite caches retain one response per key and are not required to inspect the published numbers. They are excluded by `.gitignore` and are not included in this repository. No claim of complete archival cache coverage is made.

The main manifest has 305 eligible development conversations. Clean, repeated-clean and free stages have 1,258 rows per model; injected stages have 4,209 later-turn rows per model. The separate locked training-split evaluation has 600 conversations and 2,538 turns per answerer. Development versions and smoke runs remain distinguishable from confirmatory results.

Raw ConvFinQA data remain upstream rather than being redistributed. The source is https://github.com/czyssrs/ConvFinQA; its MIT license is reproduced in `third_party/ConvFinQA-LICENSE.txt`. IDs, questions, quoted report text or annotations present in derived records remain attributable to the upstream benchmark. Provider/model terms continue to apply where relevant; Apache-2.0 covers the original code and is not a claim to ownership of third-party material.

## Benchmark files used locally

- `dev.json`: SHA-256 `9665dd60f106e5cd41d37537edac97bbd26eb7629ad63ca2390b19d4619ba51b`; 5,730,210 bytes.
- `train.json`: SHA-256 `c100932b73f3237f0474cbb6baa04e1662050cba948791f2e3a5d496b78e0b50`; 42,548,550 bytes.

## Registration and file integrity

Registration-document snapshots and source-version metadata are documented in [`provenance/`](provenance/README.md). The provenance documentation explains the limits of Git-recorded dates as evidence of preregistration. `RELEASE_MANIFEST.json` provides checksums for verifying the included artifacts.
