# Error Propagation and Verification Limits in Conversational Financial Question Answering

This repository contains the research code, saved experimental results, evaluation configurations, and reproducibility materials accompanying the manuscript “Error Propagation and Verification Limits in Conversational Financial Question Answering.”

Author: Raghavendranath Kandula, Independent Researcher. Correspondence: kraghavendranath@gmail.com.

The study measures the effect of one erroneous answer retained in otherwise gold conversational history. Each later question is evaluated separately; the injected condition does not feed generated intermediate answers forward. It also evaluates self-checking, cross-model checking and program re-execution on planted and naturally occurring errors. The principal result is a large detection gap and net loss of correctness when program outputs replace flagged natural answers. The revised verifier does not meet its prospectively specified improvement criterion.

## Reproduce without paid experiments

Start with [REPRODUCING.md](REPRODUCING.md). The saved JSONL replies are the authoritative record. Existing summaries and figures can be inspected directly; no API key is needed for the offline tests or integrity check.

```bash
python -m pip install -e '.[dev,figures]'
python scripts/verify_release.py
python -m pytest -q
```

All 105 offline tests passed on 22 October 2026. Numerical results may vary slightly across software environments.

## Materials

| Location | Contents |
|---|---|
| `src/cascade/` | Data loading, dependency heuristic, injections, numeric matching, prompts, model client, verification and analysis |
| `configs/` | Frozen/main, smoke, exploratory and revised-verifier configurations; the locked evaluation sample |
| `results/main/` | Main-run replies, manifest, generated tables and summary |
| `results/exploratory_guard/` | Guarded-prompt experiment |
| `results/change4/` | Locked 600-conversation revised-verifier evaluation |
| `results/change4_dev/` | All three development versions, including unsuccessful versions |
| `results/parser_sensitivity/` | Corrected-parser copies and sensitivity results |
| `results/audit/` | Annotation guide and primary/GPT/second-annotator labels |
| `results/spend/` | Aggregate study account-activity record |
| `PREREGISTRATION.md` | Registration and dated amendments |
| `CHANGELOG.md` | Public summary of scientifically relevant changes |
| `provenance/` | Registration snapshots and source commit metadata |
| `paper/figures/` | Six original scientific figures, PNG and vector PDF |
| `RELEASE_MANIFEST.json` | File checksums and source-version metadata |

The request-keyed SQLite caches and downloaded benchmark are intentionally not tracked. Identical requests sometimes produced distinct outputs, so replaying a cache is not a substitute for the result-level JSONL records. See the manuscript's reproducibility section and `ARTIFACTS.md`.

## Data and licensing

Download ConvFinQA from its original release using `cascade fetch`, or inspect the saved experiment outputs without downloading it. The upstream benchmark is not bundled here. Its MIT license and attribution are retained in `third_party/ConvFinQA-LICENSE.txt` and `NOTICE`. Original code is licensed under Apache-2.0; the code license does not replace third-party rights or provider terms.

## Study limitations and implementation record

The frozen parser mishandles scientific notation in seven stored answers. Corrected-parser sensitivity changes four correctness outcomes and no reported hypothesis decision; one historical answer would require a new model call for complete correction. No such call was made in preparing this release. The dependency graph is heuristic and its precision is limited. Verification excludes unparseable natural answers and treats unusable checks as no flag. Offline replacement is not a test of downstream conversational repair. Hosted temperature-zero outputs can vary.

## AI assistance

Claude Code (Anthropic) assisted with implementation, offline analyses and earlier manuscript drafting. GPT 6.1 Sol drafted the primary dependency labels for 47 conversations, reviewed by the author. An OpenAI AI assistant assisted with manuscript revision, formatting and submission preparation. The author remains responsible for the research, software, analyses and final text.

This manuscript has not yet been accepted or published. Cite the software using `CITATION.cff`; no journal DOI or acceptance is claimed.
