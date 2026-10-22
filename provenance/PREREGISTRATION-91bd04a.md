# Pre-registration

Status: FROZEN, 12 Oct 2026, at commit `6dfe505` (code, prompts, `configs/main.yaml` and the graph
audit as of that commit). Drafted 6 Oct 2026. The freeze commit itself only edits this file,
CHANGELOG.md and memory.md. No main-run result existed at freezing. Changes after freezing are listed
at the end with reasons, and the paper reports them.

## Models
The five open models in `configs/main.yaml`: qwen/qwen3-8b, qwen/qwen3-32b, meta-llama/llama-4-maverick,
mistralai/mistral-large-2512, deepseek/deepseek-chat-v3-0324, with the provider pins and request settings
given there. No closed frontier model is included (decided at freezing; the commented placeholder in the
config is not used). The size comparison (H3) rests on the two Qwen3 sizes.

## Data
ConvFinQA dev split, all conversations with at least 3 turns whose gold programs reproduce their
gold answers (`cascade check-data` reports how many are excluded and why). Train split is used
only for the optional out-of-split check of the transmission model (`configs/fit.yaml`).

## Primary metric
Strict numeric match (0.5% relative tolerance, or equal after rounding to 5 decimals, because ConvFinQA stores gold answers rounded to 5 decimals). Lenient match (percent/ratio equivalence) is a
sensitivity analysis only.

## Hypotheses (each could go either way)
- **H1 (cascade exists).** An injected wrong answer raises the error rate of later turns that depend
  on it, compared with the clean history, for every model.
- **H2 (structure predicts spread).** Extra error is concentrated in dependent turns; turns with no
  dependency path show extra error close to zero (s small). The path-transmission model predicts
  held-out error better than a model that ignores the graph (Brier score; paired bootstrap).
- **H3 (size).** Larger models catch injected errors more often (lower a). *Alternative:* larger
  models trust the conversation history more and copy the wrong number (higher a). We report
  whichever holds.
- **H4 (breakers).** Program re-execution catches more injected errors than self-checking at a
  similar or lower false-alarm rate; cross-model checking sits between them.
- **H5 (error type).** Scale and period errors spread more than sign errors, which models notice more often.

## Analysis choices fixed in advance
- Conversation-level cluster bootstrap, 2,000 resamples, 95% intervals.
- Fit / held-out split by SHA-256 of the conversation id (50/50).
- Shrinkage of the per-turn clean error toward the model mean: lam = 0.5.
- Temperature 0, seed 0, max 8,192 output tokens (answers and breaker checks; raised from 4,096 after the
  smoke run, see CHANGELOG.md R13); `types_per_turn` = 2, seed 0.
- Every model and breaker that is run is reported, including failures.
- Answer prompt version v2 (`prompts.py`): unrounded values, ratios and percentage changes as decimals unless the
  question asks to multiply by 100 or convert to a percentage, amounts in the report's units. The smoke run under
  v1 is kept in `results/smoke_v1/` and reported as a pilot.
- A reply with no parseable answer counts as wrong under both matchers. Truncated replies
  (`finish_reason = length`) are counted and reported per model.
- Run-to-run noise floor: for every model, `clean` is asked a second time (`clean_repeat`, seed 1; at
  temperature 0 the seed only makes the request distinct from the cached one). The share of turns whose
  outcome flips, with a conversation-level bootstrap interval, is reported next to every extra-error
  estimate. An extra error inside the noise floor is not claimed as a cascade, but it is still reported.
- Error mechanisms (exploratory, rules fixed here, `analysis.mechanism`): every later turn that is right with
  the clean history and wrong after an injection is assigned the first rule that applies: `loop` (cut off by
  max_tokens, no answer); `no_answer` (finished, nothing parseable); `copied` (equals the planted wrong value,
  strict match); `convention` (equals the gold value multiplied or divided by 100, 1,000, 10^6 or 10^9, relative
  tolerance 0.5%, i.e. a percent or unit convention carried over); `other`. Counts are reported per model for
  dependent and independent turns. A planted value that happens to fall within 0.5% of the model's own wrong
  answer is counted as `copied`; this is reported, not corrected. These counts describe how errors spread;
  they are not hypothesis tests.
- Dependency-graph audit: 50 dev conversations (seed 7, `cascade audit-graph`, 219 turns), labelled following
  `results/audit/GUIDE.md`, completed before any main-run result existed. Labels for 47 conversations were
  drafted by GPT 6.1 Sol (high reasoning) and reviewed by the author; 3 conversations were labelled by the
  author alone (CHANGELOG.md R24, R25). It is not an independent hand audit, and no agreement statistic is
  reported. Two labels per turn: value
  parents (the graph's target) and reference-only parents (outside the graph's definition). Reported:
  edge-level precision and recall of the graph against value parents, with conversation bootstrap 95%
  intervals, and the number of reference-only links. The graph is not changed after the audit; if it is
  poor, that is reported as a limitation of H2.
- Provider pins and any per-model request settings (`providers`, `request_extra` in `configs/main.yaml`) are
  fixed in the frozen config.

## Changes after freezing
1. **13 Oct 2026: qwen/qwen3-8b replaced by qwen/qwen3-14b.** OpenRouter retired Qwen3 8B on 9 Oct 2026
   (HTTP 404, "deprecated"), three days before freezing; the model IDs were not re-checked at freezing. No
   Qwen3 8B reply was ever collected. Qwen3 14B is the smallest remaining dense model of the same Qwen3
   release as Qwen3 32B, served by the same host (DeepInfra, fp8) with the same reasoning cap. The four other
   models' results were collected before this choice and do not depend on it. Effect on H3: the
   within-family size contrast is 14B to 32B (2.3 times) instead of 8B to 32B (4 times), so a size effect
   is harder to detect. The new model gets a 20-conversation smoke check (`configs/smoke_qwen3_14b.yaml`)
   before its main stages. CHANGELOG.md R29.
2. **14 Oct 2026: exploratory runs added after the main results (specified here before any of them is run).**
   These are not tests of H1 to H5. They were chosen after the main results were known. The paper labels them
   "exploratory, added after freezing" and reports every one of them, whatever it shows. CHANGELOG.md R31.
   - *Breakers on DeepSeek V3's answers.* The `exploratory_breakers` list in `configs/main.yaml`:
     `self_deepseek` (self; answerer and checker deepseek/deepseek-chat-v3-0324), `cross_deepseek_by_llama`
     (cross; checker meta-llama/llama-4-maverick, the same checker as for Qwen3 32B, so only the answerer changes)
     and `program_deepseek` (program; checker DeepSeek V3). The prompts, the parsing, the rule for unusable
     checks and the scores are the same as for the pre-registered breakers. The checked items are the same
     injected and gold answers, plus DeepSeek V3's own `free` answers from `results/main`.
   - *Guarded answer prompt (a mitigation aimed at the spread mechanism).* Prompt version `v2-guard`: the v2
     system prompt with this paragraph added after the number-format rules and before the report:
     "Earlier answers in this conversation may contain mistakes. When the latest question needs an earlier
     value, recompute it from the report instead of reusing the earlier answer. Decide units and number format
     from the latest question alone, not from the format of earlier answers."
     Config `configs/exploratory_guard.yaml`: identical to `configs/main.yaml` except `prompt: v2-guard`,
     output `results/exploratory_guard`, models deepseek/deepseek-chat-v3-0324 and qwen/qwen3-32b, no breakers.
     Stages clean, inject and free (no clean_repeat; the noise floor from the main run is used). A 20-conversation
     smoke (`configs/smoke_guard.yaml`, clean only) is run first to check that answers still parse; the wording
     above is not changed after the smoke. If the smoke shows a parse failure the prompt is not revised;
     the failure is reported and the guarded run is dropped.
   - *Measures for the guarded prompt.* Per model, guarded minus v2 for: clean accuracy, dependent extra
     error, independent extra error and free (own-history) accuracy. Each is a mean of paired row differences
     (same conversation, turn and injection), with a conversation cluster bootstrap 95% interval (2,000
     resamples, seed 0). A lower clean accuracy is a cost and is reported next to any gain. The error-mechanism
     counts are also reported for the guarded run.
3. **17 Oct 2026: post-hoc breaker analyses and a second, independent graph audit (both after the main results).**
   - *Program breaker matcher.* The frozen program breaker compares its recomputed value with the checked answer
     under the lenient matcher, which treats x100 and /100 as equal; a planted percent error is exactly gold x100 or
     /100, so a correct program cannot trip on it. The pre-registered scores are kept as the primary H4 result.
     Post-hoc (not pre-registered): the stored values are rescored with the strict matcher, catch rates are given
     by error type for every breaker, every check is reported as agree / disagree / unable to verify, and program
     repair is evaluated offline against gold (wrong answers repaired, wrong answers replaced by another wrong value,
     right answers corrupted). No model calls. CHANGELOG.md R32.
   - *Second labeller.* A second person labelled the 50-conversation audit sheet by hand after the main run, following
     `results/audit/GUIDE.md` (no AI, no access to the GPT draft or the primary labels). They saw the author's labels
     for 3 conversations as worked examples, so those 3 are excluded from every figure below. The primary audit
     (`graph_audit.csv`) stays the pre-registered one. Added: graph precision and recall against the second
     labeller's value parents, and agreement with the primary audit (share of turns with identical parent sets;
     Cohen's kappa over candidate edges with a conversation bootstrap interval), on the same 47 conversations. This
     replaces the statement above that no agreement statistic is reported: that statement referred to the
     GPT-drafted sheet, which is not independent. CHANGELOG.md R33.
     Amended 17 Oct 2026, before any of it was committed (CHANGELOG.md R34): the author chose all 50
     conversations as the headline, because the second sheet was made without the GPT draft; for the 3 example
     conversations it holds the author's own labels, which is stated, and the 47-conversation figures are kept as
     a sensitivity analysis.
