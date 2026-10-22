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
(none yet)
