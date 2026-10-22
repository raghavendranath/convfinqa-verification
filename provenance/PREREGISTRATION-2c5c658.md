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
     Addendum 17 Oct 2026 (CHANGELOG.md R36), post-hoc, no model calls: a scoring audit of the program breaker
     (how each program value differs from the checked answer; trip rates under the lenient, strict and a
     format-aware bound that uses gold's order of magnitude; natural errors split into format-only and substantive
     for every breaker). The pre-registered scores stay primary.
4. **17 Oct 2026: a revised program breaker, `program_v2` (a prospectively specified exploratory extension
   informed by earlier results).** It is separate from the pre-registered H4 experiment, is not part of the
   original confirmatory study, and changes no pre-registered call. CHANGELOG.md R37, R38.
   - *Part A (fixed 17 Oct 2026, before any code was built and before any model call on the evaluation sample).*
     The full text is `paper/CHANGE4.md` as committed in `2ccc2c5` (SHA-256
     `dbbadff83b4fe5f144da7fda8e8094e6aa01a256c6da387fdc1e96216b6f1c5e`). In short:
     - Design: the model resolves what is asked, the unit and the number format (the answer prompt's rule); cites
       every input as a table cell (row and column label), a verbatim quote from the report text, or a verbatim
       quote from the question; writes a DSL program using only cited numbers and constants; or declares
       "unable". Python validates every citation and that every program literal is cited. Outcomes: agree, disagree
       (trips; strict matcher), unable to verify (any failed check, with a reason code; counts as no flag, stays
       in every denominator, rate reported for wrong and right answers).
     - Sample: `configs/change4_sample.txt`, 600 ConvFinQA train conversations (fit sample and every dev filing
       excluded; one per report page; seed 2), SHA-256 `597f495dc8454de850498c1de16efe1ca08cf481832c576cf0f7d84ed38177dd`.
     - Runs: the `free` stage (v2 prompt) for Qwen3 32B and DeepSeek V3, then the frozen program breaker and
       `program_v2` on the same planted, gold and natural items with the same answers and histories.
     - Primary pair per model: natural-error recall and natural false-alarm rate. Primary comparator: the frozen
       breaker rescored with the strict matcher; secondary: its original (lenient) scoring. Paired row
       differences, conversation-level cluster bootstrap, two-sided 95% interval, 2,000 resamples, seed 0.
       Recall improved if the lower bound is above 0. "No materially higher false-alarm rate" if the upper
       bound is below +0.02 (δ = 0.02 absolute); an interval crossing +0.02 is inconclusive. Improved for a model
       only if both hold against the primary comparator; across models only if both models meet it.
     - Failure handling, development limits (dev smoke conversations only, at most 5 prompt versions), the power
       check (a recall gain near +0.05 may be inconclusive; about +0.10 is detectable) and the cost estimate
       (about $16; stop at $40) are as in `paper/CHANGE4.md`.
   - *Part B (frozen 18 Oct 2026, before the first final-evaluation call).* Development used 3 of the 5 allowed
     versions on the 20 dev smoke conversations (`results/change4_dev/`; CHANGELOG.md R39 to R41). The stop rule
     fixed before version 3 was run chose `pv2-dev3`: mean unable rate over both models 0.120, against 0.135 for
     `pv2-dev2`; catch and false-alarm rates on dev were not used. Frozen prompt: `PROGRAM_V2_PROMPTS["pv2-dev3"]`
     in `src/cascade/prompts.py`, version stamp `pv2-dev3`, SHA-256 of the template string
     `6a2249340a2ac19d4a24a16ca81bbd580aba4a0ce819ecb0ee15f4aa05852adf`. The placeholders {context}, {history} and {question} are filled
     as for the frozen program breaker. Verbatim:

       ```
       Here is a company report, a conversation about it, and the latest question.

       REPORT
       {context}

       CONVERSATION SO FAR
       {history}

       LATEST QUESTION
       {question}

       Recompute the answer to the latest question from the REPORT (not from earlier answers), in four steps.

       1. Resolve the question. Write three lines:
       Asked: <what the latest question asks for>
       Unit: <the unit of the answer>
       Format: <decimal ratio, percent number, amount in the report's units, count, or yes/no>
       Format rules: give ratios and percentage changes as decimals (e.g. 0.14218, not 14.218), even if the question says 'percent' or 'percentage', unless the question explicitly asks to multiply by 100 or to convert the value to a percentage. Give amounts in the units the report uses, unless the question asks for another unit.

       2. Cite every number you will use, one per line, in one of these forms:
       Input: table: <row label> | <column label> = <number>
       Input: text: "<exact words from the report text, including the number>" = <number>
       Input: question: "<exact words from the latest question, including the number>" = <number>
       Rules for citations:
       - Row label: the first cell of the table row. Column label: the cell in the table's first row above the number (write - if that cell is empty). Copy both exactly.
       - Numbers in the table are cited as table cells, never as text.
       - Quotes: copy a short passage from one sentence, word for word, with nothing left out or changed (no '...'); the passage must contain the number. Only the latest question may be quoted, and only when it contains the number itself. Earlier questions and answers cannot be cited: recompute any earlier value from the report.
       - Write each number exactly as it appears in its source, before any conversion: 18 for '18 percent', 173.2 for '$ 173.2 million', 40 for '40 cents', -45 for '( 45 )'.

       3. Write a program in this exact syntax: one operation per step, steps separated by commas, each step op(a, b) where a and b are cited numbers, constants (const_100, const_1000, const_1000000) or #n, the result of step n (counting from 0). Allowed operations: add, subtract, multiply, divide, exp, greater. No nested operations and no '#0 ='. Example: subtract(206588, 181001), divide(#0, 181001). Do unit and percent conversions in the program, e.g. divide(18, const_100) or multiply(173.2, const_1000000). For a plain lookup, write just the cited number. The result must already be in the format from step 1.

       4. End with one final line that starts with 'Program:' followed by the program. If the REPORT does not contain the numbers needed, end instead with the line 'Verdict: unable'.
       ```
     After this point neither the prompt, the parser and validation rules, nor the analysis is changed, except for
     code defects handled as Part A's failure rules describe.
5. **18 Oct 2026: post-hoc re-check of the change 4 replies with column label "-" accepted (after the final
   results; not pre-registered; no call).** In the change 4 final evaluation most unable checks were table
   citations that failed to match, and many of those were in ConvFinQA tables whose first row is data, not a
   header: the models wrote "-" for the column, which the frozen prompt allows only for an empty first-row cell.
   The stored replies are checked again with one rule relaxed (a "-" column label is valid when the row label
   matches and the cited number is in that row); no new calls, and the prompt, the other checks, the change 4
   analysis and its calls are unchanged. Reported as a labelled subsection of the change 4 section of
   `results/change4/TABLES.md`. CHANGELOG.md R44, R45.
6. **20 Oct 2026: post-hoc analyses of the history design, the graph and the answer signatures (after all results;
   not pre-registered; no call).** Three analyses of stored replies, in a new section of `results/main/TABLES.md`
   ("Post-hoc analyses of the history design, graph and answer signatures"; code `src/cascade/posthoc_spread.py`):
   (a) own history: each later free-stage turn paired with the same clean-stage turn, grouped by whether the
   model's own earlier answers are all right, include a wrong answer on a turn the graph links to this one, or are
   wrong only on unlinked turns (observational; the free stage is the only sequential rollout, and the inject
   stage keeps intervening answers gold); (b) graph sensitivity: on the 50 audited conversations, dependence and
   path counts rebuilt from each audit's value-parent labels, the cascade split recomputed, and the path model
   compared with the graph-blind model by leave-one-conversation-out Brier (same grid and shrinkage as the frozen
   fit); (c) answer signatures: one signature added to the mechanism rule, before `convention`: the answer equals
   turn k's gold program re-executed with turn j's gold value replaced by the planted value (literal or
   intermediate result), with counts of overlap with `convention` and of `convention` answers offset by the
   planted factor. The pre-registered mechanism counts, H1 to H5 and every earlier number are unchanged; the
   pre-registered categories are reported as answer signatures, not causes. CHANGELOG.md R50.
7. **20 Oct 2026: implementation corrections after an implementation audit (no call; frozen results unchanged).**
   (a) `match.extract_answer` dropped the exponent of an answer in scientific notation ("6.4748e+06" read as 6.4748):
   7 of 58,441 stored answers, 4 correctness outcomes. The parser is corrected for any future run; the frozen result
   rows keep the values parsed when they were run and remain the reported results. A labelled sensitivity analysis
   (`cascade report --config configs/parser_sensitivity.yaml`, `results/parser_sensitivity/`) re-reads every stored
   reply, re-runs every report on corrected copies, and lists each changed number; natural breaker items whose
   checked answer changed are re-scored from stored program values with each breaker's frozen trip rule, and self or
   cross verdicts are kept; a later free turn with a misparsed value in its history cannot be re-asked without a call.
   No call changes. (b) The change 6 reuse signature fixes substitution sites from the original gold computation with
   a near-exact value match (equal after rounding to 5 decimals, or within 1e-6 relative); the first version tested
   already-modified intermediates with the 0.5% matcher. Sites found with the 0.5% matcher are reported as a
   sensitivity. (c) The change 4 quote check removes punctuation before matching, so a sign changed in a quotation
   would pass; the frozen check is not changed, and a post-hoc sign-preserving re-check of every accepted text and
   question citation against its matched source span is added to `results/change4/TABLES.md` (no failures among
   6,705). CHANGELOG.md R53.
