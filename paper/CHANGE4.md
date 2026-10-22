# PREREGISTRATION change 4: a revised program breaker (final text, 17 Oct 2026)

Status: finalised with the author's decisions of 17 Oct 2026, before any code is built and before any model
call on the evaluation sample. It goes into PREREGISTRATION.md "Changes after freezing" as item 4, in two
parts:
- **Part A** (this document, everything except the prompt text) is fixed now.
- **Part B** (the revised prompt text and its SHA-256) is appended after development on dev conversations and
  before the first final-evaluation call.
Both parts are committed before the final evaluation runs.

**Label in the paper:** "a prospectively specified exploratory extension informed by earlier results".
- It is separate from the pre-registered H4 experiment.
- It is not part of the original confirmatory study, and this amendment does not make it so.
- It changes no pre-registered call.

## 1. Why

- The frozen program breaker compares with the lenient matcher, so it cannot see planted percent errors
  (0.160 as run).
- Strict rescoring raises planted catch (Qwen3 32B 0.767 to 0.944) but also gold false alarms (0.149 to
  0.208).
- The scoring audit (results/main/TABLES.md, "Scoring audit: number format") traces those extra false alarms to
  the program prompt, which states no number format.
- On the model's own errors every breaker is weak, and repair would do net harm for both answering models.
- The revision targets these three points: it resolves the format from the question, cites and validates its
  inputs, and allows an explicit "unable to verify".

## 2. The revised breaker (`program_v2`)

The frozen `program` breaker, its prompt and its stored results are not changed.

### 2.1 Steps asked of the model (one reply)

1. **Resolve the question.** State what is asked, the unit, and the number format. The format follows the
   same rule as the answer prompt: ratios and percentage changes as decimals unless the question asks to
   multiply by 100 or convert to a percentage; amounts in the report's units unless the question asks for
   another unit.
2. **Cite the inputs.** List every number the program uses, each with its source:
   - `table: <row label> | <column label> = <number>`;
   - `text: "<verbatim quote from the report text containing the number>" = <number>`;
   - `question: "<verbatim quote from the latest question>" = <number>`.

   Earlier answers may not be cited: inputs come from the report, as in the frozen breaker. The allowed DSL
   constants (`const_100`, `const_1000`, `const_1000000`, `const_1` and the like) need no citation.
3. **Write the program** in the same DSL as the frozen breaker, using only the cited numbers and constants.
4. **Or declare unable.** If the report does not contain the needed inputs, the model writes
   `Verdict: unable` instead of a program.

### 2.2 Validation in Python, before the program value is accepted

Each citation is checked against its source.

- **Number matching.** A cited number matches when it equals the number as written in the source. `$`,
  commas and `%` are ignored, and parentheses mean negative. The comparison uses the strict matcher.
- **Table citation.**
  - The row label must match the first cell of a table row, and the column label a header cell. Both are
    compared after lower-casing, collapsing whitespace and removing punctuation.
  - The cell at that row and column must match the cited number.
- **Text citation.** The quote, normalised the same way, must be a substring of the report's text (before and
  after the table), and the cited number must appear in the quote. Values from the report prose are allowed,
  so questions answered from the text are not excluded.
- **Question citation.** The same check against the latest question.
- **Coverage.** Every non-constant literal in the executed program must equal a cited number.

Any failed check makes the outcome **unable to verify**. The program value is not accepted. Reason codes are
stored separately, one per check:
- `cell_not_found`, `cell_value_mismatch`;
- `text_quote_not_found`, `number_not_in_quote`;
- `question_quote_not_found`;
- `uncited_literal`;
- `no_program`, `program_error`;
- `model_unable`;
- `truncated`.

### 2.3 Verdict and trip rule

- **Agree:** valid program, and its value matches the checked answer under the strict matcher (the format was
  resolved in step 1).
- **Disagree:** valid program, values differ. The check **trips**.
- **Unable to verify:** any reason in 2.2. For the primary detection metrics, unable counts as **no flag**,
  the same rule as the frozen breaker's unusable checks.
  - Unable cases stay in every denominator.
  - The unable rate is reported separately for wrong and right answers, with reason counts.

### 2.4 Answerers, checkers, settings

- Qwen3 32B and DeepSeek V3, each checking its own answers (as `program_qwen32b` and `program_deepseek` do).
- Same providers as configs/main.yaml, temperature 0, max_tokens 8,192. Qwen3 reasoning settings as in the
  main run.

## 3. Evaluation sample (selected and locked before any model output exists)

- **Source:** the ConvFinQA train split, never used in this study. The optional out-of-split fit
  (`configs/fit.yaml`) was never run.
- **Rule** (`scripts/change4_sample.py`):
  1. eligible conversations: the gold programs reproduce the gold answers and there are at least 3 turns
     (2,289);
  2. leave out the 300 conversations of the fit sample (seed 1), so that check stays possible;
  3. leave out every conversation from a filing (company and year) that appears in the dev split, so no
     underlying annual report is shared with the inspected conversations (1,346 remain, on 831 report pages);
  4. at most one conversation per report page;
  5. a fixed-seed sample of 600 pages, seed 2.
- **Locked file:** `configs/change4_sample.txt`, 600 conversations from 364 filings, SHA-256
  `597f495dc8454de850498c1de16efe1ca08cf481832c576cf0f7d84ed38177dd`. The file and hash are committed with
  Part A. If the file changes, the run is invalid.
- **Size of the planned items:**
  - 2,538 turns;
  - 3,868 planted items (the study's injection plan, seed 0, two types per turn);
  - 1,936 gold items;
  - up to 2,538 natural items (the answerer's own answers; replies with no parseable answer are skipped and
    counted, as in the frozen design).
- **Dev conversations are used only for development and smoke tests** (section 6). No dev result enters the
  final evaluation.

## 4. Runs on the sample

1. `free` stage for each answerer with the v2 answer prompt, unchanged. This gives the fresh own answers and
   histories.
2. The frozen `program` breaker and `program_v2` on the same items: the planted and gold items with gold
   history, and the natural items with the answerer's own history. Same answers, same histories, same items.
3. The strict comparator is obtained by rescoring the frozen breaker's stored values with the strict matcher.
   No extra calls are needed.
4. **No comparison with historical (dev) rates.** Every comparison is paired, on the fresh sample.

## 5. Outcomes, comparisons and success criteria

- **Primary pair, per answering model:**
  - natural-error recall: the trip rate on the answerer's own wrong answers;
  - natural false-alarm rate: the trip rate on its own right answers.
- **Comparators:**
  - **primary:** the frozen program breaker rescored with the strict matcher;
  - **secondary:** the frozen breaker as originally scored (lenient).
- **Analysis.**
  - For each outcome, the estimate is the mean of paired row differences (program_v2 minus comparator; same
    item), pooled over rows.
  - The interval is a two-sided 95% percentile interval from a conversation-level cluster bootstrap (2,000
    resamples, seed 0). This is the study's standard.
- **Recall classification:**
  - *improved* if the lower bound is above 0;
  - *worse* if the upper bound is below 0;
  - otherwise *inconclusive*.
- **False-alarm classification, "no materially higher false-alarm rate" (δ = 0.02 absolute):**
  - *non-inferior* if the upper bound is below +0.02;
  - *materially higher* if the lower bound is above +0.02;
  - otherwise *inconclusive*.
  - An interval that crosses +0.02 is inconclusive and is never reported as non-inferiority.
- **Success criteria.**
  - **Per model:** "improved" only if recall is *improved* and false alarms are *non-inferior* against the
    primary comparator.
  - **Across both models:** improvement is claimed only if both models meet the per-model criterion.
    Otherwise each model's classification is reported as it stands.
  - **Secondary comparator:** the same analysis and classifications, reported without a success call.
- **Secondary outcomes (descriptive, both breakers):**
  - planted catch, overall and by error type;
  - gold false alarms;
  - unable rates by group and reason;
  - the three outcomes (agree, disagree, unable);
  - offline repair (wrong repaired, wrong replaced by another wrong, right corrupted);
  - recall on substantive natural errors (format-only errors removed);
  - the share of citations by source (table, text, question).
- **Sensitivity analyses (reported, no call):**
  - unable counted as a flag;
  - program_v2 scored with the lenient matcher;
  - the bootstrap clustered by filing instead of conversation, since some filings contribute several pages.
- No multiplicity correction. The success criterion is a joint condition, and the whole extension is labelled
  exploratory.

## 6. Development, freeze and procedure

1. **Build** `program_v2`: prompt, citation parser, validation and reason codes, three-way verdict. Add tests,
   and selection of the sample from `configs/change4_sample.txt`. The frozen code is untouched (checked by the
   existing prompt byte-identity test, and by the SHA-256 of the stored breaker files before and after).
2. **Develop** the prompt only on the 20 smoke conversations (`configs/smoke.yaml`, dev, seed 0), at most 5
   versions. Every version and its dev smoke results are kept and listed in the appendix. The last version
   must parse on the dev smoke: every reply yields a verdict, a valid program or a reason code.
3. **Freeze** the prompt.
   - Append it verbatim, with its SHA-256, as Part B of change 4 in PREREGISTRATION.md.
   - Log CHANGELOG and commit (code, tests, config, sample file).
   - Only then make the first call on the sample.
4. **Run the final evaluation once.** No prompt or scoring change afterwards.
5. **Report** with `cascade report`, a new section labelled as this extension.

## 7. Failure handling (fixed in advance)

- **Unable checks and parse failures** are outcomes, not reasons to revise. The prompt is not changed after
  the freeze.
- **Code defects.** These are a crash, or a parser that fails on a format the frozen prompt explicitly asks
  for. They may be fixed, and the affected checks rerun. Each fix is logged with the number of affected rows
  and listed in the paper. A fix may not depend on whether the checked answers are right.
- **Provider errors and rate limits.** These are retried; stages resume by key. A row that still fails after
  retries is rerun later.
- **Model availability.** If a model becomes unavailable on its pinned provider, the run stops and this is
  reported. Any substitute would be a further, labelled change, not a silent swap.
- **Truncation** at max_tokens is recorded as unable (reason `truncated`).
- **Incomplete run.** If the run cannot be completed, no success call is made on a partial sample. What exists
  is reported as incomplete.
- **Budget.** Estimate in section 9. If spending reaches $40, the run stops and the state is reported. The
  sample and criteria are not changed.

## 8. Power check (done before freezing, no model calls)

`scripts/change4_power.py`, output in `results/change4_power.txt`.
- **Method.**
  - It uses the dev cluster structure of the frozen breaker's natural items, with the strict-rescored trips as
    the baseline.
  - A revised breaker is simulated by flipping baseline outcomes, so that the true paired difference takes a
    chosen value.
  - Conversations are resampled to the planned size.
  - 300 simulated studies, each with a 1,000-resample paired cluster bootstrap.
  - Two discordance levels: base (the revised breaker loses 10% of baseline catches and adds alarms on 3% of
    baseline non-alarms) and high (25% and 8%).
- **Results at the planned 600 conversations:**

| | Qwen3 32B, base / high | DeepSeek V3, base / high |
|---|---|---|
| Recall improvement shown, true gain +0.10 | 1.00 / 0.99 | 1.00 / 0.96 |
| Recall improvement shown, true gain +0.05 | 0.90 / 0.55 | 0.80 / 0.53 |
| False alarms non-inferior, true difference 0 | 0.97 / 0.67 | 0.98 / 0.66 |
| False alarms non-inferior, true difference -0.05 | 1.00 / 1.00 | 1.00 / 1.00 |
| False positive call, base (recall gain 0; false-alarm difference +0.02) | 0.04; 0.03 | 0.04; 0.02 |

- **Reading.**
  - The sample supports the criteria for a recall gain of about +0.10 or more, at either discordance level.
  - A gain near +0.05 may well come out inconclusive.
  - Non-inferiority is very likely if program_v2's false alarms are lower than the strict comparator's. The
    format-aware bound suggests they can be much lower (Qwen3 32B 0.059 against 0.192 strict on dev).
  - Non-inferiority is uncertain if the false alarms are equal and many verdicts change.
- **Flagged limitation, not a reason to relax the criteria.**
  - Small true gains may be classified as inconclusive.
  - The pool has 831 eligible pages, so a larger sample with one conversation per page is not possible.
  - The paper states the detectable effect size.

## 9. Cost estimate

Per-call costs are measured from the reply cache: mean cost per 1,000 calls of each kind in the main run. The
revised breaker writes longer replies; it is assumed to cost 1.3 times the frozen breaker for Qwen3 32B (mostly
reasoning tokens already) and 1.6 times for DeepSeek V3.

| Item | Calls | Qwen3 32B | DeepSeek V3 |
|---|---|---|---|
| `free` (answers; $0.307 / $0.200 per 1k) | 2,538 | $0.78 | $0.51 |
| Frozen program breaker ($0.347 / $0.317 per 1k) | 8,342 | $2.89 | $2.64 |
| `program_v2` (1.3x / 1.6x) | 8,342 | $3.76 | $4.23 |
| Final evaluation | 19,222 | $7.43 | $7.38 |
| Development on dev (5 versions x about 270 checks) | about 1,350 | about $0.60 | about $0.70 |

- **Total estimate:** about $16. With a 50% margin, about $24.
- **Stop rule:** $40 (section 7).
- These are estimates. Actual spend is reported.

## 10. What stays the same

- The pre-registered H4 experiment, its scores and its call.
- The frozen breakers and their stored results.
- The exploratory runs of change 2 and the post-hoc analyses of change 3.
- The final evaluation adds new files only. The main report's existing sections are not changed.
