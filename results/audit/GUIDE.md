# Dependency-graph audit: labelling guide

The sheet `graph_audit.csv` has 219 turns from 50 dev conversations (sampled with seed 7 by
`cascade audit-graph`). `parents_found` is what the automatic graph (`deps.py`) says. Fill in the two
columns for **every** turn without changing anything else. `cascade report` scores the sheet once
every turn has a value in `value_parents`.

Judge from the conversation as a reader would: the question's wording and what it asks for. Use the
gold program only to see which numbers are involved. Do not copy `parents_found`; the point is to
check it.

## Columns

**`value_parents (fill in)`**: the earlier turns whose *answer, the number itself*, this turn needs.
If that earlier answer were wrong, this turn's answer would be wrong too. Turn numbers start at 0;
separate them with spaces (`0 2`). Write `-` if there are none. Turn 0 is always `-`.

**`reference_parents (fill in)`**: the earlier turns whose *wording* this question points back to
without using their number: a pronoun, "this/that/the same", or a left-out subject that you can only
fill in from an earlier question ("this average volume for JCB", "what was it in 2016?"). List the
earlier turn(s) you would need to read to understand the question; a turn that only asked the same
thing before is not needed. Test: could you answer the question correctly from the
question and the report alone, without reading the earlier question? If yes, `-`. So a fully stated
question has `-` even if it opens with a conversational "what about..." (HII/2017 t1) or is about a
similar quantity. Write `-` if there are none. A turn listed in `value_parents` is not
repeated here.

**`note`**: optional, for anything unclear. Notes are kept and can be quoted as examples.

## Rules

0. **The test for value parents:** if *only* turn j's answer were wrong (every other earlier answer shown
   correctly), would turn k's answer change? Only then is j a value parent. This mirrors the experiment,
   which corrupts one answer at a time; a turn that matters only through another turn's result is not a
   value parent (GS/2017 t4 uses turn 2's change, so turn 0 is not listed). Dependence runs from an earlier answer into a later question. Sharing an
   *input* is not dependence: in HII/2017, 448,859 is an ingredient of turn 0's total (448,859 +
   4,087,587), but turn 1 reads 448,859 from the table, so turn 1 has no value parent.
1. **Using a number counts; re-looking it up does not.** If turn k asks for a value again ("what was
   the 2016 value?") and the answer equals an earlier turn's answer, turn k is a fresh lookup:
   `value_parents = -`. A later turn that then uses this number depends on the turns it draws on.
2. **The same number given twice.** If the number turn k uses was answered in two earlier turns
   (e.g. a lookup and a repeat lookup), list both. An error in either could reach turn k.
3. **Carried-along program steps do not count.** ConvFinQA programs often repeat earlier steps
   that the question does not need. Only list turns whose answers the question actually needs.
4. **Pronouns and "this/that".** "What proportion does *this* represent?" uses the number that
   "this" points to: a value parent.
5. **Coincidental equal values are not parents.** For example, a constant like 100 in a stock
   return table that happens to equal an earlier answer.
6. If you cannot decide, pick your best reading and write why in `note`.
7. **Reading programs.** ConvFinQA writes every program in full and never refers to earlier turns, even
   when the question uses them (HII/2017 t2 recomputes turn 0's total with `add(448859, 4087587)`). So a
   program that does not mention an earlier turn is no evidence of independence. `#0`, `#1` refer to
   steps of the same program, not to turns 0, 1. Decide from the question first; use the program only to
   see where numbers come from.

## Worked examples from the sheet

| Conversation, turn | Question (short) | `parents_found` | value | reference | Why |
|---|---|---|---|---|---|
| HII/2017 t2 | what proportion does this represent? | 0 1 | `0 1` | `-` | "this" is turn 1's count; the total is turn 0's answer |
| GS/2017 t1 | what was it in 2016? | (none) | `-` | `0` | fresh lookup; "it" is the quantity named in turn 0 |
| GS/2017 t3 | what was the 2016 value? | 1 | `-` | `0` | fresh lookup of the same number (rule 1); "the 2016 value" of credit net, which only turn 0 names (turn 1 just asked the same thing before) |
| GS/2017 t4 | what was the change over the 2016 value? | 0 1 2 3 | `1 2 3` | `0` | uses the change (turn 2) and the 2016 value (turns 1 and 3, rule 2); turn 0's number only via turn 2, but its wording (credit net) is needed to understand the question |
| LKQ/2016 t2 | change in the S&P 500 return 2011 to 2016? | 0 1 | `-` | `-` | new lookup and subtraction; the LKQ steps are only carried along (rule 3); the question is fully stated |

GS t4 shows both columns at work: turn 0's number reaches t4 only through turn 2's answer (so it is not a
value parent, rule 0), but turn 0's wording is still needed to know the question is about credit net.

## What the scores mean

- **Precision**: of the edges the graph found, the share you confirmed in `value_parents`.
- **Recall**: of your `value_parents` edges, the share the graph found.
- **Reference-only links**: counted separately. The graph is defined by values, so these are outside
  its scope. Their frequency is reported because the smoke run showed errors can travel through them
  (CHANGELOG.md R16, V/2008).

Both intervals resample conversations (bootstrap, 2,000 draws).
