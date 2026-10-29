# Stop rule — written 2026-10-23, before any candidate was tested

This file is the loop's pre-registration. It exists so that the condition for stopping was
fixed *before* any result existed. A stop rule written after a PASS is not a stop rule.

## The rule

The loop stops when **one** candidate satisfies all three of:

1. **Gate PASS.** Its own `PLAN.md` gate, written before data was touched, returns PASS as
   written. Not softened, not widened, not a secondary result promoted after the primary
   failed.
2. **Control cleared.** The PASS survives the correct null for its comparison type, taken
   from the control catalogue in `CLAUDE.md` — not a null chosen after seeing the result.
   The measured permutation floor from *this* run is recorded, never one quoted from another.
3. **Replicated.** The effect reproduces on a disjoint split or a different seed, with the
   CI reported for the replication separately.

All three, or the loop continues.

## Why it is not "stop on any positive"

The requested rule was "stop when we get a positive result." As written that selects for
false positives. This project reports bootstrap 95% CIs, so each independent gate has about
a 5% chance of passing under the null. A loop that draws candidates until one passes has

    P(at least one spurious PASS) = 1 - 0.95^k

which is 26% by k=6, 51% by k=14, and 76% by k=28. The stopping rule *is* the selection
mechanism. Conditions 2 and 3 above exist to make the FAIL branch reachable after a PASS,
which is the same check this project already applies to individual gates — the first
width-router gate passed spuriously because `adaptive == fixed` made its contrast vacuous.

## Family-wise accounting

`state.json` carries `gates_tested`. Whatever stops this loop is reported as **"PASS #1 of N
gates tested"**, and N goes in the paper. This is not optional bookkeeping; it is the
difference between a result and an artefact of searching.

## What also halts the loop (hand back to the human, do not stop silently)

- A decision that cannot be defaulted without guessing at intent.
- Any GPU work that would require taking a card another user is on. Queue it with
  `09_distributed_evidence/run_when_free.sh` semantics and move to CPU work instead.
- An assembly check against paper 3 failing — the arrays are then not what they claim and
  nothing below that is worth reading.
- The candidate queue emptying with no PASS. Generate new candidates; do not lower the bar.

## Scope note

A PASS here is a positive result for an **IEEE ICDE 2027** submission. ICDE is a data
engineering venue. A VSR accuracy improvement is not in scope there however strong it is;
the contribution has to be data-engineering-shaped. `CANDIDATES.md` holds the live queue
and every candidate's novelty verdict.
