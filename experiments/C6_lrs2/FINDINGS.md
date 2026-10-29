# CONFIRMED — a curve fitted on vision, text and tabular data predicts LRS2 out of sample

Third gate of the ICDE loop. `gates_tested` = 3. **This does not stop the loop**: see §6.

## The gate, quoted from `PLAN.md` §3 before any number was read

> **CONFIRMED** requires both:
> 1. **Point prediction.** At r within ±10% of 0.1611, the measured `Δ_noise` bootstrap 95%
>    CI **overlaps** the pre-registered interval **[+0.0149, +0.0238]**.
> 2. **Control.** `Δ_shuffle` bootstrap 95% CI **contains 0**.

The interval was fitted on five external points — covtype, 20news and digits — and frozen
before any LRS2 accuracy was read. It was never adjusted.

## Result

Ten phone pairs, `n_train` fixed at exactly 5333 so every pair sits on r = 0.1611, base =
the 512-d frozen stem, candidate = `crop_px` (1024 cols), known-null = that block with rows
shuffled. Three independent pair selections:

| pair selection | realised r | **measured Δ_noise** [95% CI] | control Δ_shuffle [95% CI] | verdict |
|---|---:|---|---|---|
| first 10 in phone order | 0.1611 | **+0.0274** [+0.0200, +0.0350] | −0.0006 [−0.0037, +0.0024] | CONFIRMED |
| random, seed 11 | 0.1611 | **+0.0267** [+0.0205, +0.0324] | −0.0009 [−0.0060, +0.0042] | CONFIRMED |
| random, seed 22 | 0.1610 | **+0.0233** [+0.0169, +0.0292] | −0.0012 [−0.0042, +0.0015] | CONFIRMED |

Predicted point: **+0.0167**. Predicted interval: **[+0.0149, +0.0238]**.

A curve fitted on three non-speech corpora predicts, out of sample, the magnitude of a
probing artefact on a visual-speech corpus it never saw — to within overlapping uncertainty,
three times, under two random pair draws. The control is clean in all three: the
own-rows-shuffled null sits at zero while the Gaussian null does not.

## The three caveats that belong next to the headline, not below it

**1. The curve under-predicts by roughly half.** Point estimates are +0.0274 / +0.0267 /
+0.0233 against a predicted +0.0167 — about 55% high on average, and in two of three runs
the measured point estimate sits *above* the predicted interval's upper bound. The gate
asked for CI overlap and got it, but "the prediction was right" overstates this. LRS2's
artefact is systematically **larger** than the external curve says it should be. The obvious
suspect is base dimensionality: 512 here against 16 in every external dataset. **r transfers
as a first-order predictor and is not sufficient.**

**2. Pair-to-pair variance at fixed r is large and unexplained.** At identical r = 0.1611,
Δ_noise runs from **+0.0082** (AA-S) to **+0.0461** (AA-T) — a 5.6× spread with r held
constant by construction. Whatever else governs the artefact is of the same order as r
itself.

**3. Same-viseme pairs may sit systematically lower.** The voided attempt 1 used
same-viseme pairs; its two cells at r ≈ 0.153 gave −0.0004 and +0.0107, well below the
+0.023…+0.027 seen here on mostly vowel pairs. If that holds up it matters for paper 3,
because the confusable pairs paper 3 cares about are exactly the same-viseme ones. **Not
tested directly and not claimed** — it is a difference between a voided run and a valid one,
which is not a comparison. It is the obvious next measurement.

## Process notes that must travel with this result

- **Attempt 1 VOIDed on its own guard.** It assumed every pair reaches experiment 05's
  `PER_CLASS` cap of 8000; only 2 of 8 did, so the realised mean r was 0.2903 rather than
  0.1611 and the guard refused a verdict. Third time a pre-registered guard has caught a
  construction error on this loop before it became a result.
- **Blindness was compromised before attempt 2.** Attempt 1 printed per-pair Δ values before
  voiding. What protects the test is that the prediction interval was frozen from external
  data and could not be moved; only the sampling was corrected. Any write-up must disclose
  this.
- **Pair selection was not pre-registered.** Attempt 2 took the first ten qualifying pairs
  in phone order. That was caught after the fact and tested directly with two random draws
  from the 157 qualifying pairs; the result is robust to it. Reported because it was a real
  hole, not because it changed the answer.
- `CLAUDE.md`'s "noise ~1024" is the nominal width. At n_train = 5333 the effective rank is
  **859**, giving r = 0.1611 rather than the ≈0.19 estimated from the catalogue.

## What this does and does not establish

- **Does:** the Gaussian-null artefact is real on LRS2 at the regime experiment `05` used,
  is specific to that null (control clean), and its magnitude is predictable to within a
  factor of ~1.5 from data in entirely different domains.
- **Does:** retro-explain `05`'s three false positives quantitatively.
- **Does not:** establish r as sufficient. Two independent signs say it is not — the
  systematic under-prediction and the 5.6× within-r spread.
- **Does not:** rescue C6's failed monotonicity replication. That stands.
- **Does not:** extend to non-linear probes or other readouts.

## 6. Why the loop does not stop here

`STOP_RULE.md` requires the stopping candidate to clear **all three**: gate PASS, control
cleared, and replication. C6 stands at **PASS / CLEARED / FAILED** — the monotonicity
replication failed on a disjoint half and that has not changed.

This keystone is strong evidence, and it is tempting to count it as the replication. It is
not: it reproduces a *point prediction on a new corpus*, while what failed was *monotone
dependence on r*. Retrofitting one as the other would be exactly the move the stop rule
exists to prevent.

The honest position: the claim that survives everything measured is narrower than C6's
original — **"r predicts artefact magnitude to first order across domains, with a positive
residual in high-base-dimension regimes"** — and it was formed *after* seeing results. Under
this project's own rules it needs its own pre-registered test on fresh data before it can be
a headline. `gates_tested` = 3, and that number goes in any paper.
