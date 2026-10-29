# DIFFERENCE by the gate — but the viseme reading is not available

Fourth gate of the ICDE loop. `gates_tested` = 4.

## The gate, quoted from `PLAN.md` §4 before any number was read

> Let `D = mean Δ_noise(OTHER) − mean Δ_noise(SV)`, bootstrap 95% CI over pairs.
> - **DIFFERENCE** if the CI for `D` **excludes 0**.
> - **NO DIFFERENCE** if the CI contains 0 **and** its half-width is **< 0.010**.
> - **UNDERPOWERED** if the CI contains 0 **and** its half-width is **≥ 0.010**.
>
> **Confound reported either way:** Spearman ρ between stem-only baseline accuracy and
> Δ_noise across all pairs. If that is strong, any viseme effect may be a difficulty effect
> wearing a label, and the write-up must say so rather than claim the viseme reading.

## Result

r = 0.1611 for every pair, n_train fixed at 5333. 4 same-viseme and 12 other pairs yielded
usable speaker-grouped splits (5 and 15 qualified on sample size; the rest failed the split).

| group | n | mean Δ_noise | mean Δ_shuffle (control) | baseline acc range |
|---|---:|---:|---:|---|
| same-viseme | 4 | **+0.0105** | **+0.0087** [−0.0008, +0.0182] | 0.562 – 0.609 |
| other | 12 | **+0.0261** | −0.0006 [−0.0063, +0.0048] | 0.658 – 0.915 |

`D = +0.0156`, CI **[+0.0041, +0.0285]**, half-width 0.0122.
**ρ(baseline accuracy, Δ_noise) over all 16 pairs = +0.453.**

**Gate verdict: DIFFERENCE.** The CI excludes 0, as written.

## Why that verdict should not be reported as a viseme effect

**1. The groups do not overlap on the confound at all.** Same-viseme baseline accuracies run
0.562–0.609; other pairs run 0.658–0.915. The maximum of one group is below the minimum of
the other. "Same-viseme" and "hard" are the *same variable* in this sample, and ρ = +0.453
says difficulty tracks Δ_noise. The pre-registration committed to this in advance: with a
strong confound the viseme reading is not available. It is not available.

This is not fixable by more pairs. The map holds 10 same-viseme pairs in total and they are
confusable *because* they are hard — the confound is a property of what a viseme is, not of
the sample. Separating the two would need same-viseme pairs that are easy, which do not
exist.

**2. On same-viseme pairs the two nulls agree, which is this project's signature failure.**
Δ_noise = +0.0105 against Δ_shuffle = +0.0087 — nearly the same number. Compare the
overturned claim in `EXPERIMENTS.md`: *`adaptive − permuted` was +0.0042 and meaningless
because `fixed − permuted` was +0.0045, the same value.* When a contrast and its own control
land on top of each other, the contrast is not measuring what it claims.

The defensible statement is therefore **not** "the artefact is smaller on same-viseme pairs"
but: **on same-viseme pairs there is no clear artefact at all, and what little is measured
is not specific to the Gaussian null.** On other pairs the separation is clean — +0.0261
against a control at −0.0006.

**3. The control is marginal in the same-viseme group.** Δ_shuffle [sv] CI is
[−0.0008, +0.0182] — it contains 0 by 0.0008. The per-pair values (−0.0057, +0.0142,
+0.0044, +0.0221) are badly spread. The gate's VOID condition was "CI contains 0", and it
does, so the run is valid as written. But calling that control "clean" would overstate it.

**4. The interval is wide.** [+0.0041, +0.0285] is a 7× range, half-width 0.0122 — above
the 0.010 precision threshold the gate used for its NO DIFFERENCE branch. The DIFFERENCE
branch has no width requirement, correctly as pre-registered, but a reader should see the
width next to the verdict.

## What this means for paper 3 — the reason the test was run

Paper 3's claims are about same-viseme pairs. Three readings, in order of confidence:

- **Supported:** experiment `05`'s Gaussian-null false positives are **least** attributable
  to this artefact on the pairs paper 3 is about. The artefact is concentrated on easier,
  higher-baseline pairs.
- **Supported:** the keystone's +0.023…+0.027 figure, measured on mostly vowel pairs, should
  **not** be transferred to paper 3's pairs. They measure ~+0.010, with a control at the
  same level.
- **Not supported:** that any of this is *about* visemes. It tracks baseline difficulty, and
  the two cannot be separated in this dataset.

## What it means for the ICDE claim

It is a limit, and an honest one: the C6 curve predicts well on the aggregate but the
artefact is **not uniform within a corpus** — it varies with pair difficulty by at least
2.5×, on top of the 5.6× within-r spread already reported in `C6_lrs2/FINDINGS.md`. Any
paper built on r as a predictor has to carry that, and it is a second independent sign that
**r is first-order and not sufficient**.

## Process

- Power ceiling was stated before running: 10 same-viseme pairs exist, 5 qualified, 4 ran.
- The gate's third branch (UNDERPOWERED) was added precisely because of that ceiling, and
  the selftest verified it was live at n = 5 vs 15 (half-width 0.0113). It was not needed —
  the CI excluded 0 — but it existed before the answer did.
- Selftest also showed 1/12 same-population draws return DIFFERENCE, the expected ~5%.
  With a single run at this width, that false-positive rate is worth remembering.
