# Does the row-shuffle baseline actually make better KEEP/DROP decisions?

**Pre-registered 2026-10-30, before any graded-signal block was built or scored.**

## 1. Why this exists

An external review made the sharpest technical criticism this project has had:

> Your known null is the real block with rows shuffled. Your recommended baseline
> is the same block with rows shuffled again. These two are draws from the same
> distribution. So `a(C̃) − a(C̃′)` has mean zero by symmetry. "23 of 24 intervals
> contain zero" is close to a tautology. It does not validate the replacement.

**It is correct.** `C̃ = Π₁C` and `C̃′ = Π₂Π₁C` are both uniform row permutations
of the same `C`, scored on the same base, split and labels. They are
exchangeable, so `E[Δ_shuffle] = 0` exactly, by construction.

So `C7`/`C8` establish that the Gaussian baseline **errs**, and in which
direction. They do **not** establish that the row shuffle is a *better
instrument*. That requires measuring decisions, not deltas.

## 2. What is actually in question

A practitioner does not report `Δ`. They decide **keep this block or drop it**.
A baseline is good if that decision is right often.

The shuffle baseline is correctly calibrated at zero information *by the same
symmetry the reviewer identified* — that half is not in question and we will not
claim it as a finding. **What is genuinely unknown is whether it retains power
when the block carries real signal**, because the comparison arm
`shuffle(C_p)` destroys exactly the signal being tested for. If that costs too
much power, a calibrated-but-blind baseline is not an improvement, and the
paper's recommendation fails.

## 3. Construction: graded signal with the geometry held fixed

Take a real block `C` whose rows are paired with their labels. For
`p ∈ {0, 0.25, 0.5, 0.75, 1}`, select a uniformly random fraction `p` of rows and
permute **only those rows among themselves**, leaving the rest in place.

- `p = 0` — the block intact: full label information. Truth is **KEEP**.
- `p = 1` — a known null: zero label information. Truth is **DROP**.
- in between — partial information, decreasing monotonically in `p`.

Every column marginal, the covariance, the spectrum and the effective rank are
**identical at every `p`**, because only rows move. The sole quantity varying is
label information. This is the same device as the known null, graded.

## 4. The two decision rules under test

For each baseline, over 4 blocks × 3 seeds = 12 units, bootstrap a 95% CI and
decide:

| baseline | statistic | KEEP iff |
|---|---|---|
| Gaussian | `a(base ⊕ C_p) − a(base ⊕ G)` | CI excludes 0 **and** lies above it |
| shuffle | `a(base ⊕ C_p) − a(base ⊕ shuffle(C_p))` | CI excludes 0 **and** lies above it |

A *replicate* is one such decision under a fresh master seed. `R = 10`
replicates per (corpus, readout, `p`).

## 5. Metrics, fixed now

- **FPR** = fraction of KEEP decisions at `p = 1` (truth: DROP).
- **Power** = fraction of KEEP decisions at `p = 0` (truth: KEEP).
- The full `P(KEEP)` curve against `p` is reported whatever the verdict.

A baseline is **calibrated** in a cell if `FPR ≤ 0.10` (nominal 0.05, slack for
`R = 10`), and **powered** if `Power ≥ 0.80`.

## 6. Cells

Three readouts × three public corpora = **9 cells**.

| readout | family | why this one |
|---|---|---|
| `logistic` | fits weights | the project's standard; anti-conservative in C7 |
| `knn5` | measures distances | reversed sign in C7/C8 |
| `rf` | **neither** | random forest. ARDA's own RIFS ranks with an ensemble of random forests and sparse regression, so this is the readout the motivating system actually uses, and it belongs to neither family. |

## 7. Gate

- **SHUFFLE BETTER** — shuffle is calibrated **and** powered in ≥ 7 of 9 cells,
  **and** Gaussian fails calibration or power in ≥ 3 of 9 cells.
- **TIE** — both are calibrated and powered in ≥ 7 of 9 cells.
- **SHUFFLE WORSE** — shuffle fails calibration or power in ≥ 3 of 9 cells while
  Gaussian does not.
- **INCONCLUSIVE** — anything else. Reported as inconclusive, not resolved.

**The FAIL branch is the live one.** Shuffle can lose on power: its comparison
arm destroys the very signal under test. If `Power < 0.80` for shuffle in 3+
cells, the paper's recommendation is downgraded from "use this instead" to
"use this when a false positive costs more than a false negative", and
\S IV-C is rewritten accordingly.

**VOID (non-degeneracy)** — power is meaningless if the intact block carries no
detectable signal. For every cell, `a(base ⊕ C_0) − a(base)` must exceed
`+0.01`, asserted **before** any verdict is read. A cell failing this is
reported VOID and excluded, and if fewer than 6 cells survive the whole run is
VOID.

**FAIL-branch reachability** — demonstrated before the verdict is read by
running the identical gate with the "real" block replaced by a known null at
every `p`. Then neither baseline can be powered, the VOID guard should fire, and
the run must not return SHUFFLE BETTER.

## 8. What each outcome does to the paper

- **SHUFFLE BETTER** — \S IV-C stops resting on a tautology and rests on decision
  quality instead. This is the result the recommendation has been missing.
- **TIE** — the recommendation survives on cost (one line, no assumptions) but
  not on accuracy, and must say so.
- **SHUFFLE WORSE / INCONCLUSIVE** — reported. The `Δ_noise` sign finding in
  \S IV is unaffected either way; only the recommendation is at stake.

## 9. Resources

**CPU only, no GPU — rail 1b does not trigger.** Cost is measured by a smoke
test before the full run, not estimated, and reported to the human first.
`OMP_NUM_THREADS=4` on a shared box. Checkpoints per cell.
