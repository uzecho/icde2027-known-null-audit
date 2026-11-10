# INCONCLUSIVE — but the question the experiment was built to ask is answered

Seventh gate of the ICDE loop. `gates_tested` = 7.

## The gate, quoted from `PLAN.md` §7 before any number was read

> - **SHUFFLE BETTER** — shuffle is calibrated **and** powered in ≥ 7 of 9 cells,
>   **and** Gaussian fails calibration or power in ≥ 3 of 9 cells.
> - **TIE** — both are calibrated and powered in ≥ 7 of 9 cells.
> - **SHUFFLE WORSE** — shuffle fails in ≥ 3 of 9 while Gaussian does not.
> - **INCONCLUSIVE** — anything else.
>
> Calibrated: `FPR ≤ 0.10` at `p = 1`. Powered: `Power ≥ 0.80` at `p = 0`.

**Verdict: INCONCLUSIVE.** 3 of 9 cells voided on the non-degeneracy guard,
leaving 6; of those, shuffle was calibrated-and-powered in 4 and Gaussian in 3.

## Result

`P(KEEP)` over 10 replicates per level. Truth is KEEP at `p = 0` and DROP at
`p = 1`.

| cell | | p=0 | p=0.25 | p=0.5 | p=0.75 | **p=1** |
|---|---|---|---|---|---|---|
| digits · logistic | Gauss | 10/10 | — | — | — | **10/10** |
| | shuffle | 10/10 | — | — | — | **1/10** |
| 20news · logistic | Gauss | 10/10 | 10/10 | 10/10 | 8/10 | **6/10** |
| | shuffle | 10/10 | 10/10 | 6/10 | 4/10 | **1/10** |
| covtype · logistic | Gauss | 10/10 | 10/10 | 10/10 | 9/10 | **10/10** |
| | shuffle | 10/10 | 9/10 | 6/10 | 1/10 | **3/10** |
| covtype · knn5 | Gauss | 10/10 | 10/10 | 4/10 | 1/10 | **0/10** |
| | shuffle | 10/10 | 10/10 | 8/10 | 3/10 | **0/10** |
| digits · rf | Gauss | 10/10 | — | — | — | **1/10** |
| | shuffle | 10/10 | — | — | — | **3/10** |
| covtype · rf | Gauss | 10/10 | 10/10 | 9/10 | 4/10 | **1/10** |
| | shuffle | 10/10 | 9/10 | 6/10 | 2/10 | **1/10** |

Summarised against the pre-registered bars:

| cell | Gauss FPR | Gauss power | ok | shuffle FPR | shuffle power | ok |
|---|---:|---:|:--:|---:|---:|:--:|
| digits · logistic   | **1.00** | 1.00 | ✗ | 0.10 | 1.00 | ✓ |
| 20news · logistic   | **0.60** | 1.00 | ✗ | 0.10 | 1.00 | ✓ |
| covtype · logistic  | **1.00** | 1.00 | ✗ | 0.30 | 1.00 | ✗ |
| covtype · knn5      | 0.00 | 1.00 | ✓ | 0.00 | 1.00 | ✓ |
| digits · rf         | 0.10 | 1.00 | ✓ | 0.30 | 1.00 | ✗ |
| covtype · rf        | 0.10 | 1.00 | ✓ | 0.10 | 1.00 | ✓ |

## The live question from `PLAN.md` §2 is answered, and favourably

The plan said the genuinely unknown part was **power**, because the shuffle
baseline's comparison arm destroys the very signal under test:

> If that costs too much power, a calibrated-but-blind baseline is not an
> improvement, and the paper's recommendation fails.

**It does not cost power. Both baselines detect the intact block in 10 of 10
replicates, in all six surviving cells.** The pre-registered failure mode for
the recommendation did not occur.

## What separates them is the false-positive rate, on weight-fitting readouts

Not the verdict — reported as an observation, because the gate says
INCONCLUSIVE and the gate was written first.

On `logistic`, the Gaussian baseline recommends **keeping a block that carries
zero label information** at a rate of 1.00, 0.60 and 1.00 across the three
corpora. The shuffle baseline on the same cells: 0.10, 0.10, 0.30. On `rf` and
`knn5` the two are close and both near nominal.

That is the sign finding of `C7`/`C8` expressed as decisions rather than deltas,
and it is the first evidence in this project that the difference changes what a
practitioner would *do*.

## A gate-design error, found after the run

**The PASS branch was unreachable once three cells voided.** The thresholds were
written as absolute counts out of nine (≥ 7 good, ≥ 3 bad) while the VOID guard
allowed the run to proceed on as few as six surviving cells. With six alive, no
outcome could reach 7, so `SHUFFLE BETTER` and `TIE` were both impossible before
any data was seen.

`CLAUDE.md` requires checking that the FAIL branch is reachable. It was, and the
selftest demonstrated it. **I did not check that the PASS branch stays reachable
under the realised VOID count**, which is the same class of error.

**It did not change the answer.** Re-scoring with the thresholds as the same
*proportions* of the surviving cells (⌈7/9 × 6⌉ = 5 good, ⌈3/9 × 6⌉ = 2 bad):
shuffle good 4 < 5, Gaussian good 3 < 5, shuffle bad 2 < 3 — **still
INCONCLUSIVE**. The verdict is robust to the flaw, which is the only reason it
can be reported at all rather than voided outright.

## Two further honest notes

**The shuffle baseline is not perfectly calibrated in practice.** Its FPR
exceeds the 0.10 bar in two cells (0.30 twice), despite `E[Δ_shuffle] = 0`
holding exactly by symmetry. The likely cause is percentile-bootstrap
undercoverage at 12 units, which affects **both** arms equally — so the
*difference* between arms is interpretable while the absolute level is not.
A larger unit count, or a BCa interval, would test this.

**Three cells voided because the intact block carried no detectable signal**
for that readout: `digits·knn5` (−0.0003), `20news·knn5` (−0.0121),
`20news·rf` (+0.0007). Appending 64 raw pixel columns to a 16-d PCA base does
not help kNN, which already reaches 95.4% on the base alone. The guard did its
job; the cost is that the design lost a third of its cells and with them the
ability to reach a positive verdict.

## What this does to the paper

- §IV-C **cannot** be rewritten to rest on decision quality yet. The pre-registered
  verdict is INCONCLUSIVE and must be reported as such.
- The reviewer's criticism stands and is now sharper: the control is
  tautological, *and* a direct test of the replacement did not clear its bar.
- What can be added honestly: the recommendation **does not cost power**, and on
  weight-fitting readouts the Gaussian baseline's false-positive rate reaches
  1.00 in two of three corpora. Both are statements about measured quantities,
  neither is the gate's verdict.

## What is worth running next

A re-run with the design errors fixed, **pre-registered before it runs** and
disclosing that blindness is already compromised (as `C6_lrs2` did):

1. thresholds as **proportions of surviving cells**, not absolute counts;
2. readout/corpus pairs chosen so the intact block has signal **for that
   readout** — screen on `a(base ⊕ C) − a(base)` before the cell enters;
3. more units per decision (24 rather than 12) to reduce bootstrap undercoverage;
4. finer `p` grid near the boundary, where the curves actually separate.

That is a different experiment from this one and must be counted as another
gate, not as a rerun of this one.
