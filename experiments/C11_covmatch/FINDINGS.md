# C11 — VERDICT: **MIXED**. The mechanism holds for weight-fitting readouts and does not fully hold for distance-based ones.

Run 2026-11-05. `gates_tested` becomes **8**; L10's family-wise figure moves from
$1-0.95^7 = 30\%$ to $1-0.95^8 = 33.7\%$.

## 1. The stopping rule, quoted from PLAN.md §5 before any number was read

> - **MECHANISM SUPPORTED** — $\Delta_{\text{rank}}$ CI excludes zero with sign $s$, **and**
>   $\Delta_{\text{cov}}$ CI contains zero, in $\geq 4$ of 6 cells.
> - **MECHANISM REFUTED** — $\Delta_{\text{cov}}$ CI excludes zero with sign $s$ **and**
>   $|\Delta_{\text{cov}}| > |\Delta_{\text{rank}}|$, in $\geq 4$ of 6 cells.
> - **MIXED** — both parts contribute and neither dominates in $\geq 4$ cells.
> - **VOID** — GATE A fails, or fewer than 6 cells produce a usable interval.

**Scored as written: 3 cells SUPPORTED, 1 REFUTED, need 4. Verdict MIXED.**

## 2. GATE A

`d_noise` reproduced all six published cells from `C7_readout/audit_cluster/result_n50.json`
within 0.25 points. The design, splits and seeds are identical to Table II's, so this is the
same experiment with one arm added.

## 3. The numbers

Points of balanced accuracy, 95% cluster bootstrap over 50 seeds, blocks averaged within a
seed first. $\Delta_{\text{noise}} = \Delta_{\text{cov}} + \Delta_{\text{rank}}$ exactly.

| corpus | readout | $\Delta_{\text{noise}}$ | $\Delta_{\text{cov}}$ (survives covariance matching) | $\Delta_{\text{rank}}$ (correlation structure) | verdict |
|---|---|---|---|---|---|
| digits | logistic | $+4.76$ | $-0.42\ [-1.12,+0.28]$ | $\mathbf{+5.18\ [+4.35,+6.04]}$ | SUPPORTED |
| 20news | logistic | $+2.69$ | $+0.20\ [-0.39,+0.80]$ | $\mathbf{+2.49\ [+1.81,+3.15]}$ | SUPPORTED |
| covtype | logistic | $+4.25$ | $+0.05\ [-0.59,+0.68]$ | $\mathbf{+4.20\ [+3.50,+4.93]}$ | SUPPORTED |
| digits | kNN $k{=}5$ | $-10.59$ | $\mathbf{-4.79\ [-5.39,-4.20]}$ | $\mathbf{-5.79\ [-6.52,-5.04]}$ | neither |
| 20news | kNN $k{=}5$ | $-0.20$ | $-0.30\ [-0.84,+0.24]$ | $+0.10\ [-0.66,+0.84]$ | neither (effect itself n.s.) |
| covtype | kNN $k{=}5$ | $-1.45$ | $\mathbf{-0.75\ [-1.38,-0.14]}$ | $-0.70\ [-1.56,+0.18]$ | REFUTED |

**Share of the published effect attributable to correlation structure,
$|\Delta_{\text{rank}}| / |\Delta_{\text{noise}}|$:**

| weight-fitting | | distance-based | |
|---|---|---|---|
| digits / logistic | **109%** | digits / kNN-5 | **55%** |
| covtype / logistic | **99%** | 20news / kNN-5 | **52%** |
| 20news / logistic | **92%** | covtype / kNN-5 | **48%** |

## 4. What this establishes

**For weight-fitting readouts, §IV-F's mechanism is supported cleanly and in all three
corpora.** Matching the comparator's covariance removes the entire effect:
$\Delta_{\text{cov}}$ is indistinguishable from zero in every case ($-0.42$, $+0.20$,
$+0.05$), while $\Delta_{\text{rank}}$ carries 92–109% of it with intervals far from zero.
The account — a full-rank Gaussian block supplies more effectively independent directions than
a real block does, so the Gaussian arm is damaged more — survives its own test.

**For distance-based readouts it does not.** On the paper's largest negative cell
(digits / kNN-5, $-10.59$), **$-4.79$ points survive covariance matching** with a CI nowhere
near zero. Correlation structure accounts for roughly half the effect; the rest comes from the
ways the arms differ that §III-C now reports — discreteness (13.6 distinct values per column
against a continuous Gaussian), skewness up to 8.53, excess kurtosis up to 154.7, and
degenerate columns. §IV-F's distance account (correlated features concentrate pairwise
distances) is at best half the story for this family.

## 5. What this does **not** establish

- **It does not touch §IV's measurement.** $\Delta_{\text{noise}}$ is what a practitioner
  following the recommended practice obtains, whatever causes it. The 23-of-24 result, the
  sign pattern and the decision-rule figures are unchanged. This experiment explains the
  effect; it does not revise it.
- **The verdict is MIXED, not "supported for one family".** The 3/3 split along readout
  family is a **post-hoc reading**: the gate was pre-registered to score six cells pooled, not
  stratified by family. The pattern is striking and follows the paper's existing organising
  axis, but it was not the registered scoring rule and is reported as an observation.
- **It does not isolate which non-covariance property drives the distance residual.**
  Discreteness, tail weight and degeneracy move together across these blocks. Separating them
  needs a marginal-matched (Gaussian-copula) comparator, which was not run.

## 6. Control problems caught during the run

**The reachability selftest failed first, and was right to.** Both branches returned MIXED on
the initial synthetics. Diagnosis: the synthetic blocks were $d=16$ at $n_{\text{train}}=200$,
i.e. $r \approx 0.075$, while the effect under test lives at $r \approx 0.235$. Effects of
0.07–0.33 points against CIs of $\pm0.30$ cannot resolve anything. Rebuilt at $d=64$,
$n_{\text{train}}=205$ ($r \approx 0.237$); both branches then returned their expected verdict.
The guard was right and the synthetic was wrong — the same pattern as C10 v1.

**A degenerate contrast was caught before it ran.** The first selftest duplicated the
`logistic` cell to reach six, which would have let the 4-of-6 threshold be met by copies of a
single measurement. Replaced with both readout families, genuinely measured. This is the
width-router failure mode and it was self-inflicted.

**GATE A crashed on a key that did not exist** (`cluster15`, which lives in the points-valued
working file rather than the published fractions one) after computing a corpus of cells. Fixed,
and a startup assertion added so the same class of failure now costs a second.

## 7. What it means for the paper

1. **§IV-F can be upgraded from a bare hypothesis to a tested one**, with the split reported:
   supported for weight-fitting readouts, not fully for distance-based. That is a stronger and
   more interesting position than "we offer this as a hypothesis".
2. **§III-C's closing sentence is now false** — it says "The control that would separate them
   --- a Gaussian drawn with the block's own covariance --- is not run here." It has been run
   and must be updated.
3. **L10 and §III-F**: `gates_tested` 7 → 8, family-wise 30% → 33.7%.
4. The reviewer objection this was built to answer (R2 and R3 both asked for exactly this
   control) is now answered with a measurement rather than a concession.

## 8. Still worth running

A **marginal-matched (Gaussian-copula) comparator** — match $\hat\Sigma_C$ *and* the per-column
empirical quantiles — would split the distance-based residual into "tail/discreteness" versus
"everything else". Only worth it if the distance half of this result is going into the paper
as a claim rather than as a limitation. Estimated ~5 min, same machinery.
