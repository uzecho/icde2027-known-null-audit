# C12 — VERDICT: **MARGINALS DO NOT EXPLAIN**. For distance-based readouts the error is neither correlation structure nor marginal shape.

Run 2026-11-05. `gates_tested` becomes **9**; the multiplicity figure moves from
$1-0.95^8 = 34\%$ to $1-0.95^9 = 37\%$.

## 1. The stopping rule, quoted from PLAN.md §5 before any number was read

> - **MARGINALS EXPLAIN** — $\Delta_{\text{marg}}$ CI contains zero **and**
>   $|\Delta_{\text{marg}}| < |\Delta_{\text{cov}}|$, in $\geq 2$ of 3 cells.
> - **MARGINALS DO NOT EXPLAIN** — $\Delta_{\text{marg}}$ CI excludes zero with sign $s$
>   **and** $|\Delta_{\text{marg}}| \geq 0.5\,|\Delta_{\text{cov}}|$, in $\geq 2$ of 3 cells.
> - **PARTIAL** — anything else.
> - **VOID** — GATE A fails, or the Pearson drift exceeds its measured floor by more than
>   $\max(1.5\times, +0.05)$ in $\geq 2$ of 3 distance cells.

**Scored as written: 2 cells NOT EXPLAIN, 1 EXPLAIN, need 2. Verdict MARGINALS DO NOT
EXPLAIN.**

## 2. GATE A

Both $\Delta_{\text{noise}}$ (against Table II) and $\Delta_{\text{cov}}$ (against C11)
reproduced in all six cells within 0.25 points. Same design, same seeds, one arm added.

## 3. The numbers

Points of balanced accuracy, 95% cluster bootstrap over 50 seeds.

| corpus | readout | $\Delta_{\text{noise}}$ | $\Delta_{\text{cov}}$ (C11) | $\Delta_{\text{marg}}$ (this gate) | cell verdict |
|---|---|---|---|---|---|
| digits | logistic | $+4.76$ | $-0.44$ | $+0.33\ [-0.20,+0.88]$ | — (already $\approx 0$) |
| 20news | logistic | $+2.69$ | $+0.21$ | $+0.25\ [-0.36,+0.87]$ | — |
| covtype | logistic | $+4.24$ | $+0.04$ | $+0.41\ [-0.16,+0.99]$ | — |
| digits | kNN $k{=}5$ | $-10.59$ | $-4.74$ | $\mathbf{-3.71\ [-4.44,-2.97]}$ | **NOT EXPLAIN** |
| 20news | kNN $k{=}5$ | $-0.20$ | $-0.30$ | $+0.26\ [-0.24,+0.76]$ | EXPLAIN (effect itself null) |
| covtype | kNN $k{=}5$ | $-1.45$ | $-0.75$ | $\mathbf{-1.45\ [-2.18,-0.73]}$ | **NOT EXPLAIN** |

**Full decomposition of the paper's largest cell** (digits / kNN-5, $-10.59$ points):

| component | points | share |
|---|---|---|
| correlation structure | $-5.85$ | 55% |
| marginal shape | $-1.03$ | 10% |
| **neither** | $\mathbf{-3.71}$ | **35%** |

## 4. The consistency check passes

On all three weight-fitting cells $\Delta_{\text{marg}}$ sits at $+0.33$, $+0.25$, $+0.41$,
every interval containing zero. C11 had already found $\Delta_{\text{cov}} \approx 0$ there,
and matching marginals as well does not disturb it. **The quantile transform therefore
introduced no artefact**, which is what makes the distance-side reading trustworthy. Marginal
fit is exact in every cell ($\mathrm{KS} = 0.000$).

## 5. The caveat the verdict rests on, stated plainly

The two cells carrying the verdict are **not equally clean**.

| cell | Pearson drift | measured floor | excess |
|---|---|---|---|
| digits / kNN-5 | 0.096 | 0.090 | **$+0.006$** |
| 20news / kNN-5 | 0.092 | 0.052 | $+0.039$ |
| covtype / kNN-5 | 0.127 | 0.048 | **$+0.079$ (2.6$\times$ the floor)** |

For **digits** the marginal-matched arm matches covariance as well as $G'$ does, so
$\Delta_{\text{marg}} = -3.71$ is a clean reading. For **covtype** it does not: $G''$ is 2.6
times further from $C$'s correlation than the floor, so part of its $\Delta_{\text{marg}}$ may
be covariance mismatch rather than marginals failing to explain anything. That cell should be
discounted.

**The gate passed it anyway**, because the VOID condition needs 2 of 3 distance cells over the
bar and only 1 was. That is the gate as written, and the verdict stands as written — but a
per-cell drift bar would have been the better design, and **the verdict effectively rests on
digits alone**. We report it that way rather than claiming two independent cells.

## 6. What this establishes

**For distance-based readouts, neither of the two leading explanations accounts for the
error.** Correlation structure takes 55% of the largest cell (C11), marginal shape takes a
further 10%, and **35% is explained by neither**. §IV-F's mechanism is not merely incomplete
on this family; its natural successor hypothesis — that the residual is marginal shape, the
discreteness and heavy tails measured in §III-C — is now also largely ruled out.

The remaining suspects, named in PLAN.md §5 before the run: **dependence beyond second
moments** (a Gaussian copula has zero tail dependence, and the selftest confirmed that
construction produces exactly this NOT-EXPLAIN signature) and **the off-manifold base pairing**
that L9 concedes and that `paper_improvement/02_statistical_audit.md` F6 shows does not cancel
for $\Delta_{\text{noise}}$.

## 7. What this does **not** establish

- **It does not touch §IV's measurement.** $\Delta_{\text{noise}}$ is unchanged and
  reproduced; the 23-of-24 result and the sign pattern stand.
- **It does not identify the residual**, only narrows it. Tail dependence is a hypothesis
  supported by a synthetic, not a measurement on these corpora.
- **It rests on one clean cell**, for the reason in §5.

## 8. Control problems caught during the run

The selftest **VOIDed on both branches twice** before clearing, and all three causes were the
instrument. All were found on synthetic data with no real corpus touched; PLAN.md §5a records
them.

1. **The drift guard was below the estimation floor.** An absolute bar of 0.25, against a
   measured floor of 0.255–0.257 — a $d\times d$ correlation matrix at this $n$ carries that
   much relative error between two samples of the *same* distribution. The guard was measuring
   noise and nothing could clear it. Re-expressed as an excess over the measured floor. Same
   shape of error as C10 v1's bar sitting below its own chance rate.
2. **$G''$ did not correct its input correlation.** The quantile map preserves ranks, not
   Pearson. Fixed with NORTA; the excess over the floor fell to $\approx 0$ on Gaussian
   marginals (0.100 vs 0.101).
3. **One branch of the gate was unreachable.** The `copula` synthetic linked columns through
   $x \mapsto x^2$, which is not monotone, so the pair's rank correlation is $\approx 0$ and
   the dependence was invisible to Pearson *and* rank — $G''$ reproduced it by accident and
   the branch returned `EXPLAIN`. **The gate could not express one of its own verdicts.**
   Replaced with $t$-copula dependence under Gaussian marginals.

## 9. What it means for the paper

§IV-F currently says the marginal differences of §III-C "are the suspects, and this design does
not separate them." **They are now separated, and they are largely not the answer.** The
sharpened statement: for weight-fitting readouts the correlation account is complete; for
distance-based readouts neither correlation nor marginals explains a third of the largest
effect in the paper.

`gates_tested` 8 → 9; multiplicity 34% → 37%.

## 10. Still worth running

A **$t$-copula comparator** — match marginals, match Pearson, and add tail dependence — would
test the surviving hypothesis directly. The machinery exists (the selftest already builds one).
Estimated ~5 min. Worth it only if the distance-side residual is going into the paper as a
claim rather than as an open question.
