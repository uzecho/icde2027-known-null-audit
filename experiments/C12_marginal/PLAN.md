# C12 — What accounts for the distance-side residual C11 left open?

Written 2026-11-05, **before any number from `00_run.py` on a real corpus existed**.

## 1. The question C11 handed over

C11 partitioned the published effect exactly:

```
Delta_noise = [a(C~) - a(G')] + [a(G') - a(G)]
            =   Delta_cov     +   Delta_rank
```

For weight-fitting readouts `Delta_cov` vanished ($-0.4$, $+0.2$, $+0.1$ points, every
interval containing zero) and correlation structure carried 92–109% of the effect. **For
distance-based readouts it carried only 48–55%**, and on the paper's largest negative cell
($k$NN $k{=}5$ on digits, $-10.6$) **$-4.8\ [-5.4,-4.2]$ points survived covariance
matching**. C11 §5 states plainly that it cannot say what produces that residual: discreteness,
tail weight and degeneracy move together across these blocks.

This gate adds the next rung of the ladder and asks whether **marginal shape** is the answer.

## 2. The third comparator

| arm | construction | matches the known null in |
|---|---|---|
| $G$ | $\mathcal{N}(0, I)$ | shape only |
| $G'$ | $\mathcal{N}(0, \hat\Sigma_C)$ | + covariance, spectrum, effective rank |
| $G''$ | $G'$ pushed through each column's empirical quantile function | **+ every marginal distribution, exactly** |

$G''$ is built by replacing each column of $G'$ with the empirical quantiles of the
corresponding column of $C$, taken at $G'$'s own ranks. A per-column monotone map preserves
ranks, so $G''$ inherits $G'$'s **rank** correlation structure while acquiring $C$'s marginals
**exactly** (up to ties).

**What this does not do, stated up front.** The transform is nonlinear, so $G''$'s **Pearson**
correlation is not $\hat\Sigma_C$. The run therefore reports two fit diagnostics per cell:
per-column KS distance to $C$'s marginals (expected $\approx 0$) and relative Frobenius error
of $\mathrm{corr}(G'')$ against $\mathrm{corr}(C)$ (expected $> 0$). **A large Pearson drift is
a confound and must be read as one**, not as evidence about marginals.

New quantity: $\Delta_{\text{marg}} = a(\tilde C) - a(G'')$ — what survives matching covariance
**and** marginals.

## 3. Units and statistics

Identical to C11: unit is the **seed**; 95% cluster bootstrap over **50 seeds**, blocks
averaged within a seed first, 20 000 resamples. Two readouts at the family extremes
(`logistic`, `knn5`), three public corpora, four blocks, `n_train` as in Table II.
`n_seeds = 50` is fixed here and **not revisited after seeing the result** (the L10-iii
lesson).

## 4. GATE A — reproduction

The run **refuses** unless, per cell, $\Delta_{\text{noise}}$ reproduces
`C7_readout/audit_cluster/result_n50.json` **and** $\Delta_{\text{cov}}$ reproduces
`C11_covmatch/result.json`, each to within **0.25 points**. Same design, same seeds, one arm
added; a mismatch means this is not the same experiment.

## 5. The gate, frozen

Scored on the **three distance-based cells**, which is where C11 found the residual. Let
$s = \mathrm{sign}(\Delta_{\text{cov}})$ in that cell.

- **MARGINALS EXPLAIN** — $\Delta_{\text{marg}}$ CI contains zero **and**
  $|\Delta_{\text{marg}}| < |\Delta_{\text{cov}}|$, in $\geq 2$ of 3 cells.
  Matching marginals removes what matching covariance did not: the residual is marginal shape.
- **MARGINALS DO NOT EXPLAIN** — $\Delta_{\text{marg}}$ CI excludes zero with sign $s$ **and**
  $|\Delta_{\text{marg}}| \geq 0.5\,|\Delta_{\text{cov}}|$, in $\geq 2$ of 3 cells.
  The residual survives matching both covariance and marginals, so it is neither — the
  remaining suspects are dependence beyond second moments and the off-manifold base pairing
  that L9 concedes.
- **PARTIAL** — anything else.
- **VOID** — GATE A fails, or the Pearson drift diagnostic exceeds its **measured floor** by
  more than $\max(1.5\times, +0.05)$ in $\geq 2$ of 3 distance cells, since the arm would then
  not be the comparator this plan describes.

### 5a. Two repairs made 2026-11-05, on synthetic data only, before any real corpus was touched

The §6 selftest **VOIDed both branches** on the first two attempts. Both failures were the
instrument, not the hypothesis, and both are recorded here rather than quietly fixed.

**(i) The drift guard was measuring noise.** The original bar was an absolute $0.25$. A
$d\times d$ correlation matrix estimated at this $n$ carries roughly that much relative
Frobenius error between two samples of the *same* distribution — with $d=64$, $n=900$, each
off-diagonal has SE $\approx 0.033$ over $\approx 4000$ entries — so the guard fired on
sampling error and no construction could ever clear it. **Measured floor: $0.255$–$0.257$,
against drifts of $0.258$–$0.311$.** $G'$ has $C$'s covariance by construction, so
$\mathrm{drift}(C, G')$ *is* that floor; the bar is now on the **excess over it**, which also
self-calibrates across corpora whose $n$ and $d$ differ by an order of magnitude. This follows
the project's standing rule to report the floor one measured rather than quote a number.

**(ii) $G''$ was built without correcting the input correlation.** Feeding the quantile map a
Gaussian drawn at $\mathrm{corr}(C)$ preserves *ranks*, not Pearson, so matching marginals cost
the covariance match. Now solved by **NORTA** (Cario \& Nelson): a fixed point for the input
Gaussian correlation whose *output*, after the transform, carries $C$'s Pearson correlation,
solved once per block. After the fix the excess over the floor is $\approx 0$ on
Gaussian-marginal synthetics ($0.100$ vs $0.101$) and $0.03$ on heavy-tailed ones. NORTA has a
known defect — some targets are unreachable for given marginals — and if that occurs the drift
diagnostic stays high and the gate VOIDs, which is the honest outcome.

**(iii) One reachability synthetic was wrong.** The `copula` case originally linked column
pairs through $x \mapsto x^2$, which is **not monotone**, so the pair's rank correlation is
$\approx 0$, the dependence was invisible to Pearson *and* to rank, and $G''$ reproduced it by
accident — the branch returned `EXPLAIN` when it should have returned the opposite. Replaced
with **$t$-copula dependence under Gaussian marginals**: marginals matchable exactly, Pearson
matchable, but a Gaussian copula has zero tail dependence, so the tail clustering a
distance-based readout responds to cannot be reproduced. Both branches now return their
expected verdict.

**Consistency check, reported but not gated:** on the three weight-fitting cells C11 already
found $\Delta_{\text{cov}} \approx 0$, so $\Delta_{\text{marg}}$ should also sit near zero. If
it does not, the quantile transform has introduced an artefact and the distance-side reading
is not trustworthy. This is a guard on the instrument, not a test of the hypothesis.

## 6. Reachability — both branches demonstrated

`--selftest`, at the regime the effect lives in ($d{=}64$, $n_{\text{train}}{=}205$,
$r \approx 0.237$ — the correction C11's selftest forced):

- **marginal-driven**: heavy-tailed, discretised columns, near-isotropic covariance. Matching
  marginals should remove the effect. Expected **MARGINALS EXPLAIN**.
- **copula-driven**: columns with strong *nonlinear* dependence that a rank-preserving
  transform of a Gaussian cannot reproduce. Expected **MARGINALS DO NOT EXPLAIN**.

The run refuses to proceed unless each returns its expected branch.

## 7. Bookkeeping

On completion `gates_tested` becomes **9**, and the multiplicity figure moves from
$1-0.95^8 = 34\%$ to $1-0.95^9 = 37\%$. That edit is owed whatever this returns.

## 8. Cost

CPU only; no GPU. One arm more than C11, which measured **76 s** on a quiet box at 2 workers.
**The box is currently at load ~90 with other users holding ~32 of 32 cores**, so this will be
much slower in wall-clock. `--jobs 2`, `nice -n 19`, single-threaded BLAS — `logistic` and
`knn5` are threading-invariant (verified this session), so the footprint does not move the
numbers and GATE A checks that.
