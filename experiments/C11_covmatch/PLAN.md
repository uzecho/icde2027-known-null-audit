# C11 — Is the sign flip caused by correlation structure, or by the other ways the arms differ?

Written 2026-11-05, **before any number from `00_run.py` on a real corpus existed**.

## 1. Why this control

§III-C of the paper used to assert that after standardisation a real block and its Gaussian
comparator "differ in exactly one respect: correlation structure between columns, and hence
effective rank". **That sentence was false and has been corrected.** Measured over the same
blocks (`C4_generality/02_moments.py`): the real columns take as few as **13.6** distinct
values where $G$ is continuous, mean sample skewness reaches **8.53**, mean excess kurtosis
**154.7** against $G$'s $\approx 0$, and up to **six** columns per block are constant where
none of $G$'s is.

So the arms differ in correlation structure **and** in discreteness, skewness, kurtosis and
degeneracy. §IV-F's mechanism attributes the sign flip to the first. Nothing in the paper
separates it from the rest, and both simulated reviewers R2 and R3 asked for this control.

## 2. The decomposition

Let $\tilde C$ be the known null, $G \sim \mathcal{N}(0, I)$ the isotropic comparator the paper
uses, and $G' \sim \mathcal{N}(0, \hat\Sigma_{C})$ a comparator drawn with the **block's own
covariance**. Row permutation preserves covariance exactly, so $\hat\Sigma_{\tilde C} =
\hat\Sigma_{C}$ and $G'$ matches the known null in second-moment structure — including its
degenerate columns — while remaining Gaussian in every higher-order marginal respect.

Then, as an identity:

```
Delta_noise = a(C~) - a(G)            <- what the paper reports
Delta_cov   = a(C~) - a(G')           <- what survives once covariance is matched
Delta_rank  = a(G') - a(G)            <- attributable to correlation structure alone
Delta_noise = Delta_cov + Delta_rank  <- exact, by construction
```

The two parts **partition** the published effect. That is what makes this decisive rather
than suggestive: it is not a second opinion, it is an accounting of the first.

## 3. Units and statistics

Unit of analysis is the **seed**, not the (block, seed) pair: a seed fixes the split and the
draw of $G$, and all four blocks share them. 95% **cluster bootstrap over 50 seeds**, blocks
averaged within a seed first, 20 000 resamples. This is the estimator corrected in
`C7_readout/audit_cluster/`; it is reused, not re-derived.

Two readout families at their extremes: `logistic` (fits weights) and `knn5` (measures
distances). Three public corpora, four blocks, `n_train` as in Table II.

## 4. GATE A — reproduction

The run **refuses** unless $\Delta_{\text{noise}}$ reproduces the published 50-seed values in
`C7_readout/audit_cluster/result_n50.json` to within **0.25 points** per cell. The design,
seeds and splits are identical, so a mismatch means this is not the same experiment and
nothing below it is readable.

## 5. The gate, frozen

Scored over the 6 (corpus × family) cells, with $s = \mathrm{sign}(\Delta_{\text{noise}})$ in
that cell:

- **MECHANISM SUPPORTED** — $\Delta_{\text{rank}}$ CI excludes zero with sign $s$, **and**
  $\Delta_{\text{cov}}$ CI contains zero, in $\geq 4$ of 6 cells.
  Correlation structure accounts for the effect; §III-C's old sentence was right in substance
  and §IV-F's mechanism is supported.
- **MECHANISM REFUTED** — $\Delta_{\text{cov}}$ CI excludes zero with sign $s$ **and**
  $|\Delta_{\text{cov}}| > |\Delta_{\text{rank}}|$, in $\geq 4$ of 6 cells.
  The effect survives matching covariance, so it is not driven by correlation structure.
  §IV-F's mechanism must be withdrawn; the measurement in §IV is untouched either way.
- **MIXED** — both parts contribute and neither dominates in $\geq 4$ cells.
- **VOID** — GATE A fails, or fewer than 6 cells produce a usable interval.

Thresholds are **4 of 6**, a proportion of the cells that exist, not an absolute count. The
C9 lesson (L10-i) was a gate whose passing branch was unreachable because its counts were
written against a cell total the guard could reduce.

## 6. Reachability — both branches must be demonstrated

`--selftest` runs two synthetic constructions with known answers and refuses to proceed unless
each returns its expected branch:

- **correlation-driven**: a low-rank block with *Gaussian* marginals. The arms then differ
  only in covariance, so the expected verdict is **SUPPORTED**.
- **marginal-driven**: a full-rank block with heavy-tailed, discretised marginals. The arms
  then differ only in marginal shape, so the expected verdict is **REFUTED**.

If either case fails, the instrument cannot express that branch and the real run does not
start.

## 7. What this does and does not decide

It decides **which difference between the arms carries the published effect**. It does **not**
re-open §IV's measurement: $\Delta_{\text{noise}}$ is what a practitioner following the
recommended practice would obtain, whatever its cause, and the 23-of-24 result stands under
every outcome below.

**Family-wise bookkeeping:** on completion `gates_tested` becomes **8**, and L10's figure
moves from $1 - 0.95^7 = 30\%$ to $1 - 0.95^8 = 33.7\%$. That edit is owed whatever this
returns.

## 8. Cost

CPU only; no GPU, so the GPU approval rail does not apply. Estimated from the measured 50-seed
Table II run (187 s for 24 cells × 3 arms on 3 workers × 8 threads): this adds one arm and
restricts to 2 readouts, so **≈3–6 min**. `--jobs` capped at 3 workers × 8 threads on a
shared box.
