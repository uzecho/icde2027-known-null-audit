# Audit: Table II's twelve units are three

**Not a gate.** No pre-registration, because this is an audit of an existing result rather
than a new experiment. Stated plainly rather than dressed as a gate it never had.

## The defect

`C7_readout/00_run.py:86-92`, reused verbatim by `C8_mechanism/00_run.py:60`:

```python
for _, C in blocks.items():
    for seed in seeds:
        rng = np.random.default_rng(seed)     # <- seed only, not (block, seed)
        idx = rng.permutation(len(y))
        ...
        G = rng.standard_normal(C.shape)      # <- same shape for every block
```

The split is drawn from `seed` alone, so within a seed **all four blocks share the same
training rows**. And because every block is kept at the corpus width (§III of the paper),
`C.shape` is identical across blocks, so `G` is the **same array** for all four. Since
`d_noise = a(C~) - a(G)`, the four units of a seed share their subtrahend exactly.

The paper (§III-E) says "the unit of analysis is a (block, seed) pair ... twelve units per
cell" and bootstraps those twelve as independent. The independent unit is the seed. **n is
3, not 12.** This is the same pseudo-replication already recorded for C4 in
`C10_rifs/PLAN.md` §2 — it was not noticed that C7 and C8 inherit it.

## What it costs

`cluster_audit.py` reuses `c7.cells()` verbatim, so every point estimate reproduces
exactly; only the aggregation changes.

| | resolving cells | note |
|---|---|---|
| naive bootstrap over 12 pairs (published) | **18 / 24** | |
| cluster bootstrap over 3 seeds | **16 / 24** | median CI inflation ×1.38 |

Two cells stop resolving, both weight-fitting on covtype:

| cell | published | clustered |
|---|---|---|
| covtype / logistic | $+4.45\ [+2.02,+6.68]$ | $+4.45\ [-0.88,+7.27]$ |
| covtype / linsvc | $+4.23\ [+1.46,+6.78]$ | $+4.23\ [-2.10,+8.07]$ |

**What survives untouched:**

- **No cell resolves against its family** under clustering either. The paper's central
  claim is not overturned.
- **9 of 9 distance-based cells still resolve negative.** The strongest sub-claim is
  unaffected.
- Both extremes survive: linsvc/digits $+7.54\ [+6.20,+9.29]$, knn5/digits
  $-8.53\ [-12.80,-3.95]$. The $16$-point gap in the Conclusion stands.

## The corrected measurement (50 seeds, clustered over seeds)

`15seed_rerun.py` with `NSEEDS=50`. GATE A asserts that seeds 0-2 reproduce Table II as
printed before anything else is read; all 24 cells clear it at 0.06 pts, which is the
rounding bound of a one-decimal table. Design unchanged; only `seeds` and the
aggregation differ.

| | published (3 seeds, naive) | corrected (50 seeds, clustered) |
|---|---|---|
| cells resolving | 18 / 24 | **23 / 24** |
| resolving against family | 0 | **0** |
| weight-fitting as implied | 9 / 15 | **15 / 15** |
| distance-based as implied | 9 / 9 | **8 / 9** |
| own-shuffle control dirty | 1 / 24 (centroid/20news) | 1 / 24 (knn5/covtype) |
| largest $+$ / largest $-$ | $+7.5$ / $-8.5$ | $+5.23$ / $-10.89$ |
| gap between extremes | 16 pts | **16 pts** |

The published point estimates carried a standard error of **~1.7 points** (per-seed
sd 2.7-3.4 over 3 seeds) while the table prints half-widths of 1.0-1.7. The naive
12-unit bootstrap understated the interval by roughly $4\times$. At 50 seeds the median
SE is 0.42.

Two substantive reversals, both against earlier text:

- **20news is not a null corpus.** All five weight-fitting readouts now resolve positive
  ($+2.69, +2.13, +1.68, +2.39, +1.17$). §IV-G "Where the error does not appear", L8,
  and §VI-C's "one measurement here does earn its place" are all built on 20news showing
  nothing, and at 3 seeds it merely lacked power.
- **mlp2 is no longer unexplained.** It resolves positive on 3 of 3 corpora
  ($+3.09, +1.17, +1.36$), where §IV-F calls it "unexplained --- it fits weights, and on
  two corpora it did not resolve positive."

One cell of 24 (`digits/knn`) shows a seed-index trend surviving Bonferroni
($p = 0.001$, slope $-0.10$/seed). No mechanism identified. It does not change that
cell's sign or resolution ($-10.89\ [-11.80,-9.97]$) and is recorded, not explained.

## DO NOT re-score the §IV-E gate at the new seed count

At 50 seeds all four pre-registered readouts match in $\geq 2$ of 3 corpora, which is
the gate's `PREDICTS` branch. **That must not be reported as the verdict.** The seed
count was raised *after* the original scoring, and at the intermediate 15 seeds the same
gate gives 2 of 4 -- `PARTIAL`. A verdict that moves with an $n$ chosen after the fact is
not a pre-registration, and picking the $n$ that upgrades it is the exact move this
project's rules forbid.

What is honest: report the corrected **intervals**, keep the pre-registered **verdict**
at `PARTIAL` (3 of 4) as scored on the design that was registered, and state separately
-- labelled as a post-hoc observation, never as the gate -- that all four match once the
measurement is powered. The same caution applies to any other gate whose verdict would
move at 50 seeds.

## The part that is not fixable by re-aggregating

A percentile bootstrap over **three** clusters is itself not a trustworthy interval.
covtype/centroid returns $[-4.72,-4.54]$, an inflation factor of ×0.05 — an artefact of
three clusters happening to agree, not a tight measurement. So the cluster bootstrap is
the *honest* aggregation of this design, but the design is too thin for either.

**The fix is more seeds, not a different estimator.** At 21 s per readout-corpus cell,
raising `SEEDS` from 3 to 15 and cluster-bootstrapping over seeds costs roughly
24 cells x 21 s x 5 = **~45 min of CPU**, and would make the interval defensible at
whatever count it then reports.

## Scope, and where else the defect reaches

Measured on Table II's 24 cells (`C7_readout` + `C8_mechanism`). The same pattern is
present elsewhere and is **recorded but not quantified**:

- **§V, `C6_ratio/00_run.py:70`** — identical construction: `rng = default_rng(seed)`,
  so `G` is shared across the four blocks of every `(n_train, seed)`. `evaluate()` then
  resamples all 140 cells independently (line 104). The independent units are
  7 sizes x 5 seeds = **35 clusters, not 140 cells**, so the $\rho$ intervals are
  roughly $2\times$ too narrow. $\rho = +0.627\ [+0.504,+0.723]$ would likely still
  exclude zero at that width, and the replication arm already FAILED its gate, so the
  verdicts probably stand — but "probably" is not measured and is not a claim.
- **§VI, `C4_generality`** — already recorded in `C10_rifs/PLAN.md` §2 as bootstrapping
  20 `(block, seed)` units that share rows.

So the defect is **systemic across C4, C6_ratio, C7 and C8**, not local to Table II. It
originates in one idiom -- seeding the split from `seed` alone while looping blocks
outside it -- that was copied forward. Any future experiment reusing `c7.cells()` or
`c4.blocks_for()` inherits it unless the split is keyed on `(block, seed)` or the
aggregation clusters on `seed`.
