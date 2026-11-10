# C13 — **This is not a gate.** It is a defect repair, and it is filed here so that it cannot be mistaken for one.

Written 2026-11-05. Every other folder in this artifact holds a pre-registration with a
frozen inequality and a named verdict per branch. This one deliberately does not, and the
build refused until that was stated rather than left implicit.

## 1. Why there is no gate here

A gate tests a hypothesis that could come out either way. This work does something else:
it corrects an abscissa that was computed three different ways, and reports how far the
existing fit moves as a result. There is no hypothesis to falsify — only an inconsistency
to remove — so inventing a threshold after the fact would be dressing a repair up as a
test, which is exactly what this project's rules forbid.

**`gates_tested` is therefore unchanged at 9.** C13 does not enter the family-wise count
and does not appear in §III-F's enumeration.

## 2. The defect

`r = eff_rank(G)/n_train` is computed by three conventions across the experiments that
feed Eq. (6):

| experiment | code | convention |
|---|---|---|
| `C4_generality:132,138` | `effective_rank(C)`, `effective_rank(G)` | full sample |
| `C6_ratio:73` | `effective_rank(G[:min(len(G),4000)])` | ≤ 4000 rows |
| `C6_lrs2/01_rerun.py:127` | `effective_rank(G[tr])` | at `n_train` |
| `C7_readout:98` | `effective_rank(G[tr[:3000]])` | at `n_train` |

Table I's caption documents only the last. Eq. (6) is therefore **fitted on points from the
first two conventions and evaluated out of sample on a point from the third**.

## 3. What the repair may and may not conclude

**May:** recompute every fitted point's `r` under the single convention the paper
documents, refit, and report how far the prediction moves. `Δ` values are untouched and no
probe is refitted, so nothing here can alter a measurement.

**May not: re-score the LRS2 gate.** That gate was registered against the line as
originally fitted and returned its verdict there. A line refitted after the result is known
is a robustness check, and the paper reports it as one, beside the registered verdict
rather than in place of it. This is the same discipline L10 applies to the seed count in
§IV-E.

**May not: rescue the within-corpus sweep.** Within a corpus the numerator of `r` is
constant, so `r ∝ 1/n_train` under *any* convention. That confound is separate and
survives this repair.

## 4. Reproducibility note, recorded before the result was used

`eff_rank(G)` is a random draw, so `r` is itself a random quantity and recomputing it gives
slightly different values run to run. Expect the published slope to reproduce to about
three decimals and the intercept to about 0.01. Any future claim about `r` should fix the
comparator's seed or report the spread.

## 5. Cost

CPU only, seconds. No probe refits — only effective-rank draws and a two-parameter fit.
