# C13 — Defect repair: `r` was computed three ways. Under one convention the out-of-sample prediction improves markedly.

**Not a gate.** No pre-registration, because this is a repair to an abscissa, not a test of
a hypothesis. Saying so plainly matters more than the result below: a line refitted after
the fact cannot re-score a gate that was registered against the original.

`gates_tested` stays **9**.

## 1. The defect

`r = eff_rank(G)/n_train` is computed by three conventions across the experiments feeding
Eq. (6):

| experiment | code | convention |
|---|---|---|
| `C4_generality:132,138` | `effective_rank(C)`, `effective_rank(G)` | full sample |
| `C6_ratio:73` | `effective_rank(G[:min(len(G),4000)])` | ≤4000 rows |
| `C6_lrs2/01_rerun.py:127` | `effective_rank(G[tr])` | **at `n_train`** |
| `C7_readout:98` | `effective_rank(G[tr[:3000]])` | **at `n_train`** |

Table I's caption documents only the last. **Eq. (6) is fitted on points from the first two
conventions and evaluated out of sample on a point from the third.** For digits the same
comparator reads 48.7 at `n_train` and 61.6 under the sweep's convention — a 27% gap in the
quantity the paper regresses on.

## 2. What changes

`Δ` values are untouched; only the abscissa moves. No probe was refitted.

| point | `d` | `n_train` | `r` as used | `r` at `n_train` | `Δ` (pts) |
|---|---|---|---|---|---|
| C4 digits | 64 | 1257 | 0.0490 | 0.0485 | $+0.79$ |
| C4 20news | 128 | 7919 | 0.0157 | 0.0159 | $-0.01$ |
| C4 covtype | 54 | 14000 | 0.0038 | 0.0038 | $+0.25$ |
| C6 top-`r` digits | 64 | 104 | **0.5928** | **0.3798** | $+5.89$ |
| C6 top-`r` covtype | 54 | 177 | **0.3010** | **0.2313** | $+3.01$ |

The three C4 points barely move — their `n_train` is large, so the two conventions nearly
coincide. **The two sweep points move a lot**, because the top of each sweep is its
*smallest* `n_train`, which is exactly where the participation ratio's finite-sample bias
bites (Chun et al., arXiv:2509.26560). The defect therefore distorts the two points with the
most leverage on the slope.

## 3. The result

| | fit | residual sd | LRS2 prediction at `r = 0.1611` |
|---|---|---|---|
| mixed abscissa (as published) | $\Delta = 9.739\,r + 0.112$ | 0.205 | $+1.68\ [+1.27,+2.09]$ |
| **one convention (at `n_train`)** | $\Delta = 15.025\,r - 0.055$ | 0.335 | $\mathbf{+2.37\ [+1.70,+3.04]}$ |

Measured on LRS2, three independent pair draws: **$+2.7$, $+2.7$, $+2.3$**.

**All three measured draws fall inside the unified-convention interval.** Under the mixed
abscissa two of the three sat above it, which is the "the line under-predicts by roughly
$55\%$" that L2 records. Under one convention the under-prediction is about **10%**.

**The defect was making the paper's own out-of-sample prediction look worse than it is.**

## 4. What this does and does not license

- **It does not re-score the LRS2 gate.** That gate was registered against the line as
  originally fitted and returned its verdict there. The unified-convention fit is a
  robustness check computed after the fact, and is reported as one. The registered verdict
  stands unchanged.
- **It does not rescue §V's within-corpus sweep.** That remains confounded: within a corpus
  `r ∝ 1/n_train` whichever convention is used, because the numerator is constant across the
  sweep (`paper_improvement/02_statistical_audit.md` F2). This repair is about the
  *cross-corpus* fit only.
- **It does strengthen L2**, which should now say the under-prediction is convention-
  dependent and roughly 10% once `r` is computed consistently.

## 5. A reproducibility wrinkle worth recording

`eff_rank(G)` is itself a **random draw**, so `r` is a random quantity and recomputing it
gives slightly different values run to run. This script reproduces the published slope to
three decimals (9.739 against 9.74) and the residual sd to two (0.205 against 0.20), but the
intercept lands at 0.112 against the published 0.10 — a difference inside the variation of
drawing a fresh `G`. Any future statement about `r` should either fix the comparator's seed
or report the spread. Nothing in the paper turns on the intercept.

## 6. What the paper owes

1. **State the convention once, in §III**, and use it everywhere.
2. **§V-C**: report the unified-convention fit as a robustness check beside the registered
   one, with all three draws inside the interval.
3. **L2**: the $55\%$ under-prediction is convention-dependent; under one convention it is
   about $10\%$.
4. `gates_tested` is unchanged at 9 — this is not a gate.
