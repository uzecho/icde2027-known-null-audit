# C6 — gate PASS, control CLEARED, **replication FAILED**. The loop does not stop.

Second gate of the ICDE loop. `gates_tested` = 2.

## The three conditions, from `STOP_RULE.md`, quoted before any number was read

> 1. **Gate PASS.** Its own `PLAN.md` gate, written before data was touched, returns PASS as
>    written.
> 2. **Control cleared.** The PASS survives the correct null for its comparison type.
> 3. **Replicated.** The effect reproduces on a disjoint split or a different seed.
>
> All three, or the loop continues.

**Result: 1 ✓, 2 ✓, 3 ✗.** The loop continues.

## 1. Main gate — PASS (2 of 3 datasets, as pre-registered)

Hypothesis: the Gaussian-null artefact is governed by `r = eff_rank(noise) / n_train`, not
by rank spread. Sweep holds the block set, Gaussian width and rank spread fixed and varies
only `n_train`.

| dataset | ρ(r, Δ_noise) [95% CI] | Δ_noise at top r [95% CI] | verdict |
|---|---|---|---|
| digits / vision | **+0.627** [+0.504, +0.723] | **+0.0589** [+0.0462, +0.0721] | PASS |
| 20news / text | +0.184 [−0.004, +0.355] | +0.0187 [−0.0041, +0.0448] | FAIL |
| covtype / tabular | **+0.510** [+0.345, +0.664] | **+0.0301** [+0.0138, +0.0473] | PASS |

Gate required both conditions in ≥2 of 3. Met. VOID guards passed (spans 12.1×, 19.2×,
79.2×; top r ≥ 0.30 everywhere).

## 2. Control — CLEARED, and this is the strongest part of the result

The alternative explanation was that *every* probe degrades at small n, making this
small-sample instability rather than anything about the Gaussian null — the exact shape of
the trap where `adaptive − permuted` was meaningless because `fixed − permuted` matched it.
The discriminating measurement is the same sweep for the **correct** null, where both arms
are provably uninformative with identical geometry.

| dataset | ρ(r, **Δ_noise**) | ρ(r, **Δ_shuffle**) ← control | Δ_noise @ top r | Δ_shuffle @ top r |
|---|---|---|---|---|
| digits | +0.627 [+0.504, +0.723] | **−0.050** [−0.237, +0.129] | +0.0589 | +0.0071 [−0.0051, +0.0187] |
| covtype | +0.510 [+0.345, +0.664] | **−0.072** [−0.262, +0.127] | +0.0301 | −0.0127 [−0.0305, +0.0040] |

The correct null is **flat in r** — negative point estimates, CIs spanning zero — while the
Gaussian null degrades steeply. At the top ratio the Gaussian gap is 8× the own-shuffle gap
on digits, and on covtype the own-shuffle gap runs the *opposite* way. Small-sample
instability is ruled out: the degradation is specific to the Gaussian null.

## 3. Replication — FAILED, in both datasets

Disjoint half, different seed family (100–104 vs 0–4). `PLAN.md` §4c required both passing
datasets to still satisfy the main gate.

| dataset | ρ(r, Δ_noise) [95% CI] | Δ_noise at top r [95% CI] | monotone ρ>0.5? | magnitude >0.02? |
|---|---|---|---|---|
| digits | +0.384 [+0.222, +0.522] | +0.0409 [+0.0210, +0.0591] | **✗ (0.384)** | ✓ |
| covtype | +0.403 [+0.217, +0.566] | +0.0341 [+0.0137, +0.0523] | **✗ (0.403)** | ✓ |

**The magnitude replicates cleanly. The monotonicity does not.** Both ρ values fall below
the pre-registered 0.5 by a modest but real margin, in both datasets, with CIs that exclude
zero but sit under the threshold. Judged as written: NOT REPLICATED.

The gate was not widened and the magnitude result was not promoted to headline to cover the
monotonicity failure.

## 4. Two mechanical observations — offered as hypotheses, not as rescues

The replication halves the data, which moves the realised `r` range, and the two datasets
moved differently:

- **digits**: main sweep spanned r 0.049–0.594; the replication spanned **0.095–1.150**,
  nearly 2× higher at the top. If the relationship **saturates** at high r — plausible, since
  both arms must converge toward chance once capacity swamps the sample — then extending the
  sweep into the plateau attenuates a rank correlation computed over the whole range.
- **covtype**: the top of the range was unchanged at 0.3011, but the bottom rose from 0.0038
  to 0.0076, halving the span from 79× to 40×. Spearman ρ over a compressed range is
  attenuated by range restriction alone.

Both are testable, and neither is evidence that the gate should have passed. They point at
the same revision: **the relationship is probably threshold-like rather than monotone**, and
a gate written around monotonicity was the wrong gate for a saturating effect.

## 5. What this does and does not establish

- **Does:** the Gaussian-null artefact is real, specific to that null, and large at high
  capacity-to-sample ratio — +0.0589 and +0.0301 at top r, replicating at +0.0409 and
  +0.0341. Rank spread does not explain it; `r` does, at least in part.
- **Does:** retro-explain `C4_generality`'s near-nulls. Those were measured at r ≈ 0.051,
  0.017, 0.004 — the low end of this curve, where the gate here shows the artefact is small.
  Two independent experiments now agree on that.
- **Does not:** establish monotone dependence on r. That is what failed.
- **Does not:** establish anything about LRS2 directly. The claim that `05`'s regime
  (r ≈ 0.19) is where this bites is **inference from external data**, not a measurement on
  LRS2, and must be labelled as such.
- **Does not:** cover non-linear probes or other readouts.

## 6. Family-wise position

`gates_tested` = **2**. Anything that eventually stops this loop is reported as
"PASS #1 of N gates tested", and N includes both C4's FAIL and this one. A third gate
rewritten around saturation would make N = 3, and the ledger must say so — the whole point
of the counter is that re-gating until something passes is visible rather than hidden.

## 7. What is worth running, and the honest caution attached to it

A threshold-shaped gate — "there exists r* above which Δ_noise > 0.02 with CI excluding 0,
and below which it does not" — fits what both experiments show and would likely pass. **That
is also exactly how a loop talks itself into a false positive.** It would be the third gate
on the same underlying data, written after seeing two results. If it is run, it needs
genuinely fresh datasets rather than these three, and the count reported.
