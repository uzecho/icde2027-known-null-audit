# PARTIAL — 3 of 4. The mechanism is supported but not established.

Sixth gate of the ICDE loop. `gates_tested` = 6.

## The gate, quoted from `PLAN.md` §3 before any of the four readouts was run

> A readout **matches** if its `Δ_noise` CI excludes 0 **with the predicted sign**, in at
> least 2 of 3 corpora.
> - **MECHANISM PREDICTS** if **all 4** readouts match.
> - **PARTIAL** if 2 or 3 match.
> - **MECHANISM FAILS** if 0 or 1 match.

Signs were assigned from the fit-vs-distance mechanism **before** measurement.

## Result

| readout | family | predicted | digits | 20news | covtype | matched |
|---|---|---|---|---|---|---|
| `linsvc` | fits weights | **+** | +0.0754 [+0.0591,+0.0930] | +0.0066 [−0.0231,+0.0372] | +0.0423 [+0.0146,+0.0678] | **2/3 ✓** |
| `mlp2` | fits weights | **+** | +0.0497 [+0.0375,+0.0627] | −0.0175 [−0.0448,+0.0086] | −0.0030 [−0.0229,+0.0174] | 1/3 ✗ |
| `centroid` | measures distances | **−** | −0.0651 [−0.0927,−0.0369] | −0.0722 [−0.1049,−0.0386] | −0.0462 [−0.0629,−0.0279] | **3/3 ✓** |
| `knn5` | measures distances | **−** | −0.0853 [−0.1074,−0.0628] | −0.0437 [−0.0614,−0.0246] | −0.0240 [−0.0425,−0.0057] | **3/3 ✓** |

**Verdict: PARTIAL (3/4).** Per `PLAN.md` §5, the mechanism is therefore **not** reported as
an established explanation. The headline finding — the sign differs by readout family — is
unaffected; only the causal story is at stake, and it does not get told as fact.

## What the misses actually are, stated precisely

Across the 12 cells: **9 matched the predicted sign with a CI excluding zero, 3 were
inconclusive, and 0 contradicted the predicted sign.** Every failure is a confidence
interval spanning zero, never a reversal.

- The **distance-based** prediction is unbroken: `centroid` 3/3 and `knn5` 3/3 here, plus
  `knn` 3/3 in `C7_readout` — **9 of 9 cells negative with CIs excluding zero**, across
  three distinct distance-based readouts.
- The **fits-weights** prediction is corpus-dependent: `logistic`, `ridge`, `mlp` and
  `linsvc` each 2/3; `mlp2` 1/3. No fitted readout has ever produced a negative CI
  excluding zero in any experiment in this series.
- The misses cluster. **20news fails for every fitted readout here**, and it has never shown
  the effect in `C4_generality` (−0.0001), `C6_ratio` (ρ = 0.184, the one corpus that failed
  there) or `C7_readout`. It enters as SVD-128 of tf-idf — already a dense, low-dimensional
  representation — so there may be little rank headroom for the mechanism to act on.
  **Hypothesis, untested.**

This is a stronger pattern than the gate's PARTIAL conveys. It is reported here rather than
used to upgrade the verdict, because the gate was written first and says 3/4 is PARTIAL.

## A side observation worth a line in the paper

The first version of the reachability selftest used labels independent of every feature,
expecting no readout could show an artefact. **Two fitted readouts showed it anyway**
(`linsvc` +0.0236 [+0.0124,+0.0341], `mlp2` +0.0148 [+0.0011,+0.0277]).

That is consistent with the mechanism rather than against it: the artefact is about
*overfitting capacity*, which does not require the labels to be learnable. A Gaussian block
supplies more effectively independent directions to overfit than a real block does, whether
or not there is any signal to find. It also means a practitioner cannot escape this by
arguing their task is hard.

## Process notes

- The first selftest had a hole: with a single synthetic corpus, `MIN_CORPORA = 2` voided
  every readout, so the aggregate scoring path was never exercised and the reachability
  demonstration was weaker than it appeared. Fixed to use two corpora before the real run;
  the corrected selftest returns MECHANISM FAILS through the scoring path.
- `centroid` on 20news had a dirty own-shuffle control. Per the gate it is scored because it
  was clean in 2 of 3 corpora, which is the rule as written. Flagged here so a reader can
  discount that one cell.

## For the paper

- **State as measured:** the sign of `Δ_noise` differs systematically between fitted and
  distance-based readouts. 24 cells across two experiments, 0 contradictions.
- **State as hypothesis, explicitly labelled:** the overfit-capacity vs distance-concentration
  mechanism. It is consistent with every cell measured and it predicted 9 of 12 correctly in
  advance, but it did not clear its own pre-registered bar and `mlp2` is unexplained.
- **Do not state:** that the mechanism is established. `CLAUDE.md`'s rank-spread story was
  exactly this kind of confident post-hoc explanation and `C4_generality` overturned it.
