# Does the fit-vs-distance mechanism PREDICT the sign on readouts it has never seen?

**Pre-registered 2026-10-28, before any of the four new readouts was run.**

## 1. Why this exists

`C7_readout/FINDINGS.md` found that three weight-fitting readouts show a **positive**
`Δ_noise` and one distance-based readout shows a **negative** one, consistently. It then
offered a mechanism:

> A readout that **fits weights** overfits in proportion to the number of effectively
> independent directions, so the full-rank Gaussian arm is hurt more → null is
> anti-conservative. A readout that **measures distances** is harmed by correlated real
> features, which concentrate distances, so the real arm is hurt more → null is
> conservative.

That explanation was written **after** seeing the signs. This project has been burned by
exactly that once already: `CLAUDE.md` attached a confident rank-spread mechanism to a real
effect, and `C4_generality` overturned it. A mechanism that only explains the data it was
built from is a story.

So: state the prediction first, on readouts not yet run, and see whether it holds.

## 2. The prediction

For each new readout, the mechanism assigns a sign **before** it is measured:

| readout | family | **predicted sign of Δ_noise** |
|---|---|---|
| `linsvc` — LinearSVC, hinge loss | fits weights | **positive** |
| `mlp2` — MLP, two hidden layers (64, 32) | fits weights | **positive** |
| `centroid` — NearestCentroid | measures distances | **negative** |
| `knn5` — kNN, k = 5 | measures distances | **negative** |

`linsvc` and `mlp2` are different loss and different depth from anything in C7. `centroid`
fits no weights at all and is the cleanest distance-based case available.

## 3. Gate

Measured on the three public corpora at the same high-r regime as C7, same blocks, same
known-null construction, same bootstrap CIs.

A readout **matches** if its `Δ_noise` CI excludes 0 **with the predicted sign**, in at
least **2 of 3** corpora.

- **MECHANISM PREDICTS** if **all 4** readouts match.
- **PARTIAL** if 2 or 3 match.
- **MECHANISM FAILS** if 0 or 1 match.

A readout whose CI contains 0 in 2+ corpora counts as **not matching** — the mechanism
predicts a sign, and no sign is a miss, not a pass.

**Control:** `Δ_shuffle` CI must contain 0 for each readout in at least 2 of 3 corpora. A
readout whose own-shuffle null is dirty cannot adjudicate anything and is reported as VOID
for that readout rather than scored.

**Honest note on difficulty:** this gate is hard to pass by luck — four pre-assigned signs,
each needing a CI that excludes zero on the correct side, is a 1-in-16 coin flip before the
CI requirement is even applied. That is the point.

**FAIL-branch reachability:** demonstrated before the verdict is read, by running the gate
on unlearnable labels, where no readout can show a signed effect; it must return
MECHANISM FAILS.

## 4. Resources

**CPU only.** 3 corpora × 4 readouts × 4 blocks × 3 seeds ≈ 144 cells, two arms each. C7's
equivalent took 21 s total; `mlp2` is the slow one. Expect under 10 minutes.
`OMP_NUM_THREADS=8`. Rail 1b does not trigger.

## 5. What either outcome does

- **MECHANISM PREDICTS:** the paper's headline section states a mechanism that was tested
  out of sample on four readouts, not a post-hoc rationalisation. This is the single
  cheapest thing available that raises the paper's standard of evidence.
- **PARTIAL / FAILS:** the mechanism is removed from the paper as an explanation and the
  finding is reported as what it is — a measured sign difference between readout families,
  cause unknown. The headline survives either way; only the explanation is at stake.
