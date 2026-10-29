# SURVIVES — and the sign of the artefact depends on the readout

Fifth gate of the ICDE loop. `gates_tested` = 5.

## The gate, quoted from `PLAN.md` §4 before any number was read

> **SURVIVES** requires, for **at least 3 of the 4 readouts**, in **at least 2 of the 3
> corpora**: (1) `Δ_noise` CI excludes 0 and mean > +0.02; (2) `Δ_shuffle` CI contains 0.
> **READOUT-SPECIFIC** if fewer than 3 readouts clear both.

## Result — `Δ_noise` [95% CI], at r ≈ 0.235

| readout | digits / vision | 20news / text | covtype / tabular | clears |
|---|---|---|---|---|
| logistic | **+0.0656** [+0.0488, +0.0828] | +0.0087 [−0.0261, +0.0426] | **+0.0445** [+0.0202, +0.0668] | 2/3 |
| ridge | **+0.0540** [+0.0408, +0.0665] | +0.0093 [−0.0241, +0.0429] | **+0.0516** [+0.0337, +0.0677] | 2/3 |
| mlp (64h, ReLU) | **+0.0360** [+0.0214, +0.0483] | +0.0005 [−0.0184, +0.0172] | **+0.0209** [+0.0097, +0.0319] | 2/3 |
| knn (k=15) | **−0.0548** [−0.0727, −0.0373] | **−0.0324** [−0.0452, −0.0189] | **−0.0292** [−0.0403, −0.0180] | 0/3 |

Control `Δ_shuffle` contains 0 in **every one of the twelve cells**.

**Verdict: SURVIVES.** Three readouts — logistic, ridge, mlp — clear in 2 of 3 corpora.

## The result that is worth more than the verdict

**kNN does not fail to show the artefact. It shows it reversed**, in all three corpora, with
every CI excluding zero and its control clean throughout. That is not a null and it is not
noise.

A coherent mechanism accounts for both directions, and it is testable:

- A readout that **fits weights** overfits in proportion to the number of effectively
  independent directions available. A full-rank Gaussian block supplies ~d of them; a real
  block supplies far fewer. So the Gaussian arm is hurt *more*, and the null is
  **anti-conservative** — it makes an uninformative block look informative.
- A readout that **measures distances** is harmed by correlated real features, which
  concentrate pairwise distances and destroy neighbourhood structure. Isotropic Gaussian
  noise does this *less*. So the real arm is hurt more, and the null is **conservative** —
  it makes an uninformative block look actively harmful.

So the practical statement is not "the Gaussian null is confounded" but:

> **The sign of the error depends on the readout.** With a fitted probe you overstate a
> block's value; with a distance-based one you understate it. Neither is a property of the
> block.

The own-rows-shuffled null is correct in all twelve cells, which is what makes it the
recommendation rather than merely the alternative.

## Scoping, honestly

- **L7 largely closes.** The MLP is non-linear and clears in 2 of 3 corpora, so this is not
  an artefact of linear probing. It does **not** close for distance-based readouts, where
  the finding is different in kind rather than absent.
- **20news shows nothing for any fitted readout** — +0.0087, +0.0093, +0.0005, all CIs
  containing 0. That is consistent with `C4_generality` (−0.0001) and `C6_ratio` (ρ = 0.184,
  the one corpus that failed there too). 20news enters as SVD-128 of tf-idf, i.e. already a
  dense low-dimensional representation, so there may be little rank headroom left for the
  mechanism to act on. **Stated as a hypothesis; not tested.**
- The mechanism in the section above is an **explanation offered after seeing the signs**,
  not a pre-registered prediction. It is consistent with all twelve cells and with the
  `C6_ratio` capacity argument, but it has not been tested on its own terms. Any write-up
  must label it as such — this project has been burned once already by attaching a
  confident mechanism to a real effect (`CLAUDE.md`'s rank-spread story, overturned in
  `C4_generality`).

## What it changes for the submission

The headline moves off the `r` predictor, which failed its own replication, and onto
something measured in 12 of 12 cells with clean controls. `r` becomes a supporting section
about magnitude, where its limitations (L1–L3) are proportionate rather than fatal.

The reproducible surface also improves: all three corpora here are public, so a reviewer can
rerun the entire headline result without a licence. LRS2 remains the out-of-sample
confirmation of the magnitude claim, which is a far better place for an unshippable dataset
than under the headline.
