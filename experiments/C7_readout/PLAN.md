# Does the Gaussian-null artefact survive a change of readout?

**Pre-registered 2026-10-28, before any cross-readout number was computed.** This is L7 in
`EVIDENCE.md` — the largest open scope limitation and the first question an EAB reviewer
asks: is this a property of *nulls*, or a property of *one probe*?

It is also a kill test for the paper's headline. If the artefact is readout-specific,
Claim A does not survive and the submission has to be rebuilt around something else. Better
to learn that on day 1 of 14 than on day 10.

## 1. Claim under test

**Claim A** (`EVIDENCE.md` §1): appending a block with provably zero label information to an
informative base beats a Gaussian block of the same shape, while it does **not** beat that
same block's own rows re-shuffled.

Everything measured so far used one readout: `probelib.fit`, a regularised L2 logistic probe
fitted by LBFGS. If the artefact is an artefact *of that probe*, the paper's claim is far
narrower than stated.

## 2. Readouts

Four, chosen to span inductive biases rather than to agree:

| id | readout | why it is here |
|---|---|---|
| `logistic` | L2 logistic, LBFGS — `probelib.fit` | the project's standard; the baseline all existing numbers use |
| `ridge` | ridge classifier, closed form | same linear family, completely different fitting path and regularisation |
| `mlp` | one hidden layer (64 units), ReLU | **non-linear** — the case Belinkov's probing literature cares about most |
| `knn` | k-nearest neighbours, k = 15 | non-parametric, distance-based; no fitted weights at all |

`knn` is included knowing it may behave very differently — distance concentration in high
dimensions is a real phenomenon and it may show a *larger* artefact, or a reversed one.
That is information either way and it is not grounds for dropping it after the fact.

## 3. Design

- Corpora: `digits`, `20news`, `covtype` — the three public, fully reproducible ones.
- Blocks, known-nulls, base and Gaussian comparator exactly as `C4_generality/00_run.py`.
- Measured at the **high-r regime**, where the effect is largest. If it does not appear
  there it does not appear anywhere, so this is the fair place to test.
- Per readout, per corpus: `Δ_noise` and `Δ_shuffle`, bootstrap 95% CI over (block, seed).

## 4. Gate

**SURVIVES** requires, for **at least 3 of the 4 readouts**, in **at least 2 of the 3
corpora**:

1. `Δ_noise` bootstrap 95% CI **excludes 0** and the mean is **> +0.02**;
2. `Δ_shuffle` bootstrap 95% CI **contains 0**.

**READOUT-SPECIFIC** if fewer than 3 readouts clear both. In that case Claim A must be
restated as a property of the readouts that showed it, named individually, and the paper
reframed accordingly — **not** quietly scoped to "linear probes" after the fact.

**VOID** if the realised high-r regime differs from `C6_ratio`'s by more than 10%, or if any
readout fails to fit (which is a bug, not a result).

**FAIL-branch reachability**: demonstrated before the verdict is read, by running the gate
on a base whose labels are unlearnable, where no readout can show an artefact — it must
return READOUT-SPECIFIC.

## 5. Resources

**CPU only.** Three corpora × four readouts × four blocks × three seeds ≈ 144 cells, two
arms each. `logistic` measured at 0.8 s per fit; `ridge` and `knn` are faster, `mlp` slower.
Expect 15–30 minutes. `OMP_NUM_THREADS=8` on a shared box. Checkpoints per (corpus,
readout). Rail 1b does not trigger.

## 6. What either outcome does to the paper

- **SURVIVES:** L7 closes. Claim A becomes "a property of the null, not of the probe",
  which is a materially stronger headline and exactly what an EAB reviewer wants to see
  tested. This goes in the body as its own section.
- **READOUT-SPECIFIC:** the headline narrows to the readouts that showed it. Still
  publishable — "this baseline is confounded for linear probes" is a real finding — but the
  framing changes and the limitations section grows. Worth knowing before drafting.
