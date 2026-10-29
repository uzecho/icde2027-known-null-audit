# C4 kill test — does the feature-block null artefact exist outside visual speech?

**Pre-registered 2026-10-26, before any external data was loaded or any number computed.**
Candidate C4, lead for an ICDE 2027 EAB submission. Gate is in §5 and is the only thing that
decides PASS/FAIL.

## 1. Hypothesis

Comparing candidate feature blocks against a **Gaussian-noise** null is confounded by
effective rank, and this is a property of linear probes and feature geometry — **not** a
quirk of LRS2. The correct null is the block's **own rows, shuffled**, which preserves the
block's covariance and rank exactly and destroys only the row↔label correspondence.

Mechanism: a Gaussian block of the same shape has effective rank ≈ its full dimensionality,
so it dilutes a regularised linear probe badly and sits near chance. Any real block has far
lower effective rank and so dilutes *less*, scoring above noise **whether or not it carries
label information**. "Beats noise" therefore measures rank, not information.

## 2. Novelty position

Nearest prior work is Belinkov, *Probing Classifiers* (arXiv 2102.12452), whose catalogued
controls — control tasks (Hewitt & Liang 1909.03368), control functions (Pimentel 2020b),
selectivity — all vary **the probe `g`** or **the labels `D_P`**. Its random-baseline
discussion compares a *learned* representation against a *randomised* one. Verified by grep
that neither it nor Hewitt & Liang varies **the object being compared**. That is this
catalogue's axis.

## 3. Collision with `EXPERIMENTS.md`

- `05` — *More visual evidence (pixels / flow / landmarks / full face)*: all four
  **indistinguishable from own shuffle**.
- `05` ⚠ — *Extra evidence blocks beat the control*: overturned, **Gaussian noise was the
  wrong null, effective-rank artefact**, 3 false positives.

So the artefact is established **on LRS2 only**. The EAB bar is "wide interest". This test
asks the one question that decides whether the paper exists, and asks it first.

## 4. The cheapest kill test

The hard part of any such test is knowing the ground truth. We avoid estimating it:

> **Construct blocks with exactly zero label information, by shuffling the rows of a real
> block.** Such a block keeps the real covariance, spectrum and effective rank of its
> source, and carries provably no information about the labels.

A null that is working must score these at chance. So, for each known-null block `B₀`:

- `Δ_noise   = acc(B₀) − acc(Gaussian, same shape)`
- `Δ_shuffle = acc(B₀) − acc(shuffle_rows(B₀))`

A correct null gives ≈ 0 for a block that carries nothing. **`Δ_noise > 0` is a false
positive, by construction.**

**Data — three domains, none of them visual speech**, real feature blocks only, no block
chosen or tuned for its rank:

| id | dataset | domain | blocks |
|---|---|---|---|
| D1 | sklearn `digits` | vision, dense pixels | raw, PCA-16, random-projection, per-pixel standardised |
| D2 | `20newsgroups_vectorized` | text, sparse high-dim | raw tf-idf, SVD-64, hashed, binary-presence |
| D3 | an OpenML tabular set | tabular, mixed | raw, standardised, PCA-8, one-hot expanded |

Probe: `probelib.fit_folds`, the project's standard readout, balanced accuracy, CPU.
CIs: `probelib.boot` over blocks. Floors measured **in this run**, never quoted.

## 5. Gate — the stopping rule

Read this before any number. Judge as written.

**PASS** requires **both**, in **every one of the three datasets**:

1. `mean Δ_noise` over known-null blocks **> +0.02** balanced accuracy, bootstrap 95% CI
   **excluding 0** — the Gaussian null calls a provably uninformative block informative.
2. `mean Δ_shuffle` over the same blocks has bootstrap 95% CI **containing 0** — the
   proposed null is not itself broken.

**FAIL** if either fails in any dataset. In particular, a `Δ_noise` CI containing 0
anywhere means the artefact does **not** generalise, and C4's EAB claim of wide interest
dies with it.

**VOID — non-degeneracy assert, checked before the gate is read.** In every dataset, every
real block's effective rank must be **< 25%** of the Gaussian block's effective rank. If the
spread is not there, the contrast is vacuous and the run answers nothing. This is the
width-router lesson: that gate passed spuriously because `adaptive == fixed` made its only
contrast empty. The build **asserts** this and refuses rather than reporting.

**FAIL-branch reachability** must be demonstrated before the real run, by feeding the gate a
synthetic dataset whose real blocks are full-rank — where the artefact cannot appear — and
confirming it returns FAIL, not PASS. A gate that has not been shown to fail is not a gate.

## 6. Resources

**CPU only.** Linear probes over sklearn-scale matrices; no encoder forward passes, so no
GPU is requested for this test. If a later stage needs a card, hard rail 1b applies: report
peak memory, wall-clock, card, and the prior run the estimate is anchored to, then wait.

## 7. What a PASS would and would not establish

- **Would:** that the effective-rank confound in feature-block nulls is general across
  domains and not an LRS2 artefact — the EAB "wide interest" bar, for this one catalogue
  entry.
- **Would not:** validate the other catalogue entries (mapping nulls, masking nulls, context
  nulls). Each needs its own generality evidence or must be scoped to this corpus in the
  paper. Scope every claim to what was measured.
- **Would not:** say anything about non-linear probes. The claim is scoped to regularised
  linear probes, which is this project's standard readout.
