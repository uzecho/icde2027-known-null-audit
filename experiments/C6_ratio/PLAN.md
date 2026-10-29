# C6 kill test — is the Gaussian-null artefact governed by d_eff / n?

**Pre-registered 2026-10-26, before the curve was measured.** Supersedes the first sketch of
C6 ("frozen representations"), which named a *regime* without a mechanism. This names the
mechanism and is falsifiable.

## 0. What had already been seen when this was written — stated so it can be discounted

From `C4_generality`, one `Δ_noise` value per dataset, at **one** ratio each:

| dataset | noise eff. rank | approx. n_train | ratio r | Δ_noise measured |
|---|---:|---:|---:|---:|
| digits | 61.8 | ~1200 | 0.051 | +0.0079 |
| 20news | 126.5 | ~7500 | 0.017 | −0.0001 |
| covtype | 53.9 | ~13300 | 0.004 | +0.0025 |

Nothing about the **curve** has been observed — not its slope, not its value at any other
ratio, and in particular nothing at LRS2's ratio of ≈0.19. The gate below is about the
curve. The three points above are consistent with the hypothesis at the extremes (digits
highest r, largest effect) and **inconsistent in the middle** (20news has 4× covtype's ratio
and a smaller effect), so this is not a hypothesis fitted to the data it will be judged on.

## 1. Hypothesis

The Gaussian-null artefact is governed by **capacity-to-sample ratio**

    r = effective_rank(noise block) / n_train

not by rank spread between the real block and the noise block. `C4_generality` showed rank
spread does **not** predict it: covtype had the most extreme spread (0.035) and the second
smallest effect. A full-rank Gaussian block supplies ≈d independent directions to overfit;
how much that costs depends on how many of them there are **per training row**. LRS2's
experiment `05` sat at r ≈ 0.19, roughly 4× higher than anything C4 tested, which would
explain why the artefact was large there and small or absent everywhere else.

If true, the practical rule is quantitative and general: *the choice of null matters above a
measurable ratio and is negligible below it.* That is a "new way to evaluate existing
methods", which is the EAB bar.

## 2. The cheapest kill test

**Hold everything constant except n.** The block set, the Gaussian width and the rank spread
all stay fixed; only the number of training rows is subsampled. That varies r without
touching the confound C4 already eliminated.

- Datasets: `digits`, `20news`, `covtype` — the same three, so C4's points sit on the curve.
- Blocks: the same four as C4 (`raw`, `randproj`, `smoothed`, `pca_recon`), unchanged.
- Sweep: n_train subsampled to give **r spanning at least 10×** per dataset, bracketing
  LRS2's r ≈ 0.19 at the top where the data allows.
- Known-null blocks and readout exactly as in C4: real block with rows shuffled, appended to
  an informative base, `probelib.fit` as the readout, `probelib.boot` for CIs.
- 5 seeds per (block, ratio) cell.

## 3. Gate — the stopping rule

Read before any number. Judge as written.

**PASS** requires **both**, in **at least 2 of the 3 datasets**:

1. **Monotone dependence.** Spearman ρ between `r` and `Δ_noise` across the sweep is
   **> +0.5**, with a bootstrap 95% CI over cells **excluding 0**.
2. **It bites where LRS2 sat.** At the highest ratio tested in that dataset (which must be
   **≥ 0.15**), `mean Δ_noise` is **> +0.02** with bootstrap 95% CI **excluding 0** — the
   same threshold C4 used, so the two gates are commensurable.

**FAIL** if either condition holds in fewer than 2 datasets.

**VOID — non-degeneracy asserts, checked before the gate is read.**
- The realised `r` range must span **≥ 10×** within each dataset, else the correlation is
  being read off a point cloud with no leverage.
- The top ratio must actually reach **≥ 0.15**, else condition 2 is untested rather than
  failed, and claiming FAIL would be dishonest.

**FAIL-branch reachability** must be demonstrated before the real run: feed the gate a
synthetic case where `Δ_noise` is flat in `r`, and confirm it returns FAIL rather than PASS.

## 4. Resources

**CPU only.** Same scale as C4, which ran all three datasets in 1m49s; the sweep multiplies
cell count by the number of ratios, so expect single-digit minutes. No GPU requested, so
hard rail 1b does not trigger.

## 4b. Control gate — written 2026-10-26 **after** the main gate returned PASS (2/3) and **before** the control was run

`STOP_RULE.md` condition 2. The main result is a rising `Δ_noise` as `n_train` falls. The
obvious alternative explanation is that **every** probe degrades at small n, so the gap is
small-sample instability rather than anything about the Gaussian null. This is the exact
shape of the trap this project already fell into once, where `adaptive − permuted` was
+0.0042 and meaningless because `fixed − permuted` was +0.0045.

The discriminating measurement is the same sweep for the **correct** null,
`Δ_shuffle = acc(base ⊕ C₀) − acc(base ⊕ shuffle(C₀))`, where both arms are provably
uninformative and share identical geometry.

**CONTROL CLEARED** requires, in **each dataset that passed the main gate**:

1. `ρ(r, Δ_shuffle)` is **not** itself strong: either ρ ≤ +0.5, or its bootstrap 95% CI
   contains 0.
2. `Δ_shuffle` at top r is **less than half** of `Δ_noise` at top r in the same dataset.

**CONTROL FAILED** if either is violated anywhere — the r-dependence is then a property of
small samples, not of the Gaussian null, and C6's PASS is hollow and must be reported as
such.

## 4c. Replication — `STOP_RULE.md` condition 3

Re-run the full sweep on a **disjoint half** of each dataset with a different seed family.
Both passing datasets must still satisfy the main gate, with their own CIs reported
separately. A PASS that does not reproduce does not stop the loop.

## 5. What a PASS would and would not establish

- **Would:** a quantitative, cross-domain rule for when the Gaussian null misleads, which
  retro-explains both `05`'s three false positives and C4's three near-nulls.
- **Would not:** establish it for non-linear probes, or for readouts other than this one.
- **Would not:** rescue C4's original claim. C4 failed; this is a different and narrower
  claim about *when* the entry applies, and it must be reported as superseding C4 rather
  than as a second attempt at the same gate.
- **Would not:** on its own satisfy the stop rule — a PASS still has to clear its control
  and replicate on a disjoint split before the loop may stop (`STOP_RULE.md`).
