# Do same-viseme pairs show a smaller Gaussian-null artefact at fixed r?

**Pre-registered 2026-10-28, before any comparison was computed.** Caveat 3 of
`C6_lrs2/FINDINGS.md`. This matters for paper 3 regardless of what happens with ICDE,
because the pairs paper 3 is about are exactly the same-viseme ones.

## 1. Why ask

The keystone measured Δ_noise = +0.023…+0.027 at r = 0.1611 on mostly vowel pairs. The
voided attempt 1 used same-viseme pairs and its two valid-r cells gave −0.0004 and +0.0107.
Those are not comparable — one run was VOID — so the difference is a **suspicion, not a
result**. This tests it directly, with r held constant by construction.

If same-viseme pairs do show a smaller artefact, then `05`'s false positives were driven
less by this artefact on precisely the pairs paper 3 cares about, and the correction already
made to `CLAUDE.md` needs scoping again.

## 2. Hard limit on power, stated before running

The map contains **10 same-viseme pairs in total**. At the sample size needed to fix
n_train = 5333 (6133 paired samples), **5 qualify**: D-N, D-T, M-P, N-T, S-Z. That is the
ceiling — not a choice, and not something more compute can fix.

So this test can be underpowered, and the gate must be able to say so. `verdict/SKILL.md`:
*an underpowered run is not a null.*

## 3. Design

- **Group SV:** all 5 qualifying same-viseme pairs.
- **Group OTHER:** 15 pairs drawn at random (seed fixed, recorded) from the 230 qualifying
  others. Random because the keystone's first attempt used phone order, which was a
  selection hole caught after the fact.
- `n_train` fixed at **5333** for every pair, so r = 0.1611 throughout; n_test 800.
- Per pair also measure **baseline accuracy from the stem alone** — same-viseme pairs are
  harder, and Δ_noise may track difficulty rather than viseme status.
- Base, candidate, known-null and both nulls exactly as in `C6_lrs2/01_rerun.py`.

## 4. Gate

Let `D = mean Δ_noise(OTHER) − mean Δ_noise(SV)`, bootstrap 95% CI over pairs.

- **DIFFERENCE** if the CI for `D` **excludes 0**.
- **NO DIFFERENCE** if the CI contains 0 **and** its half-width is **< 0.010** — i.e. there
  was enough precision to have seen a difference of the size suggested (~0.018) and it was
  not there.
- **UNDERPOWERED** if the CI contains 0 **and** its half-width is **≥ 0.010**. Reported as
  inconclusive. Not written up as a null, not written up as support.

**Control:** `Δ_shuffle` CI must contain 0 in **both** groups. If the correct null is not
clean in one of them, the between-group comparison is contaminated and the run is VOID.

**Confound reported either way:** Spearman ρ between stem-only baseline accuracy and
Δ_noise across all 20 pairs. If that is strong, any viseme effect may be a difficulty effect
wearing a label, and the write-up must say so rather than claim the viseme reading.

**FAIL-branch reachability:** demonstrated before the verdict is read, by feeding the gate
two groups drawn from the same population — which must return NO DIFFERENCE or
UNDERPOWERED, never DIFFERENCE.

## 5. Resources

**CPU only.** 20 pairs × 3 seeds × 4 arms ≈ 240 probes at the 5333×1536 shape, ~0.8 s each
measured in `C6_lrs2`, so roughly 5–10 minutes plus NAS reads. `OMP_NUM_THREADS=8`, shared
box. Checkpoints per pair. Rail 1b does not trigger.

## 6. What either outcome would mean

- **DIFFERENCE:** the artefact is weaker exactly where paper 3 makes its claims. Good for
  paper 3's robustness, and a limit on how far the C6 curve generalises *within* a corpus.
- **NO DIFFERENCE:** the artefact is uniform across pair types, so the keystone's number
  transfers to paper 3's pairs directly.
- **UNDERPOWERED:** says nothing, and will be recorded as saying nothing. With 5 pairs
  available in the entire dataset, this is a live outcome and not a failure of the design.
