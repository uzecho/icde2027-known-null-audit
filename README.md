# A Known-Null Audit of Noise-Injection Feature Checks

Artifact for the ICDE 2027 submission *"A Known-Null Audit of Noise-Injection
Feature Checks in Data Augmentation Pipelines"*
[Experiment, Analysis, and Benchmark].

**Three of the four corpora are public and reproduce end to end on a laptop
CPU in about half an hour, no GPU.** The fourth (LRS2, visual speech) is licence-restricted; its
derived features cannot be redistributed, and `LRS2_REGENERATION.md` documents
how to rebuild them from a licensed copy. This is stated here rather than
claimed away: the paper's headline result does not depend on it.

## The claim, in one sentence

Comparing a candidate feature block against a block of Gaussian noise does not
merely add variance to your answer -- it biases it, and **the direction of the
bias is set by your readout**. Every readout that fits weights reports an
uninformative block as informative; every readout that measures distances
reports it as actively harmful. Of the 24 cells, 18 resolve, all 18 on the side
their family implies and none against it.

**What the own-rows-shuffled null does and does not buy you.** Its `Delta_shuffle`
sits at zero in 23 of the 24 cells, but that is close to a tautology and we do not
offer it as evidence: the two arms are exchangeable row permutations of the same
block, so `E[Delta_shuffle] = 0` holds by construction. The non-tautological
evidence is the decision experiment in `experiments/C9_power/`, and it is narrower:
at `p = 1` the logistic readout's false-keep rate falls from 1.00 / 0.60 / 1.00
(Gaussian, on digits / 20news / covtype) to 0.10 / 0.10 / 0.30 (shuffle), with no
loss of power on an intact block. **That gate's own verdict was INCONCLUSIVE** --
its passing branch turned out to be unreachable as written, which
`experiments/C9_power/FINDINGS.md` states before any number.

## Quick start

```bash
pip install -r requirements.txt
./reproduce.sh          # the three public corpora, all public gates
python verify.py        # check the numbers against the ones in the paper
```

`reproduce.sh` took **1651 s** (27.5 min) end to end with `OMP_NUM_THREADS=8` on
a CPU-only machine; roughly 18 min of that is `C9_power`, which `SKIP_C9=1`
leaves out. `verify.py` then reported **224 of 224 checks agreeing with the
paper, 0 disagreeing, 0 missing**: every published confidence interval contains
the rerun's point estimate, and every pre-registered verdict is unchanged.

All 224 are checks of files that run actually rewrote. That is not a given: an
earlier version of this artifact checked 272, of which 56 were on files no rerun
touched, so they compared a shipped file against itself and could not fail. Gate
A11 now refuses the build unless every checked file has a generator in the tree
*and* `reproduce.sh` invokes it. No GPU, no licensed data, no download beyond
what scikit-learn fetches.

## What is in here

| path | what |
|---|---|
| `probelib.py` | the probe, the controls, `effective_rank`, `shuffle_rows`, `permute_labels` |
| `experiments/C4_generality/` | does the artefact generalise beyond one corpus? **FAIL**. `01_ranks.py` recomputes its effective ranks after standardisation and withdraws its secondary reading |
| `experiments/C6_ratio/` | does magnitude track `r = eff_rank/n_train`? PASS, control cleared, **replication failed** |
| `experiments/C6_lrs2/` | out-of-sample point prediction on LRS2. **CONFIRMED 3x** *(needs licensed data)* |
| `experiments/C6_viseme/` | is it uniform within a corpus? DIFFERENCE, **unreadable as a viseme effect** *(needs licensed data)* |
| `experiments/C7_readout/` | does it survive a change of readout? **SURVIVES, and kNN reverses the sign** |
| `experiments/C8_mechanism/` | does the mechanism *predict* unseen readouts? **PARTIAL 3/4** |
| `experiments/C9_power/` | do the two baselines differ as KEEP/DROP decision rules? **INCONCLUSIVE** by its own gate, whose passing branch was unreachable; the false-keep contrast is the fix's only non-tautological evidence |
| `experiments/C11_covmatch/` | is the sign flip caused by correlation structure? **MIXED** -- it holds for weight-fitting readouts only |
| `experiments/C12_marginal/` | does matching the marginals explain the residual? **MARGINALS DO NOT EXPLAIN** |
| `experiments/C13_rconvention/` | defect repair: `r` had been computed three ways; this fixes one convention and re-reads what changes |
| `check_stop.py`, `STOP_RULE.md` | the machine-checked stopping rule, pre-registered before any candidate was tested |

Every experiment folder contains, in this order of precedence:

1. `PLAN.md` -- the **pre-registration**, written and frozen before the data was
   touched. The gate is quoted verbatim in `FINDINGS.md` before any number.
2. `00_run.py` -- the runner. `--selftest` demonstrates the FAIL branch is
   reachable; this is run **before** the real run, every time.
3. `result.json` -- what came out.
4. `FINDINGS.md` -- the verdict judged against the gate as written, not softened.

## Five of the nine gates did not pass cleanly

The family-wise count is **`gates_tested` = 9**, and the paper states it. One
returned FAIL outright (`C4_generality`); one passed its gate but failed its own
pre-registered replication (`C6_ratio`); one returned PARTIAL (`C8_mechanism`);
one returned INCONCLUSIVE (`C9_power`); and one returned MIXED (`C11_covmatch`).
All nine ship with their pre-registrations, their runners and their results,
exactly like the four that passed. At a 95% interval nine gates carry roughly a
37% chance that at least one passes spuriously, which is why `STOP_RULE.md`
requires a gate to pass, its control to clear *and* the result to replicate
before a claim counts as settled.

## Reproducibility, scoped honestly

- **Public (C4, C6_ratio, C7, C8, C9, C11, C12, C13):** `./reproduce.sh`
  downloads the corpora via scikit-learn and reruns every gate. No licensed
  data, no GPU.
- **Licensed (C6_lrs2, C6_viseme):** needs features derived from LRS2. Set
  `$LRS2_DERIVED` after following `LRS2_REGENERATION.md`. The runners refuse
  with an explanation rather than failing obscurely if it is unset.
- Bootstrap CIs are seeded; `verify.py` asserts each rerun lands inside the
  interval the paper reports, and prints both numbers either way.
