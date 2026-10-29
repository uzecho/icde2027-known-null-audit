# The Sign of Your Error Depends on Your Probe

Artifact for the ICDE 2027 submission *"The Sign of Your Error Depends on Your
Probe: Gaussian Null Baselines in Representation Evaluation"*
[Experiment, Analysis & Benchmark].

**Three of the four corpora are public and reproduce end to end on a laptop
CPU in minutes.** The fourth (LRS2, visual speech) is licence-restricted; its
derived features cannot be redistributed, and `LRS2_REGENERATION.md` documents
how to rebuild them from a licensed copy. This is stated here rather than
claimed away: the paper's headline result does not depend on it.

## The claim, in one sentence

Comparing a candidate feature block against a block of Gaussian noise does not
merely add variance to your answer -- it biases it, and **the direction of the
bias is set by your readout**. Every readout that fits weights reports an
uninformative block as informative; every readout that measures distances
reports it as actively harmful. The own-rows-shuffled null is correct in all 24
cells measured.

## Quick start

```bash
pip install -r requirements.txt
./reproduce.sh          # the three public corpora, all public gates
python verify.py        # check the numbers against the ones in the paper
```

`reproduce.sh` took **106 s** end to end with `OMP_NUM_THREADS=4` on a CPU-only
machine, and `verify.py` then reported **82 of 82 checks agreeing with the
paper**: every published confidence interval contains the rerun's point
estimate, and every pre-registered verdict is unchanged. No GPU, no licensed
data, no download beyond what scikit-learn fetches.

## What is in here

| path | what |
|---|---|
| `probelib.py` | the probe, the controls, `effective_rank`, `shuffle_rows`, `permute_labels` |
| `experiments/C4_generality/` | does the artefact generalise beyond one corpus? **FAIL** |
| `experiments/C6_ratio/` | does magnitude track `r = eff_rank/n_train`? PASS, control cleared, **replication failed** |
| `experiments/C6_lrs2/` | out-of-sample point prediction on LRS2. **CONFIRMED 3x** *(needs licensed data)* |
| `experiments/C6_viseme/` | is it uniform within a corpus? DIFFERENCE, **unreadable as a viseme effect** *(needs licensed data)* |
| `experiments/C7_readout/` | does it survive a change of readout? **SURVIVES, and kNN reverses the sign** |
| `experiments/C8_mechanism/` | does the mechanism *predict* unseen readouts? **PARTIAL 3/4** |
| `check_stop.py`, `STOP_RULE.md` | the machine-checked stopping rule, pre-registered before any candidate was tested |

Every experiment folder contains, in this order of precedence:

1. `PLAN.md` -- the **pre-registration**, written and frozen before the data was
   touched. The gate is quoted verbatim in `FINDINGS.md` before any number.
2. `00_run.py` -- the runner. `--selftest` demonstrates the FAIL branch is
   reachable; this is run **before** the real run, every time.
3. `result.json` -- what came out.
4. `FINDINGS.md` -- the verdict judged against the gate as written, not softened.

## Two of the six gates failed and one was partial

`C4_generality` FAILED and `C6_ratio`'s replication FAILED. They ship with their
pre-registrations, their runners and their results, exactly like the ones that
passed. The family-wise count is **`gates_tested` = 6**, and the paper states it.

## Reproducibility, scoped honestly

- **Public (C4, C6_ratio, C7, C8):** `./reproduce.sh` downloads the corpora via
  scikit-learn and reruns every gate. No licensed data, no GPU.
- **Licensed (C6_lrs2, C6_viseme):** needs features derived from LRS2. Set
  `$LRS2_DERIVED` after following `LRS2_REGENERATION.md`. The runners refuse
  with an explanation rather than failing obscurely if it is unset.
- Bootstrap CIs are seeded; `verify.py` asserts each rerun lands inside the
  interval the paper reports, and prints both numbers either way.
