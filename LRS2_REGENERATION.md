# Regenerating the LRS2-derived features

The LRS2-BBC corpus is distributed under a licence that does not permit
redistribution, and that restriction extends to features derived from it. Two
experiments here (`C6_lrs2`, `C6_viseme`) therefore ship their code, their
pre-registrations and their per-pair results, but not their inputs.

**The paper's headline result does not depend on these two experiments.** They
supply the out-of-sample confirmation of the magnitude claim and the
within-corpus non-uniformity result. Sections 4 and 6 reproduce entirely from
the public corpora.

## What is needed

| artefact | shape | what it is |
|---|---|---|
| `ambiguity_map/frames/*.npy` | per-arm frame features | frozen 512-d encoder features and the pixel/landmark arms, one file per arm |
| `ambiguity_map/align_mfa.jsonl` | one record per utterance | Montreal Forced Aligner phone and word alignments |
| `frontend_ceiling/packed/` | packed arrays | the packed per-frame blocks the probes read |

Point `$LRS2_DERIVED` at the directory holding `ambiguity_map/` and
`frontend_ceiling/`.

## How they were built

1. Obtain LRS2-BBC from the Visual Geometry Group under their licence.
2. Run the Auto-AVSR preprocessing to 96x96 mouth crops at 25 fps.
3. Extract frozen encoder features with the `vsr_trlrs2lrs3vox2avsp_base`
   checkpoint. One encoder step is one video frame (40 ms); there is no
   subsampling, so MFA labels align to frames directly.
4. Align audio with the Montreal Forced Aligner to produce `align_mfa.jsonl`.
5. Pack per-arm frame features into `ambiguity_map/frames/`.

Steps 2-5 are the standard Auto-AVSR feature pipeline. If you have a licensed
copy of LRS2 and hit a mismatch, open an issue: the per-pair result JSONs in
`experiments/C6_lrs2/` give the exact `n_train`, `r` and accuracy for every pair,
so a partial rebuild can still be checked against ours.
