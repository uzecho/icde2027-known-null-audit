# C6 keystone — does the external curve PREDICT LRS2?

**Pre-registered 2026-10-27. The prediction below was computed from external data only and
is frozen before any LRS2 accuracy has been read.** Shapes and CPU timings were measured,
and the timing was run on synthetic arrays of the LRS2 shape precisely so that no real value
was seen.

This is a **confirmation of an out-of-sample point prediction**, not a third re-gate. That
distinction is the whole reason for running it: `C6_ratio/FINDINGS.md` §7 warned that a
third gate written after seeing two results is how a loop manufactures a false positive. A
prediction made before the measurement, on a corpus not used to fit it, carries the opposite
property — it can only be confirmed by being right.

## 1. The prediction

Five external points, all from the same protocol, none from LRS2:

| source | dataset | r | Δ_noise |
|---|---|---:|---:|
| C4 | covtype | 0.0040 | +0.0025 |
| C4 | 20news | 0.0168 | −0.0001 |
| C4 | digits | 0.0516 | +0.0079 |
| C6 | covtype (top r) | 0.3011 | +0.0301 |
| C6 | digits (top r) | 0.5943 | +0.0589 |

Least squares: **Δ_noise = 0.0974·r + 0.0010**, residual sd **0.0020**.

LRS2's operating point is **measured, not assumed**. Experiment `05` used `PER_CLASS = 4000`
and `NFOLD = 3`, so n_train ≈ 5333; its widest candidates (`crop_px`, `face_px`) are 1024
columns, and a 1024-wide Gaussian at n = 5333 has effective rank **859.1**, giving

    r_LRS2 = 859.1 / 5333 = 0.1611

(`CLAUDE.md` says "noise ~1024", which is the nominal width; the effective rank is 859.)

> ### PREDICTION
> **Δ_noise at r = 0.1611 is +0.0167, bootstrap 95% interval [+0.0149, +0.0238].**

Note this interval straddles the +0.02 threshold used in C4 and C6. That is deliberate and
is not a hedge: the question here is whether `r` **predicts**, not whether LRS2's artefact
is large. A confirmed prediction below +0.02 would still confirm the curve.

## 2. What is being measured

Experiment `05`'s own protocol, reused rather than reinvented:

- base: `X_front512.npy`, the 512-d frozen stem — `05`'s 0.596 baseline
- candidate: `crop_px` (1024 columns), from `frontend_ceiling/packed/`
- paired sampling within utterance, `PER_CLASS = 4000`, 3 folds, L2 = 1.0
- **known-null block**: `crop_px` with rows shuffled — real covariance, spectrum and
  effective rank, provably zero label information
- `Δ_noise   = acc(stem ⊕ crop_px₀) − acc(stem ⊕ Gaussian₁₀₂₄)`
- `Δ_shuffle = acc(stem ⊕ crop_px₀) − acc(stem ⊕ reshuffle(crop_px₀))`
- several phone pairs as the unit of analysis; `probelib.boot` for CIs

A sweep over n_train is also recorded, so the LRS2 curve can be compared in shape to the
external ones. **The sweep is secondary. The gate is the point prediction.**

## 3. Gate

**CONFIRMED** requires both:

1. **Point prediction.** At r within ±10% of 0.1611, the measured `Δ_noise` bootstrap 95% CI
   **overlaps** the pre-registered interval **[+0.0149, +0.0238]**.
2. **Control.** `Δ_shuffle` bootstrap 95% CI **contains 0** — the correct null is clean on
   LRS2 too.

**NOT CONFIRMED** if either fails. In particular, a measured `Δ_noise` far above the
interval would mean `r` is *not* sufficient and something else about this regime — base
dimensionality is the obvious suspect, 512 here against 16 externally — carries the effect.
That outcome is informative and must be reported as a falsification of the sufficiency
claim, not quietly folded into "the artefact is real".

**VOID** if the realised r deviates from 0.1611 by more than 10%, or if fewer than 5 phone
pairs yield usable samples.

**FAIL-branch reachability**: the gate FAILS whenever the measured CI misses a fixed
interval of width 0.009, which any sufficiently different result does. Demonstrated
numerically before the verdict is read.

## 4. Resources — CPU only, no GPU requested

Measured, not guessed: **0.8 s per probe** at the LRS2 shape (5333 × 1536) on CPU, **33 MB
per arm**. The sweep is ~90–150 probes, so single-digit minutes of compute; the real cost is
scattered reads from the NAS memmaps, which `probelib.gather` handles with a sequential
pass. Hard rail 1b does not trigger. If any later stage needs a card, it stops and asks with
peak memory, wall-clock, card and the prior run behind the estimate.

## 5. What confirmation would and would not establish

- **Would:** that a curve fitted on vision, text and tabular data predicts, out of sample,
  the magnitude of the artefact on a visual-speech corpus it never saw — which is the
  "wide interest" evidence the EAB bar wants, and which neither C4 nor C6 supplied.
- **Would:** retro-explain `05`'s three false positives quantitatively rather than by story.
- **Would not:** rescue C6's failed monotonicity replication. That failure stands.
- **Would not:** extend to non-linear probes or to readouts other than this one.
- **Would not:** on its own stop the loop. `STOP_RULE.md` still requires the stopping
  candidate to clear its control and replicate; this confirms a prediction, and the loop's
  bookkeeping must show it as such.
