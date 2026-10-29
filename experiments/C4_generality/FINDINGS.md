# FAIL — the effective-rank artefact does not generalise at the magnitude pre-registered

Candidate C4, the lead for an ICDE 2027 EAB submission. First gate of the ICDE loop.
`gates_tested` = 1.

## The stopping rule, quoted from `PLAN.md` §5 before any number was read

> **PASS** requires **both**, in **every one of the three datasets**:
> 1. `mean Δ_noise` over known-null blocks **> +0.02** balanced accuracy, bootstrap 95% CI
>    **excluding 0** — the Gaussian null calls a provably uninformative block informative.
> 2. `mean Δ_shuffle` over the same blocks has bootstrap 95% CI **containing 0** — the
>    proposed null is not itself broken.
>
> **FAIL** if either fails in any dataset.

## What ran

Three domains, none visual speech. Four real blocks each, all at the source width so the
Gaussian comparator is the same width; 5 seeds per block; **n = 20 units per dataset**.
Known-null blocks are real blocks with rows shuffled — real covariance, spectrum and
effective rank, provably zero label information. Readout is `probelib.fit`, the project's
own L2 logistic probe, which standardises columns on the train split, so the Gaussian
comparator matches the real block in marginals and differs **only** in correlation
structure.

| dataset | eff. rank, blocks | eff. rank, Gaussian | **Δ_noise** [95% CI] | **Δ_shuffle** [95% CI] | verdict |
|---|---|---|---|---|---|
| digits / vision | 6.43 – 13.17 | 61.79 | **+0.0079** [+0.0060, +0.0098] | +0.0013 [−0.0020, +0.0048] | FAIL |
| 20news / text | 4.58 – 23.49 | 126.51 | **−0.0001** [−0.0020, +0.0019] | −0.0019 [−0.0029, −0.0010] | FAIL |
| covtype / tabular | 1.57 – 1.87 | 53.86 | **+0.0025** [+0.0018, +0.0033] | +0.0001 [−0.0004, +0.0008] | FAIL |

**Verdict: FAIL**, against the gate as written, in all three datasets. Condition 1 fails
everywhere: the artefact is 2.5× below the threshold on vision, 8× below on tabular, and
**absent on text** (CI contains 0). Condition 2 additionally fails on text.

The gate was not softened and no secondary result was promoted. The threshold of +0.02 was
chosen before any external data was loaded.

## The result that matters more than the verdict

**Rank spread does not predict the artefact.** covtype has the most extreme spread measured
here — block effective rank 1.6–1.9 against a Gaussian 53.9, a ratio of **0.035** — and
produces the *second smallest* effect, +0.0025. digits has a far milder spread (ratio
0.10–0.21) and produces three times the effect. The ordering is backwards.

That matters because the catalogue's stated mechanism in `CLAUDE.md` is:

> Gaussian noise. Effective rank differs ~100×, so "beats noise" only means "dilutes less".

These three datasets say rank spread alone is **not** sufficient to produce the artefact at
a magnitude that would mislead anyone. Whatever drove the three false positives in `05` is
not captured by "the effective ranks differ", or is not captured by it alone. The catalogue
entry is still *correct advice* — own-rows-shuffled is a strictly safer null, and it came
out clean in 2 of 3 domains — but the explanation attached to it is not supported outside
LRS2 and should not be stated as a general mechanism.

## A second, smaller self-correction

On text, `Δ_shuffle` = −0.0019, CI [−0.0029, −0.0010], which **excludes 0**. The
own-rows-shuffled null — the one this project recommends — is itself not exactly neutral
there. The effect is tiny and in the conservative direction (the null scores *above* the
known-null block, so it would suppress a claim rather than invent one), but it is real and
should be reported rather than rounded to zero.

## A design flaw caught mid-run, and how

The first block set mixed widths. A 32-column block gets a 32-column Gaussian comparator
whose effective rank is ≈31, so no real block can sit below 25% of it and the VOID guard
fired on the guard's own arithmetic rather than on the data. The block set was amended to
keep every block at the source width.

This amendment was made **while blind to every effect size**: `evaluate()` returns at the
VOID check before any bootstrap is computed, so no `Δ` existed when the blocks were changed.
That is the only reason the amendment is legitimate. The surviving scope limit is real and
belongs in any write-up: **the artefact needs a block substantially lower-rank than its own
nominal width**, which narrow blocks cannot be.

Both non-PASS branches were demonstrated reachable before the real run — FAIL via a
near-separable base, VOID via an iid full-rank block. A gate that has not been shown to fail
is not a gate.

## What this does and does not establish

- **Does:** on three non-speech domains, with the project's own linear readout, the
  Gaussian-null artefact is either an order of magnitude smaller than on LRS2 or absent.
  C4's claim of "wide interest" — the EAB bar — is **not supported** for this entry.
- **Does:** rank spread is not the mechanism, or not the whole mechanism.
- **Does not:** show the artefact is absent on LRS2. It was measured there and overturned
  three claims; that stands.
- **Does not:** test the catalogue's other entries (mapping nulls, masking nulls, context
  nulls). Each would need its own generality evidence.
- **Does not:** say anything about non-linear probes, or about high-dimensional frozen
  representations of the kind `05` used. The regime where the artefact was found — a
  512-d frozen encoder base with ~1024-rank noise — is not among the three tested here, and
  that gap is the most plausible explanation for the discrepancy.

## What it means for the papers

- **ICDE 2027 EAB:** this was the cheapest question that could kill the submission, and it
  killed it. A catalogue whose headline entry does not generalise beyond one corpus does not
  meet "fundamentally new insights ... of wide interest".
- **Paper 3:** unaffected as to its own results, but the sentence attributing the `05`
  false positives to effective rank should be scoped to that setting rather than stated as a
  general mechanism. That is a correction to our own text, not to our own numbers.
- **`CLAUDE.md` control catalogue:** keep the recommendation, soften the explanation. The
  row should say the mechanism is unverified outside this corpus.

## Still worth running, if anyone returns to this

The one regime not tested is the one the artefact was found in: a high-dimensional frozen
representation as the base. Re-running this gate with a 512-d base and a ~1024-wide noise
block would say whether the discrepancy is about regime or about the claim. That needs the
cached `front512/` features and probably a GPU — and under hard rail 1b it needs an estimate
and a go-ahead first.
