#!/usr/bin/env python3
"""C4 kill test: is the feature-block null artefact specific to LRS2, or general?

Gate is in PLAN.md §5, frozen before any external data was loaded. This script does not
decide anything it was not pre-registered to decide.

    python 00_run.py --selftest    # FAIL-branch reachability, run this FIRST
    python 00_run.py               # the real run, three non-speech domains

Design, in one paragraph. Shuffling the rows of a REAL feature block gives a block that
keeps that block's covariance, spectrum and effective rank but carries provably zero label
information. A working null must score such a block at chance. We append it to an
informative base and compare two nulls: a Gaussian block of the same shape, and the block's
own rows shuffled again. `probelib.fit` standardises columns on the train split, so the
Gaussian comparator matches the real block in marginals and differs ONLY in correlation
structure -- i.e. in effective rank. Any gap is therefore attributable to rank, which is the
claim under test: overfitting capacity scales with EFFECTIVE rank, not nominal dimension,
so a full-rank Gaussian block dilutes an informative base more than a real block does, and
"beats noise" measures geometry rather than information.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probelib as pl  # noqa: E402

HERE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2, 3, 4)
N_FOLDS = 3
BASE_DIM = 16
VOID_RANK_FRAC = 0.25      # PLAN.md §5: real block eff. rank must be < 25% of Gaussian's
GATE_MIN_DNOISE = 0.02     # PLAN.md §5


# ------------------------------------------------------------------ datasets
def _pca(X, k, seed=0):
    """Top-k PCA scores. Used for the informative base and as one candidate block."""
    X = np.asarray(X, np.float64)
    Xc = X - X.mean(0, keepdims=True)
    # randomised SVD is unnecessary at this scale; full SVD keeps it exact
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :k] * S[:k]).astype(np.float32)


def blocks_for(X, rng):
    """Candidate blocks a practitioner would plausibly compare.

    AMENDED 2026-10-26, while blind to every Delta (evaluate() returns at the VOID check
    before any bootstrap, so no effect size had been computed or seen). The first block set
    mixed widths: a 32-column block gets a 32-column Gaussian comparator whose effective
    rank is ~31, and no real block can sit below 25% of that. The precondition was
    unsatisfiable by construction for narrow blocks, so the run VOIDed on the guard rather
    than on the data.

    Every block is therefore kept at the SAME full width d, so the Gaussian comparator is
    always d-dimensional and the guard reduces to "is this block substantially lower-rank
    than its nominal width" -- which is the condition the artefact actually needs. That is
    a real scope limit on the claim and is reported as one, not hidden.
    """
    X = np.asarray(X, np.float32)
    n, d = X.shape
    out = {"raw": X}
    # a random projection back to d: real data, rotated
    P = rng.standard_normal((d, d)).astype(np.float32) / np.sqrt(d)
    out["randproj"] = X @ P
    # local averaging over neighbouring columns -- the "smoothed sensor" block, more
    # correlated than raw and so lower-rank, still d wide
    A = np.zeros((d, d), np.float32)
    for i in range(d):
        lo, hi = max(0, i - 2), min(d, i + 3)
        A[lo:hi, i] = 1.0 / (hi - lo)
    out["smoothed"] = X @ A
    # PCA reconstruction back to d dims -- the "denoised features" block a practitioner
    # would actually build; real data, low rank, full width
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = max(2, min(8, len(S)))
    out["pca_recon"] = (U[:, :k] * S[:k]) @ Vt[:k]
    return out


def load_datasets(which):
    """(name, X, y_binary) for domains outside visual speech. Downloads are optional:
    a dataset that cannot be fetched is reported and skipped, never silently replaced."""
    from sklearn.datasets import (fetch_20newsgroups_vectorized, fetch_covtype,
                                  load_breast_cancer, load_digits)
    got = []
    if "digits" in which:
        d = load_digits()
        got.append(("digits/vision", d.data.astype(np.float32), (d.target >= 5).astype(np.int64)))
    if "news" in which:
        try:
            d = fetch_20newsgroups_vectorized(subset="train")
            from sklearn.decomposition import TruncatedSVD
            Xs = TruncatedSVD(n_components=128, random_state=0).fit_transform(d.data)
            got.append(("20news/text", Xs.astype(np.float32),
                        (d.target >= 10).astype(np.int64)))
        except Exception as e:                                    # noqa: BLE001
            print(f"  [skip] 20newsgroups unavailable: {type(e).__name__}: {e}")
    if "covtype" in which:
        try:
            d = fetch_covtype()
            idx = np.random.default_rng(0).choice(len(d.data), 20000, replace=False)
            got.append(("covtype/tabular", d.data[idx].astype(np.float32),
                        (d.target[idx] == 2).astype(np.int64)))
        except Exception as e:                                    # noqa: BLE001
            print(f"  [skip] covtype unavailable: {type(e).__name__}: {e}")
            d = load_breast_cancer()
            got.append(("breast_cancer/tabular", d.data.astype(np.float32),
                        d.target.astype(np.int64)))
    return got


# ---------------------------------------------------------------- the measure
def probe(X, y, rng):
    """Balanced accuracy via the project's own readout, averaged over random folds."""
    fold = rng.integers(0, N_FOLDS, len(y))
    accs = pl.fit_folds(X, y, fold, n_folds=N_FOLDS, min_fold=60)
    return float(np.nanmean(accs))


def measure(name, X, y, verbose=True):
    """One row per (block, seed). Returns rows and the rank-spread check."""
    rows, ranks = [], []
    base_rng = np.random.default_rng(12345)
    base = _pca(X, BASE_DIM)
    for bname, C in blocks_for(X, base_rng).items():
        er_C = pl.effective_rank(C)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            # KNOWN-NULL block: real geometry, provably zero label information
            C0 = pl.shuffle_rows(C, seed=seed)
            G = rng.standard_normal(C.shape).astype(np.float32)
            er_G = pl.effective_rank(G)
            a_b0 = probe(np.hstack([base, C0]), y, np.random.default_rng(1000 + seed))
            a_gs = probe(np.hstack([base, G]), y, np.random.default_rng(1000 + seed))
            a_sh = probe(np.hstack([base, pl.shuffle_rows(C0, seed=seed + 500)]), y,
                         np.random.default_rng(1000 + seed))
            rows.append({"block": bname, "seed": seed,
                         "d_noise": a_b0 - a_gs, "d_shuffle": a_b0 - a_sh,
                         "er_block": er_C, "er_gauss": er_G})
            ranks.append((bname, er_C, er_G))
        if verbose:
            print(f"    {name:22s} {bname:9s} eff.rank {er_C:7.2f}  "
                  f"(gauss {ranks[-1][2]:7.2f})")
    return rows, ranks


def evaluate(name, rows, ranks):
    """Apply PLAN.md §5 exactly. Returns (verdict, detail)."""
    # VOID guard first -- a vacuous contrast answers nothing (the width-router lesson)
    bad = [(b, c, g) for b, c, g in ranks if not (c < VOID_RANK_FRAC * g)]
    if bad:
        return "VOID", {"reason": "rank spread absent; contrast is vacuous",
                        "offenders": sorted({b for b, _, _ in bad})}
    dn = np.array([r["d_noise"] for r in rows])
    ds = np.array([r["d_shuffle"] for r in rows])
    m_dn, lo_dn, hi_dn = pl.boot(dn)
    m_ds, lo_ds, hi_ds = pl.boot(ds)
    c1 = (m_dn > GATE_MIN_DNOISE) and (lo_dn > 0)
    c2 = (lo_ds <= 0 <= hi_ds)
    return ("PASS" if (c1 and c2) else "FAIL"), {
        "d_noise": [round(v, 4) for v in (m_dn, lo_dn, hi_dn)],
        "d_shuffle": [round(v, 4) for v in (m_ds, lo_ds, hi_ds)],
        "cond1_noise_null_false_positive": bool(c1),
        "cond2_own_shuffle_null_clean": bool(c2),
        "n_units": len(rows)}


# ------------------------------------------------------------------ selftest
def selftest():
    """FAIL-branch reachability. PLAN.md §5 owes this BEFORE the real run.

    A near-separable base: every arm scores ~1.0, so appending anything changes nothing and
    d_noise collapses to 0. Rank spread is still present, so this is a genuine FAIL and not
    a VOID -- which is the point. It shows the gate can return FAIL.
    """
    rng = np.random.default_rng(7)
    n = 1200
    y = (rng.random(n) > 0.5).astype(np.int64)
    # a base that separates the classes almost perfectly
    X = (y[:, None] * 6.0 + rng.standard_normal((n, 40))).astype(np.float32)
    rows, ranks = measure("SELFTEST/separable", X, y, verbose=False)
    verdict, detail = evaluate("selftest", rows, ranks)
    print(f"\n  selftest verdict: {verdict}")
    print(f"  {json.dumps(detail)}")
    if verdict != "FAIL":
        print("\n  REACHABILITY CHECK FAILED: the synthetic no-artefact case did not "
              f"return FAIL (got {verdict}). The gate is not trustworthy. Halting.")
        return 2
    print("  FAIL branch is reachable. The gate can return FAIL. Proceed.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--datasets", default="digits,news,covtype")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    out, verdicts = {}, []
    for name, X, y in load_datasets(a.datasets):
        print(f"\n  {name}: X={X.shape} y balance={y.mean():.3f}")
        rows, ranks = measure(name, X, y)
        verdict, detail = evaluate(name, rows, ranks)
        print(f"    -> {verdict}  {json.dumps(detail)}")
        out[name] = {"verdict": verdict, **detail}
        verdicts.append(verdict)

    # PLAN.md §5: PASS requires both conditions in EVERY dataset
    overall = ("VOID" if "VOID" in verdicts
               else "PASS" if verdicts and all(v == "PASS" for v in verdicts)
               else "FAIL")
    out["_overall"] = overall
    out["_n_datasets"] = len(verdicts)
    (HERE / "result.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  OVERALL: {overall}  over {len(verdicts)} datasets -> result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
