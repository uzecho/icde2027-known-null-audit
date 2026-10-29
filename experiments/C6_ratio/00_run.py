#!/usr/bin/env python3
"""C6 kill test: is the Gaussian-null artefact governed by r = eff_rank(noise)/n_train?

Gate is PLAN.md §3, frozen before the curve was measured.

    python 00_run.py --selftest   # FAIL-branch reachability, run FIRST
    python 00_run.py              # the sweep

Method: hold the block set, the Gaussian width and the rank spread fixed; subsample n_train
only. That moves r without touching the confound C4 already eliminated (rank spread does
not predict the artefact).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "C4_generality"))
import probelib as pl  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "c4", Path(__file__).resolve().parent.parent / "C4_generality" / "00_run.py")
c4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c4)

HERE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2, 3, 4)
BASE_DIM = c4.BASE_DIM
GATE_MIN_RHO = 0.5
GATE_MIN_DNOISE = 0.02
GATE_TOP_R = 0.15
VOID_MIN_SPAN = 10.0


def probe_at(X, y, n_train, seed):
    """Balanced accuracy with a held-out split of fixed train size n_train."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    tr, te = idx[:n_train], idx[n_train:n_train + max(300, n_train)]
    if len(te) < 100 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return float("nan")
    import torch
    t = lambda a, d: torch.as_tensor(np.asarray(a, d))          # noqa: E731
    return pl.fit(t(X[tr], np.float32), t(y[tr], np.int64),
                  t(X[te], np.float32), t(y[te], np.int64))


def sweep(name, X, y, verbose=True):
    """One row per (block, ratio, seed)."""
    base = c4._pca(X, BASE_DIM)
    blocks = c4.blocks_for(X, np.random.default_rng(12345))
    # Sweep range is chosen to SATISFY the two pre-registered VOID guards (span >= 10x,
    # top r >= 0.15), not to shape the answer. Set 2026-10-26 while blind to every curve.
    # The naive range failed both: digits spanned 9x and covtype topped out at r = 0.077,
    # which would have VOIDed on the guards' own arithmetic rather than on the data.
    n_max = int(0.7 * len(y))
    er_ref = pl.effective_rank(np.random.default_rng(0).standard_normal(
        (min(len(y), 4000), next(iter(blocks.values())).shape[1])))
    n_min = int(min(max(100, er_ref / 0.30), n_max / 12))   # top r ~0.30, span >= 12x
    sizes = sorted({int(s) for s in np.geomspace(n_min, n_max, 7)})
    rows = []
    for bname, C in blocks.items():
        d = C.shape[1]
        for n_tr in sizes:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                C0 = pl.shuffle_rows(C, seed=seed)
                G = rng.standard_normal(C.shape).astype(np.float32)
                er_G = pl.effective_rank(G[:min(len(G), 4000)])
                a_b0 = probe_at(np.hstack([base, C0]), y, n_tr, 1000 + seed)
                a_gs = probe_at(np.hstack([base, G]), y, n_tr, 1000 + seed)
                if np.isnan(a_b0) or np.isnan(a_gs):
                    continue
                rows.append({"block": bname, "n_train": n_tr, "seed": seed,
                             "r": er_G / n_tr, "d_noise": a_b0 - a_gs})
        if verbose:
            rs = [x["r"] for x in rows if x["block"] == bname]
            print(f"    {name:20s} {bname:10s} d={d:4d}  r {min(rs):.4f} .. {max(rs):.4f}")
    return rows


def evaluate(rows):
    if not rows:
        return "VOID", {"reason": "no usable cells"}
    r = np.array([x["r"] for x in rows])
    dn = np.array([x["d_noise"] for x in rows])

    span = r.max() / max(r.min(), 1e-12)
    if span < VOID_MIN_SPAN:
        return "VOID", {"reason": f"r span {span:.1f}x < {VOID_MIN_SPAN}x; no leverage"}
    if r.max() < GATE_TOP_R:
        return "VOID", {"reason": f"top r {r.max():.3f} < {GATE_TOP_R}; condition 2 untested"}

    from scipy.stats import spearmanr
    rho = float(spearmanr(r, dn).statistic)
    # bootstrap rho over cells
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(r), len(r))
        if len(np.unique(r[i])) > 2:
            bs.append(float(spearmanr(r[i], dn[i]).statistic))
    lo_rho, hi_rho = np.percentile(bs, [2.5, 97.5])

    top = r >= np.percentile(r, 90)
    m_top, lo_top, hi_top = pl.boot(dn[top])

    c1 = (rho > GATE_MIN_RHO) and (lo_rho > 0)
    c2 = (m_top > GATE_MIN_DNOISE) and (lo_top > 0)
    return ("PASS" if (c1 and c2) else "FAIL"), {
        "rho": round(rho, 3), "rho_CI": [round(lo_rho, 3), round(hi_rho, 3)],
        "r_span": round(span, 1), "r_max": round(float(r.max()), 4),
        "d_noise_at_top_r": [round(v, 4) for v in (m_top, lo_top, hi_top)],
        "cond1_monotone": bool(c1), "cond2_bites_at_high_r": bool(c2),
        "n_cells": len(rows)}


def selftest():
    """FAIL-branch reachability: Δ_noise flat in r must return FAIL, not PASS."""
    rng = np.random.default_rng(11)
    n, d = 4000, 60
    y = (rng.random(n) > 0.5).astype(np.int64)
    # labels depend on a feature NOT in the block, so every arm sits at chance at every r
    X = rng.standard_normal((n, d)).astype(np.float32)
    rows = sweep("SELFTEST/flat", X, y, verbose=False)
    v, d_ = evaluate(rows)
    print(f"\n  selftest verdict: {v}\n  {json.dumps(d_)}")
    if v != "FAIL":
        print(f"  REACHABILITY CHECK FAILED: flat case returned {v}, not FAIL. Halting.")
        return 2
    print("  FAIL branch reachable. Proceed.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--datasets", default="digits,news,covtype")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    out, n_pass = {}, 0
    for name, X, y in c4.load_datasets(a.datasets):
        print(f"\n  {name}: X={X.shape}")
        rows = sweep(name, X, y)
        v, det = evaluate(rows)
        print(f"    -> {v}  {json.dumps(det)}")
        out[name] = {"verdict": v, **det}
        n_pass += (v == "PASS")
    # PLAN.md §3: PASS requires both conditions in >= 2 of 3 datasets
    out["_overall"] = "PASS" if n_pass >= 2 else "FAIL"
    out["_n_pass"] = n_pass
    (HERE / "result.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  OVERALL: {out['_overall']}  ({n_pass}/3 datasets passed) -> result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
