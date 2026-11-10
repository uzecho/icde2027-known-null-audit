#!/usr/bin/env python3
"""Does the row-shuffle baseline make better KEEP/DROP decisions? Gate: PLAN.md §7.

    python 00_run.py --selftest   # FAIL-branch reachability, run FIRST
    python 00_run.py --smoke      # one cell, to measure cost before committing
    python 00_run.py
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probelib as pl  # noqa: E402

import importlib.util  # noqa: E402
HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("c7", HERE.parent / "C7_readout" / "00_run.py")
c7 = importlib.util.module_from_spec(_s); _s.loader.exec_module(c7)
c4 = c7.c4

P_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
R_REPLICATES = 10
SEEDS = (0, 1, 2)                       # units within one decision, with 4 blocks
FPR_MAX = 0.10                          # PLAN.md §5
POWER_MIN = 0.80                        # PLAN.md §5
MIN_CELLS_GOOD = 7                      # PLAN.md §7
MIN_CELLS_BAD = 3
VOID_SIGNAL_MIN = 0.01                  # PLAN.md §7
MIN_CELLS_ALIVE = 6


# ------------------------------------------------------------------ readouts
def _readouts():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    def fit(make):
        def f(Xtr, ytr, Xte, yte):
            a, b = c7._std(Xtr, Xte)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return c7._bal(yte, make().fit(a, ytr).predict(b))
        return f

    return {
        "logistic": c7.r_logistic,
        "knn5": fit(lambda: KNeighborsClassifier(n_neighbors=5)),
        # ARDA's RIFS ranks with an ensemble of random forests and sparse
        # regression, so this is the readout the motivating system uses.
        "rf": fit(lambda: RandomForestClassifier(n_estimators=120, n_jobs=1,
                                                 random_state=0)),
    }


# ----------------------------------------------------- graded-signal blocks
def degrade(C, p, rng):
    """Permute a fraction p of rows among themselves; leave the rest in place.

    Every column marginal, the covariance, the spectrum and the effective rank
    are identical for every p -- only rows move. The sole quantity varying is
    label information: p=0 keeps it all, p=1 destroys it all.
    """
    C = np.asarray(C)
    n = len(C)
    k = int(round(p * n))
    if k < 2:
        return C.copy()
    idx = rng.choice(n, k, replace=False)
    out = C.copy()
    out[idx] = C[idx[rng.permutation(k)]]
    return out


# ------------------------------------------------------------------ measure
def one_decision(X, y, fn, n_train, p, master_seed):
    """12 units -> two bootstrap CIs -> two KEEP/DROP decisions, plus signal."""
    base = c4._pca(X, c4.BASE_DIM)
    blocks = c4.blocks_for(X, np.random.default_rng(12345))
    dg, ds, sig = [], [], []
    for _, C in blocks.items():
        for s in SEEDS:
            rng = np.random.default_rng(master_seed * 1000 + s)
            idx = rng.permutation(len(y))
            tr, te = idx[:n_train], idx[n_train:n_train + max(400, n_train)]
            if len(te) < 150 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            Cp = degrade(C, p, rng)
            G = rng.standard_normal(C.shape).astype(np.float32)
            SH = pl.shuffle_rows(Cp, seed=int(rng.integers(1 << 30)))
            Xp, Xg, Xs = (np.hstack([base, Z]) for Z in (Cp, G, SH))
            a_p = fn(Xp[tr], y[tr], Xp[te], y[te])
            dg.append(a_p - fn(Xg[tr], y[tr], Xg[te], y[te]))
            ds.append(a_p - fn(Xs[tr], y[tr], Xs[te], y[te]))
            sig.append(a_p - fn(base[tr], y[tr], base[te], y[te]))
    if not dg:
        return None
    keep = {}
    for name, d in (("gauss", dg), ("shuffle", ds)):
        _, lo, _ = pl.boot(np.array(d))
        keep[name] = bool(lo > 0)
    return keep, float(np.mean(sig))


def run_cell(X, y, fn, n_train, label, t0):
    out = {"p": {}, "signal_at_p0": None}
    for p in P_LEVELS:
        kg = ks = 0
        for r in range(R_REPLICATES):
            got = one_decision(X, y, fn, n_train, p, master_seed=r + 1)
            if got is None:
                continue
            keep, sig = got
            kg += keep["gauss"]; ks += keep["shuffle"]
            if p == 0.0 and out["signal_at_p0"] is None:
                out["signal_at_p0"] = sig
        out["p"][f"{p:.2f}"] = {"gauss_keep": kg / R_REPLICATES,
                                "shuffle_keep": ks / R_REPLICATES}
        print(f"      p={p:.2f}  gauss KEEP {kg}/{R_REPLICATES}   "
              f"shuffle KEEP {ks}/{R_REPLICATES}   [{time.time()-t0:.0f}s]")
    return out


def judge(cells):
    """PLAN.md §7, applied exactly. VOID guard first."""
    alive = {k: v for k, v in cells.items()
             if v.get("signal_at_p0") is not None
             and v["signal_at_p0"] > VOID_SIGNAL_MIN}
    voided = sorted(set(cells) - set(alive))
    if len(alive) < MIN_CELLS_ALIVE:
        return "VOID", {"reason": f"only {len(alive)} cells carry detectable "
                                  f"signal at p=0 (need {MIN_CELLS_ALIVE})",
                        "voided": voided}, alive

    def ok(v, who):
        fpr = v["p"]["1.00"][f"{who}_keep"]
        pwr = v["p"]["0.00"][f"{who}_keep"]
        return (fpr <= FPR_MAX) and (pwr >= POWER_MIN), fpr, pwr

    sg = {k: ok(v, "gauss") for k, v in alive.items()}
    ss = {k: ok(v, "shuffle") for k, v in alive.items()}
    n_s_good = sum(a for a, _, _ in ss.values())
    n_g_good = sum(a for a, _, _ in sg.values())
    n_g_bad = len(alive) - n_g_good
    n_s_bad = len(alive) - n_s_good

    if n_s_good >= MIN_CELLS_GOOD and n_g_bad >= MIN_CELLS_BAD:
        v = "SHUFFLE BETTER"
    elif n_s_good >= MIN_CELLS_GOOD and n_g_good >= MIN_CELLS_GOOD:
        v = "TIE"
    elif n_s_bad >= MIN_CELLS_BAD and n_g_bad < MIN_CELLS_BAD:
        v = "SHUFFLE WORSE"
    else:
        v = "INCONCLUSIVE"
    det = {"n_cells": len(alive), "voided": voided,
           "shuffle_good": n_s_good, "gauss_good": n_g_good,
           "per_cell": {k: {"shuffle": {"ok": ss[k][0], "fpr": ss[k][1],
                                        "power": ss[k][2]},
                            "gauss": {"ok": sg[k][0], "fpr": sg[k][1],
                                      "power": sg[k][2]}} for k in alive}}
    return v, det, alive


def run(datasets, label, readouts=None, p_levels=None, reps=None):
    global P_LEVELS, R_REPLICATES
    if p_levels is not None:
        P_LEVELS = p_levels
    if reps is not None:
        R_REPLICATES = reps
    ro = readouts if readouts is not None else _readouts()
    cells, t0 = {}, time.time()
    for name, X, y in c4.load_datasets(datasets):
        er = pl.effective_rank(np.random.default_rng(0).standard_normal(
            (min(len(y), 3000), c4.blocks_for(X, np.random.default_rng(1))["raw"].shape[1])))
        n_train = int(min(max(100, er / 0.30), int(0.7 * len(y)) / 2))
        print(f"\n  {name}  n_train={n_train}")
        for rname, fn in ro.items():
            print(f"    {rname}")
            cells[f"{name}|{rname}"] = run_cell(X, y, fn, n_train, rname, t0)
            (HERE / f"partial_{label}.json").write_text(
                json.dumps(cells, indent=2) + "\n")
    verdict, det, _ = judge(cells)
    print(f"\n  VERDICT: {verdict}")
    print(f"  {json.dumps({k: v for k, v in det.items() if k != 'per_cell'})}")
    return verdict, det, cells


def selftest():
    """Real block replaced by a known null at every p: nobody can be powered."""
    print("  selftest: the 'real' block carries zero information at every p,\n"
          "            so power is impossible and the VOID guard should fire")
    real_blocks = c4.blocks_for

    def dead(X, rng):
        return {k: pl.shuffle_rows(v, seed=7)
                for k, v in real_blocks(X, rng).items()}

    c4.blocks_for = dead
    try:
        ro = _readouts()
        v, det, _ = run("digits", "selftest", readouts={"logistic": ro["logistic"]},
                        p_levels=(0.0, 1.0), reps=6)
    finally:
        c4.blocks_for = real_blocks
    if v == "SHUFFLE BETTER":
        print(f"  REACHABILITY CHECK FAILED: got {v} on a dead block. Halting.")
        return 2
    print(f"  Non-BETTER branch reachable (got {v}). Proceed.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", default="digits,news,covtype")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.smoke:
        ro = _readouts()
        t = time.time()
        run("digits", "smoke", readouts={"rf": ro["rf"]},
            p_levels=(0.0, 1.0), reps=3)
        el = time.time() - t
        full = el * (3 / 1) * (5 / 2) * (10 / 3) * 3        # readouts, p, reps, corpora
        print(f"\n  smoke: {el:.0f}s for 1 readout x 2 p x 3 reps on digits")
        print(f"  extrapolated full run: ~{full/60:.0f} min "
              f"(9 cells x 5 p x 10 reps), CPU only")
        return 0
    verdict, det, cells = run(a.datasets, "real")
    (HERE / "result.json").write_text(json.dumps(
        {"verdict": verdict, **det, "cells": cells}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
