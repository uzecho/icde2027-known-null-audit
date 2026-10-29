#!/usr/bin/env python3
"""Does the Gaussian-null artefact survive a change of readout? Gate is PLAN.md §4.

    python 00_run.py --selftest   # FAIL-branch reachability, run FIRST
    python 00_run.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402

torch.set_num_threads(8)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probelib as pl  # noqa: E402

import importlib.util  # noqa: E402
HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("c4", HERE.parent / "C4_generality" / "00_run.py")
c4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(c4)

SEEDS = (0, 1, 2)
GATE_MIN = 0.02
MIN_READOUTS, MIN_CORPORA = 3, 2


# ------------------------------------------------------------------ readouts
def _bal(yte, pred):
    return float(np.mean([(pred[yte == c] == c).mean() if (yte == c).any() else 0.5
                          for c in (0, 1)]))


def r_logistic(Xtr, ytr, Xte, yte):
    t = lambda a, d: torch.as_tensor(np.asarray(a, d))                  # noqa: E731
    return pl.fit(t(Xtr, np.float32), t(ytr, np.int64),
                  t(Xte, np.float32), t(yte, np.int64))


def _std(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-6] = 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def r_ridge(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import RidgeClassifier
    a, b = _std(Xtr, Xte)
    return _bal(yte, RidgeClassifier(alpha=1.0).fit(a, ytr).predict(b))


def r_mlp(Xtr, ytr, Xte, yte):
    from sklearn.neural_network import MLPClassifier
    a, b = _std(Xtr, Xte)
    m = MLPClassifier(hidden_layer_sizes=(64,), max_iter=120, random_state=0,
                      early_stopping=False, alpha=1e-3)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(a, ytr)
    return _bal(yte, m.predict(b))


def r_knn(Xtr, ytr, Xte, yte):
    from sklearn.neighbors import KNeighborsClassifier
    a, b = _std(Xtr, Xte)
    return _bal(yte, KNeighborsClassifier(n_neighbors=15).fit(a, ytr).predict(b))


READOUTS = {"logistic": r_logistic, "ridge": r_ridge, "mlp": r_mlp, "knn": r_knn}


# -------------------------------------------------------------------- measure
def cells(X, y, readout, n_train, seeds=SEEDS):
    """One row per (block, seed) at a fixed n_train, i.e. a fixed r."""
    base = c4._pca(X, c4.BASE_DIM)
    blocks = c4.blocks_for(X, np.random.default_rng(12345))
    fn = READOUTS[readout]
    rows = []
    for _, C in blocks.items():
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(y))
            tr, te = idx[:n_train], idx[n_train:n_train + max(400, n_train)]
            if len(te) < 150 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            C0 = pl.shuffle_rows(C, seed=seed)                  # provably zero information
            G = rng.standard_normal(C.shape).astype(np.float32)
            SH = pl.shuffle_rows(C0, seed=seed + 500)
            Xb, Xg, Xs = (np.hstack([base, Z]) for Z in (C0, G, SH))
            a0 = fn(Xb[tr], y[tr], Xb[te], y[te])
            rows.append({"d_noise": a0 - fn(Xg[tr], y[tr], Xg[te], y[te]),
                         "d_shuffle": a0 - fn(Xs[tr], y[tr], Xs[te], y[te]),
                         "er_gauss": pl.effective_rank(G[tr[:min(len(tr), 3000)]])})
    return rows


def judge(rows):
    dn = np.array([r["d_noise"] for r in rows])
    ds = np.array([r["d_shuffle"] for r in rows])
    m, lo, hi = pl.boot(dn)
    ms, los, his = pl.boot(ds)
    ok = (m > GATE_MIN and lo > 0) and (los <= 0 <= his)
    return ok, {"d_noise": [round(v, 4) for v in (m, lo, hi)],
                "d_shuffle": [round(v, 4) for v in (ms, los, his)], "n": len(rows)}


def run(datasets, label, seeds=SEEDS):
    out, t0 = {}, time.time()
    for name, X, y in c4.load_datasets(datasets):
        n_max = int(0.7 * len(y))
        # high-r regime, matched to C6_ratio's top end
        er = pl.effective_rank(np.random.default_rng(0).standard_normal(
            (min(len(y), 3000), c4.blocks_for(X, np.random.default_rng(1))["raw"].shape[1])))
        n_train = int(min(max(100, er / 0.30), n_max / 2))
        print(f"\n  {name}  n_train={n_train}  (target r ~0.30)")
        out[name] = {}
        for ro in READOUTS:
            rows = cells(X, y, ro, n_train, seeds)
            if not rows:
                print(f"    {ro:9s} no usable cells")
                continue
            ok, det = judge(rows)
            r_real = float(np.mean([r["er_gauss"] for r in rows])) / n_train
            out[name][ro] = {"clears": bool(ok), "r": round(r_real, 4), **det}
            print(f"    {ro:9s} r={r_real:.3f}  d_noise {det['d_noise'][0]:+.4f} "
                  f"{det['d_noise'][1:]}  d_shuffle {det['d_shuffle'][0]:+.4f} "
                  f"{det['d_shuffle'][1:]}  -> {'clears' if ok else 'no'}  "
                  f"[{time.time()-t0:.0f}s]")
            (HERE / f"partial_{label}.json").write_text(json.dumps(out, indent=2) + "\n")

    n_by_readout = {ro: sum(1 for c in out.values() if c.get(ro, {}).get("clears"))
                    for ro in READOUTS}
    survivors = [ro for ro, n in n_by_readout.items() if n >= MIN_CORPORA]
    verdict = "SURVIVES" if len(survivors) >= MIN_READOUTS else "READOUT-SPECIFIC"
    print(f"\n  corpora cleared per readout: {n_by_readout}")
    print(f"  readouts clearing in >= {MIN_CORPORA} corpora: {survivors}")
    print(f"\n  VERDICT: {verdict}")
    return verdict, out, n_by_readout


def selftest():
    """Unlearnable labels: no readout can show an artefact -> must be READOUT-SPECIFIC."""
    print("  selftest: labels independent of every feature, so no artefact can exist")

    def fake_load(_):
        rng = np.random.default_rng(3)
        n = 1500
        X = rng.standard_normal((n, 48)).astype(np.float32)
        return [("synthetic/unlearnable", X, (rng.random(n) > 0.5).astype(np.int64))]

    real, c4.load_datasets = c4.load_datasets, fake_load
    try:
        v, _, n_by = run("synthetic", "selftest", seeds=(0, 1))
    finally:
        c4.load_datasets = real
    if v != "READOUT-SPECIFIC":
        print(f"  REACHABILITY CHECK FAILED: got {v} on unlearnable labels. Halting.")
        return 2
    print("  READOUT-SPECIFIC branch is reachable. Proceed.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--datasets", default="digits,news,covtype")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    verdict, out, n_by = run(a.datasets, "real")
    (HERE / "result.json").write_text(json.dumps(
        {"verdict": verdict, "corpora_cleared_per_readout": n_by,
         "detail": out}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
