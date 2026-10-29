#!/usr/bin/env python3
"""Does the fit-vs-distance mechanism predict the sign on unseen readouts? Gate: PLAN.md §3.

    python 00_run.py --selftest   # FAIL-branch reachability, run FIRST
    python 00_run.py
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np                                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probelib as pl  # noqa: E402

import importlib.util  # noqa: E402
HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("c7", HERE.parent / "C7_readout" / "00_run.py")
c7 = importlib.util.module_from_spec(_s); _s.loader.exec_module(c7)
c4 = c7.c4

SEEDS = (0, 1, 2)
MIN_CORPORA = 2

# PLAN.md §2 — signs assigned BEFORE measurement.
PREDICTED = {"linsvc": +1, "mlp2": +1, "centroid": -1, "knn5": -1}
FAMILY = {"linsvc": "fits weights", "mlp2": "fits weights",
          "centroid": "measures distances", "knn5": "measures distances"}


def _fit(make):
    def f(Xtr, ytr, Xte, yte):
        a, b = c7._std(Xtr, Xte)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return c7._bal(yte, make().fit(a, ytr).predict(b))
    return f


def _readouts():
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
    return {
        "linsvc": _fit(lambda: LinearSVC(C=1.0, max_iter=3000)),
        "mlp2": _fit(lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=120,
                                           random_state=0, alpha=1e-3)),
        "centroid": _fit(NearestCentroid),
        "knn5": _fit(lambda: KNeighborsClassifier(n_neighbors=5)),
    }


def run(datasets, label):
    ro = _readouts()
    c7.READOUTS = ro                       # reuse C7's cell machinery verbatim
    out, t0 = {}, time.time()
    for name, X, y in c4.load_datasets(datasets):
        n_max = int(0.7 * len(y))
        er = pl.effective_rank(np.random.default_rng(0).standard_normal(
            (min(len(y), 3000), c4.blocks_for(X, np.random.default_rng(1))["raw"].shape[1])))
        n_train = int(min(max(100, er / 0.30), n_max / 2))
        print(f"\n  {name}  n_train={n_train}")
        out[name] = {}
        for r in ro:
            rows = c7.cells(X, y, r, n_train, SEEDS)
            if not rows:
                continue
            dn = np.array([x["d_noise"] for x in rows])
            ds = np.array([x["d_shuffle"] for x in rows])
            m, lo, hi = pl.boot(dn)
            ms, los, his = pl.boot(ds)
            signed = (lo > 0 and PREDICTED[r] > 0) or (hi < 0 and PREDICTED[r] < 0)
            ctrl = los <= 0 <= his
            out[name][r] = {"d_noise": [round(v, 4) for v in (m, lo, hi)],
                            "d_shuffle": [round(v, 4) for v in (ms, los, his)],
                            "matches_predicted_sign": bool(signed), "control_clean": bool(ctrl)}
            pred = "+" if PREDICTED[r] > 0 else "-"
            print(f"    {r:9s} pred {pred}  d_noise {m:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                  f"ctrl {'ok' if ctrl else 'DIRTY'}  -> "
                  f"{'MATCH' if signed else 'miss'}   [{time.time()-t0:.0f}s]")
            (HERE / f"partial_{label}.json").write_text(json.dumps(out, indent=2) + "\n")

    scored, void = {}, []
    for r in ro:
        cells_ = [c[r] for c in out.values() if r in c]
        if sum(c["control_clean"] for c in cells_) < MIN_CORPORA:
            void.append(r)
            continue
        scored[r] = sum(c["matches_predicted_sign"] for c in cells_) >= MIN_CORPORA
    n = sum(scored.values())
    verdict = ("MECHANISM PREDICTS" if n == 4 else
               "PARTIAL" if n >= 2 else "MECHANISM FAILS")
    print(f"\n  matched: {[r for r, v in scored.items() if v]}"
          f"{'   VOID (dirty control): ' + str(void) if void else ''}")
    print(f"\n  VERDICT: {verdict}   ({n}/4 readouts matched their predicted sign)")
    return verdict, out, scored


def selftest():
    print("  selftest: unlearnable labels, so no readout can show a signed effect")

    def fake(_):
        # TWO corpora, or MIN_CORPORA=2 voids every readout and the aggregate scoring
        # path is never exercised -- the first version of this selftest had that hole.
        out = []
        for i, d in enumerate((48, 64)):
            rng = np.random.default_rng(3 + i)
            n = 1200
            out.append((f"synthetic/unlearnable{i}",
                        rng.standard_normal((n, d)).astype(np.float32),
                        (rng.random(n) > 0.5).astype(np.int64)))
        return out

    real, c4.load_datasets = c4.load_datasets, fake
    try:
        v, _, _ = run("synthetic", "selftest")
    finally:
        c4.load_datasets = real
    if v == "MECHANISM PREDICTS":
        print(f"  REACHABILITY CHECK FAILED: got {v} on unlearnable labels. Halting.")
        return 2
    print(f"  Non-PREDICTS branch reachable (got {v}). Proceed.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--datasets", default="digits,news,covtype")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    verdict, out, scored = run(a.datasets, "real")
    (HERE / "result.json").write_text(json.dumps(
        {"verdict": verdict, "predicted_signs": PREDICTED, "family": FAMILY,
         "matched": scored, "detail": out}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
