#!/usr/bin/env python3
"""Table II at 15 seeds, cluster-bootstrapped over seeds.

WHAT THIS CHANGES, AND WHAT IT DOES NOT.

Not changed: the design. Within a seed the four blocks still share the training
split and the Gaussian comparator G, exactly as C7_readout/00_run.py:86-92 does
it. That sharing is defensible -- it holds the split fixed while the block
varies -- and the error was never there. The error was aggregating the four
resulting values as four independent units.

Changed: `seeds` 3 -> 15, and the interval is a cluster bootstrap over seeds
rather than a naive bootstrap over (block, seed) pairs. The estimand is
identical, so the published point estimates must reproduce; GATE A below
refuses to continue if they do not.

Why not also de-correlate the design: giving each block its own split and its
own G would change what is measured, the published numbers would no longer
reproduce, and this would be a new experiment rather than a correction to the
interval of an existing one.

CPU only. No GPU.
"""
import os
# Must precede numpy/sklearn, and must MATCH THE ORIGINAL RUN: C8_mechanism
# sets OMP_NUM_THREADS=8. The first attempt here used 1 thread to allow 8
# workers, and GATE A caught the consequence -- the two MLP readouts moved by
# ~0.05 points (digits/mlp2 +4.95 vs the published +5.00, 20news/mlp +0.05 vs
# +0.10) because BLAS reduction order changes MLP training, while logistic,
# ridge, SVC, kNN and centroid were bit-identical. Matching the original
# environment removes the confound instead of widening the gate around it.
# 3 workers x 8 threads = 24 of 32 cores, leaving the box's other users room.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
N_JOBS = 3

import importlib.util            # noqa: E402
import json                      # noqa: E402
import sys                       # noqa: E402
import time                      # noqa: E402
from pathlib import Path         # noqa: E402

import numpy as np               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# self-locating: .../C7_readout/audit_cluster/<this> -> icde2027 in the source
# tree, and -> experiments/ in the packaged artifact, where the sibling
# experiment folders sit at the same depth. A2 refuses an absolute path here.
ICDE = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent

NSEEDS = int(os.environ.get("NSEEDS", "15"))
SEEDS = tuple(range(NSEEDS))
PUBLISHED_SEEDS = (0, 1, 2)
N_BOOT = 20000                   # the paper's figure
# Table II prints one decimal place, so a correctly rounded entry already admits
# |true - printed| <= 0.05 with no numerical difference at all. 0.06 is that
# rounding bound plus a hair; it is set from how the table is printed, not from
# how far the re-run happened to land.
REPRO_TOL = 0.06                 # points of balanced accuracy

FAMILY = {"logistic": "+", "ridge": "+", "mlp": "+", "linsvc": "+", "mlp2": "+",
          "knn": "-", "knn5": "-", "centroid": "-"}
LABEL = {"logistic": "logistic", "ridge": "ridge", "mlp": "MLP $(64)$",
         "linsvc": "linear SVC$^\\dagger$", "mlp2": "MLP $(64,32)^\\dagger$",
         "knn": "$k$NN $k{=}15$", "knn5": "$k$NN $k{=}5^\\dagger$",
         "centroid": "nearest centroid$^\\dagger$"}
ORDER = ["logistic", "ridge", "mlp", "linsvc", "mlp2", "knn", "knn5", "centroid"]

# Table II as printed, in points. GATE A checks the re-run against these.
PUBLISHED = {
    "digits/vision":   {"logistic": +6.6, "ridge": +5.4, "mlp": +3.6, "linsvc": +7.5,
                        "mlp2": +5.0, "knn": -5.5, "knn5": -8.5, "centroid": -6.5},
    "20news/text":     {"logistic": +0.9, "ridge": +0.9, "mlp": +0.1, "linsvc": +0.7,
                        "mlp2": -1.8, "knn": -3.2, "knn5": -4.4, "centroid": -7.2},
    "covtype/tabular": {"logistic": +4.5, "ridge": +5.2, "mlp": +2.1, "linsvc": +4.2,
                        "mlp2": -0.3, "knn": -2.9, "knn5": -2.4, "centroid": -4.6},
}


def _load():
    s7 = importlib.util.spec_from_file_location("c7", ICDE / "C7_readout" / "00_run.py")
    c7 = importlib.util.module_from_spec(s7); s7.loader.exec_module(c7)
    s8 = importlib.util.spec_from_file_location("c8", ICDE / "C8_mechanism" / "00_run.py")
    c8 = importlib.util.module_from_spec(s8); s8.loader.exec_module(c8)
    return c7, c8


def _cell(corpus, ro, X, y, n_train, seeds):
    """One (corpus, readout) cell. Re-imports inside the worker."""
    c7, c8 = _load()
    c7.READOUTS = {**c7.READOUTS, **c8._readouts()}
    rows = c7.cells(X, y, ro, n_train, seeds)
    return corpus, ro, [r["d_noise"] for r in rows], [r["d_shuffle"] for r in rows]


def naive_ci(v, seed=0):
    """What the paper printed: resample the (block, seed) pairs as if independent."""
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(v, (N_BOOT, len(v)), replace=True).mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def cluster_ci(v, seed_of, seed=0):
    """Resample SEEDS, carrying all of a seed's blocks together."""
    v = np.asarray(v, float)
    sids = sorted(set(seed_of))
    per = np.array([v[[i for i, s in enumerate(seed_of) if s == sd]].mean()
                    for sd in sids])
    rng = np.random.default_rng(seed)
    b = rng.choice(per, (N_BOOT, len(per)), replace=True).mean(1)
    return per.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), per


def fail(gate, msg):
    print(f"\n  {gate} REFUSED: {msg}\n")
    sys.exit(2)


def main():
    from joblib import Parallel, delayed
    import probelib as pl
    c7, _ = _load()
    c4 = c7.c4
    t0 = time.time()

    jobs, meta = [], {}
    for name, X, y in c4.load_datasets("digits,news,covtype"):
        n_max = int(0.7 * len(y))
        er = pl.effective_rank(np.random.default_rng(0).standard_normal(
            (min(len(y), 3000),
             c4.blocks_for(X, np.random.default_rng(1))["raw"].shape[1])))
        n_train = int(min(max(100, er / 0.30), n_max / 2))
        meta[name] = {"n_train": n_train, "X": X, "y": y}
        print(f"  {name}: n_train={n_train}  readouts={len(ORDER)}  seeds={len(SEEDS)}",
              flush=True)
        for ro in ORDER:
            jobs.append((name, ro))

    print(f"\n  running {len(jobs)} cells on {N_JOBS} workers x 8 threads ...", flush=True)
    out = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_cell)(c, ro, meta[c]["X"], meta[c]["y"], meta[c]["n_train"], SEEDS)
        for c, ro in jobs)

    # ---- GATE A: the first three seeds must reproduce Table II as printed.
    # The estimand is unchanged by adding seeds, so a mismatch means the re-run
    # is not measuring the published quantity and nothing below it is readable.
    res, bad = {}, []
    for corpus, ro, dn, ds in out:
        seed_of = [SEEDS[i % len(SEEDS)] for i in range(len(dn))]
        sub = [v for v, s in zip(dn, seed_of) if s in PUBLISHED_SEEDS]
        got, want = np.mean(sub) * 100, PUBLISHED[corpus][ro]
        if abs(got - want) > REPRO_TOL:
            bad.append(f"{corpus}/{ro}: seeds 0-2 give {got:+.2f}, paper prints {want:+.2f}")
        res.setdefault(corpus, {})[ro] = (dn, ds, seed_of)
    if bad:
        fail("GATE A (reproduction)", "the re-run does not reproduce Table II:\n    "
             + "\n    ".join(bad))
    print(f"\n  GATE A ok: all {len(out)} cells reproduce Table II on seeds 0-2 "
          f"(tol {REPRO_TOL} pts)\n")

    rows_out, n_naive, n_clu, against = {}, 0, 0, []
    print(f"  {'corpus':16s} {'readout':10s} {'published (3 seeds)':>24s} "
          f"{'15 seeds, clustered':>24s}")
    for corpus in res:
        rows_out[corpus] = {}
        for ro in ORDER:
            dn, ds, seed_of = res[corpus][ro]
            nm, nlo, nhi = naive_ci(dn)
            cm, clo, chi, per = cluster_ci(dn, seed_of)
            sm, slo, shi, _ = cluster_ci(ds, seed_of)
            rn = (nlo > 0) or (nhi < 0)
            rc = (clo > 0) or (chi < 0)
            n_naive += rn; n_clu += rc
            wrong = rc and ((FAMILY[ro] == "+" and chi < 0) or
                            (FAMILY[ro] == "-" and clo > 0))
            if wrong:
                against.append(f"{corpus}/{ro}")
            rows_out[corpus][ro] = {
                "family": FAMILY[ro], "n_units": len(dn), "n_seeds": len(set(seed_of)),
                "naive15": [nm * 100, nlo * 100, nhi * 100],
                "cluster15": [cm * 100, clo * 100, chi * 100],
                "d_shuffle_cluster15": [sm * 100, slo * 100, shi * 100],
                "resolves_naive15": bool(rn), "resolves_cluster15": bool(rc),
                "published_point": PUBLISHED[corpus][ro],
                "per_seed": (per * 100).tolist(),
            }
            print(f"  {corpus:16s} {ro:10s} {PUBLISHED[corpus][ro]:+8.1f}"
                  f"{'':16s} {cm*100:+7.2f} [{clo*100:+6.2f},{chi*100:+6.2f}]"
                  f"{'  R' if rc else '  .'}", flush=True)

    tot = sum(len(v) for v in rows_out.values())
    print(f"\n  resolving at 15 seeds: naive {n_naive}/{tot}   clustered {n_clu}/{tot}")
    print(f"  resolving AGAINST family (clustered): {against or 'none'}")
    dist = [v for c in rows_out.values() for ro, v in c.items() if v["family"] == "-"]
    print(f"  distance-based cells resolving negative: "
          f"{sum(v['resolves_cluster15'] for v in dist)}/{len(dist)}")
    dirty = [f"{c}/{ro}" for c, cc in rows_out.items() for ro, v in cc.items()
             if not (v['d_shuffle_cluster15'][1] <= 0 <= v['d_shuffle_cluster15'][2])]
    print(f"  own-shuffle control cells excluding zero: {len(dirty)}/{tot} {dirty}")

    (HERE / f"rerun_n{NSEEDS}.json").write_text(json.dumps(
        {"seeds": list(SEEDS), "n_boot": N_BOOT, "per_cell": rows_out,
         "n_resolving_naive15": n_naive, "n_resolving_cluster15": n_clu,
         "n_cells": tot, "against_family": against,
         "n_distance_resolving": int(sum(v["resolves_cluster15"] for v in dist)),
         "dirty_control_cells": dirty,
         "wall_s": round(time.time() - t0, 1)}, indent=2, default=float) + "\n")

    # ready-to-paste Table II body, clustered
    L = []
    for fam, head in (("+", "fits weights"), ("-", "measures distances")):
        L.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{head}}}}} \\\\")
        for ro in ORDER:
            if FAMILY[ro] != fam:
                continue
            cells = []
            for corpus in ("digits/vision", "20news/text", "covtype/tabular"):
                m, lo, hi = rows_out[corpus][ro]["cluster15"]
                b = rows_out[corpus][ro]["resolves_cluster15"]
                v = f"\\mathbf{{{m:+.1f}}}" if b else f"{m:+.1f}"
                cells.append(f"${v}$ & {{\\scriptsize$[{lo:+.1f},{hi:+.1f}]$}}")
            L.append(f"{LABEL[ro]}\n  & " + "\n  & ".join(cells) + " \\\\")
    (HERE / f"table2_clustered_n{NSEEDS}.tex").write_text("\n".join(L) + "\n")
    print(f"\n  -> rerun_n{NSEEDS}.json, table2_clustered_n{NSEEDS}.tex   ({time.time()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
