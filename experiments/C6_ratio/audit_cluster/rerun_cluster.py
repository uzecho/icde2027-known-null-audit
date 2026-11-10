#!/usr/bin/env python3
"""Section V's sweep, re-aggregated with a cluster bootstrap.

THE DEFECT. C6_ratio/00_run.py:70 and 01_control.py:42 both seed the RNG from
`seed` alone with the block loop OUTSIDE it, so within a (n_train, seed) all four
blocks are scored against the same split and the same Gaussian comparator G --
and because every block is held at the corpus width, G is the same matrix for all
four, hence so is r = eff_rank(G)/n_train. The four cells of a cluster therefore
carry IDENTICAL r and share the subtrahend of d_noise. evaluate() and rho_ci()
then resample all 140 cells independently. The real units are
7 sizes x 5 seeds = 35 clusters, not 140 cells.

WHAT THIS CHANGES. Only the estimator. The design, the seed family (0..4), the
sizes and the blocks are untouched, so every point estimate must reproduce --
GATE A refuses otherwise. This is deliberately NOT a re-run at a larger n: the
gate's branch would then move with a sample size chosen after the original
scoring, which is the objection L18 raises against Section IV-E. Computing a
95% interval with the right estimator is not the same as re-powering it, and
only the former is done here.

CPU only: probe_at() builds plain torch tensors, so pl.fit runs on CPU.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
HERE = Path(__file__).resolve().parent
C6 = HERE.parent
import probelib as pl  # noqa: E402

_s = importlib.util.spec_from_file_location("c6", C6 / "00_run.py")
c6 = importlib.util.module_from_spec(_s); _s.loader.exec_module(c6)
_sc = importlib.util.spec_from_file_location("c6ctrl", C6 / "01_control.py")
c6ctrl = importlib.util.module_from_spec(_sc); _sc.loader.exec_module(c6ctrl)
c4 = c6.c4

N_BOOT = 2000                      # 01_control.py's figure
# Published in result.json / the paper, for GATE A.
PUB_RHO = {"digits/vision": 0.627, "20news/text": 0.184, "covtype/tabular": 0.510}
PUB_TOP = {"digits/vision": 0.0589, "covtype/tabular": 0.0300}
TOL_RHO, TOL_TOP = 0.02, 0.004


def both_arms_tagged(name, X, y, seeds, verbose=True):
    """Verbatim copy of c6ctrl.both_arms, with (block, n_train, seed) recorded.

    The published rows carry only r and the two deltas; a cluster bootstrap needs
    to know which cells share a draw, and rows are dropped on NaN so the cluster
    cannot be recovered from the row index after the fact. Nothing else differs.
    """
    base = c4._pca(X, c6.BASE_DIM)
    blocks = c4.blocks_for(X, np.random.default_rng(12345))
    n_max = int(0.7 * len(y))
    er_ref = pl.effective_rank(np.random.default_rng(0).standard_normal(
        (min(len(y), 4000), next(iter(blocks.values())).shape[1])))
    n_min = int(min(max(100, er_ref / 0.30), n_max / 12))
    sizes = sorted({int(s) for s in np.geomspace(n_min, n_max, 7)})
    rows = []
    for bname, C in blocks.items():
        for n_tr in sizes:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                C0 = pl.shuffle_rows(C, seed=seed)
                G = rng.standard_normal(C.shape).astype(np.float32)
                er_G = pl.effective_rank(G[:min(len(G), 4000)])
                S = pl.shuffle_rows(C0, seed=seed + 500)
                a_b0 = c6.probe_at(np.hstack([base, C0]), y, n_tr, 1000 + seed)
                a_gs = c6.probe_at(np.hstack([base, G]), y, n_tr, 1000 + seed)
                a_sh = c6.probe_at(np.hstack([base, S]), y, n_tr, 1000 + seed)
                if any(np.isnan(v) for v in (a_b0, a_gs, a_sh)):
                    continue
                rows.append({"r": er_G / n_tr, "d_noise": a_b0 - a_gs,
                             "d_shuffle": a_b0 - a_sh,
                             "block": bname, "n_train": n_tr, "seed": seed})
    if verbose:
        print(f"    {name}: {len(rows)} cells in "
              f"{len({(x['n_train'], x['seed']) for x in rows})} clusters, r "
              f"{min(x['r'] for x in rows):.4f} .. {max(x['r'] for x in rows):.4f}",
              flush=True)
    return rows


def clusters_of(rows):
    cid = {}
    for i, x in enumerate(rows):
        cid.setdefault((x["n_train"], x["seed"]), []).append(i)
    return list(cid.values())


def rho_cluster(rows, key, n=N_BOOT, seed=0):
    """Spearman over all cells; interval by resampling CLUSTERS."""
    r = np.array([x["r"] for x in rows])
    v = np.array([x[key] for x in rows])
    rho = float(spearmanr(r, v).statistic)
    cl = clusters_of(rows)
    rng, bs = np.random.default_rng(seed), []
    for _ in range(n):
        pick = rng.integers(0, len(cl), len(cl))
        idx = np.concatenate([cl[j] for j in pick])
        if len(np.unique(r[idx])) > 2:
            bs.append(float(spearmanr(r[idx], v[idx]).statistic))
    return rho, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def top_cluster(rows, key, n=N_BOOT, seed=0):
    """Mean at the top decile of r; interval by resampling the clusters there.

    r is constant within a cluster, so the top-r set is a union of whole
    clusters and the bootstrap is well defined -- but there are very few of
    them, and the returned cluster count says how few.
    """
    r = np.array([x["r"] for x in rows])
    v = np.array([x[key] for x in rows])
    keep = r >= np.percentile(r, 90)
    sub = [x for x, k in zip(rows, keep) if k]
    cl = clusters_of(sub)
    vv = np.array([x[key] for x in sub])
    per = np.array([vv[c].mean() for c in cl])
    rng = np.random.default_rng(seed)
    b = rng.choice(per, (n, len(per)), replace=True).mean(1)
    return (float(vv.mean()), float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)), len(cl))


def fail(msg):
    print(f"\n  GATE A REFUSED: {msg}\n")
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="digits,news,covtype")
    args = ap.parse_args()
    t0, out = time.time(), {}

    for name, X, y in c4.load_datasets(args.datasets):
        print(f"\n  {name}", flush=True)
        rows = both_arms_tagged(name, X, y, c6.SEEDS)

        # ---- GATE A: the point estimates must reproduce. Same design, same
        # seeds; only the interval estimator changes, so a shift here means the
        # tagged copy is not the published sweep and nothing below is readable.
        rho_n, lo_n, hi_n = rho_cluster(rows, "d_noise")
        if abs(rho_n - PUB_RHO[name]) > TOL_RHO:
            fail(f"{name}: rho(r,d_noise) = {rho_n:+.3f}, published "
                 f"{PUB_RHO[name]:+.3f} (tol {TOL_RHO})")
        m_dn, lo_dn, hi_dn, ncl = top_cluster(rows, "d_noise")
        if name in PUB_TOP and abs(m_dn - PUB_TOP[name]) > TOL_TOP:
            fail(f"{name}: d_noise at top r = {m_dn:+.4f}, published "
                 f"{PUB_TOP[name]:+.4f} (tol {TOL_TOP})")

        rho_s, lo_s, hi_s = rho_cluster(rows, "d_shuffle")
        m_ds, lo_ds, hi_ds, _ = top_cluster(rows, "d_shuffle")

        # the gate of PLAN.md §3, with the interval computed correctly
        c1 = (rho_n > c6.GATE_MIN_RHO) and (lo_n > 0)
        c2 = (m_dn > c6.GATE_MIN_DNOISE) and (lo_dn > 0)
        verdict = "PASS" if (c1 and c2) else "FAIL"

        print(f"    GATE A ok (rho and top-r reproduce)")
        print(f"    rho(r,d_noise)    {rho_n:+.3f} [{lo_n:+.3f},{hi_n:+.3f}]"
              f"   cond1(monotone) {c1}")
        print(f"    d_noise at top r  {m_dn*100:+.2f} [{lo_dn*100:+.2f},"
              f"{hi_dn*100:+.2f}] pts over {ncl} clusters   cond2 {c2}")
        print(f"    rho(r,d_shuffle)  {rho_s:+.3f} [{lo_s:+.3f},{hi_s:+.3f}]"
              f"   <- the control")
        print(f"    d_shuf at top r   {m_ds*100:+.2f} [{lo_ds*100:+.2f},"
              f"{hi_ds*100:+.2f}] pts")
        print(f"    VERDICT (clustered): {verdict}", flush=True)

        entry = {"n_cells": len(rows), "n_clusters": len(clusters_of(rows)),
                 "rho_d_noise": [rho_n, lo_n, hi_n],
                 "rho_d_shuffle": [rho_s, lo_s, hi_s],
                 "d_noise_at_top_r": [m_dn, lo_dn, hi_dn],
                 "d_shuffle_at_top_r": [m_ds, lo_ds, hi_ds],
                 "n_top_clusters": ncl,
                 "cond1_monotone": bool(c1), "cond2_bites_at_high_r": bool(c2),
                 "verdict_clustered": verdict}

        # ---- replication, disjoint half and a disjoint seed family
        if name in PUB_TOP:
            rng = np.random.default_rng(777)
            half = rng.permutation(len(y))[: len(y) // 2]
            rep = both_arms_tagged(name + "/replication", X[half], y[half],
                                   c6ctrl.REPL_SEEDS)
            rho_r, lo_r, hi_r = rho_cluster(rep, "d_noise")
            m_r, lo_r2, hi_r2, nclr = top_cluster(rep, "d_noise")
            repl = ("REPLICATED" if (rho_r > 0.5 and lo_r > 0 and m_r > 0.02
                                     and lo_r2 > 0) else "NOT REPLICATED")
            print(f"    replication: rho {rho_r:+.3f} [{lo_r:+.3f},{hi_r:+.3f}], "
                  f"top-r {m_r*100:+.2f} [{lo_r2*100:+.2f},{hi_r2*100:+.2f}] pts "
                  f"-> {repl}", flush=True)
            entry["replication"] = {"rho": [rho_r, lo_r, hi_r],
                                    "d_noise_at_top_r": [m_r, lo_r2, hi_r2],
                                    "n_top_clusters": nclr, "verdict": repl}
        out[name] = entry

    (HERE / "result_cluster.json").write_text(json.dumps(
        {"note": "Section V re-aggregated with a cluster bootstrap over "
                 "(n_train, seed). Same design and seeds as the published sweep; "
                 "only the interval estimator differs. Units: FRACTIONS.",
         "n_boot": N_BOOT, "seeds": list(c6.SEEDS), "per_corpus": out,
         "wall_s": round(time.time() - t0, 1)}, indent=2, default=float) + "\n")
    print(f"\n  -> result_cluster.json   ({time.time()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
