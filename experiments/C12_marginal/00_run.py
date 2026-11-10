#!/usr/bin/env python3
"""C12: does matching the comparator's MARGINALS explain C11's distance-side residual?

Gate is PLAN.md §5, frozen before any output of this script existed.

    python 00_run.py --selftest    # BOTH branches reachable. Run this FIRST.
    python 00_run.py               # the real run

Three rungs of one ladder, each adding a constraint on the comparator:

    G    ~ N(0, I)                              shape only
    G'   ~ N(0, Sigma_C)                        + covariance / spectrum / eff. rank
    G''  = quantile_C( rank(G') )               + every marginal, exactly

    Delta_noise = a(C~) - a(G)     (Table II)
    Delta_cov   = a(C~) - a(G')    (C11)
    Delta_marg  = a(C~) - a(G'')   (this gate)

A per-column monotone map preserves ranks, so G'' keeps G''s rank-correlation structure
while taking C's marginals exactly. It does NOT keep Pearson covariance; the run reports
that drift per cell and VOIDs if it is large (PLAN.md §5).

CPU only. No GPU is touched.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse          # noqa: E402
import importlib.util    # noqa: E402
import json              # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402
from pathlib import Path # noqa: E402

import numpy as np       # noqa: E402

HERE = Path(__file__).resolve().parent
ICDE = HERE.parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probelib as pl    # noqa: E402

_s7 = importlib.util.spec_from_file_location("c7", ICDE / "C7_readout" / "00_run.py")
c7 = importlib.util.module_from_spec(_s7); _s7.loader.exec_module(c7)
_s8 = importlib.util.spec_from_file_location("c8", ICDE / "C8_mechanism" / "00_run.py")
c8 = importlib.util.module_from_spec(_s8); _s8.loader.exec_module(c8)
_s11 = importlib.util.spec_from_file_location("c11", ICDE / "C11_covmatch" / "00_run.py")
c11 = importlib.util.module_from_spec(_s11); _s11.loader.exec_module(c11)
c4 = c7.c4

SEEDS = tuple(range(50))                 # PLAN.md §3: fixed in advance
N_BOOT = 20_000
REPRO_TOL = 0.25                         # points; PLAN.md §4
# PLAN.md §5 VOID condition, REPAIRED 2026-11-05 on synthetic data only.
# The original absolute bar of 0.25 fired on the estimation floor: a d x d
# correlation matrix at this n carries ~0.26 relative Frobenius error between two
# samples of the SAME distribution, so the guard was measuring noise and VOIDed
# both branches. G' has C's covariance by construction, so drift(C,G') IS that
# floor; the bar is now on the excess over it, which is self-calibrating across
# corpora whose n and d differ by an order of magnitude.
DRIFT_EXCESS = 0.05
DRIFT_RATIO = 1.5
NEED = 2                                 # of 3 distance cells
DIST = "knn5"
WEIGHT = "logistic"


# ------------------------------------------------------------------ comparator
def norta_factor(C, rng):
    """Cholesky-like factor A with A @ A.T == R*, for sampling N(0, R*)."""
    R = norta_R(C, rng)
    w, V = np.linalg.eigh((R + R.T) / 2.0)
    return V * np.sqrt(np.clip(w, 0.0, None))


def _corr(M):
    M = np.asarray(M, np.float64)
    s = M.std(0).copy(); s[s < 1e-12] = 1.0
    Z = (M - M.mean(0)) / s
    return (Z.T @ Z) / len(Z)


def gauss_marginal(C, Gp):
    """G'': G' pushed through each column's empirical quantile function.

    Rank-preserving per column, so G' 's rank-correlation survives and C's
    marginals are reproduced exactly (up to ties).
    """
    C = np.asarray(C, np.float64)
    Gp = np.asarray(Gp, np.float64)
    out = np.empty_like(Gp)
    for j in range(C.shape[1]):
        order = np.argsort(Gp[:, j], kind="stable")
        out[order, j] = np.sort(C[:, j])
    return out.astype(np.float32)


def norta_R(C, rng, iters=15, n_mc=4000):
    """NORTA (Cario & Nelson): the INPUT Gaussian correlation whose output,
    after the quantile transform, has C's Pearson correlation.

    FIX (2026-11-05): the first version fed the quantile transform a Gaussian
    drawn at corr(C) directly. The selftest VOIDed on both branches with the
    pre-registered drift guard reading 0.26-0.29 -- the monotone map preserves
    RANKS, not Pearson, so matching marginals cost the covariance match and the
    arm was not the comparator PLAN.md describes. The guard was right and the
    construction was wrong; no real corpus was touched.

    Fixed point: R <- R + (target - corr(transform(N(0,R)))), projected back to a
    valid correlation matrix each step. Solved once per BLOCK -- it depends only
    on C's marginals and the target, not on the seed -- so it costs nothing per cell.

    NORTA has a known defect: some targets are unreachable for given marginals.
    If that happens the drift diagnostic stays high and the gate VOIDs, which is
    the honest outcome rather than a silently mismatched arm.
    """
    C = np.asarray(C, np.float64)
    # gauss_marginal assigns C's sorted values at Z's ranks, so the Monte-Carlo
    # draw must have exactly as many rows as C. Subsample large blocks: the
    # marginals and the target correlation are estimated well from n_mc rows.
    if len(C) > n_mc:
        C = C[rng.choice(len(C), n_mc, replace=False)]
    m, d = C.shape
    target = _corr(C)
    R = target.copy()
    best, best_err = R.copy(), np.inf
    for _ in range(iters):
        w, V = np.linalg.eigh((R + R.T) / 2.0)
        A = V * np.sqrt(np.clip(w, 0.0, None))
        Z = rng.standard_normal((m, d)) @ A.T
        cur = _corr(gauss_marginal(C, Z))
        err = np.linalg.norm(cur - target) / max(np.linalg.norm(target), 1e-9)
        if err < best_err:
            best, best_err = R.copy(), err
        R = R + (target - cur)
        np.fill_diagonal(R, 1.0)
        R = np.clip(R, -0.999, 0.999)
        np.fill_diagonal(R, 1.0)
        w, V = np.linalg.eigh((R + R.T) / 2.0)          # project to PSD
        R = V @ np.diag(np.clip(w, 1e-8, None)) @ V.T
        dg = np.sqrt(np.diag(R)); dg[dg < 1e-12] = 1.0
        R = R / np.outer(dg, dg)
    return best


def fit_diagnostics(C, Gpp, Gp=None):
    """How well G'' actually matches, reported rather than assumed (PLAN.md §2)."""
    C = np.asarray(C, np.float64); Gpp = np.asarray(Gpp, np.float64)
    ks = float(np.mean([np.max(np.abs(np.sort(C[:, j]) - np.sort(Gpp[:, j])))
                        / max(np.ptp(C[:, j]), 1e-9) for j in range(C.shape[1])]))
    def corr(M):
        s = M.std(0); s[s < 1e-12] = 1.0
        Z = (M - M.mean(0)) / s
        return (Z.T @ Z) / len(Z)
    A, B = corr(C), corr(Gpp)
    drift = float(np.linalg.norm(A - B) / max(np.linalg.norm(A), 1e-9))
    # The FLOOR: G' has C's covariance by construction, so whatever drift it
    # shows is estimation noise in a d x d correlation matrix at this n, not
    # mismatch. Reported so the guard can be read against its own null rather
    # than against an absolute number (CLAUDE.md: report the floor you measured).
    floor = float("nan")
    if Gp is not None:
        Bp = corr(np.asarray(Gp, np.float64))
        floor = float(np.linalg.norm(A - Bp) / max(np.linalg.norm(A), 1e-9))
    return ks, drift, floor


# ------------------------------------------------------------------- one cell
def cell(X, y, base, block, n_train, seed, readout, A_star, fn):
    """Split and G are drawn exactly as C7/C11 do, so both reproduce.

    A_star is the NORTA factor for this block (A_star @ A_star.T == R*), solved
    once per block outside the seed loop.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    tr, te = idx[:n_train], idx[n_train:n_train + max(400, n_train)]
    if len(te) < 150 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None
    C0 = pl.shuffle_rows(block, seed=seed)
    G = c11.gauss_iso(block, rng)
    Gp = c11.gauss_cov(C0, rng)
    Z = rng.standard_normal((len(C0), C0.shape[1])) @ A_star.T
    Gpp = gauss_marginal(C0, Z)          # NORTA-corrected input correlation
    ks, drift, floor = fit_diagnostics(C0, Gpp, Gp)
    out = {}
    for nm, Z in (("b0", C0), ("g", G), ("gp", Gp), ("gpp", Gpp)):
        Xz = np.hstack([base, Z])
        out[nm] = fn(Xz[tr], y[tr], Xz[te], y[te])
    return {"d_noise": out["b0"] - out["g"], "d_cov": out["b0"] - out["gp"],
            "d_marg": out["b0"] - out["gpp"], "ks": ks, "drift": drift,
            "drift_floor": floor}


def run_cells(tasks, fn, jobs):
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=jobs)(delayed(cell)(*t, fn) for t in tasks)
    return [o for o in out if o is not None]


def summarise(rows, seed_of):
    d = {k: c11.cluster_ci(rows, seed_of, k)
         for k in ("d_noise", "d_cov", "d_marg")}
    d["ks"] = float(np.mean([r["ks"] for r in rows]))
    d["drift"] = float(np.mean([r["drift"] for r in rows]))
    d["drift_floor"] = float(np.mean([r["drift_floor"] for r in rows]))
    return d


def _nest(cells):
    """corpus -> readout -> quantity, in fractions. See C11's _nest: a flat
    'corpus|readout' key contains verify.py's path separator and breaks lookup."""
    out = {}
    for k, v in cells.items():
        corpus, ro = k.split("|")
        out.setdefault(corpus, {})[ro] = {
            kk: [x / 100.0 for x in vv]
            for kk, vv in v.items() if kk.startswith("d_")}
    return out


def judge(cells):
    """PLAN.md §5, applied exactly. Scored on the distance cells only."""
    dist = {k: v for k, v in cells.items() if k.endswith(f"|{DIST}")}
    over = sum(v["drift"] > max(DRIFT_RATIO * v["drift_floor"],
                                v["drift_floor"] + DRIFT_EXCESS)
               for v in dist.values())
    if over >= NEED:
        return "VOID", {"reason": f"Pearson drift exceeds its measured floor by "
                                  f"more than max({DRIFT_RATIO}x, +{DRIFT_EXCESS}) "
                                  f"in >= {NEED} distance cells; G'' is not the "
                                  f"described arm", "n_dist": len(dist)}
    exp, no = [], []
    for name, v in dist.items():
        dm, dc = v["d_marg"], v["d_cov"]
        s = np.sign(dc[0])
        marg_zero = dm[1] <= 0 <= dm[2]
        marg_signed = (dm[1] > 0) if s > 0 else (dm[2] < 0)
        if marg_zero and abs(dm[0]) < abs(dc[0]):
            exp.append(name)
        if marg_signed and abs(dm[0]) >= 0.5 * abs(dc[0]):
            no.append(name)
    if len(exp) >= NEED:
        v = "MARGINALS EXPLAIN"
    elif len(no) >= NEED:
        v = "MARGINALS DO NOT EXPLAIN"
    else:
        v = "PARTIAL"
    return v, {"explain_cells": exp, "not_explain_cells": no, "need": NEED,
               "n_dist": len(dist)}


# -------------------------------------------------------------------- selftest
def _synth(kind, rng, n=900, d=64):
    """PLAN.md §6, at the regime the effect lives in (d=64, n_train=205)."""
    y = (rng.random(n) > 0.5).astype(np.int64)
    base = y[:, None] * 1.2 + rng.standard_normal((n, c4.BASE_DIM))
    if kind == "marginal":
        # heavy-tailed + discretised, near-isotropic covariance: matching the
        # marginals should remove it -> MARGINALS EXPLAIN
        B = rng.standard_t(2, size=(n, d))
        B = np.round(B * 1.5) / 1.5
    else:
        # FIX (2026-11-05): the first version linked pairs through a SQUARED term.
        # x -> x^2 is not monotone, so the rank correlation of the pair is ~0 and
        # the dependence is invisible to Pearson AND to rank -- G'' reproduced it
        # by accident and the branch returned EXPLAIN. Replaced with t-copula
        # dependence under GAUSSIAN marginals: marginals are matchable exactly,
        # Pearson is matchable, but a Gaussian copula has ZERO tail dependence,
        # so the tail clustering kNN responds to cannot be reproduced.
        L = rng.standard_normal((d, 6)) / np.sqrt(6)
        R = L @ L.T + 0.35 * np.eye(d)
        dg = np.sqrt(np.diag(R)); R = R / np.outer(dg, dg)
        w, V = np.linalg.eigh(R)
        A = V * np.sqrt(np.clip(w, 0.0, None))
        Zc = rng.standard_normal((n, d)) @ A.T
        W = rng.chisquare(2, size=(n, 1))
        T = Zc / np.sqrt(W / 2.0)                       # multivariate t: tail dep.
        B = np.empty_like(T)                            # -> Gaussian marginals
        from scipy.special import ndtri
        for j in range(d):
            r = np.argsort(np.argsort(T[:, j], kind="stable"), kind="stable")
            B[:, j] = ndtri((r + 0.5) / n)
    return base, B.astype(np.float32), y


def selftest(args):
    readouts = {**c7.READOUTS, **c8._readouts()}
    print("\n  PLAN.md §6: branch reachability.\n")
    ok = True
    for kind, expect in (("marginal", "MARGINALS EXPLAIN"),
                         ("copula", "MARGINALS DO NOT EXPLAIN")):
        cells = {}
        for ci, corpus in enumerate(("synthA", "synthB", "synthC")):
            base, B, y = _synth(kind, np.random.default_rng(200 + ci))
            A_star = norta_factor(pl.shuffle_rows(B, seed=0),
                                  np.random.default_rng(7))
            for ro in (WEIGHT, DIST):
                tasks, seed_of = [], []
                for sd in range(args.selftest_seeds):
                    tasks.append((None, y, base, B, 205, sd, ro, A_star))
                    seed_of.append(sd)
                rows = run_cells(tasks, readouts[ro], args.jobs)
                cells[f"{corpus}|{ro}"] = summarise(rows, seed_of[:len(rows)])
        v, det = judge(cells)
        hit = v == expect
        ok &= hit
        print(f"    {kind:9s} -> {v:<26s} expected {expect:<26s} "
              f"{'OK' if hit else 'MISS'}")
        for n_, d_ in [(k, v2) for k, v2 in cells.items() if k.endswith(DIST)]:
            print(f"       {n_:16s} cov {d_['d_cov'][0]:+6.2f}  "
                  f"marg {d_['d_marg'][0]:+6.2f} "
                  f"[{d_['d_marg'][1]:+6.2f},{d_['d_marg'][2]:+6.2f}]  "
                  f"ks {d_['ks']:.3f}  drift {d_['drift']:.3f} "
                  f"(floor {d_['drift_floor']:.3f})")
    if not ok:
        print("\n  REACHABILITY CHECK FAILED. Halting before the real run.\n")
        return 2
    print("\n  Both branches reachable.\n")
    return 0


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--datasets", default="digits,news,covtype")
    ap.add_argument("--selftest-seeds", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    if args.selftest:
        return selftest(args)

    readouts = {**c7.READOUTS, **c8._readouts()}
    pub = json.loads((ICDE / "C7_readout" / "audit_cluster" /
                      "result_n50.json").read_text())["per_cell"]
    c11r = json.loads((ICDE / "C11_covmatch" / "result.json").read_text())["per_cell"]
    for _c in ("digits/vision", "20news/text", "covtype/tabular"):
        for _r in (WEIGHT, DIST):
            if "d_noise" not in pub.get(_c, {}).get(_r, {}) or \
               "d_cov" not in c11r.get(f"{_c}|{_r}", {}):
                print(f"\n  REFUSED: cannot evaluate GATE A for {_c}/{_r}\n")
                return 2

    t0, cells, bad = time.time(), {}, []
    for corpus, X, y in c4.load_datasets(args.datasets):
        base = c4._pca(X, c4.BASE_DIM)
        blocks = c4.blocks_for(X, np.random.default_rng(12345))
        er = pl.effective_rank(np.random.default_rng(0).standard_normal(
            (min(len(y), 3000), blocks["raw"].shape[1])))
        n_train = int(min(max(100, er / 0.30), int(0.7 * len(y)) / 2))
        astar = {b: norta_factor(pl.shuffle_rows(blocks[b], seed=0),
                                 np.random.default_rng(7)) for b in blocks}
        print(f"  {corpus}: n_train={n_train}  (NORTA solved for {len(astar)} blocks)",
              flush=True)
        for ro in (WEIGHT, DIST):
            tasks, seed_of = [], []
            for b in blocks:
                A_star = astar[b]
                for sd in SEEDS:
                    tasks.append((X, y, base, blocks[b], n_train, sd, ro, A_star))
                    seed_of.append(sd)
            rows = run_cells(tasks, readouts[ro], args.jobs)
            d = summarise(rows, seed_of[:len(rows)])
            cells[f"{corpus}|{ro}"] = d
            w1 = pub[corpus][ro]["d_noise"][0] * 100.0
            w2 = c11r[f"{corpus}|{ro}"]["d_cov"][0]
            if abs(d["d_noise"][0] - w1) > REPRO_TOL:
                bad.append(f"{corpus}/{ro}: d_noise {d['d_noise'][0]:+.2f} vs {w1:+.2f}")
            if abs(d["d_cov"][0] - w2) > REPRO_TOL:
                bad.append(f"{corpus}/{ro}: d_cov {d['d_cov'][0]:+.2f} vs {w2:+.2f}")
            print(f"    {ro:9s} noise {d['d_noise'][0]:+6.2f}  "
                  f"cov {d['d_cov'][0]:+6.2f}  "
                  f"marg {d['d_marg'][0]:+6.2f} "
                  f"[{d['d_marg'][1]:+6.2f},{d['d_marg'][2]:+6.2f}]   "
                  f"ks {d['ks']:.3f} drift {d['drift']:.3f}", flush=True)

    if bad:
        print("\n  GATE A REFUSED:\n    " + "\n    ".join(bad) + "\n")
        return 2
    print(f"\n  GATE A ok: d_noise and d_cov both reproduce (tol {REPRO_TOL} pts)")

    verdict, det = judge(cells)
    out = {"verdict": verdict, **det, "seeds": len(SEEDS), "n_boot": N_BOOT,
           "note": "Units: POINTS. result_fractions.json is the fractions view.",
           "per_cell": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                            for kk, vv in v.items()} for k, v in cells.items()},
           "wall_s": round(time.time() - t0, 1)}
    (HERE / "result.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    frac = {"note": "C12 in FRACTIONS, matching every other result.json here.",
            "verdict": verdict,
            "per_cell": _nest(cells)}
    (HERE / "result_fractions.json").write_text(
        json.dumps(frac, indent=1, default=float) + "\n")
    print(f"\n  VERDICT: {verdict}   "
          f"(explain {len(det.get('explain_cells', []))}/3, "
          f"not-explain {len(det.get('not_explain_cells', []))}/3, need {NEED})")
    print(f"  -> result.json   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
