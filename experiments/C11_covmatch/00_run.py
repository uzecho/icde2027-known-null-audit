#!/usr/bin/env python3
"""C11: does matching the comparator's COVARIANCE remove the sign flip?

Gate is PLAN.md §5, frozen before any output of this script existed.

    python 00_run.py --selftest    # BOTH branches reachable. Run this FIRST.
    python 00_run.py               # the real run

The decomposition (PLAN.md §2), exact by construction:

    Delta_noise = a(C~) - a(G)      G  ~ N(0, I)          <- the paper's figure
    Delta_cov   = a(C~) - a(G')     G' ~ N(0, Sigma_C)    <- survives covariance matching
    Delta_rank  = a(G') - a(G)                            <- correlation structure alone
    Delta_noise = Delta_cov + Delta_rank

Row permutation preserves covariance exactly, so G' matches the known null in second-moment
structure while remaining Gaussian in every higher-order marginal respect. Whatever is left
in Delta_cov is therefore NOT correlation structure.

CPU only. No GPU is touched.
"""
import os
# match the original runs: C8_mechanism sets OMP_NUM_THREADS=8, and the MLP/BLAS
# reduction order moves results if this differs (found via GATE A in C7's re-run).
os.environ.setdefault("OMP_NUM_THREADS", "8")

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
c4 = c7.c4

SEEDS = tuple(range(50))                 # PLAN.md §7: fixed in advance, not revisited
N_BOOT = 20_000
N_JOBS = 3
REPRO_TOL = 0.25                         # points; PLAN.md §4
NEED = 4                                 # of 6 cells; PLAN.md §5
FAMILY = {"logistic": "+", "knn5": "-"}


# ------------------------------------------------------------------ comparators
def gauss_iso(C, rng):
    """The paper's comparator: isotropic, same shape."""
    return rng.standard_normal(C.shape).astype(np.float32)


def gauss_cov(C, rng):
    """N(0, Sigma_C): matches the block's covariance, including degenerate columns.

    eigh, not cholesky: the real blocks are rank-deficient (up to six constant
    columns per block, measured in C4_generality/02_moments.py), so Sigma is
    singular and a Cholesky factorisation would fail or silently jitter.
    """
    C = np.asarray(C, np.float64)
    S = np.cov(C, rowvar=False)
    S = np.atleast_2d(S)
    w, V = np.linalg.eigh((S + S.T) / 2.0)
    w = np.clip(w, 0.0, None)
    A = V * np.sqrt(w)                                    # A @ A.T == Sigma
    Z = rng.standard_normal((len(C), C.shape[1]))
    return (Z @ A.T).astype(np.float32)


# ------------------------------------------------------------------- one cell
def cell(X, y, base, block, n_train, seed, readout, fn):
    """One (block, seed). Split and the isotropic G are drawn exactly as C7 does,
    so Delta_noise reproduces; G' is an extra draw from the same stream."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    tr, te = idx[:n_train], idx[n_train:n_train + max(400, n_train)]
    if len(te) < 150 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None
    C0 = pl.shuffle_rows(block, seed=seed)                # the known null
    G = gauss_iso(block, rng)                             # same draw order as C7
    Gp = gauss_cov(C0, rng)                               # covariance-matched
    out = {}
    for nm, Z in (("b0", C0), ("g", G), ("gp", Gp)):
        Xz = np.hstack([base, Z])
        out[nm] = fn(Xz[tr], y[tr], Xz[te], y[te])
    return {"d_noise": out["b0"] - out["g"],
            "d_cov": out["b0"] - out["gp"],
            "d_rank": out["gp"] - out["g"]}


# ------------------------------------------------------------------ statistics
def cluster_ci(vals, seed_of, key, n_boot=N_BOOT, seed=0):
    """95% cluster bootstrap over SEEDS; blocks averaged within a seed first."""
    by = {}
    for v, s in zip(vals, seed_of):
        by.setdefault(s, []).append(v[key])
    per = np.array([np.mean(x) for x in by.values()])
    if len(per) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = rng.choice(per, (n_boot, len(per)), replace=True).mean(1)
    return (float(per.mean()) * 100, float(np.percentile(b, 2.5)) * 100,
            float(np.percentile(b, 97.5)) * 100)


def _nest(cells):
    """corpus -> readout -> quantity, in fractions.

    NOT a flat 'corpus|readout' key: the artifact's verify.py addresses nested
    values by splitting a path on '|', and a composite key containing that
    separator is split in the wrong place, so every check reports MISSING.
    """
    out = {}
    for k, v in cells.items():
        corpus, ro = k.split("|")
        out.setdefault(corpus, {})[ro] = {
            kk: [x / 100.0 for x in vv] for kk, vv in v.items()}
    return out


def judge(cells):
    """PLAN.md §5, applied exactly."""
    sup, ref = [], []
    for name, d in cells.items():
        dn, dc, dr = d["d_noise"], d["d_cov"], d["d_rank"]
        s = np.sign(dn[0])
        rank_signed = (dr[1] > 0) if s > 0 else (dr[2] < 0)
        cov_zero = dc[1] <= 0 <= dc[2]
        cov_signed = (dc[1] > 0) if s > 0 else (dc[2] < 0)
        if rank_signed and cov_zero:
            sup.append(name)
        if cov_signed and abs(dc[0]) > abs(dr[0]):
            ref.append(name)
    if len(sup) >= NEED:
        v = "MECHANISM SUPPORTED"
    elif len(ref) >= NEED:
        v = "MECHANISM REFUTED"
    else:
        v = "MIXED"
    return v, {"supported_cells": sup, "refuted_cells": ref, "need": NEED,
               "n_cells": len(cells)}


# -------------------------------------------------------------------- selftest
def _synth(kind, rng, n=900, d=64):
    """PLAN.md §6. Two cases with a known right answer.

    FIX (2026-11-05): the first version used d=16 at n_train=200, i.e. r ~ 0.075,
    and BOTH branches returned MIXED with effects of 0.07-0.33 points against CIs
    of +-0.30. The guard was right and the synthetic was wrong: the effect this
    gate is about lives at r ~ 0.235 (Table II's regime), so the synthetic must
    sit there too. d=64 at n_train=205 gives eff_rank(G)/n ~ 0.237.
    """
    y = (rng.random(n) > 0.5).astype(np.int64)
    base = y[:, None] * 1.2 + rng.standard_normal((n, c4.BASE_DIM))
    if kind == "correlation":
        # low rank, GAUSSIAN marginals: arms differ only in covariance -> SUPPORTED
        L = rng.standard_normal((d, 4))
        B = (rng.standard_normal((n, 4)) @ L.T) + 0.05 * rng.standard_normal((n, d))
    else:
        # full rank, heavy-tailed AND discretised: arms differ only in marginal
        # shape (covariance is near-isotropic either way) -> REFUTED
        B = rng.standard_t(2, size=(n, d))
        B = np.round(B * 1.5) / 1.5
    return base, B.astype(np.float32), y


def run_cells(tasks, fn, jobs):
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=jobs)(delayed(cell)(*t, fn) for t in tasks)
    return [o for o in out if o is not None]


def selftest(args):
    readouts = {**c7.READOUTS, **c8._readouts()}
    print("\n  PLAN.md §6: branch reachability.\n")
    ok = True
    for kind, expect in (("correlation", "MECHANISM SUPPORTED"),
                         ("marginal", "MECHANISM REFUTED")):
        cells = {}
        for ci, corpus in enumerate(("synthA", "synthB", "synthC")):
            base, B, y = _synth(kind, np.random.default_rng(100 + ci))
            # Both readout families, as in the real run. The first version
            # duplicated the logistic cell to reach six, which let NEED=4 be met
            # by copies of one measurement -- a degenerate contrast of exactly
            # the kind this project's gates exist to refuse.
            for ro in ("logistic", "knn5"):
                tasks, seed_of = [], []
                for sd in range(args.selftest_seeds):
                    tasks.append((None, y, base, B, 205, sd, ro))
                    seed_of.append(sd)
                rows = run_cells(tasks, readouts[ro], args.jobs)
                sofar = seed_of[:len(rows)]
                cells[f"{corpus}|{ro}"] = {
                    k: cluster_ci(rows, sofar, k)
                    for k in ("d_noise", "d_cov", "d_rank")}
        v, det = judge(cells)
        hit = v == expect
        ok &= hit
        print(f"    {kind:12s} -> {v:<22s} expected {expect:<22s} "
              f"{'OK' if hit else 'MISS'}")
        for n_, d in list(cells.items())[:3]:
            print(f"       {n_:18s} noise {d['d_noise'][0]:+6.2f}  "
                  f"cov {d['d_cov'][0]:+6.2f} [{d['d_cov'][1]:+6.2f},{d['d_cov'][2]:+6.2f}]  "
                  f"rank {d['d_rank'][0]:+6.2f} [{d['d_rank'][1]:+6.2f},{d['d_rank'][2]:+6.2f}]")
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
    ap.add_argument("--jobs", type=int, default=N_JOBS)
    args = ap.parse_args()
    if args.selftest:
        return selftest(args)

    readouts = {**c7.READOUTS, **c8._readouts()}
    pub = json.loads((ICDE / "C7_readout" / "audit_cluster" /
                      "result_n50.json").read_text())["per_cell"]
    # Fail in the first second, not after minutes of fits: the first version of
    # this script read a key ('cluster15') that lives in the POINTS working file
    # rather than the published FRACTIONS one, and crashed at GATE A only after
    # computing a corpus of cells.
    for _c in ("digits/vision", "20news/text", "covtype/tabular"):
        for _r in ("logistic", "knn5"):
            if "d_noise" not in pub.get(_c, {}).get(_r, {}):
                print(f"\n  REFUSED: result_n50.json has no d_noise for {_c}/{_r}; "
                      f"GATE A could not be evaluated.\n")
                return 2
    t0, cells, bad = time.time(), {}, []

    for corpus, X, y in c4.load_datasets(args.datasets):
        base = c4._pca(X, c4.BASE_DIM)
        blocks = c4.blocks_for(X, np.random.default_rng(12345))
        n_train = c7.NTRAIN[corpus] if hasattr(c7, "NTRAIN") else None
        if n_train is None:
            er = pl.effective_rank(np.random.default_rng(0).standard_normal(
                (min(len(y), 3000), blocks["raw"].shape[1])))
            n_train = int(min(max(100, er / 0.30), int(0.7 * len(y)) / 2))
        print(f"  {corpus}: n_train={n_train}", flush=True)
        for ro in ("logistic", "knn5"):
            tasks, seed_of = [], []
            for b in blocks:
                for sd in SEEDS:
                    tasks.append((X, y, base, blocks[b], n_train, sd, ro))
                    seed_of.append(sd)
            rows = run_cells(tasks, readouts[ro], args.jobs)
            sofar = seed_of[:len(rows)]
            d = {k: cluster_ci(rows, sofar, k)
                 for k in ("d_noise", "d_cov", "d_rank")}
            cells[f"{corpus}|{ro}"] = d
            # ---- GATE A (PLAN.md §4). result_n50.json stores FRACTIONS, as every
            # released result file in this project does; cluster_ci() returns points.
            want = pub[corpus][ro]["d_noise"][0] * 100.0
            if abs(d["d_noise"][0] - want) > REPRO_TOL:
                bad.append(f"{corpus}/{ro}: d_noise {d['d_noise'][0]:+.2f}, "
                           f"published {want:+.2f}")
            print(f"    {ro:9s} noise {d['d_noise'][0]:+6.2f}  "
                  f"cov {d['d_cov'][0]:+6.2f} [{d['d_cov'][1]:+6.2f},{d['d_cov'][2]:+6.2f}]  "
                  f"rank {d['d_rank'][0]:+6.2f} [{d['d_rank'][1]:+6.2f},{d['d_rank'][2]:+6.2f}]",
                  flush=True)

    if bad:
        print("\n  GATE A REFUSED: d_noise does not reproduce result_n50.json:\n    "
              + "\n    ".join(bad) + "\n")
        return 2
    print(f"\n  GATE A ok: d_noise reproduces all {len(cells)} cells "
          f"(tol {REPRO_TOL} pts)")

    verdict, det = judge(cells)
    share = {k: (abs(v["d_rank"][0]) / max(abs(v["d_noise"][0]), 1e-9))
             for k, v in cells.items()}
    out = {"verdict": verdict, **det, "seeds": len(SEEDS), "n_boot": N_BOOT,
           "note": "Units: POINTS of balanced accuracy. "
                   "Delta_noise = Delta_cov + Delta_rank by construction.",
           "per_cell": {k: {kk: list(vv) for kk, vv in v.items()}
                        for k, v in cells.items()},
           "rank_share_of_noise": share, "wall_s": round(time.time() - t0, 1)}
    (HERE / "result.json").write_text(json.dumps(out, indent=2, default=float) + "\n")

    # Also emit the FRACTIONS view. Every other released result.json in this
    # project stores fractions, and both paper/check_numbers.py and the
    # artifact's verify.py read that convention; a points-only file is either
    # matched against unrelated magnitudes or reported as missing. This must be
    # written by the reproduction path itself, not by a side script, or a
    # reviewer re-running 00_run.py cannot regenerate what the artifact checks.
    frac = {"note": "C11 in FRACTIONS of balanced accuracy, matching every other "
                    "result.json here. result.json itself is in points.",
            "verdict": verdict, "supported_cells": det["supported_cells"],
            "refuted_cells": det["refuted_cells"], "rank_share_of_noise": share,
            "per_cell": _nest(cells)}
    (HERE / "result_fractions.json").write_text(
        json.dumps(frac, indent=1, default=float) + "\n")
    print(f"\n  VERDICT: {verdict}   "
          f"(supported {len(det['supported_cells'])}/6, "
          f"refuted {len(det['refuted_cells'])}/6, need {NEED})")
    print(f"  -> result.json   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
