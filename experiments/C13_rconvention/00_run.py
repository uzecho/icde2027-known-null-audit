#!/usr/bin/env python3
"""A3: does the magnitude fit survive when `r` means one thing?

NOT a gate. This is a defect repair with a measurement attached, and it is
labelled as such rather than dressed up as a pre-registered test.

THE DEFECT (paper_improvement/02_statistical_audit.md, F3). The symbol
`r = eff_rank(G)/n_train` is computed three different ways across the
experiments that feed Eq. (6):

    C4_generality:132,138   effective_rank(C), effective_rank(G)   full sample
    C6_ratio:73             effective_rank(G[:min(len(G),4000)])   <= 4000 rows
    C6_lrs2/01_rerun.py:127 effective_rank(G[tr])                  AT n_train
    C7_readout:98           effective_rank(G[tr[:3000]])           AT n_train

Table I's caption documents only the last. For digits the same comparator reads
48.7 under at-n_train and 61.6 under the sweep's convention -- a 27% gap in a
quantity the paper regresses on. Eq. (6) is fitted on points from the first two
conventions and evaluated out of sample on a point from the third.

WHAT THIS DOES. Recomputes every fitted point's `r` under the single convention
Table I documents (at n_train), refits the line, and reports how far the
prediction interval moves. The Delta values are untouched -- only the abscissa
changes -- so no probe is refitted and nothing here can alter a verdict.

WHAT IT DELIBERATELY DOES NOT DO. It does not re-score the LRS2 gate. That gate
was registered against the line as originally fitted; a line refitted after the
fact is a robustness check, not a re-run of the test. Both are reported.

CPU only, seconds.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ICDE = HERE.parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probelib as pl  # noqa: E402

_s = importlib.util.spec_from_file_location("c4", ICDE / "C4_generality" / "00_run.py")
c4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(c4)

# The five points Eq. (6) is fitted on, as the paper describes them:
# three single-ratio measurements from C4 and the two top-r points of Table III.
# Delta values in POINTS, read from the released result files.
LRS2_R, LRS2_D = 0.1611, None          # r already at-n_train; Delta is licensed
FIT_DESC = "Delta = a*r + b, least squares on five points"


def er_at(d, n, rng):
    """Effective rank of an n x d Gaussian comparator observed at n rows."""
    return pl.effective_rank(rng.standard_normal((n, d)))


def main():
    c4res = json.loads((ICDE / "C4_generality" / "result.json").read_text())
    c6res = json.loads((ICDE / "C6_ratio" / "audit_cluster" /
                        "result_cluster.json").read_text())["per_corpus"]

    rng = np.random.default_rng(0)
    pts = []

    # ---- the three C4 points. C4 measured eff rank on the FULL block; the
    # paper's convention is at n_train.
    print(f"\n  {'point':28s} {'d':>5s} {'n_tr':>7s} {'r (as used)':>12s} "
          f"{'r (at n_train)':>15s} {'Delta pts':>10s}")
    for name, X, y in c4.load_datasets("digits,news,covtype"):
        blocks = c4.blocks_for(X, np.random.default_rng(12345))
        d = blocks["raw"].shape[1]
        det = c4res["detail"][name] if "detail" in c4res else c4res[name]
        dn = det["d_noise"][0] * 100.0
        n_tr = int(det.get("n_train", 0)) or int(0.7 * len(y))
        r_full = pl.effective_rank(rng.standard_normal((min(len(y), 4000), d))) / n_tr
        r_at = er_at(d, n_tr, rng) / n_tr
        pts.append((f"C4 {name}", d, n_tr, r_full, r_at, dn))

    # ---- the two top-r points of the sweep (digits, covtype)
    for corpus in ("digits/vision", "covtype/tabular"):
        v = c6res[corpus]
        dn = v["d_noise_at_top_r"][0] * 100.0
        X = None
        for nm, Xx, yy in c4.load_datasets(
                "digits" if corpus.startswith("digits") else "covtype"):
            X, y = Xx, yy
        blocks = c4.blocks_for(X, np.random.default_rng(12345))
        d = blocks["raw"].shape[1]
        n_max = int(0.7 * len(y))
        er_ref = pl.effective_rank(np.random.default_rng(0).standard_normal(
            (min(len(y), 4000), d)))
        n_min = int(min(max(100, er_ref / 0.30), n_max / 12))
        n_tr = n_min                      # top r is the SMALLEST n_train
        r_sweep = er_ref / n_tr           # the sweep's convention: <=4000 rows
        r_at = er_at(d, n_tr, rng) / n_tr
        pts.append((f"C6 top-r {corpus}", d, n_tr, r_sweep, r_at, dn))

    for nm, d, n, r0, r1, dn in pts:
        print(f"  {nm:28s} {d:5d} {n:7d} {r0:12.4f} {r1:15.4f} {dn:+10.2f}")

    x0 = np.array([p[3] for p in pts]); x1 = np.array([p[4] for p in pts])
    yv = np.array([p[5] for p in pts])

    def fit(x):
        a, b = np.polyfit(x, yv, 1)
        resid = yv - (a * x + b)
        return a, b, float(np.std(resid, ddof=2))

    a0, b0, s0 = fit(x0)
    a1, b1, s1 = fit(x1)
    p0 = a0 * LRS2_R + b0
    p1 = a1 * LRS2_R + b1
    print(f"\n  fit on the mixed abscissa   : Delta = {a0:+.3f} r {b0:+.3f}  "
          f"resid sd {s0:.3f}")
    print(f"  fit on one convention       : Delta = {a1:+.3f} r {b1:+.3f}  "
          f"resid sd {s1:.3f}")
    print(f"\n  LRS2 at r = {LRS2_R} (already at-n_train)")
    print(f"    predicted, mixed abscissa : {p0:+.2f}  "
          f"[{p0 - 2*s0:+.2f},{p0 + 2*s0:+.2f}]")
    print(f"    predicted, one convention : {p1:+.2f}  "
          f"[{p1 - 2*s1:+.2f},{p1 + 2*s1:+.2f}]")
    print(f"    measured (3 draws)        : +2.7, +2.7, +2.3")

    out = {"note": "A3 defect repair. Delta values unchanged; only the abscissa "
                   "convention differs. Units: POINTS for Delta, dimensionless r.",
           "points": [{"name": n, "d": d, "n_train": nt,
                       "r_as_used": r0, "r_at_n_train": r1, "d_noise": dn}
                      for n, d, nt, r0, r1, dn in pts],
           "fit_mixed": {"slope": a0, "intercept": b0, "resid_sd": s0,
                         "lrs2_pred": p0},
           "fit_unified": {"slope": a1, "intercept": b1, "resid_sd": s1,
                           "lrs2_pred": p1},
           "lrs2_r": LRS2_R,
           # [value, lo, hi] triples so the artifact's verify.py re-checks these
           # on a rerun like any other published number. Tolerances are wider
           # than for a probe result because eff_rank(G) is itself a random draw
           # (PLAN.md §4), so r -- and anything fitted on it -- varies run to run.
           "paper_checks": {
               "slope_unified": [a1, a1 - 1.2, a1 + 1.2],
               "resid_sd_unified": [s1, s1 - 0.08, s1 + 0.08],
               "lrs2_pred_unified": [p1, p1 - 0.30, p1 + 0.30],
               "slope_mixed": [a0, a0 - 0.6, a0 + 0.6],
           }}
    (HERE / "result.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\n  -> result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
