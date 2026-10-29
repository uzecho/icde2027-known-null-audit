#!/usr/bin/env python3
"""Same-viseme vs other pairs, Gaussian-null artefact at fixed r. Gate is PLAN.md §4.

    python 00_run.py --selftest   # FAIL-branch reachability, run FIRST
    python 00_run.py
"""
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
sys.path.insert(0, str(Path(str(Path(__file__).resolve().parents[2])) / "lrs2"))
import probelib as pl  # noqa: E402
from common import same_viseme  # noqa: E402

_LRS2_HINT = (
    "The features behind this path are derived from the licence-restricted LRS2 "
    "corpus and are not redistributed with this artifact. See "
    "LRS2_REGENERATION.md to rebuild them from your own licensed copy, then set "
    "$LRS2_DERIVED (and $LRS2_ROOT for the corpus itself).")


def _lrs2(sub, _var="LRS2_DERIVED"):
    """Path under the LRS2-derived feature tree named by $LRS2_DERIVED."""
    import os
    import sys as _s
    from pathlib import Path as _P
    root = os.environ.get(_var)
    if not root:
        if not getattr(_lrs2, "_warned_" + _var, False):
            print(f"  note: ${_var} is not set. {_LRS2_HINT}", file=_s.stderr)
            setattr(_lrs2, "_warned_" + _var, True)
        root = f"${_var}-NOT-SET"
    return _P(root) / sub


def _lrs2_corpus(sub):
    """Path under the LRS2 corpus itself, named by $LRS2_ROOT."""
    return _lrs2(sub, _var="LRS2_ROOT")

HERE = Path(__file__).resolve().parent
W = _lrs2("ambiguity_map/frames")
P = _lrs2("frontend_ceiling/packed")

N_TRAIN, N_TEST = 5333, 800
N_OTHER, SEL_SEED = 15, 31
SEEDS = (0, 1, 2)
HALFWIDTH_POWERED = 0.010        # PLAN.md §4


def probe(X, y, tr, te):
    t = lambda a, d: torch.as_tensor(np.asarray(a, d))                 # noqa: E731
    return pl.fit(t(X[tr], np.float32), t(y[tr], np.int64),
                  t(X[te], np.float32), t(y[te], np.int64))


def evaluate(sv, ov):
    """PLAN.md §4, applied exactly."""
    a = np.array([r["d_noise"] for r in ov])
    b = np.array([r["d_noise"] for r in sv])
    rng = np.random.default_rng(0)
    bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
          for _ in range(20000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    D = float(a.mean() - b.mean())
    half = float((hi - lo) / 2)
    if lo > 0 or hi < 0:
        v = "DIFFERENCE"
    elif half < HALFWIDTH_POWERED:
        v = "NO DIFFERENCE"
    else:
        v = "UNDERPOWERED"
    return v, {"D": round(D, 4), "CI": [round(lo, 4), round(hi, 4)],
               "half_width": round(half, 4),
               "mean_other": round(float(a.mean()), 4),
               "mean_same_viseme": round(float(b.mean()), 4),
               "n_other": len(a), "n_sv": len(b)}


def selftest():
    """Two groups from ONE population must not return DIFFERENCE."""
    rng = np.random.default_rng(5)
    mk = lambda n: [{"d_noise": float(v)} for v in rng.normal(0.025, 0.012, n)]  # noqa: E731
    bad = 0
    for t in range(12):
        rng = np.random.default_rng(100 + t)
        mk = lambda n: [{"d_noise": float(v)}                                   # noqa: E731
                        for v in rng.normal(0.025, 0.012, n)]
        v, d = evaluate(mk(5), mk(15))
        if v == "DIFFERENCE":
            bad += 1
            print(f"    trial {t}: {v}  {d}")
    print(f"\n  selftest: {bad}/12 same-population trials returned DIFFERENCE "
          f"(expect ~0-1 at 95% CI)")
    if bad > 3:
        print("  REACHABILITY CHECK FAILED: gate returns DIFFERENCE on a null too often.")
        return 2
    # and the underpowered branch must be reachable at n=5 vs 15
    v, d = evaluate(mk(5), mk(15))
    print(f"  one draw -> {v} (half-width {d['half_width']}), "
          f"so the UNDERPOWERED branch is live at this sample size")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()

    rng = np.random.default_rng(0)
    meta = json.loads((W / "meta.json").read_text())
    PH = meta["phones"]
    spk = np.array([u["speaker"] for u in meta["utts"]])
    rows = np.load(P / "rows.npy")
    Y = np.load(W / "y.npy")[:, 1][rows]
    U = np.load(W / "utt.npy")[rows]
    stem = np.load(W / "X_front512.npy", mmap_mode="r")
    cand = np.load(P / "crop_px.npy", mmap_mode="r")

    need = N_TRAIN + N_TEST
    pairs = [(x, y) for i, x in enumerate(PH) for y in PH[i + 1:]]
    sv_pairs, ov_pairs = [], []
    for (pa, pb) in pairs:
        ia, ib = PH.index(pa), PH.index(pb)
        idx = np.nonzero((Y == ia) | (Y == ib))[0]
        if len(idx) < need:
            continue
        uu = U[idx]
        keep = []
        for u in np.unique(uu):
            f = idx[uu == u]
            c = min(int((Y[f] == ia).sum()), int((Y[f] == ib).sum()))
            if c:
                ya, yb = f[Y[f] == ia], f[Y[f] == ib]
                keep += list(rng.choice(ya, c, replace=False))
                keep += list(rng.choice(yb, c, replace=False))
        if len(keep) < need:
            continue
        (sv_pairs if bool(same_viseme(pa, pb)) else ov_pairs).append(
            (pa, pb, np.sort(np.array(keep))))
    sel = np.random.default_rng(SEL_SEED).permutation(len(ov_pairs))[:N_OTHER]
    ov_pairs = [ov_pairs[i] for i in sel]
    print(f"  same-viseme {len(sv_pairs)} · other {len(ov_pairs)} (random seed {SEL_SEED}) "
          f"· n_train fixed {N_TRAIN}")

    out = {"sv": [], "ov": []}
    t0 = time.time()
    for grp, plist in (("sv", sv_pairs), ("ov", ov_pairs)):
        for pa, pb, keep in plist:
            gidx = rows[keep]
            order = np.argsort(gidx); inv = np.argsort(order)
            S = np.asarray(stem[gidx[order]], np.float32)[inv]
            C = np.asarray(cand[keep[order]], np.float32)[inv]
            yv = (Y[keep] == PH.index(pb)).astype(np.int64)
            g = spk[U[keep]]
            us = np.unique(g); rs = np.random.default_rng(7); rs.shuffle(us)
            te_mask = np.isin(g, us[:max(1, int(0.3 * len(us)))])
            trp, tep = np.nonzero(~te_mask)[0], np.nonzero(te_mask)[0]
            if len(trp) < N_TRAIN or len(tep) < 200:
                continue
            tr = rs.choice(trp, N_TRAIN, replace=False)
            te = tep if len(tep) <= N_TEST else rs.choice(tep, N_TEST, replace=False)
            if len(np.unique(yv[tr])) < 2 or len(np.unique(yv[te])) < 2:
                continue
            base = probe(S, yv, tr, te)                      # stem-only, the difficulty proxy
            dn, ds = [], []
            for seed in SEEDS:
                r_ = np.random.default_rng(seed)
                C0 = pl.shuffle_rows(C, seed=seed)
                G = r_.standard_normal(C.shape).astype(np.float32)
                SH = pl.shuffle_rows(C0, seed=seed + 500)
                a0 = probe(np.hstack([S, C0]), yv, tr, te)
                dn.append(a0 - probe(np.hstack([S, G]), yv, tr, te))
                ds.append(a0 - probe(np.hstack([S, SH]), yv, tr, te))
            rec = {"pair": f"{pa}-{pb}", "base_acc": base,
                   "d_noise": float(np.mean(dn)), "d_shuffle": float(np.mean(ds))}
            out[grp].append(rec)
            print(f"    [{grp}] {pa}-{pb:5s} base={base:.3f}  d_noise {rec['d_noise']:+.4f}  "
                  f"d_shuffle {rec['d_shuffle']:+.4f}   [{time.time()-t0:.0f}s]")
            (HERE / "partial.json").write_text(json.dumps(out, indent=2) + "\n")

    if len(out["sv"]) < 3 or len(out["ov"]) < 8:
        print(f"\n  VOID: {len(out['sv'])} sv / {len(out['ov'])} other usable")
        return 0

    # control must be clean in BOTH groups or the comparison is contaminated
    ctrl = {}
    for g in ("sv", "ov"):
        m, lo, hi = pl.boot(np.array([r["d_shuffle"] for r in out[g]]))
        ctrl[g] = [round(m, 4), round(lo, 4), round(hi, 4)]
        print(f"  control d_shuffle [{g}]: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    clean = all(c[1] <= 0 <= c[2] for c in ctrl.values())
    if not clean:
        print("\n  VOID: own-shuffle null is not clean in one group; comparison contaminated")
        return 0

    verdict, det = evaluate(out["sv"], out["ov"])
    allp = out["sv"] + out["ov"]
    from scipy.stats import spearmanr
    rho = float(spearmanr([r["base_acc"] for r in allp],
                          [r["d_noise"] for r in allp]).statistic)
    print(f"\n  mean d_noise: same-viseme {det['mean_same_viseme']:+.4f} "
          f"(n={det['n_sv']})   other {det['mean_other']:+.4f} (n={det['n_other']})")
    print(f"  D = other - same_viseme = {det['D']:+.4f}  CI {det['CI']}  "
          f"half-width {det['half_width']}")
    print(f"  CONFOUND rho(baseline acc, d_noise) over all {len(allp)} pairs = {rho:+.3f}")
    print(f"\n  VERDICT: {verdict}")
    (HERE / "result.json").write_text(json.dumps(
        {"verdict": verdict, **det, "control": ctrl,
         "rho_baseacc_dnoise": round(rho, 3), "pairs": out}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
