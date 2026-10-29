#!/usr/bin/env python3
"""C6 keystone, attempt 2. The first attempt VOIDed on its own guard.

Why it VOIDed: `00_run.py` assumed every phone pair reaches experiment 05's PER_CLASS cap
of 8000 paired samples, so that n_train = 5333 and r = 0.1611. Only 2 of 8 pairs did; the
rest were smaller, giving n_train as low as 1029 and a realised mean r of 0.2903. The
pre-registered guard caught it and refused to report a verdict.

The fix: **fix n_train at exactly 5333 by construction**, selecting only pairs that have
enough paired samples to supply it, so every pair sits on r = 0.1611 rather than wherever
its natural size lands.

DISCLOSURE: the first attempt printed per-pair Δ values before it VOIDed, so blindness to
LRS2 is compromised for this re-run. What protects the test is that the prediction interval
[+0.0149, +0.0238] was frozen in PLAN.md §1 from external data and **cannot be moved**. The
target is fixed; only the sampling is corrected.

CPU only. Checkpoints after every pair, because attempt 1 wrote nothing until the end and a
timeout would have lost it.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")      # shared box; attempt 1 took ~20 threads
import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402

torch.set_num_threads(8)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(str(Path(__file__).resolve().parents[2])) / "lrs2"))
import probelib as pl  # noqa: E402

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

N_TRAIN = 5333           # experiment 05's n_train; fixes r by construction
N_TEST = 2000
N_PAIRS = 10
SEEDS = (0, 1, 2)
PRED_LO, PRED_HI = 0.0149, 0.0238        # PLAN.md §1, frozen, not adjustable
R_TARGET, R_TOL = 0.1611, 0.10


def probe(X, y, tr, te):
    t = lambda a, d: torch.as_tensor(np.asarray(a, d))                  # noqa: E731
    return pl.fit(t(X[tr], np.float32), t(y[tr], np.int64),
                  t(X[te], np.float32), t(y[te], np.int64))


def main():
    rng = np.random.default_rng(0)
    meta = json.loads((W / "meta.json").read_text())
    PH = meta["phones"]
    spk = np.array([u["speaker"] for u in meta["utts"]])
    rows = np.load(P / "rows.npy")
    Y = np.load(W / "y.npy")[:, 1][rows]
    U = np.load(W / "utt.npy")[rows]
    stem = np.load(W / "X_front512.npy", mmap_mode="r")
    cand = np.load(P / "crop_px.npy", mmap_mode="r")

    # pairs with enough paired samples to supply N_TRAIN + N_TEST
    need = N_TRAIN + N_TEST
    pairs = [(x, y) for i, x in enumerate(PH) for y in PH[i + 1:]]
    # SELECTION-ROBUSTNESS (added 2026-10-28): the first run took the first 10 qualifying
    # pairs in phone order, which is AA-* and AE-* -- a selection never pre-registered, and
    # d_noise varies 5.6x across pairs at identical r. --random draws instead.
    if "--random" in sys.argv:
        sel_seed = int(sys.argv[sys.argv.index("--random") + 1])
        np.random.default_rng(sel_seed).shuffle(pairs)
        print(f"  pair selection: RANDOM, seed {sel_seed}")
    chosen = []
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
        if len(keep) >= need:
            chosen.append((pa, pb, np.sort(np.array(keep))))
        if len(chosen) >= N_PAIRS:
            break
    print(f"  {len(chosen)} pairs with >= {need} paired samples; n_train fixed at {N_TRAIN}")

    out, t0 = [], time.time()
    for pa, pb, keep in chosen:
        gidx = rows[keep]
        order = np.argsort(gidx); inv = np.argsort(order)
        S = np.asarray(stem[gidx[order]], np.float32)[inv]
        C = np.asarray(cand[keep[order]], np.float32)[inv]
        yv = (Y[keep] == PH.index(pb)).astype(np.int64)
        grp = spk[U[keep]]

        # speaker-grouped split, then train subsampled to exactly N_TRAIN
        us = np.unique(grp)
        rs = np.random.default_rng(7)
        rs.shuffle(us)
        cut = max(1, int(0.3 * len(us)))
        te_mask = np.isin(grp, us[:cut])
        tr_pool, te_pool = np.nonzero(~te_mask)[0], np.nonzero(te_mask)[0]
        if len(tr_pool) < N_TRAIN or len(te_pool) < 300:
            print(f"    {pa}-{pb}: skipped, split too small "
                  f"(train pool {len(tr_pool)}, test pool {len(te_pool)})")
            continue
        tr = rs.choice(tr_pool, N_TRAIN, replace=False)
        te = te_pool if len(te_pool) <= N_TEST else rs.choice(te_pool, N_TEST, replace=False)
        if len(np.unique(yv[tr])) < 2 or len(np.unique(yv[te])) < 2:
            continue

        dn, ds, ers = [], [], []
        for seed in SEEDS:
            r_ = np.random.default_rng(seed)
            C0 = pl.shuffle_rows(C, seed=seed)                 # known-null, zero info
            G = r_.standard_normal(C.shape).astype(np.float32)
            SH = pl.shuffle_rows(C0, seed=seed + 500)
            ers.append(pl.effective_rank(G[tr]))               # eff rank AT n_train
            a0 = probe(np.hstack([S, C0]), yv, tr, te)
            ag = probe(np.hstack([S, G]), yv, tr, te)
            ash = probe(np.hstack([S, SH]), yv, tr, te)
            dn.append(a0 - ag); ds.append(a0 - ash)
        rec = {"pair": f"{pa}-{pb}", "n_train": N_TRAIN, "r": float(np.mean(ers)) / N_TRAIN,
               "d_noise": float(np.mean(dn)), "d_shuffle": float(np.mean(ds))}
        out.append(rec)
        print(f"    {pa}-{pb:5s} r={rec['r']:.4f}  d_noise {rec['d_noise']:+.4f}  "
              f"d_shuffle {rec['d_shuffle']:+.4f}   [{time.time()-t0:.0f}s]")
        (HERE / "rerun_partial.json").write_text(json.dumps(out, indent=2) + "\n")

    if len(out) < 5:
        print(f"\n  VOID: {len(out)} usable pairs, PLAN.md requires >= 5")
        return 0
    r_mean = float(np.mean([o["r"] for o in out]))
    if abs(r_mean - R_TARGET) / R_TARGET > R_TOL:
        print(f"\n  VOID: realised r {r_mean:.4f} deviates >{R_TOL:.0%} from {R_TARGET}")
        return 0

    dn = np.array([o["d_noise"] for o in out]); ds = np.array([o["d_shuffle"] for o in out])
    m_dn, lo_dn, hi_dn = pl.boot(dn)
    m_ds, lo_ds, hi_ds = pl.boot(ds)
    overlaps = not (hi_dn < PRED_LO or lo_dn > PRED_HI)
    ctrl_ok = lo_ds <= 0 <= hi_ds
    verdict = "CONFIRMED" if (overlaps and ctrl_ok) else "NOT CONFIRMED"
    print(f"\n  realised r         {r_mean:.4f}  (target {R_TARGET})")
    print(f"  PREDICTED d_noise  [{PRED_LO:+.4f}, {PRED_HI:+.4f}]  (frozen, external only)")
    print(f"  MEASURED  d_noise  {m_dn:+.4f} [{lo_dn:+.4f}, {hi_dn:+.4f}] -> overlap {overlaps}")
    print(f"  control   d_shuffle {m_ds:+.4f} [{lo_ds:+.4f}, {hi_ds:+.4f}] -> clean {ctrl_ok}")
    print(f"\n  VERDICT: {verdict}   ({len(out)} pairs, {time.time()-t0:.0f}s)")
    (HERE / "rerun_result.json").write_text(json.dumps(
        {"verdict": verdict, "r_measured": round(r_mean, 4), "predicted": [PRED_LO, PRED_HI],
         "measured_d_noise": [round(v, 4) for v in (m_dn, lo_dn, hi_dn)],
         "control_d_shuffle": [round(v, 4) for v in (m_ds, lo_ds, hi_ds)],
         "overlaps": bool(overlaps), "control_clean": bool(ctrl_ok), "pairs": out},
        indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
