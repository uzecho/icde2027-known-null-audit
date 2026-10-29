#!/usr/bin/env python3
"""C6 keystone: does the external r-curve predict LRS2's artefact out of sample?

Gate is PLAN.md §3. The prediction interval was frozen from external data before any LRS2
accuracy was read. CPU only.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

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
WORK = _lrs2("ambiguity_map")
PACK = _lrs2("frontend_ceiling/packed")

OFFSET_IDX, PER_CLASS, NFOLD = 1, 4000, 3     # experiment 05's own settings
PRED_LO, PRED_HI = 0.0149, 0.0238             # PLAN.md §1, frozen
R_TARGET, R_TOL = 0.1611, 0.10                # PLAN.md §3
N_PAIRS = 8


def probe(X, y, fold, f):
    te, tr = fold == f, fold != f
    if te.sum() < 200 or tr.sum() < 200 or len(np.unique(y[te])) < 2:
        return float("nan")
    t = lambda a, d: torch.as_tensor(np.asarray(a, d))                  # noqa: E731
    return pl.fit(t(X[tr], np.float32), t(y[tr], np.int64),
                  t(X[te], np.float32), t(y[te], np.int64))


def main():
    rng = np.random.default_rng(0)
    meta = json.load(open(WORK / "frames" / "meta.json"))
    PH = meta["phones"]
    spk = np.array([u["speaker"] for u in meta["utts"]])
    rows = np.load(PACK / "rows.npy")
    Y = np.load(WORK / "frames" / "y.npy")[:, OFFSET_IDX][rows]
    U = np.load(WORK / "frames" / "utt.npy")[rows]
    stem = np.load(WORK / "frames" / "X_front512.npy", mmap_mode="r")
    cand = np.load(PACK / "crop_px.npy", mmap_mode="r")        # 1024 wide, 05's widest
    print(f"  rows {len(rows):,} · stem {stem.shape[1]}d · crop_px {cand.shape[1]}d")

    pairs = [(x, y) for i, x in enumerate(PH) for y in PH[i + 1:]]
    sv = [p for p in pairs if same_viseme(*p)]

    out, t0 = [], time.time()
    for (pa, pb) in sv[:N_PAIRS]:
        ia, ib = PH.index(pa), PH.index(pb)
        idx = np.nonzero((Y == ia) | (Y == ib))[0]
        uu = U[idx]
        keep = []
        for u in np.unique(uu):                                 # paired within utterance
            f = idx[uu == u]
            ya, yb = f[Y[f] == ia], f[Y[f] == ib]
            c = min(len(ya), len(yb))
            if c:
                keep += list(rng.choice(ya, c, replace=False))
                keep += list(rng.choice(yb, c, replace=False))
        if len(keep) < 600:
            continue
        keep = np.sort(np.array(keep))
        if len(keep) > 2 * PER_CLASS:
            keep = np.sort(rng.choice(keep, 2 * PER_CLASS, replace=False))

        gidx = rows[keep]
        order = np.argsort(gidx)                                # sequential read, not scattered
        inv = np.argsort(order)
        S = np.asarray(stem[gidx[order]], np.float32)[inv]
        C = np.asarray(cand[keep[order]], np.float32)[inv]
        yv = (Y[keep] == ib).astype(np.int64)
        grp = spk[U[keep]]
        _, gi = np.unique(grp, return_inverse=True)
        fold = gi % NFOLD                                        # speaker-grouped folds

        n_tr = int(np.mean([(fold != f).sum() for f in range(NFOLD)]))
        accs = {"b0": [], "gs": [], "sh": []}
        ers = []
        for seed in (0, 1, 2):
            r_ = np.random.default_rng(seed)
            C0 = pl.shuffle_rows(C, seed=seed)                   # known-null: zero info
            G = r_.standard_normal(C.shape).astype(np.float32)
            SH = pl.shuffle_rows(C0, seed=seed + 500)
            ers.append(pl.effective_rank(G[:min(len(G), 4000)]))
            for f in range(NFOLD):
                accs["b0"].append(probe(np.hstack([S, C0]), yv, fold, f))
                accs["gs"].append(probe(np.hstack([S, G]), yv, fold, f))
                accs["sh"].append(probe(np.hstack([S, SH]), yv, fold, f))
        m = {k: float(np.nanmean(v)) for k, v in accs.items()}
        r_meas = float(np.mean(ers)) / n_tr
        out.append({"pair": f"{pa}-{pb}", "n": len(keep), "n_train": n_tr, "r": r_meas,
                    "d_noise": m["b0"] - m["gs"], "d_shuffle": m["b0"] - m["sh"]})
        print(f"    {pa}-{pb:4s} n={len(keep):5d} n_tr={n_tr:5d} r={r_meas:.4f}  "
              f"d_noise {out[-1]['d_noise']:+.4f}  d_shuffle {out[-1]['d_shuffle']:+.4f}"
              f"   [{time.time()-t0:.0f}s]")

    if len(out) < 5:
        print(f"\n  VOID: only {len(out)} usable pairs, PLAN.md requires >= 5")
        return 0

    dn = np.array([o["d_noise"] for o in out])
    ds = np.array([o["d_shuffle"] for o in out])
    r_mean = float(np.mean([o["r"] for o in out]))
    if abs(r_mean - R_TARGET) / R_TARGET > R_TOL:
        print(f"\n  VOID: realised r {r_mean:.4f} deviates >{R_TOL:.0%} from {R_TARGET}")
        return 0

    m_dn, lo_dn, hi_dn = pl.boot(dn)
    m_ds, lo_ds, hi_ds = pl.boot(ds)
    overlaps = not (hi_dn < PRED_LO or lo_dn > PRED_HI)
    ctrl_ok = lo_ds <= 0 <= hi_ds
    verdict = "CONFIRMED" if (overlaps and ctrl_ok) else "NOT CONFIRMED"

    print(f"\n  realised r          {r_mean:.4f}  (target {R_TARGET}, +/-{R_TOL:.0%})")
    print(f"  PREDICTED  d_noise  [{PRED_LO:+.4f}, {PRED_HI:+.4f}]   (frozen, external only)")
    print(f"  MEASURED   d_noise  {m_dn:+.4f} [{lo_dn:+.4f}, {hi_dn:+.4f}]  "
          f"-> overlap {overlaps}")
    print(f"  control    d_shuffle{m_ds:+.4f} [{lo_ds:+.4f}, {hi_ds:+.4f}]  -> clean {ctrl_ok}")
    # FAIL-branch reachability, numerically: an interval of width 0.009 is missable
    print(f"  reachability: a measured CI of e.g. [+0.0400,+0.0500] would NOT overlap "
          f"[{PRED_LO:+.4f}, {PRED_HI:+.4f}] -> gate can return NOT CONFIRMED")
    print(f"\n  VERDICT: {verdict}   ({len(out)} pairs, {time.time()-t0:.0f}s)")

    (HERE / "result.json").write_text(json.dumps(
        {"verdict": verdict, "r_measured": round(r_mean, 4),
         "predicted": [PRED_LO, PRED_HI],
         "measured_d_noise": [round(v, 4) for v in (m_dn, lo_dn, hi_dn)],
         "control_d_shuffle": [round(v, 4) for v in (m_ds, lo_ds, hi_ds)],
         "overlaps": bool(overlaps), "control_clean": bool(ctrl_ok),
         "pairs": out}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
