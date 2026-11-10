#!/usr/bin/env python3
"""Recompute C4's effective ranks on the measure the probe actually sees.

Why this exists. `00_run.py` reports `pl.effective_rank(C)` on the raw block, and
its VOID guard compares that against the raw Gaussian comparator. But
`probelib.fit` z-scores every column on the train split before fitting, so the
geometry the readout is handed is the *standardised* one. Column scaling changes
a participation ratio: a block whose variance is concentrated in a few
high-variance columns is low-rank raw and much closer to full-rank once every
column is put on the same scale.

That makes the raw rank the wrong quantity for two separate claims:

  1. the VOID guard, which was meant to refuse a vacuous rank contrast;
  2. the paper's "the ordering is backwards" argument, which ranks the corpora
     by rank spread and shows the effect does not follow.

This script changes no verdict and re-runs no probe. It recomputes a descriptive
statistic over the same blocks, from the same construction, and writes
`ranks.json` so the paper can quote a released number instead of an ad hoc one.

    python 01_ranks.py          # writes ranks.json

CPU only; no GPU is touched.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probelib as pl                                            # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from importlib import import_module                              # noqa: E402

run = import_module("00_run") if (HERE / "00_run.py").exists() else None
if run is None:                                                  # pragma: no cover
    raise SystemExit("00_run.py not found; blocks must come from the run itself")


def standardise(X):
    """Exactly what probelib.fit does to its inputs: centre, divide by column sd.

    fit() takes mean/sd on the TRAIN split of each fold; we take them over all
    rows. For a spectrum statistic over 2/3 vs 3/3 of the rows the difference is
    far below the gaps reported here, and using all rows keeps this script
    independent of the fold draw.
    """
    X = np.asarray(X, np.float64)
    return (X - X.mean(0, keepdims=True)) / X.std(0, keepdims=True).clip(1e-6)


def main():
    out = {}
    for name, X, y in run.load_datasets("digits,news,covtype"):
        base_rng = np.random.default_rng(12345)       # same seed as measure()
        blocks = run.blocks_for(X, base_rng)
        per_block = {}
        print(f"\n  {name}: X={X.shape}")
        print(f"    {'block':10s} {'er raw':>8s} {'erG raw':>8s} {'ratio':>7s}   "
              f"{'er std':>8s} {'erG std':>8s} {'ratio':>7s}")
        for bname, C in blocks.items():
            er_raw, er_std = pl.effective_rank(C), pl.effective_rank(standardise(C))
            g_raw, g_std = [], []
            for seed in run.SEEDS:
                rng = np.random.default_rng(seed)
                G = rng.standard_normal(C.shape).astype(np.float32)
                g_raw.append(pl.effective_rank(G))
                g_std.append(pl.effective_rank(standardise(G)))
            gr, gs = float(np.mean(g_raw)), float(np.mean(g_std))
            per_block[bname] = {
                "er_raw": round(er_raw, 2), "er_gauss_raw": round(gr, 2),
                "ratio_raw": round(er_raw / gr, 4),
                "er_std": round(er_std, 2), "er_gauss_std": round(gs, 2),
                "ratio_std": round(er_std / gs, 4),
                "nominal_width": int(C.shape[1])}
            print(f"    {bname:10s} {er_raw:8.2f} {gr:8.2f} {er_raw / gr:7.4f}   "
                  f"{er_std:8.2f} {gs:8.2f} {er_std / gs:7.4f}")

        # What the pre-registered guard decides under each measure. The guard in
        # PLAN.md 5 voids unless EVERY block sits below VOID_RANK_FRAC of its
        # Gaussian comparator.
        frac = run.VOID_RANK_FRAC
        viol_raw = sorted(b for b, v in per_block.items() if v["ratio_raw"] >= frac)
        viol_std = sorted(b for b, v in per_block.items() if v["ratio_std"] >= frac)
        out[name] = {
            "blocks": per_block,
            "void_frac": frac,
            "ratio_raw_range": [min(v["ratio_raw"] for v in per_block.values()),
                                max(v["ratio_raw"] for v in per_block.values())],
            "ratio_std_range": [min(v["ratio_std"] for v in per_block.values()),
                                max(v["ratio_std"] for v in per_block.values())],
            "guard_raw": "VOID" if viol_raw else "pass",
            "guard_std": "VOID" if viol_std else "pass",
            "guard_std_offenders": viol_std}
        print(f"    guard as run (raw)      : {out[name]['guard_raw']}")
        print(f"    guard on what probe sees: {out[name]['guard_std']}"
              f"{'  offenders ' + ','.join(viol_std) if viol_std else ''}")

    out["_note"] = ("Descriptive re-measurement of C4's blocks. No probe was "
                    "re-run and no verdict was recomputed. guard_std is what "
                    "PLAN.md 5's non-degeneracy guard would have decided had it "
                    "been applied to the standardised rank, which is the "
                    "geometry probelib.fit hands the readout.")
    out["_any_guard_std_void"] = any(
        v["guard_std"] == "VOID" for k, v in out.items() if not k.startswith("_"))
    (HERE / "ranks.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  -> {HERE / 'ranks.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
