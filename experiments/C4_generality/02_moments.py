#!/usr/bin/env python3
"""The marginal statistics §III-C quotes, over the same blocks the paper uses.

§III-C used to assert that after standardisation a real block and its Gaussian
comparator "differ in exactly one respect: correlation structure between columns".
That is false -- standardising matches the first two MARGINAL moments and nothing
else -- and this script is the measurement that says so. It produces the figures
quoted in §III-C: distinct values per column, sample skewness, excess kurtosis,
and the count of constant columns.

Reported, not gated. It changes no verdict; it bounds what \\Delta_noise can be
read as isolating.

    python 02_moments.py        # -> moments.json

CPU only, seconds.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_s = importlib.util.spec_from_file_location("c4", HERE / "00_run.py")
c4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(c4)

NROW = 1000          # rows sampled per block; the statistics are marginal
SEED = 0


def standardise(M):
    """Exactly what every readout in the paper applies before fitting."""
    M = np.asarray(M, np.float64)
    return (M - M.mean(0, keepdims=True)) / M.std(0, keepdims=True).clip(1e-6)


def describe(M):
    M = np.asarray(M, np.float64)[:NROW]
    Z = standardise(M)
    const = int((M.std(0) < 1e-12).sum())
    # skew/kurtosis over the non-constant columns only: a constant column
    # standardises to all-zero and its higher moments are undefined, which is
    # itself part of the finding and is reported separately as n_constant_cols.
    live = M.std(0) >= 1e-12
    return {
        "n_rows": int(len(M)), "n_cols": int(M.shape[1]),
        "mean_distinct_values_per_col":
            round(float(np.mean([len(np.unique(M[:, j])) for j in range(M.shape[1])])), 2),
        "mean_skew": round(float(np.mean(skew(Z[:, live]))), 3) if live.any() else None,
        "mean_excess_kurtosis":
            round(float(np.mean(kurtosis(Z[:, live]))), 3) if live.any() else None,
        "n_constant_cols": const,
    }


def main():
    out, rng = {}, np.random.default_rng(SEED)
    for name, X, y in c4.load_datasets("digits,news,covtype"):
        blocks = c4.blocks_for(X, np.random.default_rng(12345))
        out[name] = {b: describe(C) for b, C in blocks.items()}
        shape = next(iter(blocks.values()))[:NROW].shape
        out[name]["GAUSSIAN"] = describe(rng.standard_normal(shape))

    # The four figures §III-C quotes, in the [value, lo, hi] shape the artifact's
    # harvest() reads, so verify.py checks a rerun against them like any other
    # published number. These statistics are deterministic given the seed; the
    # bounds absorb BLAS-level variation only.
    live = [(b, d) for bb in out.values() for b, d in bb.items() if b != "GAUSSIAN"]
    dig = out["digits/vision"]["raw"]["mean_distinct_values_per_col"]
    sk = max(d["mean_skew"] for _, d in live if d["mean_skew"] is not None)
    ku = max(d["mean_excess_kurtosis"] for _, d in live
             if d["mean_excess_kurtosis"] is not None)
    cc = max(d["n_constant_cols"] for _, d in live)
    out["paper_checks"] = {
        "digits_raw_distinct_per_col": [dig, dig - 0.2, dig + 0.2],
        "max_mean_skew": [sk, sk - 0.2, sk + 0.2],
        "max_mean_excess_kurtosis": [ku, ku - 2.0, ku + 2.0],
        "max_constant_cols": [cc, cc - 0.5, cc + 0.5],
    }

    (HERE / "moments.json").write_text(json.dumps(out, indent=1) + "\n")

    print(f"\n  {'corpus':16s} {'block':10s} {'distinct/col':>13s} {'skew':>8s} "
          f"{'kurtosis':>9s} {'const cols':>11s}")
    for c, bb in out.items():
        if c == "paper_checks":
            continue
        for b, d in bb.items():
            sk = f"{d['mean_skew']:+8.2f}" if d["mean_skew"] is not None else f"{'--':>8s}"
            ku = (f"{d['mean_excess_kurtosis']:+9.2f}"
                  if d["mean_excess_kurtosis"] is not None else f"{'--':>9s}")
            print(f"  {c:16s} {b:10s} {d['mean_distinct_values_per_col']:13.1f} "
                  f"{sk} {ku} {d['n_constant_cols']:11d}")
    print("\n  -> moments.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
