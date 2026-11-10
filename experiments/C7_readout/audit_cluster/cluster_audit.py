#!/usr/bin/env python3
"""Audit: Table II bootstraps 12 (block, seed) units that are not independent.

C7_readout/00_run.py:86 re-seeds the RNG from `seed` alone, so within a seed all
four blocks share the train/test split; and because every block is kept at the
corpus width, the Gaussian comparator G drawn at line 92 is the SAME array for
all four. d_noise = a(C~) - a(G) therefore shares its subtrahend across the four
units of a seed. The cluster unit is the seed, n=3, not the pair, n=12.

This reuses C7's cells() verbatim, so the point estimates reproduce exactly; the
only thing that changes is how the 12 values are aggregated.

CPU only.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# self-locating: .../C7_readout/audit_cluster/<this> -> icde2027 in the source
# tree, and -> experiments/ in the packaged artifact, where the sibling
# experiment folders sit at the same depth. A2 refuses an absolute path here.
ICDE = Path(__file__).resolve().parent.parent.parent

_s = importlib.util.spec_from_file_location("c7", ICDE / "C7_readout" / "00_run.py")
c7 = importlib.util.module_from_spec(_s); _s.loader.exec_module(c7)
_s8 = importlib.util.spec_from_file_location("c8", ICDE / "C8_mechanism" / "00_run.py")
c8 = importlib.util.module_from_spec(_s8); _s8.loader.exec_module(c8)
c4 = c7.c4

SEEDS = (0, 1, 2)
N_BOOT = 20000          # the paper's figure
FAMILY = {"logistic": "+", "ridge": "+", "mlp": "+", "linsvc": "+", "mlp2": "+",
          "knn": "-", "knn5": "-", "centroid": "-"}


def naive_ci(v, seed=0):
    """What the paper does: resample the 12 (block, seed) pairs."""
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(v, (N_BOOT, len(v)), replace=True).mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def cluster_ci(v, seed_of, seed=0):
    """Resample SEEDS, carrying all four of a seed's blocks together."""
    v = np.asarray(v, float)
    sids = sorted(set(seed_of))
    per = np.array([v[[i for i, s in enumerate(seed_of) if s == sd]].mean()
                    for sd in sids])
    rng = np.random.default_rng(seed)
    b = rng.choice(per, (N_BOOT, len(per)), replace=True).mean(1)
    return per.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    readouts = dict(c7.READOUTS)
    readouts.update(c8._readouts())
    out, flips = {}, []

    for name, X, y in c4.load_datasets("digits,news,covtype"):
        n_max = int(0.7 * len(y))
        import probelib as pl
        er = pl.effective_rank(np.random.default_rng(0).standard_normal(
            (min(len(y), 3000),
             c4.blocks_for(X, np.random.default_rng(1))["raw"].shape[1])))
        n_train = int(min(max(100, er / 0.30), n_max / 2))
        out[name] = {}
        print(f"\n  {name}  n_train={n_train}", flush=True)

        for ro, fn in readouts.items():
            c7.READOUTS = {ro: fn}
            rows = c7.cells(X, y, ro, n_train, SEEDS)
            if not rows:
                continue
            # cells() iterates block-major, seed-minor
            seed_of = [SEEDS[i % len(SEEDS)] for i in range(len(rows))]
            dn = [r["d_noise"] for r in rows]

            nm, nlo, nhi = naive_ci(dn)
            cm, clo, chi = cluster_ci(dn, seed_of)
            res_n = (nlo > 0) or (nhi < 0)
            res_c = (clo > 0) or (chi < 0)
            infl = ((chi - clo) / (nhi - nlo)) if nhi > nlo else float("nan")

            out[name][ro] = {
                "family": FAMILY[ro], "n_units": len(rows),
                "naive": [nm * 100, nlo * 100, nhi * 100],
                "cluster": [cm * 100, clo * 100, chi * 100],
                "resolves_naive": bool(res_n), "resolves_cluster": bool(res_c),
                "ci_inflation": infl,
                "per_seed": [float(np.mean([dn[i] for i, s in enumerate(seed_of)
                                            if s == sd])) * 100 for sd in SEEDS],
            }
            if res_n and not res_c:
                flips.append(f"{name}/{ro}")
            print(f"    {ro:9s} naive {nm*100:+6.2f} [{nlo*100:+6.2f},{nhi*100:+6.2f}]"
                  f"{'  R' if res_n else '  .'}   "
                  f"cluster {cm*100:+6.2f} [{clo*100:+6.2f},{chi*100:+6.2f}]"
                  f"{'  R' if res_c else '  .'}   x{infl:.2f}", flush=True)

    n_n = sum(v["resolves_naive"] for c in out.values() for v in c.values())
    n_c = sum(v["resolves_cluster"] for c in out.values() for v in c.values())
    tot = sum(len(c) for c in out.values())
    infl = [v["ci_inflation"] for c in out.values() for v in c.values()]
    # a cell contradicts its family if it resolves on the wrong side
    against = [f"{cn}/{ro}" for cn, c in out.items() for ro, v in c.items()
               if v["resolves_cluster"] and
               ((v["family"] == "+" and v["cluster"][2] < 0) or
                (v["family"] == "-" and v["cluster"][1] > 0))]

    print(f"\n  cells resolving: naive {n_n}/{tot}   cluster {n_c}/{tot}")
    print(f"  median CI inflation under clustering: x{np.median(infl):.2f}")
    print(f"  cells that stop resolving: {flips or 'none'}")
    print(f"  cells resolving AGAINST their family, clustered: {against or 'none'}")

    Path("cluster_audit.json").write_text(json.dumps(
        {"per_cell": out, "n_resolving_naive": n_n, "n_resolving_cluster": n_c,
         "n_cells": tot, "median_ci_inflation": float(np.median(infl)),
         "stopped_resolving": flips, "against_family": against}, indent=2) + "\n")
    print("  -> cluster_audit.json")


if __name__ == "__main__":
    main()
