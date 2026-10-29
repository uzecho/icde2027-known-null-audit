#!/usr/bin/env python3
"""C6 control + replication. Gates are PLAN.md §4b and §4c, both written before this ran.

§4b  Does the CORRECT null show the same r-dependence? If it does, the main PASS is
     small-sample instability wearing a costume, and it is hollow.
§4c  Does the main gate reproduce on a disjoint half with a different seed family?

    python 01_control.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probelib as pl  # noqa: E402

import importlib.util  # noqa: E402
HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("c6", HERE / "00_run.py")
c6 = importlib.util.module_from_spec(_s); _s.loader.exec_module(c6)
c4 = c6.c4

REPL_SEEDS = (100, 101, 102, 103, 104)       # disjoint seed family for §4c


def both_arms(name, X, y, seeds, verbose=True):
    """Same sweep as 00_run, measuring BOTH nulls per cell so they are commensurable."""
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
                rows.append({"r": er_G / n_tr,
                             "d_noise": a_b0 - a_gs, "d_shuffle": a_b0 - a_sh})
    if verbose:
        print(f"    {name}: {len(rows)} cells, r "
              f"{min(x['r'] for x in rows):.4f} .. {max(x['r'] for x in rows):.4f}")
    return rows


def rho_ci(r, v, n=2000):
    rho = float(spearmanr(r, v).statistic)
    rng, bs = np.random.default_rng(0), []
    for _ in range(n):
        i = rng.integers(0, len(r), len(r))
        if len(np.unique(r[i])) > 2:
            bs.append(float(spearmanr(r[i], v[i]).statistic))
    return rho, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    main_pass = ["digits/vision", "covtype/tabular"]     # from 00_run result.json
    out = {}
    for name, X, y in c4.load_datasets("digits,news,covtype"):
        if name not in main_pass:
            print(f"\n  {name}: skipped, did not pass the main gate")
            continue
        print(f"\n  {name}")

        # ---- §4b control: both nulls, same cells
        rows = both_arms(name, X, y, c6.SEEDS)
        r = np.array([x["r"] for x in rows])
        dn = np.array([x["d_noise"] for x in rows])
        ds = np.array([x["d_shuffle"] for x in rows])
        rho_n, lo_n, hi_n = rho_ci(r, dn)
        rho_s, lo_s, hi_s = rho_ci(r, ds)
        top = r >= np.percentile(r, 90)
        m_dn, lo_dn, _ = pl.boot(dn[top])
        m_ds, lo_ds, hi_ds = pl.boot(ds[top])
        c1 = (rho_s <= 0.5) or (lo_s <= 0 <= hi_s)
        c2 = abs(m_ds) < 0.5 * abs(m_dn)
        control = "CLEARED" if (c1 and c2) else "FAILED"
        print(f"    rho(r, d_noise)   = {rho_n:+.3f} [{lo_n:+.3f}, {hi_n:+.3f}]")
        print(f"    rho(r, d_shuffle) = {rho_s:+.3f} [{lo_s:+.3f}, {hi_s:+.3f}]  "
              f"<- the control")
        print(f"    at top r: d_noise {m_dn:+.4f}   d_shuffle {m_ds:+.4f} "
              f"[{lo_ds:+.4f}, {hi_ds:+.4f}]")
        print(f"    CONTROL: {control}  (cond1 {c1}, cond2 {c2})")

        # ---- §4c replication: disjoint half, different seed family
        rng = np.random.default_rng(777)
        half = rng.permutation(len(y))[: len(y) // 2]
        rep = both_arms(name + "/replication", X[half], y[half], REPL_SEEDS)
        rr = np.array([x["r"] for x in rep])
        rdn = np.array([x["d_noise"] for x in rep])
        rho_r, lo_r, hi_r = rho_ci(rr, rdn)
        rtop = rr >= np.percentile(rr, 90)
        m_r, lo_rr, hi_rr = pl.boot(rdn[rtop])
        repl = "REPLICATED" if (rho_r > 0.5 and lo_r > 0 and m_r > 0.02 and lo_rr > 0) \
            else "NOT REPLICATED"
        print(f"    replication: rho {rho_r:+.3f} [{lo_r:+.3f}, {hi_r:+.3f}], "
              f"d_noise@top {m_r:+.4f} [{lo_rr:+.4f}, {hi_rr:+.4f}] -> {repl}")

        out[name] = {
            "control": control,
            "rho_noise": [round(v, 3) for v in (rho_n, lo_n, hi_n)],
            "rho_shuffle": [round(v, 3) for v in (rho_s, lo_s, hi_s)],
            "top_d_noise": round(m_dn, 4), "top_d_shuffle": round(m_ds, 4),
            "replication": repl,
            "repl_rho": [round(v, 3) for v in (rho_r, lo_r, hi_r)],
            "repl_top_d_noise": [round(v, 4) for v in (m_r, lo_rr, hi_rr)]}

    ok = out and all(v["control"] == "CLEARED" and v["replication"] == "REPLICATED"
                     for v in out.values())
    out["_all_cleared_and_replicated"] = bool(ok)
    (HERE / "control.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  CONTROL+REPLICATION across passing datasets: "
          f"{'OK' if ok else 'NOT OK'} -> control.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
