#!/usr/bin/env python3
"""Check a rerun against every number and verdict the paper reports.

Run ./reproduce.sh first; it overwrites the result JSONs in place. This then
asserts, for every published interval, that the rerun's point estimate lands
inside it, and that every pre-registered verdict is unchanged. A miss prints
both numbers and exits non-zero -- it does not quietly pass.

    python verify.py              # the public corpora
    python verify.py --licensed   # also the LRS2 half, if you regenerated it

`expected.json` is generated from the shipped results, so the two cannot drift
apart; see package_artifact.py in the paper repository.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = json.loads((HERE / "expected.json").read_text())


def dig(obj, path):
    for k in path.split("|"):
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def check(group, verbose):
    ok = bad = miss = 0
    for rel, spec in EXP.get(group, {}).items():
        p = HERE / "experiments" / rel
        if not p.exists():
            n = len(spec["intervals"]) + len(spec["verdicts"])
            print(f"  MISSING  {rel}  ({n} checks) -- run ./reproduce.sh first")
            miss += n
            continue
        res = json.loads(p.read_text())
        for path, (lo, hi) in spec["intervals"].items():
            v = dig(res, path)
            got = v[0] if isinstance(v, list) else v
            if got is None:
                print(f"  MISSING  {rel}:{path}")
                miss += 1
            elif lo <= got <= hi:
                ok += 1
                if verbose:
                    print(f"  ok       {rel}:{path}  "
                          f"{got:+.4f} in [{lo:+.4f},{hi:+.4f}]")
            else:
                print(f"  OUTSIDE  {rel}:{path}  "
                      f"{got:+.4f} NOT in [{lo:+.4f},{hi:+.4f}]")
                bad += 1
        for path, want in spec["verdicts"].items():
            got = dig(res, path)
            if got is None:
                print(f"  MISSING  {rel}:{path}")
                miss += 1
            elif got == want:
                ok += 1
                if verbose:
                    print(f"  ok       {rel}:{path}  verdict {got}")
            else:
                print(f"  CHANGED  {rel}:{path}  verdict {got!r}, paper says {want!r}")
                bad += 1
    return ok, bad, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--licensed", action="store_true",
                    help="also check the LRS2 half (needs $LRS2_DERIVED)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    groups = ["public"] + (["licensed"] if a.licensed else [])
    ok = bad = miss = 0
    for g in groups:
        print(f"== {g} ==")
        o, b, m = check(g, a.verbose)
        ok, bad, miss = ok + o, bad + b, miss + m
    print(f"\n  {ok} agree with the paper, {bad} do not, {miss} missing")
    if not a.licensed:
        print("  (the LRS2 half is not checked; see LRS2_REGENERATION.md, then "
              "--licensed)")
    return 1 if (bad or miss) else 0


if __name__ == "__main__":
    sys.exit(main())
