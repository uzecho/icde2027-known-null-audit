#!/usr/bin/env python3
"""Decide whether the ICDE loop may stop. Machine-checked, not judged.

A limit written in a README is not enforced (see CLAUDE.md, "Build gates"). The same
applies to a stop rule: this script is the only thing allowed to say STOP, so that the
loop cannot talk itself into a PASS at 3am.

    python icde2027/check_stop.py        # prints STOP or CONTINUE
    python icde2027/check_stop.py --why  # plus the per-condition breakdown

Exit codes: 0 = CONTINUE, 10 = STOP, 2 = malformed state (a bug, halt and fix).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "state.json"

# The three conditions from STOP_RULE.md, in order.
CONDITIONS = [
    ("gate_verdict", "gate PASS as pre-registered"),
    ("control_cleared", "survives the correct null from the control catalogue"),
    ("replicated", "reproduces on a disjoint split or seed"),
]


def load():
    if not STATE.exists():
        sys.exit(f"MALFORMED: {STATE} does not exist")
    try:
        s = json.loads(STATE.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"MALFORMED: {STATE} is not valid JSON ({e})")
    for key in ("tick", "gates_tested", "candidates"):
        if key not in s:
            sys.exit(f"MALFORMED: state.json is missing {key!r}")
    if not isinstance(s["candidates"], list):
        sys.exit("MALFORMED: candidates must be a list")
    return s


def qualifies(c):
    """A candidate stops the loop only on all three conditions."""
    checks = []
    for key, label in CONDITIONS:
        v = c.get(key)
        ok = (v == "PASS") if key == "gate_verdict" else (v is True)
        checks.append((label, ok, v))
    return all(ok for _, ok, _ in checks), checks


def main():
    why = "--why" in sys.argv
    s = load()
    n = s["gates_tested"]

    winners = []
    for c in s["candidates"]:
        ok, checks = qualifies(c)
        if why:
            print(f"\n  {c.get('id', '?')}  {c.get('title', '')}")
            for label, passed, raw in checks:
                print(f"    [{'x' if passed else ' '}] {label}  (= {raw!r})")
        if ok:
            winners.append(c)

    if why:
        print()

    if winners:
        w = winners[0]
        print(f"STOP  {w.get('id')} — {w.get('title')}")
        print(f"      PASS #1 of {n} gates tested. Report N in the paper.")
        if len(winners) > 1:
            print(f"      note: {len(winners)} candidates qualify; "
                  f"the loop stops on the first.")
        return 10

    # A DEAD candidate still has gate_verdict None — it was killed on novelty, never run.
    # Counting it as "open" overstates what is left, so status is checked too.
    RETIRED = {"DEAD", "SUPERSEDED", "FAIL"}
    alive = [c for c in s["candidates"]
             if c.get("status") not in RETIRED and c.get("gate_verdict") in (None, "", "OPEN")]
    dead_n = sum(1 for c in s["candidates"] if c.get("status") in RETIRED)
    print(f"CONTINUE  tick {s['tick']}, {n} gates tested, "
          f"{len(alive)} candidates alive, {dead_n} killed on novelty")
    if alive:
        print(f"          next: {alive[0]['id']} — {alive[0]['title']}")
    else:
        print("          NO CANDIDATES LEFT — generate new ones or report back; "
              "do not lower the bar (STOP_RULE.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
