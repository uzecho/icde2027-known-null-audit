#!/usr/bin/env bash
# Rerun every gate that needs no licensed data. CPU only.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PY="${PYTHON:-python3}"

echo "== FAIL-branch reachability selftests (these run BEFORE the real runs) =="
$PY experiments/C7_readout/00_run.py --selftest
$PY experiments/C8_mechanism/00_run.py --selftest

echo
echo "== C4_generality: does the artefact generalise? (pre-registered FAIL) =="
$PY experiments/C4_generality/00_run.py

echo
echo "== C6_ratio: does magnitude track r? =="
$PY experiments/C6_ratio/00_run.py
$PY experiments/C6_ratio/01_control.py

echo
echo "== C7_readout: does it survive a change of readout? (headline) =="
$PY experiments/C7_readout/00_run.py

echo
echo "== C8_mechanism: does the mechanism predict unseen readouts? =="
$PY experiments/C8_mechanism/00_run.py

echo
echo "All public gates rerun. Now: $PY verify.py"
