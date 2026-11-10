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
echo "== C4_generality: the rank the readout is actually handed =="
$PY experiments/C4_generality/01_ranks.py
echo "== C4_generality: how the arms differ beyond correlation =="
$PY experiments/C4_generality/02_moments.py

echo
echo "== C13_rconvention: one convention for r =="
$PY experiments/C13_rconvention/00_run.py

echo
echo "== C6_ratio: does magnitude track r? =="
$PY experiments/C6_ratio/00_run.py
$PY experiments/C6_ratio/01_control.py

echo
echo "== C7_readout: does it survive a change of readout? (headline) =="
$PY experiments/C7_readout/00_run.py

echo "== C12_marginal: does matching marginals explain the residual? =="
$PY experiments/C12_marginal/00_run.py --selftest
$PY experiments/C12_marginal/00_run.py

echo "== C11_covmatch: is the sign flip caused by correlation structure? =="
echo "   (selftest first: both branches must be reachable)"
$PY experiments/C11_covmatch/00_run.py --selftest
$PY experiments/C11_covmatch/00_run.py

echo "== Section V as printed: cluster bootstrap over (n_train, seed) =="
echo "   (refuses unless rho and the top-r mean reproduce the published sweep)"
$PY experiments/C6_ratio/audit_cluster/rerun_cluster.py

echo "== Table II as printed: 50 seeds, cluster bootstrap over seeds =="
echo "   (refuses unless seeds 0-2 reproduce the 3-seed run first)"
NSEEDS=50 $PY experiments/C7_readout/audit_cluster/15seed_rerun.py

echo
echo "== C8_mechanism: does the mechanism predict unseen readouts? =="
$PY experiments/C8_mechanism/00_run.py

echo
echo "== C9_power: do the baselines differ as KEEP/DROP decision rules? =="
echo "   (the slow one: ~18 min on 8 threads. Set SKIP_C9=1 to leave it out.)"
$PY experiments/C9_power/00_run.py --selftest
if [ "${SKIP_C9:-0}" = "1" ]; then
  echo "   SKIPPED by SKIP_C9=1; verify.py will report its checks as MISSING."
else
  $PY experiments/C9_power/00_run.py
fi

echo
echo "All public gates rerun. Now: $PY verify.py"
