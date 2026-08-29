#!/usr/bin/env bash
# Run the three scenarios concurrently. Ollama batches the concurrent requests
# into one GPU batch, so three worlds cost far less than three times one world.
# Requires the server to have been started with OLLAMA_NUM_PARALLEL>=3.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-./venv/bin/python}

pids=()
for s in noad broadcast retarget; do
  $PY src/main.py --config "scenario_${s}.yaml" > "run_${s}.out" 2>&1 &
  pid=$!
  pids+=($pid)
  echo "started scenario_${s}.yaml (pid $pid)"
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "at least one run failed; see run_*.out"; exit 1; }

echo "=== $(date +%H:%M:%S) all runs done, analysing ==="
$PY scripts/ghost_analysis.py output_noad output_broadcast output_retarget \
    --json-out demo/ghost_summary.json -o demo/ghost_decomposition.png
