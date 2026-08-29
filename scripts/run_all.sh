#!/usr/bin/env bash
# Run the three scenarios back to back. Same seed, same world, only the ads differ.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-./venv/bin/python}

for s in noad broadcast retarget; do
  echo "=== $(date +%H:%M:%S) running scenario_${s}.yaml ==="
  $PY src/main.py --config "scenario_${s}.yaml"
done

echo "=== $(date +%H:%M:%S) analysing ==="
$PY scripts/ghost_analysis.py output_noad output_broadcast output_retarget \
    --json-out demo/ghost_summary.json -o demo/ghost_decomposition.png
