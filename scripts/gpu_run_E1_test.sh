#!/usr/bin/env bash
# Recovery driver: rebuild caches on a fresh box and run ONLY the test sweep + evaluation,
# using the frozen selections committed in results/tables/selections.parquet (never re-selected).
#   nohup bash scripts/gpu_run_E1_test.sh > results/gpu_run_E1_test.log 2>&1 &
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
CFG="${CFG:-configs/deit_small.yaml}"; DEV="${DEV:-cuda}"; SEEDS="${SEEDS:-20}"; M=results/manifests
stage() { local name="$1" man="$2"; shift 2; if [ -f "$man/run_manifest.json" ]; then echo "[skip] $name"; return; fi; echo "[start] $name $(date -u +%FT%TZ)"; "$@"; echo "[done]  $name $(date -u +%FT%TZ)"; }
mkdir -p results/tables results/raw
test -f results/tables/selections.parquet || { echo "frozen selections.parquet missing"; exit 1; }
stage download   "$M/download_data"          python scripts/download_data.py   --config "$CFG" --device "$DEV"
stage cache      "$M/cache_features"         python scripts/cache_features.py  --config "$CFG" --device "$DEV" --tokens-dtype float32
stage metrics    "$M/compute_metrics"        python scripts/compute_metrics.py --config "$CFG" --device "$DEV"
stage patch_test "$M/run_patching_E1_test"   python scripts/run_patching.py    --config "$CFG" --device "$DEV" --experiment E1 --split test --mask-seeds "$SEEDS"
stage evaluate   "$M/evaluate"               python scripts/evaluate.py        --config "$CFG" --device "$DEV"
echo "[all done] $(date -u +%FT%TZ)"; touch results/E1_TEST_DONE
