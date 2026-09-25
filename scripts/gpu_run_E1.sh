#!/usr/bin/env bash
# Full E1 (+E3 unseen evaluation) pipeline on a CUDA box. Run from the repo root under nohup:
#   nohup bash scripts/gpu_run_E1.sh > results/gpu_run_E1.log 2>&1 &
# Idempotent stages: a stage is skipped when its manifest exists (delete the manifest to rerun).
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
CFG="${CFG:-configs/deit_small.yaml}"
DEV="${DEV:-cuda}"
SEEDS="${SEEDS:-20}"
M=results/manifests
stage() { # name, manifest-dir, command...
  local name="$1" man="$2"; shift 2
  if [ -f "$man/run_manifest.json" ]; then echo "[skip] $name (manifest exists)"; return; fi
  echo "[start] $name  $(date -u +%FT%TZ)"; "$@"; echo "[done]  $name  $(date -u +%FT%TZ)"
}
mkdir -p results
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
stage download   "$M/download_data"            python scripts/download_data.py   --config "$CFG" --device "$DEV"
stage cache      "$M/cache_features"           python scripts/cache_features.py  --config "$CFG" --device "$DEV" --tokens-dtype float32
stage precision  "$M/check_tokens_precision"   python scripts/check_tokens_precision.py --config "$CFG" --device "$DEV" --limit 64
stage metrics    "$M/compute_metrics"          python scripts/compute_metrics.py --config "$CFG" --device "$DEV"
stage patch_val  "$M/run_patching_E1_val"      python scripts/run_patching.py    --config "$CFG" --device "$DEV" --experiment E1 --split val  --mask-seeds "$SEEDS"
stage select     "$M/select_sites"             python scripts/select_sites.py    --config "$CFG" --device "$DEV"
stage patch_test "$M/run_patching_E1_test"     python scripts/run_patching.py    --config "$CFG" --device "$DEV" --experiment E1 --split test --mask-seeds "$SEEDS"
stage evaluate   "$M/evaluate"                 python scripts/evaluate.py        --config "$CFG" --device "$DEV"
# freeze E1 artefacts before any E2 run overwrites the active tables
cp -n results/summary.md results/E1_summary.md 2>/dev/null || true
cp -n results/tables/selections.parquet results/tables/E1_selections.parquet 2>/dev/null || true
cp -n results/tables/selections.csv     results/tables/E1_selections.csv     2>/dev/null || true
echo "[all done] $(date -u +%FT%TZ)"; touch results/E1_DONE
