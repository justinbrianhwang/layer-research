#!/usr/bin/env bash
# E6 driver (DeiT-S): rebuild caches, compute extended metrics on the score split, rerun the E1 validation
# sweep (alpha=1, 10 mask seeds) so selectors incl. the extended metrics can be frozen, then wait for the
# E1 test raw table (uploaded by the PM to results/raw/patching_test.parquet) and evaluate.
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
CFG="${CFG:-configs/deit_small.yaml}"; DEV="${DEV:-cuda}"; SEEDS="${SEEDS:-10}"; M=results/manifests
stage() { local name="$1" man="$2"; shift 2; if [ -f "$man/run_manifest.json" ]; then echo "[skip] $name"; return; fi; echo "[start] $name $(date -u +%FT%TZ)"; "$@"; echo "[done]  $name $(date -u +%FT%TZ)"; }
mkdir -p results/tables results/raw
pip install -q ripser persim 2>&1 | tail -1 || true
stage download   "$M/download_data"            python scripts/download_data.py   --config "$CFG" --device "$DEV"
stage cache      "$M/cache_features"           python scripts/cache_features.py  --config "$CFG" --device "$DEV" --tokens-dtype float32
stage metrics    "$M/compute_metrics"          python scripts/compute_metrics.py --config "$CFG" --device "$DEV"
stage tasksens   "$M/compute_task_sensitivity" python scripts/compute_task_sensitivity.py --config "$CFG" --device "$DEV" --split score
stage topology   "$M/compute_topology"         python scripts/compute_topology.py --config "$CFG" --device "$DEV" --split score
stage patch_val  "$M/run_patching_E1_val"      python scripts/run_patching.py    --config "$CFG" --device "$DEV" --experiment E1 --split val --mask-seeds "$SEEDS"
stage select_ext "$M/select_sites"             python scripts/select_sites.py    --config "$CFG" --device "$DEV" --extra-metrics
touch results/E6_SELECT_DONE
echo "[wait] for results/raw/patching_test.parquet (uploaded by the PM)"
until [ -f results/raw/patching_test.parquet ] && [ -f results/raw/UPLOAD_DONE ]; do sleep 60; done
stage evaluate_ext "$M/evaluate_extended"      python scripts/evaluate.py        --config "$CFG" --device "$DEV" --selections results/tables/selections_extended.parquet --output-suffix _extended
echo "[all done] $(date -u +%FT%TZ)"; touch results/E6_DONE
