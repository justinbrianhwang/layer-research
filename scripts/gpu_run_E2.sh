#!/usr/bin/env bash
# E2 budget variant on a fresh box: CFG must be one of configs/e2_*.yaml (carries output_root and the _e2 block).
#   CFG=configs/e2_alpha025.yaml nohup bash scripts/gpu_run_E2.sh > results/gpu_run_E2.log 2>&1 &
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
CFG="${CFG:?set CFG=configs/e2_*.yaml}"; DEV="${DEV:-cuda}"
ROOT=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_root'])" "$CFG")
ALPHAS=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['_e2']['alphas'])" "$CFG")
CAP=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['_e2']['norm_cap'])" "$CFG")
SEEDS=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['_e2']['mask_seeds'])" "$CFG")
M="$ROOT/results/manifests"
stage() { local name="$1" man="$2"; shift 2; if [ -f "$man/run_manifest.json" ]; then echo "[skip] $name"; return; fi; echo "[start] $name $(date -u +%FT%TZ)"; "$@"; echo "[done]  $name $(date -u +%FT%TZ)"; }
mkdir -p "$ROOT/results/tables" results
stage download   "$M/download_data"          python scripts/download_data.py   --config "$CFG" --device "$DEV"
stage cache      "$M/cache_features"         python scripts/cache_features.py  --config "$CFG" --device "$DEV" --tokens-dtype float32
stage metrics    "$M/compute_metrics"        python scripts/compute_metrics.py --config "$CFG" --device "$DEV"
stage patch_val  "$M/run_patching_E2_val"    python scripts/run_patching.py    --config "$CFG" --device "$DEV" --experiment E2 --split val  --mask-seeds "$SEEDS" --alphas "$ALPHAS" --norm-cap "$CAP"
stage select     "$M/select_sites"           python scripts/select_sites.py    --config "$CFG" --device "$DEV"
stage patch_test "$M/run_patching_E2_test"   python scripts/run_patching.py    --config "$CFG" --device "$DEV" --experiment E2 --split test --mask-seeds "$SEEDS" --alphas "$ALPHAS" --norm-cap "$CAP"
stage evaluate   "$M/evaluate"               python scripts/evaluate.py        --config "$CFG" --device "$DEV"
echo "[all done] $(date -u +%FT%TZ)"; touch "$ROOT/results/E2_DONE"
