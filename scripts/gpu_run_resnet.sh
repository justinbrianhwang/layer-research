#!/usr/bin/env bash
# ResNet-50 E1/E3/E4 on a 60 GB box: token caches are ~22 GB per split in fp16 (16 sites, early stages are
# 256x56x56), so splits are cached, used and deleted one at a time. Run from the repo root under nohup.
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
CFG="${CFG:-configs/resnet50.yaml}"; DEV="${DEV:-cuda}"; SEEDS="${SEEDS:-10}"
ROOT=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_root'])" "$CFG"); M="$ROOT/results/manifests"
stage() { local name="$1" man="$2"; shift 2; if [ -f "$man/run_manifest.json" ]; then echo "[skip] $name"; return; fi; echo "[start] $name $(date -u +%FT%TZ)"; "$@"; echo "[done]  $name $(date -u +%FT%TZ)"; }
drop_tokens() { rm -f "$ROOT"/cache/"$1"/clean/tokens_layer*.pt; echo "[disk] dropped $1 tokens; $(df -h . | tail -1)"; }
mkdir -p results "$ROOT/results/tables"
stage download    "$M/download_data"          python scripts/download_data.py   --config "$CFG" --device "$DEV"
# score split: summaries + channel stats need the clean score tokens during caching only
stage cache_score "$M/cache_features_score"   bash -c "python scripts/cache_features.py --config $CFG --device $DEV --tokens-dtype float16 --splits score && mkdir -p $M/cache_features_score && cp $M/cache_features/run_manifest.json $M/cache_features_score/"
stage metrics     "$M/compute_metrics"        python scripts/compute_metrics.py --config "$CFG" --device "$DEV"
drop_tokens score
stage cache_val   "$M/cache_features_val"     bash -c "python scripts/cache_features.py --config $CFG --device $DEV --tokens-dtype float16 --splits val && mkdir -p $M/cache_features_val && cp $M/cache_features/run_manifest.json $M/cache_features_val/"
stage precision   "$M/check_tokens_precision" python scripts/check_tokens_precision.py --config "$CFG" --device "$DEV" --limit 64 --split val
stage patch_val   "$M/run_patching_E1_val"    python scripts/run_patching.py    --config "$CFG" --device "$DEV" --experiment E1 --split val  --mask-seeds "$SEEDS"
stage select      "$M/select_sites"           python scripts/select_sites.py    --config "$CFG" --device "$DEV" --no-extra-metrics
drop_tokens val
stage cache_test  "$M/cache_features_test"    bash -c "python scripts/cache_features.py --config $CFG --device $DEV --tokens-dtype float16 --splits test && mkdir -p $M/cache_features_test && cp $M/cache_features/run_manifest.json $M/cache_features_test/"
stage patch_test  "$M/run_patching_E1_test"   python scripts/run_patching.py    --config "$CFG" --device "$DEV" --experiment E1 --split test --mask-seeds "$SEEDS"
stage evaluate    "$M/evaluate"               python scripts/evaluate.py        --config "$CFG" --device "$DEV"
drop_tokens test
stage adapters    "$M/train_adapters"         python scripts/train_adapters.py  --config "$CFG" --device "$DEV" --widths 32 --seeds 1
echo "[all done] $(date -u +%FT%TZ)"; touch "$ROOT/results/RESNET_DONE"
