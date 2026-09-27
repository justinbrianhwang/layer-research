#!/usr/bin/env bash
# Run from the repository root on a provisioned CUDA machine.
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
CFG="${CFG:-configs/resnet50.yaml}"
DEV="${DEV:-cuda}"
python scripts/download_data.py --config "$CFG" --device "$DEV"
python scripts/cache_features.py --config "$CFG" --device "$DEV" --tokens-dtype float32
python scripts/compute_metrics.py --config "$CFG" --device "$DEV"
python scripts/run_patching.py --config "$CFG" --device "$DEV" --experiment E1 --split val --mask-seeds 10
python scripts/select_sites.py --config "$CFG" --device "$DEV"
python scripts/run_patching.py --config "$CFG" --device "$DEV" --experiment E1 --split test --mask-seeds 10
python scripts/evaluate.py --config "$CFG" --device "$DEV"
# One training seed means seed 0; all 16 sites supply the E4 reference.
python scripts/train_adapters.py --config "$CFG" --device "$DEV" --widths 32 --seeds 1
