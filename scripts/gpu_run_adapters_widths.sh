#!/usr/bin/env bash
# Adapter width extension (8 and 64, one seed) into a separate output root.
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
python - <<PY
import yaml
c = yaml.safe_load(open("configs/deit_small.yaml")); c["output_root"] = "outputs_adapters_w8_64"
yaml.safe_dump(c, open("configs/deit_small_w8_64.yaml", "w"), sort_keys=False)
PY
mkdir -p results
python scripts/download_data.py --config configs/deit_small_w8_64.yaml --device cuda
python scripts/train_adapters.py --config configs/deit_small_w8_64.yaml --device cuda --widths 8,64 --seeds 1
echo "[all done] $(date -u +%FT%TZ)"; touch outputs_adapters_w8_64/ADAPTERS_DONE
