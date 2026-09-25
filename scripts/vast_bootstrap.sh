#!/usr/bin/env bash
# Bootstrap the repo on an already-running instance and (optionally) launch a driver script.
# Usage: scripts/vast_bootstrap.sh <ssh_host> <ssh_port> [driver-script e.g. scripts/gpu_run_E1.sh]
set -euo pipefail
HOST="$1"; PORT="$2"; DRIVER="${3:-}"
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ConnectTimeout=20 -p $PORT root@$HOST"
for i in $(seq 1 30); do $SSH true 2>/dev/null && break; echo "[vast] waiting for ssh..."; sleep 10; done
$SSH 'set -e; cd /workspace 2>/dev/null || cd ~; \
  if [ ! -d layer-research ]; then git clone -q https://github.com/justinbrianhwang/layer-research.git; fi; \
  cd layer-research && git pull -q && \
  pip install -q -e . "timm>=1.0" imagecorruptions scikit-image scikit-learn opencv-python-headless pandas pyarrow pyyaml tqdm "setuptools<81" 2>&1 | tail -2; \
  python -c "import torch, timm; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0), \"timm\", timm.__version__)"; \
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | head -2; df -h . | tail -1'
if [ -n "$DRIVER" ]; then
  $SSH "cd /workspace/layer-research 2>/dev/null || cd ~/layer-research; mkdir -p results; nohup bash $DRIVER > results/gpu_run.log 2>&1 & echo launched pid \$!"
fi
