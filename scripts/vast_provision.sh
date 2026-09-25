#!/usr/bin/env bash
# Rent the cheapest reliable single 24 GB GPU on Vast.ai, wait for SSH, bootstrap the repo.
# Usage: scripts/vast_provision.sh [max_dph]      (default max price 0.25 $/h)
# Writes .vast_instance.json (gitignored) with the instance id and ssh target.
set -euo pipefail
VAST="${VAST:-$HOME/AppData/Roaming/Python/Python314/Scripts/vastai.exe}"
MAX_DPH="${1:-0.25}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime}"
DISK="${DISK:-60}"
QUERY="gpu_name in [RTX_3090,RTX_4080S,RTX_4090] num_gpus=1 gpu_ram>=20 reliability>0.98 inet_down>200 disk_space>=${DISK} cpu_cores_effective>=8 cuda_vers>=12.4 rentable=true dph<=${MAX_DPH}"

echo "[vast] searching offers: $QUERY"
# cpu_ram is filtered here (the query-language cpu_ram term matches nothing); MB units.
OFFER=$("$VAST" search offers "$QUERY" -o dph --raw | python -c "import sys,json; ex=set(int(x) for x in '${EXCLUDE_MACHINES:-}'.split(',') if x); d=[o for o in json.load(sys.stdin) if o['cpu_ram']>=30000 and o['machine_id'] not in ex]; print(d[0]['id'] if d else '')")
[ -n "$OFFER" ] || { echo "no offer under \$$MAX_DPH/h"; exit 1; }
echo "[vast] creating instance from offer $OFFER"
CREATE=$("$VAST" create instance "$OFFER" --image "$IMAGE" --disk "$DISK" --ssh --direct --raw)
echo "$CREATE"
IID=$(echo "$CREATE" | python -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
echo "{\"instance_id\": $IID, \"offer_id\": $OFFER, \"created\": \"$(date -u +%FT%TZ)\"}" > .vast_instance.json

echo "[vast] waiting for instance $IID to be running..."
for i in $(seq 1 60); do
  STATE=$("$VAST" show instance "$IID" --raw | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('actual_status',''))")
  [ "$STATE" = "running" ] && break
  sleep 10
done
[ "$STATE" = "running" ] || { echo "instance not running after 10 min (state=$STATE)"; exit 1; }
SSH_URL=$("$VAST" ssh-url "$IID")
echo "[vast] ssh: $SSH_URL"
python - "$IID" "$SSH_URL" <<'EOF'
import json, sys
d = json.load(open(".vast_instance.json")); d["ssh_url"] = sys.argv[2]
json.dump(d, open(".vast_instance.json", "w"), indent=2)
EOF

HOST=$(echo "$SSH_URL" | sed -E 's#ssh://([^:]+):([0-9]+)#\1#'); PORT=$(echo "$SSH_URL" | sed -E 's#ssh://([^:]+):([0-9]+)#\2#')
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT $HOST"
for i in $(seq 1 30); do $SSH true 2>/dev/null && break; sleep 10; done

echo "[vast] bootstrapping repo on instance"
$SSH 'set -e; cd /workspace 2>/dev/null || cd ~; \
  if [ ! -d layer-research ]; then git clone -q https://github.com/justinbrianhwang/layer-research.git; fi; \
  cd layer-research && git pull -q && \
  pip install -q -e . "timm>=1.0" imagecorruptions scikit-image scikit-learn opencv-python-headless pandas pyarrow pyyaml tqdm "setuptools<81" && \
  python -c "import torch;print(\"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0))" && \
  nvidia-smi --query-gpu=name,memory.total --format=csv'
echo "[vast] ready. instance $IID  ($SSH)"
