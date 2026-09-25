#!/usr/bin/env bash
# Destroy the instance recorded in .vast_instance.json (or all instances with --all) and verify none remain.
set -euo pipefail
VAST="${VAST:-$HOME/AppData/Roaming/Python/Python314/Scripts/vastai.exe}"
if [ "${1:-}" = "--all" ]; then
  IDS=$("$VAST" show instances --raw | python -c "import sys,json; print(' '.join(str(i['id']) for i in json.load(sys.stdin)))")
else
  IDS=$(python -c "import json; print(json.load(open('.vast_instance.json'))['instance_id'])")
fi
for IID in $IDS; do
  echo "[vast] destroying instance $IID"; "$VAST" destroy instance -y "$IID" || true
done
sleep 5
LEFT=$("$VAST" show instances --raw | python -c "import sys,json; print(len(json.load(sys.stdin)))")
echo "[vast] instances remaining: $LEFT"
[ "$LEFT" = "0" ] && rm -f .vast_instance.json
