#!/usr/bin/env bash
# Provision one instance per job and launch its driver. Jobs are "name|driver command" arguments.
# Bad machines (no internet, stuck loading) are excluded automatically; EXCLUDE_MACHINES seeds the list.
#   bash scripts/vast_launch_many.sh "e2_alpha050|CFG=configs/e2_alpha050.yaml bash scripts/gpu_run_E2.sh" ...
set -u
EX="${EXCLUDE_MACHINES:-}"
for JOB in "$@"; do
  NAME="${JOB%%|*}"; DRV="${JOB#*|}"
  ok=0
  for attempt in 1 2 3 4; do
    OUT=$(EXCLUDE_MACHINES="$EX" bash scripts/vast_provision.sh 0.25 2>&1 | grep -vE "^\s*$|WARNING: Running pip|Welcome to vast|Have fun|Skipping opencv")
    if echo "$OUT" | grep -q "EXCLUDE_MACHINE="; then M=$(echo "$OUT" | grep -oE "EXCLUDE_MACHINE=[0-9]+" | tail -1 | cut -d= -f2); EX="${EX:+$EX,}$M"; echo "[$NAME] excluded machine $M (attempt $attempt)"; continue; fi
    if [ -f .vast_instance.json ] && echo "$OUT" | grep -q "\[vast\] ready"; then
      mv .vast_instance.json ".vast_instance_$NAME.json"
      read -r IID MID HOST PORT < <(python -c "import json;d=json.load(open('.vast_instance_$NAME.json'));u=d['ssh_url'];print(d['instance_id'],d.get('machine_id'),u.split('@')[1].split(':')[0],u.split(':')[-1])")
      EX="${EX:+$EX,}$MID"
      echo "[$NAME] instance $IID (machine $MID) at $HOST:$PORT"
      bash scripts/vast_bootstrap.sh "$HOST" "$PORT" "$DRV" 2>&1 | grep -E "launched|cuda|error|fatal" | head -3
      ok=1; break
    fi
    echo "[$NAME] attempt $attempt failed: $(echo "$OUT" | tail -2 | tr '\n' ' ')"
  done
  [ $ok = 1 ] || echo "[$NAME] GAVE UP"
done
echo "=== done: $(ls .vast_instance_*.json 2>/dev/null | tr '\n' ' ')"
