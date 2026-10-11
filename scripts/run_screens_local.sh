#!/usr/bin/env bash
# Run the local models on the screenshots once GPU 0 is free, so their times
# are not slowed by someone else's work. Never stops another program: it waits.
#   nohup scripts/run_screens_local.sh screens-local-1 > results/run-screens-local-1.log 2>&1 &
# Free = no program other than the desktop's small ones (< 1 GB) holds GPU 0,
# and the GPU is under 10% busy, three checks in a row, 20 s apart.
set -euo pipefail
NAME=${1:?run name}
cd "$(dirname "$0")/.."
GPU=GPU-ed411651-10df-4a1c-981a-9cf0787065b1
free_now() {
  local big util
  big=$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1==g && $2>1024' | wc -l)
  util=$(nvidia-smi -i $GPU --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  [ "$big" -eq 0 ] && [ "$util" -lt 10 ]
}
echo "$(date -Is) waiting for GPU 0 to be free"
ok=0
while [ $ok -lt 3 ]; do
  if free_now; then ok=$((ok+1)); else ok=0; fi
  sleep 20
done
echo "$(date -Is) GPU 0 free: starting"
uv run --with pillow scripts/run_screens_local.py --name "$NAME"
(cd specialists/yoloe && uv run screens.py --name "$NAME-yoloe")
echo "$(date -Is) done"
