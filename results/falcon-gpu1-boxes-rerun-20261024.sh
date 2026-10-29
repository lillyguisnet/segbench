#!/usr/bin/env bash
# Falcon box timing, rerun on the whole of GPU 1 (2026-10-24). The first box run
# (run-finders-boxes-3-falcon) overlapped lighton-ocr-retry.timer restarting
# LightOnOCR on GPU 1 at 08:45:11; this time the timer is paused as well.
# ALWAYS restarts the services and the timer.
set -uo pipefail
cd /home/maxime/Projects/segbench

SYSTEM=(model-vllm-lighton-ocr parakeet-server chattering-semantic kokoro-tts)
USER_UNITS=(chattering-speaker myvoice-speech)
TIMER=lighton-ocr-retry.timer
GPU1=GPU-73bb8110-4d4b-f645-dce4-724b0baf767f
BASIS="whole GPU 1 (RTX 3090, same model as GPU 0, no display): its six programs stopped for the run (model-vllm-lighton-ocr, parakeet-server, chattering-semantic, kokoro-tts, chattering-speaker, myvoice-speech) and lighton-ocr-retry.timer paused, nothing else on it"

restart() {
  echo "== restarting services $(date -Is)"
  systemctl --user start "${USER_UNITS[@]}"
  sudo -n systemctl start --no-block "${SYSTEM[@]}"
  sudo -n systemctl start "$TIMER"
  echo "== restart requested $(date -Is)"
}
trap restart EXIT

echo "== pausing timer and stopping services $(date -Is)"
sudo -n systemctl stop "$TIMER"
sudo -n systemctl stop "${SYSTEM[@]}"
systemctl --user stop "${USER_UNITS[@]}"
for i in $(seq 30); do
  [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$GPU1")" -eq 0 ] && break
  sleep 2
done
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader | grep "$GPU1" \
  && { echo "GPU 1 not free: not running"; exit 1; }

# Watch GPU 1 every second during the run: any foreign process is written down.
( while true; do nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    | grep "$GPU1" | sed "s/^/$(date +%T) /"; sleep 1; done ) > results/falcon-gpu1-boxes-rerun-20261024.gpu-watch.log 2>&1 &
WATCH=$!

echo "== boxes, eleven photos, timed $(date -Is)"
uv run --with pillow scripts/time_finders_boxes.py --name finders-boxes-4-falcon --setups falcon-perception-hd --gpu 1 --basis "$BASIS"
status=$?
kill $WATCH
echo "== box run exit $status $(date -Is)"
exit $status
