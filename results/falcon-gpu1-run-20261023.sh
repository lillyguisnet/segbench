#!/usr/bin/env bash
# Falcon Perception-HD photo benchmarks on the whole of GPU 1 (user's go-ahead, 2026-10-23).
# Stops GPU 1's six services, runs the three photo runs, and ALWAYS restarts the services.
set -uo pipefail
cd /home/maxime/Projects/segbench

SYSTEM=(model-vllm-lighton-ocr parakeet-server chattering-semantic kokoro-tts)
USER_UNITS=(chattering-speaker myvoice-speech)
GPU1=GPU-73bb8110-4d4b-f645-dce4-724b0baf767f
BASIS="whole GPU 1 (RTX 3090, same model as GPU 0, no display): its six programs stopped for the run (model-vllm-lighton-ocr, parakeet-server, chattering-semantic, kokoro-tts, chattering-speaker, myvoice-speech), nothing else on it"

restart() {
  echo "== restarting services $(date -Is)"
  sudo -n systemctl start "${SYSTEM[@]}"
  systemctl --user start "${USER_UNITS[@]}"
  sleep 5
  for s in "${SYSTEM[@]}"; do echo "system $s: $(systemctl is-active "$s")"; done
  for s in "${USER_UNITS[@]}"; do echo "user $s: $(systemctl --user is-active "$s")"; done
  echo "== done $(date -Is)"
}
trap restart EXIT

echo "== stopping services $(date -Is)"
sudo -n systemctl stop "${SYSTEM[@]}"
systemctl --user stop "${USER_UNITS[@]}"
for i in $(seq 30); do
  left=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$GPU1")
  [ "$left" -eq 0 ] && break
  sleep 2
done
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep "$GPU1" \
  && { echo "GPU 1 not free: not running"; exit 1; }
nvidia-smi --id=1 --query-gpu=memory.used,utilization.gpu --format=csv

echo "== 1/3 points, first four photos, timed $(date -Is)"
uv run --with pillow scripts/time_finders.py --name finders-6 --setups falcon-perception-hd --gpu 1 --basis "$BASIS" || exit 1

echo "== 2/3 points, second batch $(date -Is)"
uv run --with pillow scripts/run_finders_batch2.py --name finders-b2-falcon --gpu 1 --models falcon-perception-hd || exit 1

echo "== 3/3 boxes, eleven photos, timed $(date -Is)"
uv run --with pillow scripts/time_finders_boxes.py --name finders-boxes-3-falcon --setups falcon-perception-hd --gpu 1 --basis "$BASIS" || exit 1

echo "== all three runs finished $(date -Is)"
