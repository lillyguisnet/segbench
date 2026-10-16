"""Live check of an lm15 upgrade: are replies read into the same records? (2026-10-16, 1.2.1 -> 1.5.0)

    uv run --no-project --with 'lm15==X' --with pillow scripts/check_lm15_replies.py OUT.jsonl

scripts/capture_wire.py showed both versions send byte-identical requests.
This sends one real call per kind of service (one model each, lowest
thinking level, the synthetic oval picture of scripts/check_boxes.py at
1015x1015, the box-a prompt for the beetles task's words) so the two
versions' records can be compared field by field: text present, usage
fields, cost, finish reason, the boxes read. Answers differ between any two
calls of the same model; the record's shape and the reading must not.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm15  # noqa: E402
from check_boxes import picture  # noqa: E402
from run import image_settings  # noqa: E402
from segbench.box_read import read_box_a  # noqa: E402
from segbench.call import call  # noqa: E402
from segbench.models import BY_KEY, levels_for  # noqa: E402
from segbench.tasks import BY_KEY as TASKS, prompt_box  # noqa: E402

# One model per service (the Muse plan is skipped: its quota is used up until 2026-10-19).
ONE_PER_SERVICE = ["luna", "haiku", "gemini-flash-lite", "glm-5.3-flash", "deepseek-flash", "kimi-k3",
                   "qwen-flash", "muse-wonder-api"]


def one(key):
    model = BY_KEY[key]
    img, _ = picture(1015, 1015)
    rec = call(model, img, "image/jpeg", prompt_box("box-a", TASKS["beetles"]), level=levels_for(key)[0],
               **image_settings(model, "box-a"))
    rec.pop("traceback", None)
    try:
        rec["boxes_read"] = len(read_box_a(rec.get("text") or "") or [])
    except Exception as e:  # noqa: BLE001
        rec["boxes_read"] = f"{type(e).__name__}: {e}"
    rec["lm15"] = lm15.__version__
    return rec


def main():
    out = Path(sys.argv[1])
    with ThreadPoolExecutor(8) as ex:
        recs = list(ex.map(one, ONE_PER_SERVICE))
    out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    for r in recs:
        print(f"{r['model']:18s} err={str(r.get('error'))[:60]!s:60s} boxes={r.get('boxes_read')} "
              f"usage={r.get('usage')} cost={r.get('cost_usd')} finish={r.get('finish_reason')}")


if __name__ == "__main__":
    main()
