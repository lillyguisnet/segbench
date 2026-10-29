# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow"]
# ///
"""Build and serve the click app for the measurement answer key (apps/measure-clicks).

uv run scripts/serve_measure_clicks.py [--port 8840] [--build-only]

Builds apps/measure-clicks/seed.js and images/ from segbench/measure_tasks.py
(both ignored by Git, like the other review apps), then serves the app on
lambda's tailnet address and on localhost. The page saves every change here,
in annotations/measure/clicks-v1.json (a copy at most every 2 minutes in
annotations/measure/history/), as well as in the browser. Opened from
anywhere else (for example as a file), it still works and saves in the
browser; Export then keeps the work.

A save carries the revision it was based on; a save from an older revision
(a second window) is refused rather than overwriting the newer one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.measure_tasks import PROMPT, QUESTIONS  # noqa: E402

APP = ROOT / "apps/measure-clicks"
OUT = ROOT / "annotations/measure"
STATE = OUT / "clicks-v1.json"
HISTORY = OUT / "history"
TAILNET_IP = "100.86.49.54"  # lambda


def build() -> dict:
    (APP / "images").mkdir(parents=True, exist_ok=True)
    questions = [asdict(q) for q in QUESTIONS]
    photos = {}
    for q in QUESTIONS:
        if q.photo in photos:
            continue
        src = ROOT / "images" / q.photo
        with Image.open(src) as im:
            if im.getexif().get(274) not in (None, 1):
                raise SystemExit(f"{q.photo}: EXIF rotation tag; photos must be stored upright")
            w, h = im.size
        shutil.copyfile(src, APP / "images" / q.photo)
        photos[q.photo] = {"src": f"images/{q.photo}", "width": w, "height": h,
                           "sha256": hashlib.sha256(src.read_bytes()).hexdigest()}
    sha = hashlib.sha256(json.dumps([questions, photos], sort_keys=True).encode()).hexdigest()
    seed = {"schema": 1, "questions_sha256": sha, "prompt": PROMPT, "questions": questions, "photos": photos}
    (APP / "seed.js").write_text("window.MEASURE_SEED = " + json.dumps(seed, separators=(",", ":")) + ";\n")
    print(f"Built apps/measure-clicks: {len(questions)} questions on {len(photos)} photos; questions {sha[:12]}")
    return seed


def check(data: dict, seed: dict) -> None:
    """Refuse anything that is not a click file for these questions."""
    if data.get("schema") != "segbench-measure-clicks/v1" or data.get("questions_sha256") != seed["questions_sha256"]:
        raise ValueError("not a click file for these questions")
    by_key = {q["key"]: q for q in seed["questions"]}
    for item in data.get("items", []):
        q = by_key.get(item.get("q"))
        if q is None:
            raise ValueError(f"unknown question {item.get('q')!r}")
        photo = seed["photos"][q["photo"]]
        pts = item.get("points")
        if not isinstance(pts, list) or len(pts) < 2 or (q["kind"] == "span" and len(pts) != 2):
            raise ValueError(f"{item.get('id')}: wrong number of points")
        for p in pts + [item.get("dot")]:
            if not (isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p)
                    and 0 <= p[0] <= photo["width"] and 0 <= p[1] <= photo["height"]):
                raise ValueError(f"{item.get('id')}: point outside the photo")


def serve(seed: dict, port: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    HISTORY.mkdir(exist_ok=True)
    lock = threading.Lock()
    last_copy = [0.0]

    def current() -> dict:
        return json.loads(STATE.read_text()) if STATE.exists() else {"revision": 0, "data": None}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(APP), **k)

        def log_message(self, fmt, *args):  # quiet: only saves are printed
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == "/api/state":
                with lock:
                    return self._json(200, current())
            return super().do_GET()

        def do_POST(self):
            if self.path != "/api/save":
                return self._json(404, {"error": "no such endpoint"})
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                check(body["data"], seed)
            except Exception as error:  # noqa: BLE001 - any bad request is refused, with the reason
                return self._json(400, {"error": str(error)})
            with lock:
                cur = current()
                if body.get("base_revision") != cur["revision"]:
                    return self._json(409, {"error": "saved meanwhile from another window", "revision": cur["revision"]})
                new = {"revision": cur["revision"] + 1,
                       "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "data": body["data"]}
                tmp = STATE.with_suffix(".tmp")
                tmp.write_text(json.dumps(new, indent=1))
                tmp.replace(STATE)
                if time.time() - last_copy[0] > 120:
                    shutil.copyfile(STATE, HISTORY / f"clicks-v1-{datetime.now():%Y%m%d-%H%M%S}.json")
                    last_copy[0] = time.time()
            n = len(body["data"].get("items", []))
            print(f"{datetime.now():%H:%M:%S} saved revision {new['revision']}: {n} items", flush=True)
            return self._json(200, {"revision": new["revision"]})

    servers = []
    for host in (TAILNET_IP, "127.0.0.1"):
        try:
            servers.append(ThreadingHTTPServer((host, port), Handler))
        except OSError as error:
            print(f"cannot listen on {host}:{port}: {error}")
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print(f"Open http://192.168.2.24/measure/ (home network, through the portal) or http://lambda.tail69222b.ts.net/measure/", flush=True)
    servers[0].serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8840)
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    seed = build()
    if not args.build_only:
        serve(seed, args.port)


if __name__ == "__main__":
    main()
