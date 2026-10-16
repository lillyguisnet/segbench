"""A feedback loop on Jev + halving's final boxes (Maxime, 2026-10-16).

For each box the halving returns:

1. Check: the box, with a margin of 25% of its size on each side, is cut from
   the full photo, fitted to Jev's limits, and Jev is asked yes / no: "Is there
   a <target> in the middle of the image (whole, or most of it)?". Yes (p >= 0.5)
   -> keep the box.
2. No -> find again. Jev gives (almost) the same box to the same request
   (checked 2026-10-16: three identical requests, boxes within 0.03), so the
   second try changes the picture: the piece of the halving where the box's
   middle lies (the smallest one) is sent again with the rejected box painted
   grey, with the halving's own two questions (how many, and the box).
   "none" -> reject. Otherwise the new box gets the same check: yes -> keep it,
   no -> reject.
3. A box kept at step 2 whose middle lies inside a box already kept is dropped
   (the same object found twice).

All boxes are checked at the same time. Written after looking at the
butterflies-and-beetles photo (a benchmark photo), so its results there are
exploratory, not a fair benchmark score.
"""
from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor

from lm15 import judgments, yes_no
from PIL import ImageDraw

from .call import call
from .jev_halving import ATTEMPTS, MODEL, fit, unpad

MARGIN = 0.25
MIN_MARGIN = 6  # pixels, so a tiny box keeps some surroundings
GREY = (128, 128, 128)
VERSION = "jev-verify-v1"


def check_question(target: str, note: str = ""):
    thing = target[5:] if target.startswith("each ") else target
    extra = f" {note}" if note else ""
    return judgments(ok=yes_no(f"Is there a {thing} in the middle of the image (whole, or most of it)?{extra}"))


class Verifier:
    def __init__(self, search, target: str, note: str = "", workers: int = 16):
        self.s = search
        self.cq = check_question(target, note)
        self.workers = workers
        self.log: list = []

    def _send(self, piece, q) -> dict:
        b = io.BytesIO()
        piece.save(b, "JPEG", quality=95)
        rec = None
        for attempt in range(ATTEMPTS):
            rec = call(MODEL, b.getvalue(), "image/jpeg", None, level="medium", response_format=q)
            rec.update(sent_size=list(piece.size), attempt=attempt, stage="verify")
            rec.pop("traceback", None)
            self.s.records.append(rec)
            if not rec.get("error") or not rec.get("retryable"):
                break
            time.sleep(min(30, 2 ** attempt))
        if rec.get("error"):
            raise RuntimeError(f"Jev call failed on a {piece.size} piece: {rec['error']}")
        return rec

    def check(self, box) -> float:
        w, h = self.s.im.size
        mx = max(MIN_MARGIN, MARGIN * (box[2] - box[0]))
        my = max(MIN_MARGIN, MARGIN * (box[3] - box[1]))
        region = (max(0, int(box[0] - mx)), max(0, int(box[1] - my)),
                  min(w, int(box[2] + mx) + 1), min(h, int(box[3] + my) + 1))
        piece, _ = fit(self.s.im.crop(region))
        rec = self._send(piece, self.cq)
        return (rec.get("probabilities") or {}).get("ok", {}).get("true", 1.0 if rec["data"]["ok"] else 0.0)

    def piece_of(self, box):
        """The smallest halving piece that gave a part and holds the box's middle; else the box, 3x as wide and tall."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inside = [p["region"] for p in self.s.raw
                  if p["region"][0] <= cx < p["region"][2] and p["region"][1] <= cy < p["region"][3]]
        if inside:
            return tuple(min(inside, key=lambda r: (r[2] - r[0]) * (r[3] - r[1])))
        w, h = self.s.im.size
        bw, bh = box[2] - box[0], box[3] - box[1]
        return (max(0, int(box[0] - bw)), max(0, int(box[1] - bh)), min(w, int(box[2] + bw) + 1), min(h, int(box[3] + bh) + 1))

    def again(self, box, region):
        """The piece with the rejected box painted grey, asked the halving's questions. -> (count, new box or None)"""
        x0, y0, x1, y1 = region
        crop = self.s.im.crop(region).copy()
        ImageDraw.Draw(crop).rectangle((box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0), fill=GREY)
        piece, pad = fit(crop)
        rec = self._send(piece, self.s.q)
        count = rec["data"]["count"]
        if count == "none":
            return count, None
        b = unpad(rec["data"]["box"], piece, pad)
        w, h = x1 - x0, y1 - y0
        return count, [x0 + b["left"] * w, y0 + b["top"] * h, x0 + b["right"] * w, y0 + b["bottom"] * h]

    def one(self, box):
        entry = {"box": list(box)}
        entry["p"] = p = self.check(box)
        if p >= 0.5:
            entry["outcome"] = "kept"
            return entry
        region = self.piece_of(box)
        entry["retry_region"] = list(region)
        entry["retry_count"], nb = self.again(box, region)
        if nb is None:
            entry["outcome"] = "rejected (none left)"
            return entry
        entry["retry_box"] = nb
        entry["retry_p"] = p2 = self.check(nb)
        entry["outcome"] = "replaced" if p2 >= 0.5 else "rejected"
        return entry

    def run(self, boxes) -> list:
        with ThreadPoolExecutor(self.workers) as pool:
            self.log = list(pool.map(self.one, boxes))
        kept = [e["box"] for e in self.log if e["outcome"] == "kept"]
        for e in self.log:
            if e["outcome"] != "replaced":
                continue
            b = e["retry_box"]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if any(k[0] <= cx <= k[2] and k[1] <= cy <= k[3] for k in kept):
                e["outcome"] = "replaced, duplicate dropped"
                continue
            kept.append(b)
        return kept
