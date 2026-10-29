"""Jev drives across San Francisco, one map at a time (Maxime, 2026-10-16).

From TypeSafe AI (255 California St) to UCSF Parnassus (505 Parnassus Ave).
Each step Jev gets one street map centred on the car and boxes the next
stretch of street to drive; the box is turned into a move on the real street
network, the car moves to the end of that stretch, and a new map is drawn
around it.

    set -a; source .env; set +a
    uv run --with osmnx==2.1.1 --with pillow scripts/jev_drive.py --tries 3 --name run1
    uv run --with osmnx==2.1.1 --with pillow scripts/jev_drive.py --tries 3 --name run2-clean \
        --map clean --one-way off --dest none

Settings: run1 = --map osm --one-way on --dest arrow (the defaults below).
run2-clean (Maxime, 2026-10-16, after run1): the clean map of scripts/clean_map.py
(streets, names, parks, water only), one-way rules ignored (every move is legal, the
question does not mention arrows), and no destination mark at all: the destination
is only named in the question.

Rules (set with Maxime before the first run, 2026-10-16):
- Map (run1): OpenStreetMap's standard map (tile.openstreetmap.org), zoom 17,
  1024 x 1024 px (about 960 m across), north up. Red dot = the car. Blue dot =
  the destination when it is on the map; otherwise a blue arrow at the edge
  points toward it (Jev cannot know where Parnassus Avenue is otherwise).
- Question: one Jev box (lm15 bounding_box); the words also say which street
  the car is on and that one-way arrows must be followed.
- Reading the box: every move the car could make from where it is (keep going
  on its street, or go along it to an intersection and turn onto another
  street, either way, up to any intersection on the map) is drawn as the
  rectangle around the stretch of the new street, widened by PAD px for the
  street's width. The move whose rectangle overlaps Jev's box best
  (intersection over union) is the one taken. Below MATCH_MIN no move fits:
  "not a road". When a legal and a forbidden move fit about equally well
  (within TIE), the legal one is taken (the two sides of a divided road).
- Wrong moves: driving against a one-way street (counted, and the car still
  drives it), or a box that fits no street the car can reach (counted, Jev is
  asked again, up to NO_ROAD_RETRIES times in a row, then the run stops: lost).
  Moving away from the destination is not wrong. Turn bans ("no left turn"
  signs) are not checked: the network has one-way rules only.
- End: arrived when the car's path passes within ARRIVE_M of the destination
  (snapped onto Parnassus Avenue); or MAX_STEPS maps; or lost.

Data: map pictures and street network (c) OpenStreetMap contributors (ODbL).
Street network downloaded once through Overpass (a mirror; the main server was
full) to ~/.cache/segbench-maps/sf-drive.graphml, 2026-10-16.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import osmnx as ox
from lm15 import bounding_box, judgments
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segbench.call import call  # noqa: E402
from segbench.models import BY_KEY  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import clean_map  # noqa: E402

OPT = {"map": "osm", "one_way": True, "dest": "arrow", "painter": None}

MODEL = BY_KEY["jev"]
CACHE = Path.home() / ".cache" / "segbench-maps"
GRAPH = CACHE / "sf-drive.graphml"
UA = {"User-Agent": "segbench-research/0.1 (github.com/lillyrivest/segbench)"}

# Nominatim, 2026-10-16
START = (37.7929589, -122.3995978, "California Street")   # 255 California St
DEST = (37.7630700, -122.4574027, "Parnassus Avenue")     # 505 Parnassus Ave
DEST_WORDS = "UCSF Parnassus Campus, 505 Parnassus Avenue"
AREAS = CACHE / "sf-areas.gpkg"

Z, SIZE = 17, 1024
PAD = 8            # px added around a street's stretch (half a street's width at zoom 17)
MATCH_MIN = 0.25   # below this overlap no move fits the box
MIN_PX = 15        # a move must drive at least this far (px) inside the box
TIE = 0.03
ARRIVE_M = 60
MAX_STEPS = 40
NO_ROAD_RETRIES = 2
EDGE_MARGIN = 6    # px: a stretch may end only at an intersection this far inside the map


# --- map pictures -------------------------------------------------------------

def world_px(lat, lon, z=Z):
    n = 256 * 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def tile(z, x, y) -> Image.Image:
    n = 2 ** z
    x %= n  # the world repeats left and right
    if not 0 <= y < n:  # above or below the map's world: blank
        return Image.new("RGB", (256, 256), (170, 211, 223))
    f = CACHE / "tiles" / str(z) / str(x) / f"{y}.png"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png", headers=UA)
        f.write_bytes(urllib.request.urlopen(req, timeout=30).read())
        time.sleep(0.05)
    return Image.open(f).convert("RGB")


class View:
    """A SIZE x SIZE map centred on (lat, lon) at zoom z."""

    def __init__(self, lat, lon, z=Z, w=SIZE, h=SIZE):
        cx, cy = world_px(lat, lon, z)
        self.z, self.w, self.h = z, w, h
        self.x0, self.y0 = cx - w / 2, cy - h / 2

    def px(self, lat, lon):
        x, y = world_px(lat, lon, self.z)
        return x - self.x0, y - self.y0

    def inside(self, lat, lon, margin=0):
        x, y = self.px(lat, lon)
        return margin <= x <= self.w - margin and margin <= y <= self.h - margin

    def picture(self) -> Image.Image:
        if OPT["map"] == "clean" and self.z == Z:  # the overview (zoom 15) stays on the standard map
            return OPT["painter"].draw(self)
        img = Image.new("RGB", (self.w, self.h))
        for tx in range(int(self.x0 // 256), int((self.x0 + self.w) // 256) + 1):
            for ty in range(int(self.y0 // 256), int((self.y0 + self.h) // 256) + 1):
                img.paste(tile(self.z, tx, ty), (round(tx * 256 - self.x0), round(ty * 256 - self.y0)))
        return img


def dot(d, x, y, r, fill):
    d.ellipse((x - r - 3, y - r - 3, x + r + 3, y + r + 3), fill="white")
    d.ellipse((x - r, y - r, x + r, y + r), fill=fill)


BLUE, RED = (30, 90, 230), (220, 30, 30)


def map_for_jev(view: View, dest) -> Image.Image:
    img = view.picture()
    d = ImageDraw.Draw(img)
    if OPT["dest"] == "none":
        pass
    elif view.inside(dest[0], dest[1], 14):
        dot(d, *view.px(dest[0], dest[1]), 11, BLUE)
    else:  # arrow at the edge, on the line from the car toward the destination
        cx, cy = view.w / 2, view.h / 2
        tx, ty = view.px(dest[0], dest[1])
        a = math.atan2(ty - cy, tx - cx)
        ux, uy = math.cos(a), math.sin(a)
        k = min((view.w / 2 - 40) / abs(ux) if ux else 1e9, (view.h / 2 - 40) / abs(uy) if uy else 1e9)
        tip = (cx + ux * k, cy + uy * k)
        L, W = 46, 22
        base = (tip[0] - ux * L, tip[1] - uy * L)
        pts = [tip, (base[0] - uy * W, base[1] + ux * W), (base[0] + uy * W, base[1] - ux * W)]
        d.polygon(pts, fill=BLUE, outline="white", width=4)
    dot(d, view.w / 2, view.h / 2, 11, RED)
    return img


# --- street network -------------------------------------------------------------

def names(e) -> frozenset:
    n = e.get("name") or e.get("ref")
    if n:
        return frozenset([n] if isinstance(n, str) else n)
    hw = e.get("highway")
    hw = hw[0] if isinstance(hw, list) else hw
    return frozenset([f"(unnamed {hw})"])


def words(ns) -> str:
    real = sorted(n for n in ns if not n.startswith("("))
    if real:
        return " / ".join(real)
    return "a ramp" if any("link" in n for n in ns) else "an unnamed road"


def coords(G, u, v, k):
    e = G.edges[u, v, k]
    if "geometry" in e:
        return [(lat, lon) for lon, lat in e["geometry"].coords]
    return [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]


def meters(a, b):
    k = math.cos(math.radians(a[0]))
    return math.hypot((a[0] - b[0]) * 111_320, (a[1] - b[1]) * 111_320 * k)


def seg_dist(p, a, b):
    """Metres from point p to segment ab (all lat, lon; flat at this scale)."""
    k = 111_320 * math.cos(math.radians(p[0]))
    P, A, B = ((q[1] * k, q[0] * 111_320) for q in (p, a, b))
    dx, dy = B[0] - A[0], B[1] - A[1]
    t = 0 if dx == dy == 0 else max(0, min(1, ((P[0] - A[0]) * dx + (P[1] - A[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(P[0] - A[0] - t * dx, P[1] - A[1] - t * dy)


def nearest_on(G, point, street):
    """The edge of `street` nearest to point, and the nearest spot on it."""
    best = None
    for u, v, k, e in G.edges(keys=True, data=True):
        if street not in names(e):
            continue
        c = coords(G, u, v, k)
        for i in range(len(c) - 1):
            d = seg_dist(point, c[i], c[i + 1])
            if best is None or d < best[0]:
                best = (d, u, v, k, i)
    d, u, v, k, i = best
    c = coords(G, u, v, k)
    a, b = c[i], c[i + 1]
    kx = math.cos(math.radians(point[0]))
    dx, dy = (b[1] - a[1]) * kx, b[0] - a[0]
    t = max(0.0, min(1.0, (((point[1] - a[1]) * kx) * dx + (point[0] - a[0]) * dy) / (dx * dx + dy * dy)))
    spot = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    return (u, v, k, i), spot, d


def split_at(G, u, v, spot, new):
    """Cut every edge between u and v (either direction) at `spot`: a new node `new` there."""
    from shapely.geometry import LineString
    line = lambda pts: LineString([(lon, lat) for lat, lon in pts])  # noqa: E731
    G.add_node(new, y=spot[0], x=spot[1])
    for a, b in ((u, v), (v, u)):
        if not G.has_edge(a, b):
            continue
        for k in list(G[a][b]):
            c = coords(G, a, b, k)
            i = min(range(len(c) - 1), key=lambda i: seg_dist(spot, c[i], c[i + 1]))
            attrs = {x: y for x, y in G.edges[a, b, k].items() if x not in ("geometry", "length")}
            G.add_edge(a, new, geometry=line(c[:i + 1] + [spot]), **attrs)
            G.add_edge(new, b, geometry=line([spot] + c[i + 1:]), **attrs)
            G.remove_edge(a, b, k)
    return new


def place_car(G, point, street):
    """A node 'car0' on the nearest edge of `street`, at the spot nearest `point`."""
    (u, v, k, i), spot, d = nearest_on(G, point, street)
    return split_at(G, u, v, spot, "car0"), d


def moves_from(G, a, street=None, exclude=None):
    """Every step out of node a: (b, points, legal, names). Against a one-way: legal False."""
    out = []
    for _, b, k, e in G.out_edges(a, keys=True, data=True):
        out.append((b, coords(G, a, b, k), True, names(e)))
    for p, _, k, e in G.in_edges(a, keys=True, data=True):
        if not G.has_edge(a, p):  # one-way street toward a: driving it from a is against the arrows
            out.append((p, coords(G, p, a, k)[::-1], False, names(e)))
    return [m for m in out if m[0] != exclude and (street is None or m[3] & street)]


def walk(G, view, start, street, first=None, limit=400):
    """Every way of driving along `street` from `start` (optionally beginning with move
    `first`), as paths {nodes, pts, segs}: segs[i] = (from, to, legal) for pts[i]->pts[i+1].
    A path may take one step past the edge of the map (it is not continued there).
    Returns (all prefixes, the paths no other one continues)."""
    def extend(path, m):
        a = path["nodes"][-1]
        return {"nodes": path["nodes"] + [m[0]], "pts": path["pts"] + m[1][1:],
                "segs": path["segs"] + [(a, m[0], m[2])] * (len(m[1]) - 1)}
    root = {"nodes": [start], "pts": [(G.nodes[start]["y"], G.nodes[start]["x"])], "segs": []}
    stack = [extend(root, first)] if first else [extend(root, m) for m in moves_from(G, start, street)]
    prefixes, ends = [], []
    while stack and len(prefixes) < limit:
        path = stack.pop()
        prefixes.append(path)
        n = path["nodes"]
        if not view.inside(G.nodes[n[-1]]["y"], G.nodes[n[-1]]["x"]):
            ends.append(path)
            continue
        nxt = [m for m in moves_from(G, n[-1], street, exclude=n[-2]) if m[0] not in n]
        if not nxt:
            ends.append(path)
        stack += [extend(path, m) for m in nxt]
    return prefixes, ends


def candidates(G, view, car, street):
    """Every move the car can make on this map: keep going on its street, or drive along
    it to an intersection on the map and turn onto another street there."""
    here = {"nodes": [car], "pts": [(G.nodes[car]["y"], G.nodes[car]["x"])], "segs": []}
    prefixes, ends = walk(G, view, car, street)
    cands = [{"kind": "keep going", "street": street, "approach": here, "stretch": p} for p in ends]
    approaches = [here] + [p for p in prefixes
                           if view.inside(G.nodes[p["nodes"][-1]]["y"], G.nodes[p["nodes"][-1]]["x"], EDGE_MARGIN)]
    for a in approaches:
        at = a["nodes"][-1]
        prev = a["nodes"][-2] if len(a["nodes"]) > 1 else None
        for m in moves_from(G, at, exclude=prev):
            if m[3] & street:
                continue
            for s in walk(G, view, at, m[3], first=m)[1]:
                cands.append({"kind": "turn", "street": m[3], "approach": a, "stretch": s})
    return cands


def clip(p, q, box):
    """Part of segment p->q (pixels) inside box: (t0, t1) or None (Liang-Barsky)."""
    t0, t1 = 0.0, 1.0
    dx, dy = q[0] - p[0], q[1] - p[1]
    for d, lo, hi, v in ((dx, box[0], box[2], p[0]), (dy, box[1], box[3], p[1])):
        if d == 0:
            if v < lo or v > hi:
                return None
            continue
        a, b = (lo - v) / d, (hi - v) / d
        a, b = min(a, b), max(a, b)
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return None
    return t0, t1


def inside(view, path, box):
    """The stretch's part inside the box: (its pixel points, length px, exit = (segment, t))."""
    pts = [view.px(*p) for p in path["pts"]]
    got, length, exit_ = [], 0.0, None
    for i in range(len(pts) - 1):
        c = clip(pts[i], pts[i + 1], box)
        if not c:
            continue
        p, q = pts[i], pts[i + 1]
        a = (p[0] + c[0] * (q[0] - p[0]), p[1] + c[0] * (q[1] - p[1]))
        b = (p[0] + c[1] * (q[0] - p[0]), p[1] + c[1] * (q[1] - p[1]))
        got += [a, b]
        length += math.dist(a, b)
        exit_ = (i, c[1])
    return got, length, exit_


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def choose(view, cands, box):
    """Score = overlap of Jev's box with the box around the part of the stretch inside it."""
    scored = []
    for c in cands:
        got, length, exit_ = inside(view, c["stretch"], box)
        if length < MIN_PX:
            continue
        xs, ys = zip(*got)
        r = [max(box[0], min(xs) - PAD), max(box[1], min(ys) - PAD), min(box[2], max(xs) + PAD), min(box[3], max(ys) + PAD)]
        segs = c["stretch"]["segs"][:exit_[0] + 1]
        legal = not OPT["one_way"] or (all(s[2] for s in c["approach"]["segs"]) and all(s[2] for s in segs))
        scored.append(dict(c, rect=r, iou=iou(r, box), inside_px=length, exit=exit_, legal=legal))
    ranked = sorted(scored, key=lambda c: (-c["iou"], -c["inside_px"], len(c["approach"]["pts"])))
    if not ranked or ranked[0]["iou"] < MATCH_MIN:
        return None, ranked
    best = ranked[0]
    if not best["legal"]:
        legal = [c for c in ranked if c["legal"] and c["iou"] >= best["iou"] - TIE]
        if legal:
            best = legal[0]
    return best, ranked


# --- Jev ----------------------------------------------------------------------

def question(street):
    on = words(street)
    if OPT["dest"] == "none" and not OPT["one_way"]:
        return judgments(box=bounding_box(
            f"the next street the car must drive on. This is a street map of San Francisco. The red dot "
            f"is a car on {on}. The car must drive to {DEST_WORDS}, San Francisco. Box only the part of "
            f"the next street that the car will drive: from where the car turns onto it up to its next "
            f"turn or the edge of the map. If the car should keep going on {on}, box the part of {on} "
            f"ahead of it."))
    assert OPT["dest"] == "arrow" and OPT["one_way"], "only the two tried settings have their words"
    return judgments(box=bounding_box(
        f"the next street the car must drive on. This is a street map. The red dot is a car on {on}. "
        f"The car must drive to the blue dot ({DEST_WORDS}); when the blue dot is not on the map, the "
        f"blue arrow at the edge of the map points toward it. Box only the part of the next street that "
        f"the car will drive: from where the car turns onto it up to its next turn or the edge of the map. "
        f"If the car should keep going on {on}, box the part of {on} ahead of it. "
        f"Arrows on a street mean it is one-way: the car may only drive in the arrows' direction."))


def ask(img, q):
    b = io.BytesIO()
    img.save(b, "PNG")
    for attempt in range(5):
        rec = call(MODEL, b.getvalue(), "image/png", None, level="medium", response_format=q)
        if not rec.get("error") or not rec.get("retryable"):
            break
        time.sleep(min(30, 2 ** attempt))
    if rec.get("error"):
        raise RuntimeError(f"Jev call failed: {rec['error']}")
    rec.pop("traceback", None)
    return rec


# --- pictures for us ------------------------------------------------------------

def font(size):
    for f in ("/run/current-system/sw/share/X11/fonts/DejaVuSans-Bold.ttf",):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_step(img, view, box, chosen, title):
    img = img.copy()
    d = ImageDraw.Draw(img, "RGBA")
    if chosen:
        if len(chosen["approach"]["pts"]) > 1:
            d.line([view.px(*p) for p in chosen["approach"]["pts"]], fill=(0, 160, 90, 200), width=9, joint="curve")
        d.line([view.px(*p) for p in chosen["driven_stretch"]],
               fill=(0, 160, 90, 230) if chosen["legal"] else (230, 0, 0, 230), width=9, joint="curve")
        d.rectangle(chosen["rect"], outline=(0, 120, 70, 255), width=2)
    d.rectangle(box, outline=(255, 0, 200, 255), width=5)
    d.rectangle((0, 0, view.w, 34), fill=(255, 255, 255, 220))
    d.text((10, 6), title, fill=(0, 0, 0, 255), font=font(20))
    return img


def overview(G, path_pts, ref_pts, out):
    lats = [p[0] for p in path_pts + ref_pts]
    lons = [p[1] for p in path_pts + ref_pts]
    z = 15
    x0, y0 = world_px(max(lats), min(lons), z)
    x1, y1 = world_px(min(lats), max(lons), z)
    w, h = int(x1 - x0) + 160, int(y1 - y0) + 160
    cy, cx = (max(lats) + min(lats)) / 2, (max(lons) + min(lons)) / 2
    view = View(cy, cx, z, w, h)
    img = view.picture()
    d = ImageDraw.Draw(img, "RGBA")
    if len(ref_pts) > 1:
        d.line([view.px(*p) for p in ref_pts], fill=(90, 90, 90, 150), width=10, joint="curve")
    if len(path_pts) > 1:
        d.line([view.px(*p) for p in path_pts], fill=(220, 30, 30, 230), width=5, joint="curve")
    dot(d, *view.px(START[0], START[1]), 9, RED)
    dot(d, *view.px(DEST[0], DEST[1]), 9, BLUE)
    img.save(out)


def reference_route():
    f = CACHE / "osrm-route.json"
    if not f.exists():
        url = (f"https://router.project-osrm.org/route/v1/driving/{START[1]},{START[0]};{DEST[1]},{DEST[0]}"
               "?overview=full&geometries=geojson&steps=true")
        f.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())
    r = json.loads(f.read_text())["routes"][0]
    return [(lat, lon) for lon, lat in r["geometry"]["coordinates"]], r


# --- the drive ----------------------------------------------------------------

def drive(G, out: Path, log):
    out.mkdir(parents=True, exist_ok=True)
    car, snap_m = place_car(G, START[:2], START[2])
    _, dest, _ = nearest_on(G, DEST[:2], DEST[2])
    street = frozenset([START[2]])
    path = [(G.nodes[car]["y"], G.nodes[car]["x"])]
    steps, wrong, status, misses = [], 0, "out of steps", 0
    log(f"start snapped {snap_m:.0f} m onto {START[2]}; destination on {DEST[2]}")
    step = 0
    while step < MAX_STEPS:
        here = (G.nodes[car]["y"], G.nodes[car]["x"])
        view = View(*here)
        img = map_for_jev(view, dest)
        name = f"{step + 1:02d}" + (f"-retry{misses}" if misses else "")
        img.save(out / f"map-{name}.png")
        q = question(street)
        rec = ask(img, q)
        b = rec["data"]["box"]
        box = [b["left"] * view.w, b["top"] * view.h, b["right"] * view.w, b["bottom"] * view.h]
        conf = rec["response"]["provider_data"]["typesafe"]["answers"]["box"].get("confidence")
        cands = candidates(G, view, car, street)
        chosen, ranked = choose(view, cands, box)
        top = [{"kind": c["kind"], "street": words(c["street"]), "fit": round(c["iou"], 3), "legal": c["legal"],
                "inside_px": round(c["inside_px"]), "turn_at": str(c["approach"]["nodes"][-1])} for c in ranked[:6]]
        entry = {"step": step + 1, "map": f"map-{name}.png", "car": here, "on": words(street),
                 "question": q, "box_px": [round(v, 1) for v in box], "box_confidence": conf,
                 "seconds": rec["seconds"], "cost_usd": rec["cost_usd"], "candidates": len(cands), "top": top}
        if chosen is None:
            wrong += 1
            misses += 1
            entry["result"] = "not a road"
            title = f"{name}  on {words(street)}: box fits no reachable street (best {top[0]['fit'] if top else 0:.2f})"
            draw_step(img, view, box, None, title).save(out / f"step-{name}.png")
            steps.append(entry)
            log(title)
            if misses > NO_ROAD_RETRIES:
                status = "lost"
                break
            continue
        misses = 0
        step += 1
        if not chosen["legal"]:
            wrong += 1
        i, t = chosen["exit"]
        sp, segs = chosen["stretch"]["pts"], chosen["stretch"]["segs"]
        spot = (sp[i][0] + t * (sp[i + 1][0] - sp[i][0]), sp[i][1] + t * (sp[i + 1][1] - sp[i][1]))
        chosen["driven_stretch"] = sp[:i + 1] + [spot]
        driven = chosen["approach"]["pts"] + chosen["driven_stretch"][1:]
        u, v, _ = segs[i]
        ends = [n for n in (u, v) if meters((G.nodes[n]["y"], G.nodes[n]["x"]), spot) < 1.0]
        car = ends[0] if ends else split_at(G, u, v, spot, f"car{step}")
        street = chosen["street"]
        entry.update(result="move" if chosen["legal"] else "wrong way on a one-way street",
                     kind=chosen["kind"], to=words(street), fit=round(chosen["iou"], 3),
                     metres=round(sum(meters(driven[k], driven[k + 1]) for k in range(len(driven) - 1))))
        title = (f"{name}  {entry['on']} -> {entry['to']}  ({entry['metres']} m, fit {chosen['iou']:.2f})"
                 f"{'  WRONG WAY' if not chosen['legal'] else ''}")
        draw_step(img, view, box, chosen, title).save(out / f"step-{name}.png")
        log(title)
        path += driven[1:]
        entry["dist_to_dest_m"] = round(meters(spot, dest))
        steps.append(entry)
        if any(seg_dist(dest, driven[k], driven[k + 1]) <= ARRIVE_M for k in range(len(driven) - 1)):
            status = "arrived"
            break
    total = sum(meters(path[k], path[k + 1]) for k in range(len(path) - 1))
    summary = {"status": status, "maps": step, "wrong_moves": wrong,
               "requests": len(steps), "cost_usd": sum(s["cost_usd"] or 0 for s in steps),
               "metres_driven": round(total), "left_m": round(meters(path[-1], dest))}
    return summary, steps, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=1)
    ap.add_argument("--name", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--map", choices=["osm", "clean"], default="osm")
    ap.add_argument("--one-way", choices=["on", "off"], default="on")
    ap.add_argument("--dest", choices=["arrow", "none"], default="arrow")
    args = ap.parse_args()
    OPT.update(map=args.map, one_way=args.one_way == "on", dest=args.dest)
    if args.map == "clean":
        import geopandas as gpd
        OPT["painter"] = clean_map.Painter(ox.load_graphml(GRAPH), gpd.read_file(AREAS), world_px, names, words)
    base = ROOT / "results" / "probe-jev-drive" / args.name
    ref_pts, ref = reference_route()
    for t in range(1, args.tries + 1):
        G = ox.load_graphml(GRAPH)
        out = base / f"try-{t}"
        lines = []

        def log(s):
            print(f"[try {t}] {s}", flush=True)
            lines.append(s)

        summary, steps, path = drive(G, out, log)
        summary.update(model=MODEL.route, map=f"{OPT['map']} (OpenStreetMap data), zoom {Z}, {SIZE}px",
                       one_way_rules=OPT["one_way"], destination_mark=OPT["dest"],
                       reference_m=round(ref["distance"]), when=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        (out / "steps.json").write_text(json.dumps({"summary": summary, "steps": steps, "path": path}, indent=1))
        overview(G, path, ref_pts, out / "overview.png")
        log(json.dumps(summary))


if __name__ == "__main__":
    main()
