"""A clean street map drawn from OpenStreetMap data (Maxime, 2026-10-16): streets and
their names, parks and water, nothing else (no shops, icons, bus stops, one-way arrows).

Same scale and placement as OpenStreetMap's own map at zoom 17 (256 px tiles), so the
car's position on it is the same pixel as on the standard map. Drawn at twice the size
and shrunk, for smooth lines and letters. Street names are written along their streets,
repeated about every LABEL_EVERY px, never over another name.

Data: (c) OpenStreetMap contributors (ODbL). Streets: the drive network of
scripts/jev_drive.py; parks and water: ~/.cache/segbench-maps/sf-areas.gpkg (Overpass,
2026-10-16).
"""
from __future__ import annotations

import math
import subprocess
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, box as sbox

S = 2  # drawn at twice the size
BG, PARK, WATER = (242, 239, 233), (214, 234, 200), (170, 211, 223)
CASING = (196, 190, 182)
ROAD = {  # highway type -> (fill, width px at the final size)
    "motorway": ((247, 190, 130), 13), "trunk": ((247, 190, 130), 12),
    "primary": ((252, 226, 150), 12), "secondary": ((253, 238, 180), 11),
    "tertiary": ((255, 255, 255), 10), "residential": ((255, 255, 255), 9),
    "unclassified": ((255, 255, 255), 9), "living_street": ((255, 255, 255), 8),
}
LINK = ((255, 255, 255), 7)
FONT_PX = 14
LABEL_EVERY = 330
TEXT, HALO = (40, 40, 40), (255, 255, 255)


@lru_cache(None)
def font_file(bold=True) -> str:
    return subprocess.run(["fc-match", "-f", "%{file}", "DejaVu Sans:bold" if bold else "DejaVu Sans"],
                          capture_output=True, text=True).stdout.strip()


def kind(e):
    hw = e.get("highway")
    hw = hw[0] if isinstance(hw, list) else hw or ""
    if hw.endswith("_link"):
        return LINK, 0
    order = list(ROAD)
    return ROAD.get(hw, ROAD["residential"]), len(order) - order.index(hw) if hw in order else 0


class Painter:
    """Precomputes world pixels of every street and area once; draws any View."""

    def __init__(self, G, areas, world_px, names, words):
        self.world_px, self.words = world_px, words
        self.streets, seen = [], set()
        for u, v, k, e in G.edges(keys=True, data=True):
            key = (min(str(u), str(v)), max(str(u), str(v)), str(e.get("osmid")))
            if key in seen:
                continue
            seen.add(key)
            if "geometry" in e:
                pts = [world_px(lat, lon) for lon, lat in e["geometry"].coords]
            else:
                pts = [world_px(G.nodes[n]["y"], G.nodes[n]["x"]) for n in (u, v)]
            (fill, width), rank = kind(e)
            label = words(names(e))
            label = None if label in ("a ramp", "an unnamed road") else label
            self.streets.append({"u": str(u), "v": str(v), "pts": pts, "fill": fill, "w": width,
                                 "rank": rank, "label": label})
        self.areas = []
        for _, row in areas.iterrows():
            g = row.geometry
            colour = WATER if (row.get("natural") in ("water", "coastline")) else PARK
            parts = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
            for p in parts:
                if p.geom_type == "Polygon":
                    self.areas.append(("poly", colour, [world_px(lat, lon) for lon, lat in p.exterior.coords]))
                elif p.geom_type == "LineString":
                    self.areas.append(("line", colour, [world_px(lat, lon) for lon, lat in p.coords]))

    def draw(self, view) -> Image.Image:
        W, H = view.w * S, view.h * S
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        to = lambda pts: [((x - view.x0) * S, (y - view.y0) * S) for x, y in pts]  # noqa: E731
        frame = sbox(-200, -200, W + 200, H + 200)

        def near(pts):
            xs = [(x - view.x0) * S for x, _ in pts]
            ys = [(y - view.y0) * S for _, y in pts]
            return max(xs) > -200 and min(xs) < W + 200 and max(ys) > -200 and min(ys) < H + 200

        for kind_, colour, pts in self.areas:
            if not near(pts):
                continue
            if kind_ == "poly" and len(pts) > 2:
                d.polygon(to(pts), fill=colour)
            elif kind_ == "line":
                d.line(to(pts), fill=WATER, width=6 * S)
        streets = sorted((s for s in self.streets if near(s["pts"])), key=lambda s: s["rank"])
        for s in streets:  # casings first, then fills, so crossings look joined
            p = to(s["pts"])
            d.line(p, fill=CASING, width=(s["w"] + 2) * S, joint="curve")
            for q in (p[0], p[-1]):
                r = (s["w"] + 2) * S / 2
                d.ellipse((q[0] - r, q[1] - r, q[0] + r, q[1] + r), fill=CASING)
        for s in streets:
            p = to(s["pts"])
            d.line(p, fill=s["fill"], width=s["w"] * S, joint="curve")
            for q in (p[0], p[-1]):
                r = s["w"] * S / 2
                d.ellipse((q[0] - r, q[1] - r, q[0] + r, q[1] + r), fill=s["fill"])
        self._labels(img, view, streets, to, frame)
        return img.resize((view.w, view.h), Image.LANCZOS)

    def _labels(self, img, view, streets, to, frame):
        font = ImageFont.truetype(font_file(), FONT_PX * S)
        W, H = img.size
        placed = []
        chains = self._chains([s for s in streets if s["label"]])
        chains.sort(key=lambda c: -c[2])  # big streets first
        for label, pts, _ in chains:
            p = to(pts)
            tw = font.getlength(label) + 6 * S
            th = FONT_PX * S * 1.3
            seg = [0.0]
            for i in range(len(p) - 1):
                seg.append(seg[-1] + math.dist(p[i], p[i + 1]))
            total, s_pos, last = seg[-1], tw / 2 + 10 * S, -1e9
            while s_pos < total - tw / 2:
                spot = self._straight(p, seg, s_pos, tw)
                if spot is None or s_pos - last < LABEL_EVERY * S:
                    s_pos += 20 * S
                    continue
                (cx, cy), ang = spot
                if not (tw / 2 < cx < W - tw / 2 and th < cy < H - th):
                    s_pos += 20 * S
                    continue
                ca, sa = math.cos(ang), math.sin(ang)
                hw, hh = tw / 2 + 4 * S, th / 2 + 2 * S
                poly = Polygon([(cx + ca * x - sa * y, cy + sa * x + ca * y)
                                for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))])
                if any(poly.intersects(q) for q in placed):
                    s_pos += 20 * S
                    continue
                placed.append(poly)
                tile = Image.new("RGBA", (int(tw + 12 * S), int(th + 8 * S)), (0, 0, 0, 0))
                td = ImageDraw.Draw(tile)
                td.text((tile.width / 2, tile.height / 2), label, font=font, fill=TEXT, anchor="mm",
                        stroke_width=3 * S, stroke_fill=HALO)
                rot = tile.rotate(-math.degrees(ang), resample=Image.BICUBIC, expand=True)
                img.paste(rot, (int(cx - rot.width / 2), int(cy - rot.height / 2)), rot)
                last = s_pos
                s_pos += LABEL_EVERY * S
        return placed

    @staticmethod
    def _straight(p, seg, s_pos, length):
        """Middle point and angle of the stretch [s_pos - length/2, s_pos + length/2] if it is straight."""
        def at(s):
            for i in range(len(seg) - 1):
                if seg[i + 1] >= s:
                    t = (s - seg[i]) / max(1e-9, seg[i + 1] - seg[i])
                    return (p[i][0] + t * (p[i + 1][0] - p[i][0]), p[i][1] + t * (p[i + 1][1] - p[i][1]))
            return p[-1]
        a, b, m = at(s_pos - length / 2), at(s_pos + length / 2), at(s_pos)
        chord = math.dist(a, b)
        if chord < 0.97 * length:
            return None
        for k in range(1, 8):
            q = at(s_pos - length / 2 + k * length / 8)
            dev = abs((b[0] - a[0]) * (a[1] - q[1]) - (a[0] - q[0]) * (b[1] - a[1])) / chord
            if dev > 3 * S:
                return None
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        if ang > math.pi / 2:
            ang -= math.pi
        elif ang <= -math.pi / 2:
            ang += math.pi
        return m, ang

    @staticmethod
    def _chains(streets):
        """Pieces of the same name joined end to end: [(label, world points, rank)]."""
        by = {}
        for s in streets:
            by.setdefault(s["label"], []).append(s)
        out = []
        for label, ss in by.items():
            ends = {}
            for i, s in enumerate(ss):
                ends.setdefault(s["u"], []).append(i)
                ends.setdefault(s["v"], []).append(i)
            used = set()
            for i, s in enumerate(ss):
                if i in used:
                    continue
                used.add(i)
                pts, head, tail = list(s["pts"]), s["u"], s["v"]
                for forward in (True, False):
                    node = tail if forward else head
                    while True:
                        nxt = [j for j in ends.get(node, []) if j not in used]
                        if len(nxt) != 1 or len(ends[node]) != 2:
                            break
                        j = nxt[0]
                        used.add(j)
                        t = ss[j]
                        piece = t["pts"] if t["u"] == node else t["pts"][::-1]
                        other = t["v"] if t["u"] == node else t["u"]
                        pts = pts + piece[1:] if forward else piece[:-1] + pts
                        node = other
                out.append((label, pts, max(x["rank"] for x in ss)))
        return out
