"""A map with no words at all, drawn from OpenStreetMap shapes (Maxime, 2026-10-16).

For scripts/jev_zoom_landmark.py: Jev must find a place by its shape, not its name.
Drawn here from OpenFreeMap's vector tiles (OpenMapTiles layout), using only the shape
layers: water, waterway, landcover, landuse, park, aeroway, transportation, building.
The layers that hold names (place, poi, *_name, *_label, mountain_peak, housenumber)
are never read, so no text can appear. Road route shields and arrows are not drawn either.

Same scale and placement as the 256 px OpenStreetMap tiles (a View of jev_drive.py),
zoom 0-19; above zoom 14 the zoom-14 shapes are drawn larger (the tiles stop at 14).
Drawn at twice the size and shrunk, for smooth edges.

Data: (c) OpenStreetMap contributors (ODbL), OpenMapTiles schema, served by OpenFreeMap
(openfreemap.org, free, no key). Tile set pinned to TILESET (read 2026-10-16).
"""
from __future__ import annotations

import gzip
import math
import urllib.request
from pathlib import Path

import mapbox_vector_tile
from PIL import Image, ImageDraw

TILESET = "https://tiles.openfreemap.org/planet/20261011_080001_pt/{z}/{x}/{y}.pbf"
CACHE = Path.home() / ".cache" / "segbench-maps" / "openfreemap-20261011"
UA = {"User-Agent": "segbench-research/0.1"}
S = 2
LAND, WATER = (242, 239, 233), (170, 211, 223)
GREEN, WOOD, SAND, URBAN, AIR = (205, 230, 190), (180, 215, 170), (240, 230, 200), (232, 226, 218), (225, 220, 230)
BUILDING = (217, 208, 201)
ROADS = {  # class -> (colour, width at the final size by zoom: (z<=8, z11, z14, z17+))
    "motorway": ((233, 144, 160), (1.2, 2.5, 5, 10)), "trunk": ((249, 178, 156), (1, 2.2, 4.5, 9)),
    "primary": ((252, 214, 164), (0, 1.6, 4, 8)), "secondary": ((247, 250, 191), (0, 1.0, 3, 7)),
    "tertiary": ((255, 255, 255), (0, 0, 2.5, 6)), "minor": ((255, 255, 255), (0, 0, 1.6, 5)),
    "service": ((255, 255, 255), (0, 0, 0.8, 3)), "rail": ((170, 170, 170), (0, 0.6, 1, 1.5)),
}
CASING = (190, 180, 170)
SHAPE_LAYERS = ("water", "waterway", "landcover", "landuse", "park", "aeroway", "transportation", "building")


def tile(z, x, y) -> dict:
    n = 2 ** z
    x %= n
    if not 0 <= y < n:
        return {}
    f = CACHE / str(z) / str(x) / f"{y}.pbf"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(TILESET.format(z=z, x=x, y=y), headers=UA)
        try:
            f.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            f.write_bytes(b"")
    b = f.read_bytes()
    if b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b)
    if not b:
        return {}
    t = mapbox_vector_tile.decode(b, default_options={"y_coord_down": True})
    return {k: v for k, v in t.items() if k in SHAPE_LAYERS}


def width(cls, z):
    w = ROADS[cls][1]
    stops = (8, 11, 14, 17)
    if z <= stops[0]:
        return w[0]
    for i in range(3):
        if z <= stops[i + 1]:
            t = (z - stops[i]) / (stops[i + 1] - stops[i])
            return w[i] + t * (w[i + 1] - w[i])
    return w[3] * 2 ** ((z - 17) * 0.5)


def draw(view) -> Image.Image:
    z = view.z
    tz = min(z, 14)
    k = 2 ** (z - tz)  # how much bigger a tile's shapes are drawn
    span = 256 * k     # px of one source tile on this view
    W, H = view.w * S, view.h * S
    img = Image.new("RGB", (W, H), LAND)
    d = ImageDraw.Draw(img)
    tiles = []
    for tx in range(int(view.x0 // span), int((view.x0 + view.w) // span) + 1):
        for ty in range(int(view.y0 // span), int((view.y0 + view.h) // span) + 1):
            t = tile(tz, tx, ty)
            if t:
                ox, oy = (tx * span - view.x0) * S, (ty * span - view.y0) * S
                tiles.append((t, ox, oy))

    def pts(coords, ox, oy, ext):
        sc = span * S / ext
        return [(ox + x * sc, oy + y * sc) for x, y in coords]

    def polys(geom):
        if geom["type"] == "Polygon":
            yield geom["coordinates"]
        elif geom["type"] == "MultiPolygon":
            yield from geom["coordinates"]

    def lines(geom):
        if geom["type"] == "LineString":
            yield geom["coordinates"]
        elif geom["type"] == "MultiLineString":
            yield from geom["coordinates"]

    def fill(layer, colour_of):
        for t, ox, oy in tiles:
            lay = t.get(layer)
            if not lay:
                continue
            ext = lay.get("extent", 4096)
            for f in lay["features"]:
                c = colour_of(f["properties"])
                if c is None:
                    continue
                for rings in polys(f["geometry"]):
                    if len(rings[0]) > 2:
                        d.polygon(pts(rings[0], ox, oy, ext), fill=c)
                    for hole in rings[1:]:  # holes back to the land colour (islands in lakes stay land)
                        if len(hole) > 2:
                            d.polygon(pts(hole, ox, oy, ext), fill=LAND)

    fill("landuse", lambda p: URBAN if p.get("class") in ("residential", "commercial", "industrial", "retail") and z >= 10 else None)
    fill("landcover", lambda p: {"wood": WOOD, "grass": GREEN, "sand": SAND, "farmland": None, "wetland": GREEN}.get(p.get("class"), None))
    fill("park", lambda p: GREEN)
    fill("aeroway", lambda p: AIR)
    fill("water", lambda p: WATER)
    for t, ox, oy in tiles:
        lay = t.get("waterway")
        if lay and z >= 9:
            for f in lay["features"]:
                for ln in lines(f["geometry"]):
                    d.line(pts(ln, ox, oy, lay.get("extent", 4096)), fill=WATER, width=max(1, round(S * (1 if z < 13 else 2))))
    if z >= 15:
        fill("building", lambda p: BUILDING)
    order = ["rail", "service", "minor", "tertiary", "secondary", "primary", "trunk", "motorway"]
    for casing in (True, False):
        for cls in order:
            w = width(cls, z)
            if w <= 0:
                continue
            for t, ox, oy in tiles:
                lay = t.get("transportation")
                if not lay:
                    continue
                ext = lay.get("extent", 4096)
                for f in lay["features"]:
                    p = f["properties"]
                    c = p.get("class")
                    c = "minor" if c in ("busway",) else c
                    if c != cls or p.get("brunnel") == "tunnel":
                        continue
                    for ln in lines(f["geometry"]):
                        q = pts(ln, ox, oy, ext)
                        if len(q) < 2:
                            continue
                        if casing:
                            if w >= 2 and cls != "rail":
                                d.line(q, fill=CASING, width=max(1, round((w + 1.5) * S)), joint="curve")
                        else:
                            d.line(q, fill=ROADS[cls][0], width=max(1, round(w * S)), joint="curve")
    return img.resize((view.w, view.h), Image.LANCZOS)
