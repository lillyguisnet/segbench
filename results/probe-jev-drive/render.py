import math, io, sys, time, urllib.request
from PIL import Image, ImageDraw
UA = {"User-Agent": "segbench-map-test/0.1 (personal research)"}
def px(lat, lon, z):
    n = 256 * 2**z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y
def render(lat, lon, z, size, out, scale=2):
    # scale=2 uses retina-style upscaling? OSM standard has no @2x; keep 256 tiles
    cx, cy = px(lat, lon, z)
    x0, y0 = cx - size/2, cy - size/2
    img = Image.new("RGB", (size, size))
    for tx in range(int(x0//256), int((x0+size)//256)+1):
        for ty in range(int(y0//256), int((y0+size)//256)+1):
            url = f"https://tile.openstreetmap.org/{z}/{tx}/{ty}.png"
            t = Image.open(io.BytesIO(urllib.request.urlopen(urllib.request.Request(url, headers=UA)).read())).convert("RGB")
            img.paste(t, (int(tx*256 - x0), int(ty*256 - y0)))
    d = ImageDraw.Draw(img); r = 11
    d.ellipse((size/2-r-3, size/2-r-3, size/2+r+3, size/2+r+3), fill="white")
    d.ellipse((size/2-r, size/2-r, size/2+r, size/2+r), fill=(220, 30, 30))
    img.save(out)
for z in (17, 18):
    render(37.7929589, -122.3995978, z, 1024, f"/tmp/maptest/start_z{z}.png")
print("ok")
