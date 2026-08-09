"""_grid.py -- overlay a labeled screen-coordinate grid on a ring's mark.png so Tim can read off
item positions from his phone. Grid every 40px (cyan), labeled (yellow) along top/bottom/left.
4x nearest-neighbour zoom for phone legibility. Usage: _grid.py [ring-timestamp]  (default newest).
Writes _grid_<tag>.png and prints the path. thor-rl:cu130 (PIL)."""
import os, glob, sys
from PIL import Image, ImageDraw
RINGS = "/work/link/sessions/ramhits_solo"; DET = "/work/train/perception/detector"
tag = sys.argv[1] if len(sys.argv) > 1 else None
d = (glob.glob(os.path.join(RINGS, f"*{tag}-hit"))[0] if tag
     else max(glob.glob(os.path.join(RINGS, "20260808-*-hit")), key=os.path.getmtime))
Z = 4
im = Image.open(os.path.join(d, "mark.png")).convert("RGB").resize((240 * Z, 160 * Z), Image.NEAREST)
dr = ImageDraw.Draw(im)
for x in range(0, 241, 40):
    dr.line([(x * Z, 0), (x * Z, 160 * Z)], fill=(0, 255, 255), width=1)
    dr.text((x * Z + 2, 2), str(x), fill=(255, 255, 0))
    dr.text((x * Z + 2, 160 * Z - 12), str(x), fill=(255, 255, 0))
for y in range(0, 161, 40):
    dr.line([(0, y * Z), (240 * Z, y * Z)], fill=(0, 255, 255), width=1)
    dr.text((2, y * Z + 2), str(y), fill=(255, 255, 0))
name = os.path.basename(d)[9:15]
out = os.path.join(DET, f"_grid_{name}.png")
im.save(out)
print("RING", os.path.basename(d), flush=True)
print("OUT", out, flush=True)
