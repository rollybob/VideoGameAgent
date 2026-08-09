"""Three screenshots for Tim's eyes on the -01 downward navigation (2026-08-03):
  1. start (-01),
  2. the moment of the FIRST room transition (room = pos>>9 changes),
  3. --after frames later (default 300 = ~5s) while STILL holding the direction.
Upscaled for phone viewing; prints the room coord + HP at each shot so the warp
(vs a plain walk-down) is visible in the numbers too.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga -w /vga/train/rl \\
      thor-rl:cu130 python3 shots_transition.py --from 01 --direction down
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PIL import Image

from alttp_ppo_env import W, H, GBA, _get_frame
from oracle import AlttpOracle
from capture_transition import build, DIR_KEY, STATES

OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "sessions", "shots")


def save(img_arr, path, scale):
    im = Image.fromarray(img_arr)
    if scale != 1:
        im = im.resize((W * scale, H * scale), Image.NEAREST)
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", default="01")
    ap.add_argument("--direction", default="down", choices=list(DIR_KEY))
    ap.add_argument("--after", type=int, default=300, help="frames after transition (~60=1s)")
    ap.add_argument("--max-frames", type=int, default=1500)
    ap.add_argument("--scale", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    core, img = build(os.path.join(STATES, "alttp_human-%s.state" % a.frm))
    orc = AlttpOracle()
    key = 1 << DIR_KEY[a.direction]

    def room():
        return orc.room_of(*orc.read_pos(core))

    p1 = os.path.join(OUT, "shot1_start.png")
    save(_get_frame(img), p1, a.scale)
    r0 = room()
    print("shot1 start: room %s HP %d -> %s" % (r0, orc.read_health(core), p1), flush=True)

    tf = None
    for f in range(a.max_frames):
        core.set_keys(raw=key)
        core.run_frame()
        if orc.read_health(core) == 0:
            print("DIED at frame %d before any transition" % f)
            break
        if room() != r0:
            tf = f
            break
    p2 = os.path.join(OUT, "shot2_transition.png")
    save(_get_frame(img), p2, a.scale)
    print("shot2 transition: frame %s, room %s -> %s HP %d -> %s"
          % (tf, r0, room(), orc.read_health(core), p2), flush=True)

    for _ in range(a.after):
        core.set_keys(raw=key)
        core.run_frame()
        if orc.read_health(core) == 0:
            break
    p3 = os.path.join(OUT, "shot3_after.png")
    save(_get_frame(img), p3, a.scale)
    print("shot3 +%df: room %s HP %d -> %s"
          % (a.after, room(), orc.read_health(core), p3), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
