"""Drive a scripted MOVE SEQUENCE through ALttP and screenshot each step, for
guided navigation (2026-08-03: reaching the outdoor/bushes area that lies past
-01's dungeon -- down to a castle rampart, then drop off the ledge to the right).

Each move is DIR*N (hold N frames) or DIR*T (hold until a room transition, max
--max-frames). Tracks room / green-fraction / HP per move, saves upscaled PNGs
to sessions/shots/, and banks the final state to --out if it reads as OUTDOORS
(green-pixel fraction), so grass is detected from numbers.

  ... python3 drive_seq.py --from 01 --moves down*T,down*T,right*150,down*150 --out 06
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PIL import Image

from alttp_ppo_env import W, H, GBA, _get_frame
from oracle import AlttpOracle
from capture_transition import build, DIR_KEY, STATES, green_frac, save_state

OUT_SHOTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "sessions", "shots")
GREEN_OUTDOOR = 0.12


def save_png(arr, path, scale):
    im = Image.fromarray(arr)
    if scale != 1:
        im = im.resize((W * scale, H * scale), Image.NEAREST)
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", default="01")
    ap.add_argument("--moves", required=True, help="comma list of DIR*N or DIR*T")
    ap.add_argument("--out", default=None, help="state number to bank if outdoors")
    ap.add_argument("--max-frames", type=int, default=1500)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--tag", default="nav", help="screenshot filename prefix")
    a = ap.parse_args()
    os.makedirs(OUT_SHOTS, exist_ok=True)

    core, img = build(os.path.join(STATES, "alttp_human-%s.state" % a.frm))
    orc = AlttpOracle()

    def room():
        return orc.room_of(*orc.read_pos(core))

    def snap(name):
        p = os.path.join(OUT_SHOTS, "%s_%s.png" % (a.tag, name))
        save_png(_get_frame(img), p, a.scale)
        return p

    snap("00_start")
    print("start: -%s room %s green %.3f HP %d"
          % (a.frm, room(), green_frac(_get_frame(img)), orc.read_health(core)), flush=True)

    dead = False
    for i, mv in enumerate(a.moves.split(","), 1):
        d, arg = mv.split("*")
        key = 1 << DIR_KEY[d]
        if arg.upper() == "T":
            r0 = room()
            for _ in range(a.max_frames):
                core.set_keys(raw=key)
                core.run_frame()
                if orc.read_health(core) == 0:
                    dead = True
                    break
                if room() != r0:
                    break
        else:
            for _ in range(int(arg)):
                core.set_keys(raw=key)
                core.run_frame()
                if orc.read_health(core) == 0:
                    dead = True
                    break
        gf = green_frac(_get_frame(img))
        p = snap("%02d_%s" % (i, mv.replace("*", "")))
        print("move %d %-9s -> room %s green %.3f HP %d %s"
              % (i, mv, room(), gf, orc.read_health(core),
                 "DIED" if dead else ("OUTDOORS" if gf >= GREEN_OUTDOOR else "")), flush=True)
        if dead:
            break

    gf = green_frac(_get_frame(img))
    if a.out and gf >= GREEN_OUTDOOR and not dead:
        dst = os.path.join(STATES, "alttp_human-%s.state" % a.out)
        save_state(core, dst)
        print("PROMOTED outdoors (green %.3f) -> %s" % (gf, os.path.basename(dst)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
