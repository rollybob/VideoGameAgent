"""Task 09 A0: list ticks where a HUD region changes across a capture's PNGs.

Ground-truth extractor for the RAM mapping: crops a screen region from each
periodic frame PNG in a ram_capture dump and reports the ticks where its
pixels change (30-tick resolution). Default region = the bottom-right rupee
counter of the Four Swords HUD. Run with the VGA venv (has cv2):

  ~/projects/VGA/.venv/bin/python hud_changes.py scratch/ram/<tape> \
      [--region x0,y0,x1,y1] [--save-crops]
"""
import argparse
import glob
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--region", default="180,140,240,160")
    ap.add_argument("--save-crops", action="store_true")
    args = ap.parse_args()
    dump = args.dump if os.path.isabs(args.dump) else os.path.join(HERE, args.dump)
    x0, y0, x1, y1 = map(int, args.region.split(","))

    pngs = sorted(glob.glob(os.path.join(dump, "f*.png")))
    prev = None
    prev_tick = None
    for p in pngs:
        tick = int(os.path.basename(p)[1:6])
        img = cv2.imread(p)
        crop = img[y0:y1, x0:x1]
        if prev is not None and (crop.shape != prev.shape or (crop != prev).any()):
            print("region changed between tick %d and %d" % (prev_tick, tick))
            if args.save_crops:
                cv2.imwrite(os.path.join(dump, "hud_%05d.png" % tick), crop)
        prev = crop
        prev_tick = tick
    print("scanned %d frames" % len(pngs))


if __name__ == "__main__":
    main()
