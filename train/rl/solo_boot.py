"""Task 09 A0: build an in-game solo ALttP savestate (bench + oracle seed).

Menu path adapted from link/sessions/drive_p4_unique.py (which creates a
file and then switches RIGHT to Four Swords -- here we stay on ALttP).
No .sav is loaded (load_path does not autoload SRAM), so the game always
boots to an empty CHOOSE A FILE screen and nothing touches Tim's real save.

Dumps a numbered PNG per step into scratch/solo_boot/ for visual
verification, then mashes A through the intro dialogs, snapshotting every
cycle. Inspect the strip, then set --bank-at to the first playable frame
cycle and re-run to bank states/alttp_ingame.state.
"""
import argparse
import os

import mgba.core
import mgba.gba
import mgba.image
import mgba.log
from mgba._pylib import ffi

mgba.log.silence()
HERE = os.path.dirname(os.path.abspath(__file__))
VGA = os.path.dirname(os.path.dirname(HERE))
ROM = os.path.join(VGA, "Emulator", "mGBA", "roms",
                   "Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
OUT = os.path.join(HERE, "scratch", "solo_boot")
STATES = os.path.join(HERE, "states")
K = mgba.gba.GBA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mash-cycles", type=int, default=24,
                    help="A-press cycles after game select (240 frames each)")
    ap.add_argument("--bank-at", type=int, default=None,
                    help="mash cycle at which to save alttp_ingame.state")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(STATES, exist_ok=True)

    core = mgba.core.load_path(ROM)
    w, h = core.desired_video_dimensions()
    img = mgba.image.Image(w, h)
    core.set_video_buffer(img)
    core.reset()

    def run(n):
        for _ in range(n):
            core.run_frame()

    def tap(key, hold=5, after=40):
        core.set_keys(key)
        run(hold)
        core.set_keys(raw=0)
        run(after)

    def taps(key, n, **kw):
        for _ in range(n):
            tap(key, **kw)

    def shot(name):
        with open(os.path.join(OUT, name + ".png"), "wb") as f:
            img.save_png(f)

    run(300)
    tap(K.KEY_START); run(90); shot("00_after_start")
    tap(K.KEY_A); run(90); shot("01_name_entry")          # empty slot -> ENTER A NAME
    tap(K.KEY_A); run(20)                                  # type letter 'A'
    taps(K.KEY_DOWN, 5, after=15); taps(K.KEY_RIGHT, 3, after=15); tap(K.KEY_A)
    run(120); shot("02_file_created")
    tap(K.KEY_A); run(120); shot("03_choose_game")         # file -> CHOOSE A GAME
    tap(K.KEY_LEFT); run(90); shot("04_game_left")         # ensure ALttP highlighted
    tap(K.KEY_A); run(150); shot("05_alttp_confirm")       # confirm ALttP
    for c in range(args.mash_cycles):
        # alternate A and START: A advances dialogs, START skips cutscenes
        tap(K.KEY_A if c % 2 == 0 else K.KEY_START, after=10)
        run(230)
        shot("m%02d" % c)
        if args.bank_at is not None and c == args.bank_at:
            # dismiss any open START menu / lingering dialog before banking
            tap(K.KEY_B, after=30); tap(K.KEY_B, after=30)
            st = core.save_raw_state()
            path = os.path.join(STATES, "alttp_ingame.state")
            with open(path, "wb") as f:
                f.write(bytes(ffi.buffer(st)))
            print("banked", path, "at mash cycle", c)
    print("done; inspect", OUT)


if __name__ == "__main__":
    main()
