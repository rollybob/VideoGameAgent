"""Rung 3 task 1 probe: is enemy addressing room-generic?

ScriptedHunter hardcodes -04's slot-3 enemy (HP 0x03253, pos 0x03852/54). For
multi-room harvest we need to (a) find live enemies in ANY room and (b) read
their position generically. Health is already known-generic: base 0x03250 + slot,
16 slots (the reward sums enemy_dmg across all 16). This probe nails POSITION.

Method (empirical, no assumptions baked as truth):
  - Load each human room state -00..-06, read HP[0..15] at 0x03250+i.
  - HYPOTHESIS (from slot-3 0x03852/54): position is array-of-structs, 4 B/slot,
    X = 0x03846 + 4*i, Y = 0x03848 + 4*i. For every live slot, print the guessed
    X/Y AND a raw u16 hexdump of 0x03840..0x03880 so we can read the true layout
    if the guess is wrong. Link pos (0x038F0/0x038F4 per task09) printed as a
    sanity anchor -- live enemies should sit within a screen of Link.

RAM is TEACHER/LABELER only here (offline layout discovery), never inference.

  docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/vga \\
      -w /vga/train/rl thor-rl:cu130 python3 -u probe_enemy_slots.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from alttp_ppo_env import AlttpPpoEnv  # noqa: E402
from harvest_key_frames import u16      # noqa: E402

HP_BASE = 0x03250                  # health slot i = HP_BASE + i (known generic)
POS_X0, POS_Y0, POS_STRIDE = 0x03846, 0x03848, 4   # array-of-structs hypothesis
LINK_X, LINK_Y = 0x038F0, 0x038F4  # task09 Link position addrs (sanity anchor)
STATES_DIR = os.path.join(HERE, "states")


def probe_state(path):
    env = AlttpPpoEnv(state_path=path, horizon=16)
    env.reset(seed=0)
    from mgba._pylib import ffi
    iw = ffi.cast("uint8_t *", env.core._native.memory.iwram)

    hp = [int(iw[HP_BASE + i]) for i in range(16)]
    live = [i for i, h in enumerate(hp) if h > 0]
    lx, ly = u16(iw, LINK_X), u16(iw, LINK_Y)

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"  Link pos (0x038F0/F4): ({lx}, {ly})")
    print(f"  HP[0..15] @0x03250+i : {hp}")
    print(f"  live slots (hp>0)    : {live}")
    for i in live:
        gx = u16(iw, POS_X0 + POS_STRIDE * i)
        gy = u16(iw, POS_Y0 + POS_STRIDE * i)
        ddx, ddy = gx - lx, gy - ly
        print(f"    slot {i:2d}: hp={hp[i]:3d}  guessX/Y(0x{POS_X0+4*i:05X}/"
              f"0x{POS_Y0+4*i:05X})=({gx},{gy})  d_from_link=({ddx},{ddy})")
    # raw u16 window so the TRUE stride is readable even if the guess is wrong
    print("  raw u16 @0x03840..0x03880:")
    for a in range(0x03840, 0x03880, 8):
        vals = " ".join(f"0x{a+2*k:05X}={u16(iw, a+2*k):5d}" for k in range(4))
        print(f"    {vals}")
    env.close() if hasattr(env, "close") else None
    return {"state": os.path.basename(path), "hp": hp, "live": live,
            "link": [lx, ly]}


def main():
    results = []
    for i in range(7):
        p = os.path.join(STATES_DIR, f"alttp_human-0{i}.state")
        if os.path.exists(p):
            results.append(probe_state(p))
    # summary: which rooms have >=1 live enemy (rung-3 candidate rooms)
    print("\n=== SUMMARY: rung-3 candidate rooms (>=1 live enemy) ===")
    for r in results:
        tag = "CANDIDATE" if r["live"] else "no live enemy"
        print(f"  {r['state']}: live={r['live']}  [{tag}]")


if __name__ == "__main__":
    main()
