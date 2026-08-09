#!/usr/bin/env python3
"""Validate a pixels-only 'screen is self-animating' sensor, deterministically, no VLM.

Rescue day 2 (2026-07-04). The R4 derail cause: the agent can't perceive that the world
has entered a transient/animating state (world-map travel), so it keeps pressing and knocks
the caravan off course. A general fix is a System-1 instability reflex: measure inter-frame
motion under SETTLED input; while it's high, hold (emit wait); when it settles, hand back to
the VLM. Pixels-only -> north-star compliant, and transfers to every cutscene/animation.

This probe measures two candidate motion metrics over idle sub-windows in three contexts:
  A STATIC IDLE MAP     - world map, no input (should read ~0)
  B STATIC OPEN MENU    - Area List open, blinking cursor (the false-positive risk: must
                          stay LOW so the reflex does NOT strand the agent in menus)
  C TRAVEL ANIMATION    - right after the golden commit, caravan walking (should read HIGH)
A usable sensor separates C >> {A,B}. Run on host venv from train/ram.
"""
import sys
import numpy as np
from PIL import Image
from emu import Emu
from oracle import for_rom

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Final Fantasy Tactics Advance (E)(Surplus).gba"
STATE = "ffta_worldmap.state"

_AH = 16   # avg-hash grid (matches plugin _ahash: 16x16 -> 256-bit)
_MD = 64   # mean-abs-diff downsample grid

def _gray(emu, size):
    im = emu.image.to_pil().convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)

def _ahash_bits(emu):
    g = _gray(emu, _AH)
    return (g >= g.mean()).astype(np.uint8).flatten()

def _hamming_bits(a, b):
    return int(np.count_nonzero(a != b))

def sample_motion(emu, oracle, windows=8, gap=4):
    """Under NO input, sample `windows` idle sub-windows of `gap` frames each; return the
    per-window aHash-hamming (0-256) and mean-abs pixel diff (0-255) between consecutive
    sub-window frames. This is exactly what an inference reflex could compute from pixels."""
    prev_a = _ahash_bits(emu)
    prev_m = _gray(emu, _MD)
    ham, mad, ovls = [], [], []
    for _ in range(windows):
        emu.run(gap)                       # advance emulated time with NO key
        a = _ahash_bits(emu)
        m = _gray(emu, _MD)
        ham.append(_hamming_bits(prev_a, a))
        mad.append(float(np.abs(m - prev_m).mean()))
        ovls.append(oracle.read_state(emu).get("mode_overlay"))
        prev_a, prev_m = a, m
    return ham, mad, ovls

def report(tag, ham, mad, ovls):
    print(f"\n[{tag}] overlays={ovls}")
    print(f"  aHash-hamming: {ham}  max={max(ham)} mean={np.mean(ham):.1f}")
    print(f"  mean-abs-diff: {[round(x,2) for x in mad]}  max={max(mad):.2f} mean={np.mean(mad):.2f}")

def fresh(oracle):
    emu = Emu(ROM)
    emu.boot(load_save=True)
    with open(STATE, "rb") as f:
        emu.core.load_raw_state(f.read())
    emu.run(30)
    return emu

def main():
    oracle = for_rom(ROM)

    # A: static idle world map (no input since load)
    emu = fresh(oracle)
    ham, mad, ovls = sample_motion(emu, oracle)
    report("A static idle map", ham, mad, ovls)

    # B: static open menu (Area List up, blinking cursor)
    emu = fresh(oracle)
    for k in ["START", "DOWN", "A"]:
        emu.tap(k, hold=6, then=30)
    st = oracle.read_state(emu)
    ham, mad, ovls = sample_motion(emu, oracle)
    report(f"B static open menu (ovl={st.get('mode_overlay')})", ham, mad, ovls)

    # C: FULL travel profile - sample motion across the whole animation next to clan_pos, so
    # we see whether the caravan-WALK phase (pos 18->23->21->20, which starts ~60 frames after
    # the commit) is detectable, not just the initial scroll. Print each window: motion + pos.
    emu = fresh(oracle)
    for k in ["START", "DOWN", "A", "DOWN", "DOWN", "A", "A"]:
        emu.tap(k, hold=6, then=30)
    st = oracle.read_state(emu)
    print(f"\n(commit done: ovl={st.get('mode_overlay')} pos={st.get('clan_pos')})")
    print("\n[C full travel profile]  win: mad  ham  ovl  pos")
    prev_a = _ahash_bits(emu); prev_m = _gray(emu, _MD)
    arrived = None
    for w in range(60):
        emu.run(4)
        a = _ahash_bits(emu); m = _gray(emu, _MD)
        h = _hamming_bits(prev_a, a); d = float(np.abs(m - prev_m).mean())
        s = oracle.read_state(emu); ovl = s.get("mode_overlay"); pos = s.get("clan_pos")
        prev_a, prev_m = a, m
        if w < 55 or pos == 20:
            print(f"  {w:2d}: {d:6.2f} {h:4d} {ovl:4d} {pos:4d}")
        if pos == 20 and arrived is None:
            arrived = w
    print(f"  -> arrived at Giza (pos=20) at window {arrived} (~{(arrived+1)*4} frames post-commit)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
