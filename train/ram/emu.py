#!/usr/bin/env python3
"""Headless libmgba harness for the F21 RAM oracle.

One process gives us: deterministic frame stepping, the framebuffer, scripted
input, and RAM -- all perfectly synced, no window/X/xdotool. This is the data
engine for auto-labeling (frame + true state) and the eval oracle.

RAM is a TRAINING+EVAL crutch ONLY; runtime perception stays pixels-only.

Usage:
    from emu import Emu
    e = Emu(rom_path); e.boot()
    e.tap('A', hold=4, then=8)            # press A 4 frames, wait 8
    snap = e.wram_snapshot()              # bytes of EWRAM
    v = e.u8(0x02000000)
    e.save_png("frame.png")
"""
import os

_MGBA_PKG = os.path.expanduser(
    "~/src/mgba/build-py/python/lib.linux-aarch64-cpython-312")
import sys
if _MGBA_PKG not in sys.path:
    sys.path.insert(0, _MGBA_PKG)

import mgba.core, mgba.image, mgba.log
mgba.log.silence()

EWRAM_BASE = 0x02000000
EWRAM_SIZE = 0x40000       # 256 KB
IWRAM_BASE = 0x03000000
IWRAM_SIZE = 0x8000        # 32 KB


class Emu:
    def __init__(self, rom_path):
        self.rom_path = rom_path
        self.core = mgba.core.load_path(rom_path)
        if self.core is None:
            raise RuntimeError("load_path failed: " + rom_path)
        w, h = self.core.desired_video_dimensions()
        self.w, self.h = w, h
        self.image = mgba.image.Image(w, h)
        self.core.set_video_buffer(self.image)
        self._keymap = {
            'A': self.core.KEY_A, 'B': self.core.KEY_B,
            'L': self.core.KEY_L, 'R': self.core.KEY_R,
            'START': self.core.KEY_START, 'SELECT': self.core.KEY_SELECT,
            'UP': self.core.KEY_UP, 'DOWN': self.core.KEY_DOWN,
            'LEFT': self.core.KEY_LEFT, 'RIGHT': self.core.KEY_RIGHT,
        }

    def boot(self, load_save=True):
        if load_save:
            self.core.autoload_save()   # battery save (in-game data)
        self.core.reset()

    # --- time ---
    def run(self, frames=1):
        for _ in range(frames):
            self.core.run_frame()

    def tap(self, key, hold=4, then=6):
        """Press a key for `hold` frames, release, then run `then` idle frames."""
        k = self._keymap[key.upper()]
        self.core.set_keys(k)
        self.run(hold)
        self.core.clear_keys(k)
        if then:
            self.run(then)

    # --- memory ---
    def u8(self, addr):
        return self.core.memory.u8[addr]

    def u16(self, addr):
        return self.core.memory.u16[addr]

    def u32(self, addr):
        return self.core.memory.u32[addr]

    def wram_snapshot(self):
        ew = self.core.memory.wram
        return bytes(ew[i] for i in range(len(ew)))

    def iwram_snapshot(self):
        iw = self.core.memory.iwram
        return bytes(iw[i] for i in range(len(iw)))

    # --- video ---
    def save_png(self, path):
        self.image.to_pil().convert("RGB").save(path)

    def frame_counter(self):
        return self.core.frame_counter
