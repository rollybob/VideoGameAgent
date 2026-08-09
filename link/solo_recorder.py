"""Solo ALttP human-play flight recorder (Task 09 A0, 2026-07-09).

The 4P link viewer (link_viewer.py) already lets a human play and F12-dump a
flight tape, but its FlightRecorder is wired to the 4-core LinkSession (it
banks four savestates). Solo ALttP had NO human-play recorder -- all its
tooling (solo_boot.py, alttp_probe.py) is headless/scripted. This is the solo
counterpart: one mGBA core booting states/alttp_ingame.state, a pygame window
for keyboard play, and an always-on rolling recorder that F12 dumps the last
~N seconds so we can mine hearts/kill/death RAM channels from real combat.

The dump format mirrors the link flight dump but single-player:
  prefail.state   one raw savestate at the START of the captured window
  inputs.json     {"start_tick": T0, "masks": [int, ...]}  one raw mask/tick
  meta.json       reason / bank_tick / end_tick / n_frames
  dump.png        screenshot at dump time (human sanity check)
Replay + RAM capture is train/rl/solo_ram_capture.py (headless, in-container).

Key convention: masks are bitmasks 1<<K.KEY_* OR'd together, passed as
set_keys(raw=mask) -- verified against mgba core._keys_to_int and matching
link_viewer (the proven human-input path). NOT the unshifted-index form.

Run on the REAL desktop (needs a display + pygame), NOT in the container:
  source ~/projects/VGA/link/env.sh
  DISPLAY=:1 "$LINK_PY" ~/projects/VGA/link/solo_recorder.py
Self-test (no display, no human -- proves boot+dump path):
  source ~/projects/VGA/link/env.sh ; "$LINK_PY" solo_recorder.py --selftest 90
"""
import argparse
import collections
import io
import json
import os
import sys
import time

import mgba.core
import mgba.gba as gba
import mgba.image
import mgba.log
from mgba._pylib import ffi

mgba.log.silence()
K = gba.GBA
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)          # so ram_ring imports when run by absolute path
from ram_ring import RamRing      # noqa: E402
VGA = os.path.dirname(HERE)
GBA_W, GBA_H = 240, 160
HUD_H = 26
DEFAULT_STATE = os.path.join(VGA, "train", "rl", "states", "alttp_ingame.state")
DEFAULT_OUT = os.path.join(HERE, "sessions", "flight_solo")
DEFAULT_RAM_OUT = os.path.join(HERE, "sessions", "ramhits_solo")
DEFAULT_STATE_OUT = os.path.join(VGA, "train", "rl", "states")

# bitmask per GBA button (1<<index); OR together, pass as set_keys(raw=...)
BIT = {name: 1 << getattr(K, "KEY_" + name) for name in
       ("A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "R", "L")}


def _state_bytes(core):
    return bytes(ffi.buffer(core.save_raw_state()))


def _frame_png(img):
    # mgba's save_png closes the fileobj when done; keep ours readable
    buf = type("KeepOpen", (io.BytesIO,), {"close": lambda self: None})()
    img.save_png(buf)
    return buf.getvalue()


class SoloRecorder:
    """Always-on black box for one core. Rolling banks of savestate + the raw
    input mask every tick; F12 dumps the oldest kept bank and the tape from it
    to now. Unlike the link recorder this is REPEATABLE -- each F12 writes a
    fresh numbered take and recording continues (this tool's job is capturing
    takes, not one-shot death diagnosis). Call reset() after any state load so
    a dump never splices a tape across a load discontinuity."""

    def __init__(self, out_root, get_state, bank_ticks, banks):
        self.out_root = out_root
        self.get_state = get_state
        self.bank_ticks = max(1, bank_ticks)
        self.banks = collections.deque(maxlen=max(2, banks))
        self.masks = []          # raw masks since the OLDEST kept bank
        self.tick_no = 0
        self.n_dumps = 0
        os.makedirs(out_root, exist_ok=True)
        self._bank()             # seed bank at tick 0

    def _bank(self):
        self.banks.append((self.tick_no, self.get_state()))
        # keep the input tape aligned with the oldest kept bank
        excess = len(self.masks) - (self.tick_no - self.banks[0][0])
        if excess > 0:
            del self.masks[:excess]

    def reset(self):
        self.banks.clear()
        self.masks = []
        self.tick_no = 0
        self._bank()

    def record(self, mask):
        """Call once per tick, AFTER run_frame(), with the applied raw mask."""
        self.tick_no += 1
        self.masks.append(int(mask))
        if self.tick_no % self.bank_ticks == 0:
            self._bank()

    def history_secs(self):
        bank_tick = self.banks[0][0]
        return (self.tick_no - bank_tick) / 60.0

    def dump(self, reason, screenshot_png=None):
        bank_tick, state = self.banks[0]
        out = os.path.join(self.out_root,
                           time.strftime("flight-%Y%m%d-%H%M%S") + "-" + reason)
        if os.path.exists(out):          # two dumps in the same second
            out += "-%d" % self.n_dumps
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "prefail.state"), "wb") as f:
            f.write(state)
        tape = self.masks[-(self.tick_no - bank_tick):] if self.tick_no > bank_tick else []
        with open(os.path.join(out, "inputs.json"), "w") as f:
            json.dump({"start_tick": bank_tick, "masks": tape}, f)
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump({"reason": reason, "bank_tick": bank_tick,
                       "end_tick": self.tick_no, "n_frames": len(tape)}, f, indent=1)
        if screenshot_png is not None:
            with open(os.path.join(out, "dump.png"), "wb") as f:
                f.write(screenshot_png)
        self.n_dumps += 1
        print("solo flight: dumped %d frames (~%.1fs) -> %s"
              % (len(tape), len(tape) / 60.0, out), flush=True)
        return out


class Recorder:
    def __init__(self, args):
        self.args = args
        self.core = mgba.core.load_path(args.rom)
        w, h = self.core.desired_video_dimensions()
        self.img = mgba.image.Image(w, h)
        self.core.set_video_buffer(self.img)
        self.core.reset()
        with open(args.state, "rb") as f:
            assert self.core.load_raw_state(f.read()), "failed to load seed state"
        self.seed_state = _state_bytes(self.core)
        self.quick_state = None
        self.rec = SoloRecorder(args.out, lambda: _state_bytes(self.core),
                                int(round(args.bank_secs * 60)), args.banks)
        # Rolling raw-RAM window for address hunting: F11 right after a visible
        # hit dumps the last few seconds so the HP byte can be diffed offline.
        self.ring = RamRing(self.core, every=args.ram_every, keep=args.ram_keep,
                            label="solo_alttp")
        self.tick_no = 0
        self.human_mask = 0
        self.running = True
        self.paused = False
        self.turbo = False
        self.msg = "ready -- F11 right after taking a hit, F8 to bank a savestate"
        self.pygame = None
        self.screen = None
        self.font = None
        self.HUMAN_KEYMAP = {}

    # ---- state juggling (reset the recorder so tapes never span a load) -----
    def load_state(self, data, what):
        if data is None:
            self.msg = "no %s to load" % what
            return
        assert self.core.load_raw_state(data)
        self.rec.reset()
        self.ring.snaps.clear()   # never let a RAM window straddle a load
        self.human_mask = 0
        self.msg = "loaded %s (recorder reset)" % what

    def save_named_state(self):
        """Write the CURRENT core state to a numbered file on disk.

        F5 only quick-saves in memory, which is useless for handing a new
        training start-point to the trainer. These are auto-numbered so a play
        session can bank several candidate spots without clobbering.
        """
        d = self.args.state_out
        os.makedirs(d, exist_ok=True)
        stem = self.args.state_name
        n = 0
        while True:
            path = os.path.join(d, "%s-%02d.state" % (stem, n))
            if not os.path.exists(path):
                break
            n += 1
        with open(path, "wb") as f:
            f.write(_state_bytes(self.core))
        print("saved state -> %s" % path, flush=True)
        return path

    def init_display(self):
        import pygame
        self.pygame = pygame
        s = self.args.scale
        self.win_w, self.win_h = GBA_W * s, GBA_H * s + HUD_H
        pygame.init()
        pygame.display.set_caption("Solo ALttP recorder - VGA Task 09")
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.font = pygame.font.SysFont("monospace", 13, bold=True)
        self.HUMAN_KEYMAP = {
            pygame.K_x: BIT["A"], pygame.K_z: BIT["B"],
            pygame.K_RETURN: BIT["START"], pygame.K_BACKSPACE: BIT["SELECT"],
            pygame.K_UP: BIT["UP"], pygame.K_DOWN: BIT["DOWN"],
            pygame.K_LEFT: BIT["LEFT"], pygame.K_RIGHT: BIT["RIGHT"],
            pygame.K_a: BIT["L"], pygame.K_s: BIT["R"],
        }

    def handle_events(self):
        pg = self.pygame
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                self.running = False
            elif ev.type == pg.KEYDOWN:
                if ev.key == pg.K_ESCAPE:
                    self.running = False
                elif ev.key == pg.K_p:
                    self.paused = not self.paused
                elif ev.key == pg.K_TAB:
                    self.turbo = not self.turbo
                elif ev.key == pg.K_F12:
                    self.rec.dump("manual", _frame_png(self.img))
                    self.msg = "dumped ~%.1fs -> %s" % (
                        self.rec.history_secs(), os.path.basename(
                            sorted(os.listdir(self.args.out))[-1]))
                elif ev.key == pg.K_F11:
                    out = self.ring.dump(self.args.ram_out, "hit",
                                         _frame_png(self.img))
                    self.msg = "HIT MARKED: %.1fs RAM -> %s" % (
                        self.ring.history_secs(), os.path.basename(out or "?"))
                elif ev.key == pg.K_F8:
                    path = self.save_named_state()
                    self.msg = "saved state -> %s" % os.path.basename(path)
                elif ev.key == pg.K_F5:
                    self.quick_state = _state_bytes(self.core)
                    self.msg = "quick-saved"
                elif ev.key == pg.K_F9:
                    self.load_state(self.quick_state, "quick state")
                elif ev.key == pg.K_r:
                    self.load_state(self.seed_state, "seed state")
                elif ev.key in self.HUMAN_KEYMAP:
                    self.human_mask |= self.HUMAN_KEYMAP[ev.key]
            elif ev.type == pg.KEYUP and ev.key in self.HUMAN_KEYMAP:
                self.human_mask &= ~self.HUMAN_KEYMAP[ev.key]

    def draw(self):
        pg = self.pygame
        s = self.args.scale
        raw = ffi.buffer(self.img.buffer)
        surf = pg.image.frombuffer(raw, (GBA_W, GBA_H), "RGBX")
        surf = pg.transform.scale(surf, (GBA_W * s, GBA_H * s))
        self.screen.blit(surf, (0, 0))
        bar = pg.Rect(0, self.win_h - HUD_H, self.win_w, HUD_H)
        pg.draw.rect(self.screen, (24, 24, 24), bar)
        state = "PAUSED" if self.paused else ("TURBO" if self.turbo else "60Hz")
        hud = "%s t=%d hist=%.0fs | F11 MARK HIT  F8 save state  F12 tape  R reset  ESC quit" % (
            state, self.rec.tick_no, self.rec.history_secs())
        self.screen.blit(self.font.render(hud, True, (220, 220, 220)),
                         (6, self.win_h - HUD_H + 5))
        self.screen.blit(self.font.render(self.msg[:70], True, (255, 255, 80)), (6, 4))
        pg.display.flip()

    def run(self):
        self.init_display()
        target_dt = 1.0 / 60.0
        next_tick = time.perf_counter()
        done = 0
        while self.running:
            self.handle_events()
            if self.paused:
                self.draw()
                time.sleep(0.03)
                continue
            self.core.set_keys(raw=self.human_mask)
            self.core.run_frame()
            self.rec.record(self.human_mask)
            self.tick_no += 1
            self.ring.record(self.tick_no)
            done += 1
            self.draw()
            if not self.turbo:
                next_tick += target_dt
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
            if self.args.ticks and done >= self.args.ticks:
                self.running = False
        print("recorder exit: ticks=%d dumps=%d" % (done, self.rec.n_dumps))

    def selftest(self, n):
        """No display, no human: hold RIGHT for n ticks, dump, replay-check."""
        for _ in range(n):
            self.core.set_keys(raw=BIT["RIGHT"])
            self.core.run_frame()
            self.rec.record(BIT["RIGHT"])
        out = self.rec.dump("selftest", _frame_png(self.img))
        # verify the dump is well-formed and deterministically replayable
        with open(os.path.join(out, "inputs.json")) as f:
            masks = json.load(f)["masks"]
        c2 = mgba.core.load_path(self.args.rom)
        img2 = mgba.image.Image(*c2.desired_video_dimensions())
        c2.set_video_buffer(img2)
        c2.reset()
        with open(os.path.join(out, "prefail.state"), "rb") as f:
            assert c2.load_raw_state(f.read())
        for m in masks:
            c2.set_keys(raw=m)
            c2.run_frame()
        replay_end = _state_bytes(c2)
        live_end = _state_bytes(self.core)
        ok = replay_end == live_end
        print("selftest: %d frames, replay==live: %s" % (len(masks), ok))
        return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser(description="Solo ALttP human-play flight recorder")
    ap.add_argument("--rom", default=os.environ.get("FOUR_SWORDS_ROM"))
    ap.add_argument("--state", default=DEFAULT_STATE,
                    help="seed savestate to boot from (in-game, controllable)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="flight dump root")
    ap.add_argument("--scale", type=int, default=3, help="pixel scale")
    ap.add_argument("--bank-secs", type=float, default=10.0,
                    help="seconds between rolling savestate banks")
    ap.add_argument("--banks", type=int, default=4,
                    help="banks kept; dump window is up to (banks-1)*bank_secs")
    ap.add_argument("--ticks", type=int, default=0, help="auto-quit after N ticks (test)")
    ap.add_argument("--ram-out", default=DEFAULT_RAM_OUT,
                    help="where F11 hit-marker RAM windows are written")
    ap.add_argument("--ram-every", type=int, default=2,
                    help="snapshot RAM every N ticks (ring resolution)")
    # 300 (~10 s), raised from 120 (~4 s) on 2026-08-01. The 4 s window MISSED
    # the room transitions: measured by changed-byte churn, only 1 of 6 dumps in
    # the key room actually contained a door crossing, because "press F11 after
    # the screen settles" often puts the crossing outside the window. 10 s also
    # lets several transitions land in ONE ring, which is far stronger evidence
    # than chaining separate dumps. Costs ~88 MB per dump (was ~35 MB).
    ap.add_argument("--ram-keep", type=int, default=300,
                    help="ring depth in snapshots (300 x 2 ticks = ~10s)")
    ap.add_argument("--state-out", default=DEFAULT_STATE_OUT,
                    help="where F8 writes named savestates")
    ap.add_argument("--state-name", default="alttp_human",
                    help="stem for F8 savestates (auto-numbered)")
    ap.add_argument("--selftest", type=int, default=0, metavar="N",
                    help="headless: run N ticks holding RIGHT, dump, replay-verify")
    args = ap.parse_args()
    if not args.rom or not os.path.exists(args.rom):
        print("ROM not found - source env.sh or pass --rom")
        return 1
    if not os.path.exists(args.state):
        print("seed state not found: %s" % args.state)
        return 1
    if args.selftest:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    r = Recorder(args)
    if args.selftest:
        return r.selftest(args.selftest)
    r.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
