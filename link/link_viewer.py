"""Four Swords 4-player viewer + input harness (Stage 4, 2026-07-07).

One real-time conductor loop around LinkSession (the link engine is already
the game loop for all 4 cores): tick at 60 Hz, blit the 4 in-memory
framebuffers into one pygame window (2x2 grid), and each tick apply whatever
each slot's controller currently wants via set_keys().

Controllers per slot: HUMAN (pygame keyboard), WANDER (scripted random-walk
bot, also the full-speed harness test), VLM (async client thread against the
local serve_vlm.py /act -- decisions take seconds, so agents write INTENTS
(key mask + duration segments) into a mailbox the 60 Hz loop consumes), or
IDLE. Spectate-all vs play-one-slot is the same program: the human slot is
just a controller assignment, switchable at runtime.

THREADING RULE: mGBA cores are not thread-safe. ONLY the conductor thread
touches cores/images. VLM threads get PNG copies the conductor encodes on
request (want_frame flag), and only write to the intent mailbox.

Hotkeys:
  1-4     take that slot as human (previous human slot reverts to --agents)
  0       release human slot -> spectate all agents
  V       flip agent slots between VLM brain and wander bots, live
  F5/F9   save/load all 4 states (p*_viewer.state bank)
  P       pause    TAB  turbo (no 60 Hz sleep)    ESC  quit
Human GBA keys (VGA convention): x=A z=B enter=Start backspace=Select
  arrows=dpad a=L s=R

Run (needs the real desktop): DISPLAY=:1, source env.sh, "$LINK_PY" link_viewer.py
Or double-click the Four Swords 4P icon on the desktop (launch_viewer.sh).
"""
import argparse
import base64
import collections
import io
import json
import os
import random
import sys
import threading
import time
import urllib.request

import mgba.gba as gba
from mgba._pylib import ffi
from link_engine import LinkSession
from ram_ring import RamRing

K = gba.GBA
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "sessions", "checkpoints")


def bank_dir(name):
    """Where the 4 states for bank `name` live.

    Original banks (coop/stage_select/link_wait) sit together in checkpoints/;
    the per-stage banks captured for Task 09 A1 each got their own
    checkpoints_<stage>/ dir. Accept both so --start works for either.
    """
    per_stage = os.path.join(HERE, "sessions", "checkpoints_" + name)
    if os.path.isdir(per_stage):
        return per_stage
    return CKPT


def resolve_bank(name):
    paths = [os.path.join(bank_dir(name), "p%d_%s.state" % (i, name))
             for i in range(4)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("checkpoint bank '%s' incomplete, missing:\n  %s\n"
                         "available banks: %s" % (
                             name, "\n  ".join(missing), ", ".join(list_banks())))
    return paths


def list_banks():
    names = set()
    sess = os.path.join(HERE, "sessions")
    for d in os.listdir(sess):
        full = os.path.join(sess, d)
        if not os.path.isdir(full) or not d.startswith("checkpoints"):
            continue
        for f in os.listdir(full):
            if f.startswith("p0_") and f.endswith(".state"):
                names.add(f[3:-len(".state")])
    return sorted(names)
GBA_W, GBA_H = 240, 160
HUD_H = 28

BIT = {
    "A": 1 << K.KEY_A, "B": 1 << K.KEY_B,
    "select": 1 << K.KEY_SELECT, "start": 1 << K.KEY_START,
    "right": 1 << K.KEY_RIGHT, "left": 1 << K.KEY_LEFT,
    "up": 1 << K.KEY_UP, "down": 1 << K.KEY_DOWN,
    "R": 1 << K.KEY_R, "L": 1 << K.KEY_L,
}
DIRS = ["up", "down", "left", "right"]

VLM_GOAL = ("You are one of four Links playing Zelda Four Swords co-op. Explore the "
            "stage, collect rupees and keys, push toward the exit, and stay near the "
            "other Links so you can help on switches and heavy objects.")


class Intent:
    """Mailbox one controller writes and the conductor consumes: a queue of
    (key mask, ticks) segments. Guarded by lock -- VLM threads write async."""
    def __init__(self):
        self.lock = threading.Lock()
        self.segments = collections.deque()

    def put(self, segments):
        with self.lock:
            self.segments.clear()
            self.segments.extend(segments)

    def current_mask(self):
        """Mask for this tick; consumes one tick from the head segment."""
        with self.lock:
            while self.segments:
                mask, ticks = self.segments[0]
                if ticks <= 0:
                    self.segments.popleft()
                    continue
                self.segments[0] = (mask, ticks - 1)
                return mask
            return 0

    def empty(self):
        with self.lock:
            return not self.segments


def wander_segments(rng):
    """Random-walk bot: mostly walk a direction, sometimes tap A/B, rest a bit."""
    r = rng.random()
    if r < 0.65:
        return [(BIT[rng.choice(DIRS)], rng.randint(20, 50))]
    if r < 0.80:
        return [(BIT["A"], 4), (0, 8)]
    if r < 0.88:
        return [(BIT["B"], 4), (0, 8)]
    return [(0, rng.randint(10, 25))]


def action_to_segments(button, repeats):
    """Map a /act decision to intent segments. Directions are HELD (repeats
    scales the walk), taps are pressed repeats times with release gaps."""
    if button in DIRS:
        return [(BIT[button], 12 * max(1, repeats))]
    if button in BIT:
        segs = []
        for _ in range(max(1, repeats)):
            segs += [(BIT[button], 4), (0, 8)]
        return segs
    return [(0, 20)]   # wait / unparsed


class VlmClient(threading.Thread):
    """One per VLM slot. Loop: request a frame from the conductor, POST /act,
    write intent segments. Never touches cores. Daemon: dies with the viewer."""
    def __init__(self, viewer, slot, url):
        super().__init__(daemon=True, name="vlm-slot%d" % slot)
        self.viewer = viewer
        self.slot = slot
        self.url = url.rstrip("/") + "/act"
        self.step = 0
        self.last_action = ""
        self.latency = None      # HUD
        self.error = None        # HUD

    def run(self):
        v = self.viewer
        while v.running:
            if v.controllers[self.slot] != "vlm":
                time.sleep(0.3)
                continue
            png = v.request_frame(self.slot)
            if png is None:      # viewer quitting or paused too long
                continue
            body = json.dumps({
                "image_b64": base64.b64encode(png).decode(),
                "goal": VLM_GOAL,
                "step": self.step,
                "last_action": self.last_action,
                "last_changed": True,
                "seed_salt": "slot%d" % self.slot,
            }).encode()
            try:
                req = urllib.request.Request(
                    self.url, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    action = json.loads(resp.read().decode())
                self.error = None
            except Exception as exc:   # server down/timeout: idle briefly, retry
                self.error = type(exc).__name__
                self.latency = None
                time.sleep(3.0)
                continue
            self.latency = action.get("latency_s")
            btn = action.get("button", "wait")
            self.last_action = btn
            self.step += 1
            v.intents[self.slot].put(action_to_segments(btn, action.get("repeats", 1)))
            # wait for the intent to mostly play out before deciding again
            while v.running and not v.intents[self.slot].empty():
                time.sleep(0.1)


class FlightRecorder:
    """Always-on black box for diagnosing live link deaths (built for the
    2026-07-07 warp-pad communication error). Keeps rolling 10 s savestate
    banks + every slot's input mask every tick; when the transfer rate
    collapses (healthy 13/tick -> ~0) it dumps: the oldest kept bank (10-20 s
    before death), the input tape from that bank onward, SIO trace tails,
    mode/refusal diagnostics, and death screenshots. link_replay.py replays
    the dump deterministically headless. F12 forces a dump manually."""
    BANK_TICKS = 600
    WINDOW = 60
    POST_DEATH_TICKS = 120

    def __init__(self, sess, out_root):
        self.sess = sess
        self.out_root = out_root
        self.banks = collections.deque(maxlen=3)   # (start_tick, transfers, [state bytes]*4)
        self.masks = []                            # per tick since oldest kept bank
        self.win_start_transfers = sess.transfers
        self.win_ticks = 0
        self.healthy = False
        self.death_tick = None
        self.dumped = False
        self._bank()

    def _bank(self):
        states = []
        for nd in self.sess.nodes:
            st = nd.core.save_raw_state()
            states.append(bytes(ffi.buffer(st)))
        self.banks.append((self.sess.tick_no, self.sess.transfers, states))
        # keep the input tape aligned with the OLDEST kept bank
        excess = len(self.masks) - (self.sess.tick_no - self.banks[0][0])
        if excess > 0:
            del self.masks[:excess]

    def record(self, masks):
        """Call once per tick, after sess.tick(), with the 4 applied masks."""
        if self.dumped:
            return
        self.masks.append(list(masks))
        sess = self.sess
        if self.death_tick is None and sess.tick_no % self.BANK_TICKS == 0:
            self._bank()
        self.win_ticks += 1
        if self.win_ticks >= self.WINDOW:
            rate = (sess.transfers - self.win_start_transfers) / float(self.win_ticks)
            if rate >= 8:
                self.healthy = True
            elif self.healthy and rate < 1 and self.death_tick is None:
                self.death_tick = sess.tick_no
                print("flight: link death detected at tick %d" % sess.tick_no, flush=True)
            self.win_start_transfers = sess.transfers
            self.win_ticks = 0
        if (self.death_tick is not None
                and sess.tick_no >= self.death_tick + self.POST_DEATH_TICKS):
            self.dump("auto")

    def dump(self, reason):
        if self.dumped:
            return
        self.dumped = True
        sess = self.sess
        out = os.path.join(self.out_root,
                           time.strftime("flight-%Y%m%d-%H%M%S") + "-" + reason)
        os.makedirs(out, exist_ok=True)
        bank_tick, bank_transfers, states = self.banks[0]
        for i, st in enumerate(states):
            with open(os.path.join(out, "p%d_prefail.state" % i), "wb") as f:
                f.write(st)
        tape = self.masks[-(sess.tick_no - bank_tick):] if sess.tick_no > bank_tick else []
        meta = {
            "reason": reason,
            "bank_tick": bank_tick,
            "bank_transfers": bank_transfers,
            "death_tick": self.death_tick,
            "end_tick": sess.tick_no,
            "end_transfers": sess.transfers,
            "mode_log": sess.mode_log[-200:],
            "refused_starts": sess.refused_starts[-200:],
        }
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        with open(os.path.join(out, "inputs.json"), "w") as f:
            json.dump({"start_tick": bank_tick, "masks": tape}, f)
        with open(os.path.join(out, "trace_tail.txt"), "w") as f:
            for nd in sess.nodes:
                t = list(nd.trace or [])[-2000:]
                f.write("== node %d writes ==\n" % nd.index)
                for tick, addr, val in t:
                    f.write("%d %03X %04X\n" % (tick, addr, val))
            f.write("== transfers ==\n")
            for tick, words in list(sess.transfer_log or [])[-400:]:
                f.write("%d %s\n" % (tick, " ".join("%04X" % w for w in words)))
        for nd in sess.nodes:
            buf = type("KeepOpen", (io.BytesIO,), {"close": lambda self: None})()
            nd.image.save_png(buf)
            with open(os.path.join(out, "c%d_death.png" % nd.index), "wb") as f:
                f.write(buf.getvalue())
        print("flight: dumped %s" % out, flush=True)


class Viewer:
    def __init__(self, args):
        import pygame   # after DISPLAY is settled
        self.pygame = pygame
        self.args = args
        states = resolve_bank(args.start)
        self.sess = LinkSession(args.rom, n=4, state_path=states, trace=True)
        # bound the always-on traces so long sessions don't grow unbounded
        for nd in self.sess.nodes:
            nd.trace = collections.deque(nd.trace, maxlen=100000)
        self.sess.transfer_log = collections.deque(self.sess.transfer_log, maxlen=50000)
        self.rearm_irq()
        self.flight = FlightRecorder(self.sess, os.path.join(HERE, "sessions", "flight"))
        # Node 0 mirrors the whole lockstep world (same reason FsOracle reads all
        # 4 players' health off core 0), so one ring covers every player.
        self.ring = RamRing(self.sess.nodes[0].core, every=args.ram_every,
                            keep=args.ram_keep, label="fs_" + args.start)
        self._msg = ""
        self.msg_t = 0.0
        self.running = True
        self.paused = False
        self.turbo = False
        self.human_slot = args.human          # None or 0..3
        self.human_mask = 0
        base = args.agents                    # controller for non-human slots
        self.controllers = [base] * 4
        if self.human_slot is not None:
            self.controllers[self.human_slot] = "human"
        self.intents = [Intent() for _ in range(4)]
        self.rngs = [random.Random(1234 + i) for i in range(4)]
        # frame hand-off to VLM threads (conductor encodes, clients wait)
        self.frame_lock = threading.Lock()
        self.want_frame = [False] * 4
        self.frame_png = [None] * 4           # (serial, png bytes)
        self.frame_serial = [0] * 4
        # one client per slot regardless of start mode -- they idle unless their
        # slot's controller is "vlm", so runtime switching needs no plumbing
        self.clients = [VlmClient(self, i, args.vlm_url) for i in range(4)]
        # display
        s = args.scale
        self.win_w, self.win_h = GBA_W * 2 * s, GBA_H * 2 * s + HUD_H
        pygame.init()
        pygame.display.set_caption("Four Swords 4P - VGA link engine")
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.font = pygame.font.SysFont("monospace", 15, bold=True)
        # pacing stats
        self.tps = 0.0
        self.trps = 0
        self._stat_t0 = time.perf_counter()
        self._stat_ticks = 0
        self._stat_transfers = self.sess.transfers

    def node_png(self, slot):
        """PNG bytes of one node's current frame (visual anchor for a RAM dump)."""
        nd = self.sess.nodes[slot]
        buf = type("KeepOpen", (io.BytesIO,), {"close": lambda self: None})()
        nd.image.save_png(buf)
        return buf.getvalue()

    @property
    def msg(self):
        return self._msg

    @msg.setter
    def msg(self, value):
        self._msg = value
        self.msg_t = time.perf_counter()

    # ---- conductor-side services -------------------------------------------
    def rearm_irq(self):
        """After any state load: the savestate keeps a raised-but-unserviced
        SIO IRQ (IF bit), but the session bookkeeping does not -- re-arm it."""
        for nd in self.sess.nodes:
            if nd.index > 0 and nd.irq_flagged:
                nd.irq_pending = True
                nd.mltsend_seen = False

    def request_frame(self, slot, timeout=5.0):
        """Called from VLM threads: ask the conductor for a fresh PNG."""
        with self.frame_lock:
            serial = self.frame_serial[slot]
            self.want_frame[slot] = True
        t0 = time.time()
        while self.running and time.time() - t0 < timeout:
            with self.frame_lock:
                if self.frame_serial[slot] > serial:
                    return self.frame_png[slot]
            time.sleep(0.02)
        return None

    def fulfill_frames(self):
        """Conductor, at frame boundary: encode PNGs for waiting clients."""
        with self.frame_lock:
            wanted = [i for i in range(4) if self.want_frame[i]]
        for i in wanted:
            # mgba's save_png closes the fileobj when done -- keep ours readable
            buf = type("KeepOpen", (io.BytesIO,), {"close": lambda self: None})()
            self.sess.nodes[i].image.save_png(buf)
            with self.frame_lock:
                self.frame_png[i] = buf.getvalue()
                self.frame_serial[i] += 1
                self.want_frame[i] = False

    def set_human(self, slot):
        if self.human_slot is not None:
            self.controllers[self.human_slot] = self.args.agents
        self.human_slot = slot
        self.human_mask = 0
        if slot is not None:
            self.controllers[slot] = "human"
            self.intents[slot].put([])

    def save_bank(self):
        """F5: bank all 4 cores under --save-name, in its own checkpoints_<name>/.

        Writing to a per-stage dir (not the shared checkpoints/) means the bank
        round-trips straight back through --start <name>, which is how new
        stage start-points get handed to the trainer.
        """
        name = self.args.save_name
        d = os.path.join(HERE, "sessions", "checkpoints_" + name)
        os.makedirs(d, exist_ok=True)
        for i, nd in enumerate(self.sess.nodes):
            st = nd.core.save_raw_state()
            with open(os.path.join(d, "p%d_%s.state" % (i, name)), "wb") as f:
                f.write(bytes(ffi.buffer(st)))
        self.msg = "banked -> checkpoints_%s/ (reload: --start %s)" % (name, name)
        print("viewer: banked 4 states -> %s" % d, flush=True)

    def load_bank(self):
        name = self.args.save_name
        paths = [os.path.join(bank_dir(name), "p%d_%s.state" % (i, name))
                 for i in range(4)]
        if not all(os.path.exists(p) for p in paths):
            return
        for i, nd in enumerate(self.sess.nodes):
            with open(paths[i], "rb") as f:
                nd.core.load_raw_state(f.read())
            nd.irq_pending = False
            nd.mltsend_seen = False
        self.rearm_irq()

    # ---- input --------------------------------------------------------------
    HUMAN_KEYMAP = None   # built in run() (needs pygame constants)

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
                elif ev.key == pg.K_F5:
                    self.save_bank()
                elif ev.key == pg.K_F9:
                    self.load_bank()
                elif ev.key == pg.K_F11:
                    # anchor on the human's own screen -- that is the player
                    # whose hearts just changed
                    out = self.ring.dump(self.args.ram_out, "hit",
                                         self.node_png(self.human_slot or 0),
                                         note="human_slot=%s" % self.human_slot)
                    self.msg = "HIT MARKED: %.1fs RAM -> %s" % (
                        self.ring.history_secs(), os.path.basename(out or "?"))
                elif ev.key == pg.K_F12:
                    self.flight.dump("manual")
                elif ev.key == pg.K_v:
                    # flip all agent slots between the real VLM brain (slow,
                    # ~10s/decision) and lively wander bots, live
                    self.args.agents = "wander" if self.args.agents == "vlm" else "vlm"
                    for i in range(4):
                        if self.controllers[i] != "human":
                            self.controllers[i] = self.args.agents
                            self.intents[i].put([])
                elif ev.key == pg.K_0:
                    self.set_human(None)
                elif ev.key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4):
                    self.set_human(ev.key - pg.K_1)
                elif ev.key in self.HUMAN_KEYMAP:
                    self.human_mask |= self.HUMAN_KEYMAP[ev.key]
            elif ev.type == pg.KEYUP and ev.key in self.HUMAN_KEYMAP:
                self.human_mask &= ~self.HUMAN_KEYMAP[ev.key]

    def slot_mask(self, i):
        c = self.controllers[i]
        if c == "human":
            return self.human_mask
        if c == "wander":
            if self.intents[i].empty():
                self.intents[i].put(wander_segments(self.rngs[i]))
            return self.intents[i].current_mask()
        if c == "vlm":
            return self.intents[i].current_mask()
        return 0

    # ---- render ---------------------------------------------------------------
    def draw(self):
        pg = self.pygame
        s = self.args.scale
        for i, nd in enumerate(self.sess.nodes):
            raw = ffi.buffer(nd.image.buffer)
            surf = pg.image.frombuffer(raw, (GBA_W, GBA_H), "RGBX")
            surf = pg.transform.scale(surf, (GBA_W * s, GBA_H * s))
            x, y = (i % 2) * GBA_W * s, (i // 2) * GBA_H * s
            self.screen.blit(surf, (x, y))
            tag = "P%d %s" % (i + 1, self.controllers[i].upper())
            c = self.clients[i] if i < len(self.clients) else None
            if self.controllers[i] == "vlm" and c is not None:
                if c.error:
                    tag += " !" + c.error
                elif c.latency is not None:
                    tag += " %.1fs" % c.latency
            self.screen.blit(self.font.render(tag, True, (255, 255, 80)), (x + 6, y + 4))
        hud = ("%s  %5.1f t/s  %5d tr/s   [1-4]slot [0]spectate [V]vlm/bots "
               "[F11]MARK HIT [F5]bank [F9]load [P]pause [TAB]turbo [ESC]quit") % (
            "PAUSED" if self.paused else ("TURBO " if self.turbo else "60Hz  "),
            self.tps, self.trps)
        # a fresh action message takes over the bar briefly, then the legend returns
        if self.msg and time.perf_counter() - self.msg_t < 6.0:
            hud = self.msg
        bar = self.pygame.Rect(0, self.win_h - HUD_H, self.win_w, HUD_H)
        self.pygame.draw.rect(self.screen, (24, 24, 24), bar)
        self.screen.blit(self.font.render(hud, True, (220, 220, 220)),
                         (8, self.win_h - HUD_H + 6))
        pg.display.flip()

    # ---- main loop --------------------------------------------------------------
    def run(self):
        pg = self.pygame
        self.HUMAN_KEYMAP = {
            pg.K_x: BIT["A"], pg.K_z: BIT["B"],
            pg.K_RETURN: BIT["start"], pg.K_BACKSPACE: BIT["select"],
            pg.K_UP: BIT["up"], pg.K_DOWN: BIT["down"],
            pg.K_LEFT: BIT["left"], pg.K_RIGHT: BIT["right"],
            pg.K_a: BIT["L"], pg.K_s: BIT["R"],
        }
        for c in self.clients:
            c.start()
        target_dt = 1.0 / 60.0
        next_tick = time.perf_counter()
        ticks_done = 0
        t_tick = t_draw = 0.0
        while self.running:
            self.handle_events()
            if self.paused:
                time.sleep(0.05)
            if not self.paused:
                t0 = time.perf_counter()
                masks = [self.slot_mask(i) for i in range(4)]
                for i in range(4):
                    self.sess.nodes[i].core.set_keys(raw=masks[i])
                self.sess.tick()
                t_tick += time.perf_counter() - t0
                ticks_done += 1
                self._stat_ticks += 1
                self.fulfill_frames()
                self.flight.record(masks)
                self.ring.record(self.sess.tick_no)
                if self.args.flight_at and ticks_done == self.args.flight_at:
                    self.flight.dump("forced")
            t0 = time.perf_counter()
            self.draw()
            t_draw += time.perf_counter() - t0
            now = time.perf_counter()
            if now - self._stat_t0 >= 1.0:
                dt = now - self._stat_t0
                self.tps = self._stat_ticks / dt
                self.trps = int((self.sess.transfers - self._stat_transfers) / dt)
                self._stat_t0 = now
                self._stat_ticks = 0
                self._stat_transfers = self.sess.transfers
            if not self.turbo and not self.paused:
                next_tick += target_dt
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()   # behind: don't spiral
            else:
                next_tick = time.perf_counter()
            if self.args.ticks and ticks_done >= self.args.ticks:
                self.running = False
        if self.args.dump:
            pg.image.save(self.screen, self.args.dump)
            print("dumped final screen to", self.args.dump)
        print("viewer exit: ticks=%d transfers=%d" % (ticks_done, self.sess.transfers))
        if ticks_done:
            print("timing: tick %.2f ms/f  draw %.2f ms/f"
                  % (1000 * t_tick / ticks_done, 1000 * t_draw / ticks_done))
        for c in self.clients:
            if c.step or c.error:
                print("vlm slot %d: %d decisions, last=%r, latency=%s, error=%s"
                      % (c.slot, c.step, c.last_action, c.latency, c.error))


def main():
    ap = argparse.ArgumentParser(description="Four Swords 4-player viewer")
    ap.add_argument("--start", default="coop",
                    help="checkpoint bank to boot from (default: straight into "
                         "co-op). Any bank under sessions/checkpoints* works, "
                         "e.g. seaoftrees / taluscave / deathmountain.")
    ap.add_argument("--list-banks", action="store_true",
                    help="print available checkpoint banks and exit")
    ap.add_argument("--save-name", default="viewer",
                    help="stem F5 writes to: checkpoints_<name>/p{0-3}_<name>.state")
    ap.add_argument("--scale", type=int, default=3, help="pixel scale per GBA screen")
    ap.add_argument("--human", default="none",
                    help="slot 1-4 the human starts on, or 'none' to spectate")
    ap.add_argument("--agents", default="wander", choices=["wander", "vlm", "idle"],
                    help="controller for non-human slots")
    ap.add_argument("--vlm-url", default="http://127.0.0.1:8077")
    ap.add_argument("--rom", default=os.environ.get("FOUR_SWORDS_ROM"))
    ap.add_argument("--ticks", type=int, default=0, help="auto-quit after N ticks (test)")
    ap.add_argument("--dump", default="", help="save final composited screen PNG (test)")
    ap.add_argument("--flight-at", type=int, default=0,
                    help="force a flight-recorder dump at tick N (pipeline test)")
    ap.add_argument("--ram-out", default=os.path.join(HERE, "sessions", "ramhits_fs"),
                    help="where F11 hit-marker RAM windows are written")
    ap.add_argument("--ram-every", type=int, default=2,
                    help="snapshot RAM every N ticks (ring resolution)")
    # Kept in step with solo_recorder.py (see the note there): 4 s was too short
    # to reliably contain the event the human was marking.
    ap.add_argument("--ram-keep", type=int, default=300,
                    help="ring depth in snapshots (300 x 2 ticks = ~10s)")
    args = ap.parse_args()
    if args.list_banks:
        print("checkpoint banks: %s" % ", ".join(list_banks()))
        return 0
    if not args.rom or not os.path.exists(args.rom):
        print("ROM not found - source env.sh or pass --rom"); return 1
    args.human = None if args.human in ("none", "") else max(0, min(3, int(args.human) - 1))
    Viewer(args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
