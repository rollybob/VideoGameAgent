"""
Deterministic, single-threaded GBA multiplayer (SIO_MULTI) link coordinator.

Unlike mGBA's built-in lockstep (multi-threaded, races on the readiness handshake and
hangs 4-player Four Swords -- see PLAN.md), this drives all N cores from ONE thread and
performs the multi-transfer synchronously. Readiness is decided by inspecting all cores
between steps, so the flapping race that breaks the GUI cannot occur.

Status: v0 -- transfer state machine + synchronous ready management. Not yet cycle-accurate
(Stage 3). Validated primitives in probes/test_plumbing.py.

Run the built-in smoke test:
    source env.sh ; "$LINK_PY" link_engine.py
"""
import os
import sys

import mgba.core
import mgba.gba
import mgba.image
import mgba.log
from mgba._pylib import ffi, lib

mgba.log.silence()

# --- GBA hardware IO register offsets (C macros; not exposed as cffi constants) ---
REG_SIOMULTI0 = 0x120
REG_SIOMULTI1 = 0x122
REG_SIOMULTI2 = 0x124
REG_SIOMULTI3 = 0x126
REG_SIOCNT    = 0x128
REG_SIOMLT_SEND = 0x12A
REG_RCNT      = 0x134
SIOCNT_START  = 0x0080   # bit 7: master start / busy

MAX_GBAS = 4


def io_read(core, reg):
    return core._native.memory.io[reg >> 1] & 0xFFFF


def io_write(core, reg, val):
    core._native.memory.io[reg >> 1] = val & 0xFFFF


class _CaptureDriver(mgba.gba.GBASIODriver):
    """Attached per-core in SIO_MULTI. The binding dispatches to camelCase writeRegister.
    We only observe here; the actual transfer is executed by LinkSession from its loop."""
    def __init__(self, node):
        super().__init__()
        self.node = node

    def writeRegister(self, address, value):
        # Record the last SIOCNT the game wrote (esp. the start bit on the master).
        if address == REG_SIOCNT:
            self.node.last_siocnt_write = value
        if self.node.trace is not None:
            self.node.trace.append((self.node.session.tick_no, address, value))
        return value


class _Node:
    def __init__(self, index, core, image, session):
        self.index = index
        self.core = core
        self.image = image
        self.session = session
        self.last_siocnt_write = None
        # trace: list of (tick, address, value) for every SIO reg write the game
        # makes (routed through the driver), or None when tracing is off.
        self.trace = [] if session.tracing else None
        self.driver = _CaptureDriver(self)
        core.attach_sio(self.driver, lib.SIO_MULTI)

    @property
    def mode(self):
        return self.core._native.sio.mode

    @property
    def in_multi(self):
        return self.core._native.sio.mode == lib.SIO_MULTI

    @property
    def siocnt(self):
        return self.core._native.sio.siocnt & 0xFFFF

    @siocnt.setter
    def siocnt(self, v):
        self.core._native.sio.siocnt = v & 0xFFFF


class LinkSession:
    def __init__(self, rom, n=2, state_path=None, trace=False):
        """state_path: None, a single path (same state cloned into every core),
        or a list of n paths (distinct state per core -- e.g. unique saves).
        trace=True records every SIO register write per node (node.trace) and
        every transfer's exchanged data (self.transfer_log)."""
        assert 2 <= n <= MAX_GBAS
        self.tracing = trace
        self.tick_no = 0
        self.transfer_log = [] if trace else None
        if isinstance(state_path, (list, tuple)):
            assert len(state_path) == n, "need one state per core"
            state_paths = list(state_path)
        else:
            state_paths = [state_path] * n
        states = []
        for p in state_paths:
            if p is None:
                states.append(None)
            else:
                with open(p, "rb") as f:
                    states.append(f.read())
        self.nodes = []
        for i in range(n):
            core = mgba.core.load_path(rom)
            if core is None:
                raise RuntimeError("could not load ROM: %s" % rom)
            w, h = core.desired_video_dimensions()
            img = mgba.image.Image(w, h)
            core.set_video_buffer(img)   # before reset (known-good order)
            core.reset()
            if states[i] is not None:
                ok = core.load_raw_state(states[i])
                if not ok:
                    raise RuntimeError("load_raw_state failed on core %d" % i)
            self.nodes.append(_Node(i, core, img, self))   # attaches SIO driver after state load
        self.transfers = 0

    # --- the deterministic ready + transfer logic (the anti-race core) ---
    def _manage_ready(self):
        """Set every MULTI-mode core's READY bit synchronously. All-ready iff every
        attached core is in MULTI mode. Single pass -> no flapping."""
        multi = [nd for nd in self.nodes if nd.in_multi]
        all_ready = len(multi) == len(self.nodes)
        for nd in multi:
            s = nd.siocnt
            s = lib.GBASIOMultiplayerSetReady(s, 1 if all_ready else 0)
            # player 0 is master (not slave); others are slaves
            s = lib.GBASIOMultiplayerSetSlave(s, 1 if nd.index > 0 else 0)
            nd.siocnt = s
        return multi, all_ready

    def _do_transfer(self, participants):
        """Synchronous multiplayer exchange: gather each core's SIOMLT_SEND into
        SIOMULTI0..3 on ALL cores, assign ids, clear busy, raise SIO IRQ if enabled."""
        send = [0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF]
        for i, nd in enumerate(participants):
            send[i] = io_read(nd.core, REG_SIOMLT_SEND)
        for pid, nd in enumerate(participants):
            io_write(nd.core, REG_SIOMULTI0, send[0])
            io_write(nd.core, REG_SIOMULTI1, send[1])
            io_write(nd.core, REG_SIOMULTI2, send[2])
            io_write(nd.core, REG_SIOMULTI3, send[3])
            nd.core._native.sio.rcnt |= 1
            s = nd.siocnt
            s = lib.GBASIOMultiplayerClearBusy(s)
            s = lib.GBASIOMultiplayerSetId(s, pid)
            nd.siocnt = s
            if lib.GBASIOMultiplayerIsIrq(s):
                lib.GBARaiseIRQ(nd.core._native, lib.GBA_IRQ_SIO, 0)
        self.transfers += 1
        if self.transfer_log is not None:
            self.transfer_log.append((self.tick_no, tuple(send)))

    def tick(self):
        """Advance one coordination step: run each core a frame, then reconcile SIO."""
        self.tick_no += 1
        for nd in self.nodes:
            nd.core.run_frame()
        multi, all_ready = self._manage_ready()
        # Master (player 0) drives the clock. If it is in MULTI, has set the start bit,
        # and everyone is ready, perform the exchange.
        if multi and multi[0].index == 0 and all_ready:
            master = multi[0]
            if master.siocnt & SIOCNT_START:
                self._do_transfer(multi)

    def press_all(self, *keys, **kwargs):
        for nd in self.nodes:
            nd.core.set_keys(*keys, **kwargs)

    def framebuffer_nonzero(self, i):
        b = bytes(mgba.image.ffi.buffer(self.nodes[i].image.buffer))
        return sum(1 for x in b if x)


def _smoke_test():
    rom = os.environ.get("FOUR_SWORDS_ROM")
    if not rom or not os.path.exists(rom):
        print("set FOUR_SWORDS_ROM (source env.sh)"); return 1
    print("booting 2-core LinkSession...")
    sess = LinkSession(rom, n=2)

    KEY_START = mgba.gba.GBA.KEY_START
    KEY_A = mgba.gba.GBA.KEY_A
    seen_multi = [False, False]
    for frame in range(1800):   # ~30s emulated
        # nudge menus: tap Start/A every ~30 frames to try to reach the link screen
        if frame % 30 == 0:
            sess.press_all(KEY_START if (frame // 30) % 2 else KEY_A)
        else:
            sess.press_all(raw=0)
        sess.tick()
        for i, nd in enumerate(sess.nodes):
            if nd.in_multi and not seen_multi[i]:
                seen_multi[i] = True
                print("  frame %4d: core %d entered SIO_MULTI (siocnt=%04X)"
                      % (frame, i, nd.siocnt))
    print("done. transfers performed: %d" % sess.transfers)
    print("entered MULTI: core0=%s core1=%s" % (seen_multi[0], seen_multi[1]))
    print("framebuffer nonzero: core0=%d core1=%d"
          % (sess.framebuffer_nonzero(0), sess.framebuffer_nonzero(1)))
    return 0


if __name__ == "__main__":
    sys.exit(_smoke_test())
