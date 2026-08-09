"""
Deterministic, single-threaded GBA multiplayer (SIO_MULTI) link coordinator.

v1 (Stage 3, 2026-07-07) -- callback-driven transfers. The 2026-07-06 trace
(sessions/trace_0707.txt) showed Four Swords sends a BURST of ~13 transfers per
frame (MLTSEND word + SIOCNT start bit each), while v0 executed one exchange
per frame at the frame boundary with the stale final MLTSEND (0000) -- so the
slaves' IRQ handlers saw garbage and never responded.

v1 therefore executes the exchange INSIDE the driver's writeRegister callback,
at the moment the master sets the start bit (the mGBA C lockstep node initiates
from the same place, src/gba/sio/lockstep.c:GBASIOLockstepNodeMultiWriteRegister):

  master writes SIOCNT|0x80
    -> service slaves: single-step any slave with a pending SIO IRQ until its
       handler stages a response (observed as its own MLTSEND write) or a step
       budget runs out  [word k's response rides word k+1, like hardware]
    -> gather every core's SIOMLT_SEND, write SIOMULTI0..3 on all cores,
       raise SIO IRQ on slaves that enabled it
    -> return the SIOCNT value with busy cleared (instant complete) and the
       read-only hardware bits (SI/ready/ID) merged in, per the C node's
       `value &= 0xFF83; value |= siocnt & 0x00FC`

Hardware bits are set once at init and re-merged on the game's own SIOCNT
writes -- v0's blanket per-tick siocnt rewrite (which fought the game) is gone.
Unlike mGBA's threaded lockstep this is one thread and one deterministic order,
so the #3286 readiness race cannot occur.

Run the built-in smoke test:
    source env.sh ; "$LINK_PY" link_engine.py
"""
import atexit
import os
import sys

import mgba.core
import mgba.gba
import mgba.image
import mgba.log
from mgba._pylib import ffi, lib

mgba.log.silence()

# --- GBA hardware IO register offsets (C macros; not exposed as cffi constants) ---
REG_IF        = 0x202
REG_SIOMULTI0 = 0x120
REG_SIOMULTI1 = 0x122
REG_SIOMULTI2 = 0x124
REG_SIOMULTI3 = 0x126
REG_SIOCNT    = 0x128
REG_SIOMLT_SEND = 0x12A
REG_RCNT      = 0x134
SIOCNT_START  = 0x0080   # bit 7: master start / busy
IRQ_SIO_BIT   = 0x0080   # bit 7 of IE/IF

MAX_GBAS = 4

# Slave IRQ-service budgets (single-step counts). A GBA SIO handler is a few
# hundred to a few thousand instructions; the cap only bounds pathological
# cases (handler never writes MLTSEND, IME off, ...).
SERVICE_STEP_CAP = 30000
SERVICE_ACK_GRACE = 2000   # extra steps after IF ack if no MLTSEND write appears
# Hard bound on WASTED service CPU per slave per tick. WHY (warp-pad
# comm-error postmortem, 2026-07-07): when a slave PARKS with its SIO IRQ
# masked for a long stretch (Four Swords does during the warp cutscene), the
# handler never acks and per-transfer stepping burned the full 30k cap 13x
# per tick = 4+ emulated FRAMES of fast-forward per tick. The slaves raced
# through their wait-for-master timers at ~5x and declared a communication
# error before the master had even generated the level. But SHORT masked
# sections are routine (handlers mask SIO inside their own critical
# sections; measured mid-frame on live traffic), so a mask test cannot
# distinguish parked from busy -- productivity can: steps that end in an IRQ
# ack (handler ran) are real hardware work and stay unbounded; steps that
# never reach an ack are fast-forward and are capped here (~5% of a frame,
# so a parked slave drifts ~1.05x realtime instead of ~5x).
SERVICE_TICK_WASTE_MAX = 6000
REG_IE  = 0x200
REG_IME = 0x208


def io_read(core, reg):
    return core._native.memory.io[reg >> 1] & 0xFFFF


def io_write(core, reg, val):
    core._native.memory.io[reg >> 1] = val & 0xFFFF


class _LinkDriver(mgba.gba.GBASIODriver):
    """Attached per-core in SIO_MULTI. The binding dispatches to camelCase
    writeRegister. All transfer logic lives here (see module docstring)."""
    def __init__(self, node):
        super().__init__()
        self.node = node

    def writeRegister(self, address, value):
        node = self.node
        sess = node.session
        if node.trace is not None:
            node.trace.append((sess.tick_no, address, value))
        if address == REG_SIOMLT_SEND:
            node.mltsend_seen = True
            if node.irq_pending:
                # natural service: the handler ran during the slave's own
                # frame (not our stepping) and staged its response
                node.irq_pending = False
            if node.session.pending_start:
                node.session.try_complete_pending()
            return value
        if address == REG_SIOCNT:
            node.last_siocnt_write = value
            if value & SIOCNT_START:
                if node.index == 0 and sess.all_multi():
                    if sess.pending_start:
                        return value   # already waiting; keep busy latched
                    sess.service_slaves()
                    if any(nd.index > 0 and nd.irq_pending and not nd.mltsend_seen
                           for nd in sess.nodes):
                        # READY-HANDSHAKE (2026-07-07 warp postmortem): a
                        # slave has not consumed the previous word (parked in
                        # a masked section). Hardware masters wait on the SD
                        # line here; completing anyway made the master read
                        # stale zeros as real INPUT WORDS -- Four Swords is
                        # input-sync lockstep, so one swallowed input forked
                        # the children's simulation from the master's and the
                        # warp's all-on-pads check blew up ("communication
                        # error"). Keep busy SET; the master polls; the
                        # transfer completes when the slave catches up.
                        sess.pending_start = True
                        sess.pending_since = sess.tick_no
                        return value   # busy stays set, hw bits already live
                    sess.do_transfer()
                else:
                    # a start bit we did NOT honor -- diagnostic for protocol
                    # phases that break the master-only/all-multi assumptions
                    sess.refused_starts.append(
                        (sess.tick_no, node.index, [nd.mode for nd in sess.nodes]))
                # instant complete (or refuse: not master / not all ready)
                value &= ~SIOCNT_START
            # merge read-only hw bits like the C node (SI/SD/ID, bits 2-6)
            value = (value & 0xFF83) | (node.hw_bits & 0x7C)
        return value


class _Node:
    def __init__(self, index, core, image, session):
        self.index = index
        self.core = core
        self.image = image
        self.session = session
        self.last_siocnt_write = None
        # read-only-to-the-game siocnt bits we are authoritative for:
        # SI (bit2, set on slaves), SD/all-ready (bit3), multi ID (bits 4-5)
        self.hw_bits = (0x4 if index > 0 else 0) | 0x8 | (index << 4)
        self.irq_pending = False    # we raised SIO IRQ; handler not serviced yet
        self.mltsend_seen = False   # game wrote MLTSEND since last IRQ raise
        self.wasted_steps = 0       # unproductive service CPU this tick (bounded)
        # trace: list of (tick, address, value) for every SIO reg write the game
        # makes (routed through the driver), or None when tracing is off.
        self.trace = [] if session.tracing else None
        self.driver = _LinkDriver(self)
        core.attach_sio(self.driver, lib.SIO_MULTI)
        # apply hw bits once; afterwards they are re-merged on each SIOCNT write
        core._native.sio.siocnt = (core._native.sio.siocnt & 0xFF83) | self.hw_bits

    @property
    def mode(self):
        return self.core._native.sio.mode

    @property
    def in_multi(self):
        return self.core._native.sio.mode == lib.SIO_MULTI

    @property
    def siocnt(self):
        return self.core._native.sio.siocnt & 0xFFFF

    @property
    def irq_flagged(self):
        return bool(io_read(self.core, REG_IF) & IRQ_SIO_BIT)


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
        self.master_word = None
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
        self.service_steps_total = 0
        self.refused_starts = []    # (tick, node index, [modes]) diagnostics
        self.mode_log = []          # (tick, node index, old mode, new mode)
        self._last_modes = [nd.mode for nd in self.nodes]
        self.pending_start = False  # master start latched, waiting on a slave
        self.pending_since = 0
        self.pending_waits = 0      # completed-after-wait count (diagnostics)
        # teardown-segfault avoidance -- see shutdown()
        atexit.register(self.shutdown)

    def shutdown(self):
        """Detach every SIO driver while cores AND drivers are still alive.

        WHY (root-caused 2026-07-07): GBASIOSetDriver in C calls the OLD
        driver's unload/deinit through its function pointers. The binding
        frees a driver's trampoline struct when its Python object dies
        (ffi.gc(..., lib.free)) and GBA.__del__ re-detaches at GC time, but
        the GBA object holds no reference to the attached driver -- so when
        the driver happens to die first (cycle-GC / interpreter-exit cascade
        order is arbitrary), that detach calls through freed memory. This was
        the long-standing 'harmless' teardown segfault. After this controlled
        detach the C driver slot is NULL, so any later __del__ detach is a
        genuine no-op. atexit-registered; idempotent; call manually before
        dropping a session mid-process. The session is unusable afterwards."""
        for nd in self.nodes:
            core = getattr(nd, "core", None)
            if core is None:
                continue
            lib.GBASIOSetDriver(ffi.addressof(core._native.sio), ffi.NULL,
                                lib.SIO_MULTI)
            core._sio.discard(lib.SIO_MULTI)
            nd.driver = None
            nd.core = None
        self.nodes = []
        atexit.unregister(self.shutdown)

    def all_multi(self):
        return all(nd.in_multi for nd in self.nodes)

    def service_slaves(self):
        """Single-step each slave with a pending SIO IRQ until its handler
        stages a response (writes MLTSEND) or the budget runs out. Called
        before gathering, so word k's exchange carries responses to word k-1.

        PERF (2026-07-07): this loop dominates gameplay cost (~33k steps/tick
        in live co-op; handlers take ~850 steps there vs ~270 in menus), so it
        avoids the Python Core.step wrapper and checks flags only every 16
        steps -- prebound cffi call in a tight inner loop. Overshooting the
        MLTSEND write by <=15 instructions is harmless: the slave just runs a
        little more of its own frame (same drift the design already allows).
        This took coop from 44 t/s to (measured below) without touching
        protocol semantics: still one serviced IRQ per exchanged word."""
        for nd in self.nodes:
            if nd.index == 0 or not nd.irq_pending:
                continue
            # only the WASTE budget is bounded; productive service is not
            budget = min(SERVICE_STEP_CAP,
                         SERVICE_TICK_WASTE_MAX - nd.wasted_steps)
            if budget <= 0:
                continue   # parked slave, budget spent: leave IF latched; the
                           # handler runs naturally once the game unmasks
            io = nd.core._native.memory.io
            native = nd.core._core
            step = native.step
            if_idx = REG_IF >> 1
            steps = 0
            grace = None
            while steps < budget and not nd.mltsend_seen:
                for _ in range(16):
                    step(native)
                steps += 16
                if grace is None:
                    if not (io[if_idx] & IRQ_SIO_BIT):   # handler acked the IRQ
                        grace = SERVICE_ACK_GRACE
                else:
                    grace -= 16
                    if grace <= 0:   # acked but never staged a response
                        break
            self.service_steps_total += steps
            if grace is None and not nd.mltsend_seen:
                # never acked: the slave is parked (long IRQ mask) -- these
                # steps were fast-forward; bill them and retry next tick
                nd.wasted_steps += steps
            else:
                nd.irq_pending = False

    def try_complete_pending(self):
        """Complete a latched master transfer once every slave has consumed
        the previous word (see the READY-HANDSHAKE comment in writeRegister).
        Called from slave MLTSEND writes and from tick()."""
        if not self.pending_start:
            return
        for nd in self.nodes:
            if nd.index > 0 and nd.irq_pending and not nd.mltsend_seen:
                return
        self.pending_start = False
        self.pending_waits += 1
        self.do_transfer()
        m = self.nodes[0].core._native.sio
        m.siocnt = m.siocnt & ~SIOCNT_START   # master's poll now sees done

    def do_transfer(self):
        """Synchronous multiplayer exchange (called from the master's SIOCNT
        start-bit write): gather each core's SIOMLT_SEND into SIOMULTI0..3 on
        ALL cores, raise SIO IRQ on slaves that enabled it. The master's busy
        bit is cleared via the callback's return value, not here."""
        send = [0xFFFF] * MAX_GBAS
        for nd in self.nodes:
            send[nd.index] = io_read(nd.core, REG_SIOMLT_SEND)
        for nd in self.nodes:
            io_write(nd.core, REG_SIOMULTI0, send[0])
            io_write(nd.core, REG_SIOMULTI1, send[1])
            io_write(nd.core, REG_SIOMULTI2, send[2])
            io_write(nd.core, REG_SIOMULTI3, send[3])
            if nd.index > 0:
                if lib.GBASIOMultiplayerIsIrq(nd.siocnt):
                    lib.GBARaiseIRQ(nd.core._native, lib.GBA_IRQ_SIO, 0)
                    nd.irq_pending = True
                    nd.mltsend_seen = False
        self.transfers += 1
        if self.transfer_log is not None:
            self.transfer_log.append((self.tick_no, tuple(send)))

    def tick(self):
        """Advance one coordination step: run each core one frame. Transfers
        happen mid-frame via the master's writeRegister callback; slaves the
        callback pre-stepped simply have less of their next frame left to run."""
        self.tick_no += 1
        # reset waste budgets BEFORE any frame runs: services fire inside the
        # master's run_frame, so resetting per-node mid-loop would misbill
        for nd in self.nodes:
            nd.wasted_steps = 0
        for nd in self.nodes:
            nd.core.run_frame()
        # ack-only handlers never write MLTSEND: detect natural service by the
        # IF bit having cleared during the slave's own frame
        for nd in self.nodes:
            if nd.index > 0 and nd.irq_pending and not nd.irq_flagged:
                nd.irq_pending = False
        self.try_complete_pending()
        if self.pending_start and self.tick_no - self.pending_since > 600:
            # safety valve: never hang the whole session on one wedged slave
            self.refused_starts.append((self.tick_no, -1, ["pending-forced"]))
            self.pending_start = False
            self.do_transfer()
            m = self.nodes[0].core._native.sio
            m.siocnt = m.siocnt & ~SIOCNT_START
        for i, nd in enumerate(self.nodes):
            m = nd.mode
            if m != self._last_modes[i]:
                self.mode_log.append((self.tick_no, i, self._last_modes[i], m))
                self._last_modes[i] = m

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
