"""Host-side attempt memory + escalation for the 3-tier drive agent -- the LOGIC LOOP.

Tim's 2026-08-09 diagnosis of the failed 3-tier runs: the agent "doesn't learn
from what it's tried. It needs a logic loop." The VLM re-decided each call from
a 6-entry button history; the anti-freeze was a fixed sweep with no memory. Both
are stateless-reactive, so the agent pushed the same wall for 25 minutes.

This module is the missing state. It tracks, per room-cell (RAM room = world
pos >> 9, the movement ORACLE convention -- control signal, not perception):
  - every decision's outcome: which button, did Link's world position move,
  - which screen EDGES (candidate exits) have been probed and how often,
  - room transitions with the button that achieved them (success receipts).

It produces:
  brief(room)    -> authoritative text for /act's task_phase slot: what has been
                    tried IN THIS ROOM, what is untried, how long stuck. The VLM
                    stops re-deciding refuted moves.
  receipts()     -> lines for /act's skills slot: strategies that actually
                    changed rooms elsewhere (transferable "what works").
  escalation(room) -> 0 normal / 1 warn (brief hardens) / 2 SWEEP: the host
                    takes over and probes the least-tried edge in bounded
                    bursts. Deterministic room-escape when reasoning has failed;
                    each burst's failure feeds back into the edge stats.

No emulator imports -- pure bookkeeping, unit-testable standalone.
"""
from collections import defaultdict

# Escalation is driven by CONSECUTIVE STUCK decisions (Link's world position not
# advancing), NOT total time-in-room. The old design escalated on total in-room
# decisions and NEVER reset: after ~23s a room was permanently in blind SWEEP and
# the VLM/walker navigation was locked out for the rest of the run (2026-08-09
# forensics: SWEEP drove 80-85% of every run; the VLM's "move left" never executed
# because it became a spectator ~23s into each room). Now a MOVING agent stays at
# L0 and the VLM/walker keeps navigating; only genuine stuckness (not moving for N
# decisions -- a wall/bush) escalates, and any real movement RESETS it, so SWEEP is
# a brief un-sticking intervention that hands control back, not a takeover.
# (Thresholds in CONSECUTIVE non-moving decisions; loop runs ~15 decisions/s.)
STUCK_L1 = 30             # ~2s not moving: harden the brief (tell the VLM it is stuck)
STUCK_L2 = 70             # ~5s not moving: host SWEEP probes edges + slashes obstacles
STUCK_L3 = 170            # ~11s not moving (sweep failed too): VLM names an exit target
SWEEP_BURST = 45          # decisions per edge-probe burst before rotating edges
EDGES = ("south", "north", "west", "east")
EDGE_DIR = {"south": "down", "north": "up", "west": "left", "east": "right"}
# Buttons worth reporting as tried/untried in the brief (movement + interact).
REPORT_BTNS = ("up", "down", "left", "right", "a", "b", "select")


class RoomMemory:
    def __init__(self):
        self.rooms = {}                 # cell -> per-room stats dict
        self.order = []                 # visit order of distinct cells
        self.transitions = []           # (from_cell, to_cell, step, button)
        self._cur = None
        self._entered_step = 0

    def _room(self, cell):
        if cell not in self.rooms:
            self.rooms[cell] = {
                "visits": 0,
                "decisions": 0,
                "tried": defaultdict(lambda: [0, 0]),   # btn -> [count, moved_count]
                "edges": defaultdict(int),               # edge -> probe decisions
                "sweep_i": 0,                            # rotating edge index
                "stuck": 0,                              # CONSECUTIVE non-moving decisions -> escalation
            }
            self.order.append(cell)
        return self.rooms[cell]

    # -- recording ----------------------------------------------------------
    def enter(self, cell, step):
        """Call when the current room-cell changes (and once at start)."""
        prev = self._cur
        if cell == prev:
            return
        r = self._room(cell)
        r["visits"] += 1
        r["stuck"] = 0                  # fresh room / re-entry starts un-escalated
        self._cur = cell
        self._entered_step = step
        return prev

    def note(self, cell, button, moved, step, edge=None):
        """Record one decision's outcome in this room."""
        self.enter(cell, step)
        r = self._room(cell)
        r["decisions"] += 1
        r["stuck"] = 0 if moved else r["stuck"] + 1     # movement RESETS escalation
        if button in REPORT_BTNS:
            t = r["tried"][button]
            t[0] += 1
            t[1] += 1 if moved else 0
        if edge:
            r["edges"][edge] += 1

    def room_changed(self, old, new, step, button):
        """A transition receipt: `button` (or the sweep edge) got us out of old."""
        self.transitions.append((old, new, step, button))
        # arriving somewhere new resets that room's staleness
        self._room(new)
        self._cur = new
        self._entered_step = step

    # -- escalation ---------------------------------------------------------
    def stuck_decisions(self, cell):
        r = self._room(cell)
        return r["decisions"] if self._cur == cell else 0

    def stuck_streak(self, cell):
        """Consecutive decisions in this room with NO movement -- the escalation
        signal. Reset to 0 by any real move (note(moved=True)) or by room entry."""
        r = self._room(cell)
        return r["stuck"] if self._cur == cell else 0

    def since_entry(self, step):
        return step - self._entered_step

    def escalation(self, cell):
        s = self.stuck_streak(cell)
        if s >= STUCK_L3:
            return 3
        if s >= STUCK_L2:
            return 2
        if s >= STUCK_L1:
            return 1
        return 0

    def sweep_edge(self, cell, step):
        """Level-2 driver: probe the LEAST-tried edge, rotating every SWEEP_BURST
        decisions so a blocked edge cannot absorb the whole escalation."""
        r = self._room(cell)
        burst = (r["decisions"] // SWEEP_BURST) % len(EDGES)
        ranked = sorted(EDGES, key=lambda e: r["edges"][e])
        edge = ranked[burst % len(ranked)]
        return edge, EDGE_DIR[edge]

    # -- prompt material ----------------------------------------------------
    def brief(self, cell):
        """Authoritative stuck-brief for /act's task_phase slot. Empty string when
        there is nothing worth saying (fresh room, or making progress)."""
        r = self._room(cell)
        d = self.stuck_streak(cell)     # the brief is about STUCKNESS, not time-in-room
        if d < STUCK_L1:
            return ""
        tried, untried = [], []
        for b in REPORT_BTNS:
            c, m = r["tried"].get(b, (0, 0))
            if c >= 8:
                if b in ("up", "down", "left", "right") and m == 0:
                    tried.append("%s x%d (Link NEVER moved - blocked)" % (b, c))
                elif b in ("up", "down", "left", "right"):
                    tried.append("%s x%d (moves, but still in this room)" % (b, c))
                else:
                    tried.append("%s x%d (no effect)" % (b, c))
            elif c == 0:
                untried.append(b)
        lines = ["You have been STUCK (not moving) in this room for %d decisions." % d]
        if tried:
            lines.append("Already tried here: " + "; ".join(tried) + ".")
        if untried:
            lines.append("NOT yet tried here: " + ", ".join(untried) + ".")
        lines.append("Do something you have NOT tried, or pick the screen edge you have "
                     "not walked to yet. Leaving this room is the ONLY progress that counts now.")
        return " ".join(lines)

    def receipts(self, limit=3):
        """Transferable success receipts for /act's skills slot."""
        out = []
        for old, new, _step, button in self.transitions[-limit:]:
            out.append("Leaving a room WORKED before: '%s' took you from room %s to %s."
                       % (button, tuple(old), tuple(new)))
        return out

    def stats(self):
        return {"rooms_visited": len(self.order),
                "transitions": len(self.transitions),
                "order": [tuple(c) for c in self.order]}


def _selftest():
    m = RoomMemory()
    A, B = (4, 5), (4, 6)
    m.enter(A, 0)
    # MOVING never escalates, no matter how long in the room (the whole point)
    for i in range(500):
        m.note(A, "left", True, i)
    assert m.escalation(A) == 0, ("moving must not escalate", m.stuck_streak(A))
    # going stuck (not moving) escalates to SWEEP
    for i in range(500, 500 + STUCK_L2 + 3):
        m.note(A, "down", False, i)
    assert m.escalation(A) == 2, m.stuck_streak(A)
    assert "STUCK" in m.brief(A) and "NOT yet tried" in m.brief(A)
    # a single real move RESETS escalation -> control returns to the VLM
    m.note(A, "left", True, 9000)
    assert m.escalation(A) == 0, ("movement must de-escalate", m.stuck_streak(A))
    # deep stuck (sweep failed too) -> L3 (VLM names a target)
    for i in range(9001, 9001 + STUCK_L3 + 3):
        m.note(A, "down", False, i)
    assert m.escalation(A) == 3
    edge, d = m.sweep_edge(A, 9200)
    assert edge in EDGES and d in ("up", "down", "left", "right")
    m.room_changed(A, B, 9300, "sweep:south")
    m.note(B, "up", True, 9301)
    assert m.escalation(B) == 0 and m.brief(B) == ""
    assert "sweep:south" in m.receipts()[0]
    assert m.stats()["rooms_visited"] == 2
    print("agent_memory selftest OK:", m.stats())


if __name__ == "__main__":
    _selftest()
