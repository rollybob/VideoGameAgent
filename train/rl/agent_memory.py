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

# Decisions-in-room thresholds. At ~7.5 decisions/s (FPS 15, VLM-paced loop is
# still per-frame decisions): L1 after ~20s stuck, L2 after ~45s stuck, L3 after
# ~2 minutes (>= 5 full sweep cycles refuted plain edge-probing -- braintest
# 2026-08-09 showed 12.5k sweep decisions in one room achieve nothing more).
L1_DECISIONS = 150
L2_DECISIONS = 340
L3_DECISIONS = 900
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
        self._cur = cell
        self._entered_step = step
        return prev

    def note(self, cell, button, moved, step, edge=None):
        """Record one decision's outcome in this room."""
        self.enter(cell, step)
        r = self._room(cell)
        r["decisions"] += 1
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

    def since_entry(self, step):
        return step - self._entered_step

    def escalation(self, cell):
        d = self.stuck_decisions(cell)
        if d >= L3_DECISIONS:
            return 3
        if d >= L2_DECISIONS:
            return 2
        if d >= L1_DECISIONS:
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
        d = self.stuck_decisions(cell)
        if d < 40:
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
        lines = ["You have been in this SAME room for %d decisions without leaving." % d]
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
    for i in range(200):
        m.note(A, "down", False, i)
    assert m.escalation(A) == 1, m.stuck_decisions(A)
    assert "down x200" in m.brief(A) and "NOT yet tried" in m.brief(A)
    for i in range(200, 360):
        m.note(A, "a", False, i)
    assert m.escalation(A) == 2
    edge, d = m.sweep_edge(A, 360)
    assert edge in EDGES and d in ("up", "down", "left", "right")
    m.room_changed(A, B, 361, "sweep:south")
    m.note(B, "up", True, 362)
    assert m.escalation(B) == 0 and m.brief(B) == ""
    assert "sweep:south" in m.receipts()[0]
    assert m.stats()["rooms_visited"] == 2
    print("agent_memory selftest OK:", m.stats())


if __name__ == "__main__":
    _selftest()
