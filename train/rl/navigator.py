"""navigator.py -- classical navigation: map + plan + locomotion for the ALttP agent.

WHY (2026-08-18): the agent has legs (the walker) and a position oracle (link_world) but NO MAP
and NO PLANNER, so it walks blind into walls and can't even leave Link's house. That is the
binding constraint behind the stuck-in-a-room failures (and it re-broke the hint-persistence A/B).
This adds the two missing pieces as CLASSICAL, deterministic components (not another RL gamble):

  WalkGrid   -- an occupancy grid built INCREMENTALLY from movement: every cell Link stands on is
                FREE; a cardinal input that produces no movement marks the cell ahead BLOCKED (a
                wall). No collision-RAM hunt needed to start; link_world is the only oracle.
  Navigator  -- frontier exploration: A*/BFS over non-blocked cells to the nearest UNKNOWN cell,
                then a closed-loop controller drives Link toward the next waypoint (recomputed each
                decision from link_world, so it self-corrects). Frontier-seeking systematically
                covers a room until Link walks onto the exit tile (room transition = success) --
                it does not need to KNOW where the door is.

This is the RAM-scaffolded TEACHER stage (link_world at inference); the north-star follow-up is to
distill the grid into a pixels walkability CNN. Pure logic + stdlib -- no emulator import, so it
unit-tests standalone against a synthetic room.

WHY (2026-08-19): a single shared WalkGrid, reset on every room change, plateaued exploration at 3
rooms. Adjacent overworld screens share one CONTINUOUS world-coordinate frame (room_cell = world
pos >> 9), so wiping the grid on re-entering an already-explored neighbor threw away real knowledge
and re-treated walked ground as a frontier, pulling Link straight back across the boundary he just
crossed -- an oscillation, not exploration. Fix: `Navigator.room_maps` persists one WalkGrid.g per
room_cell, restored on revisit instead of rebuilt. A never-seen room still starts fresh (same as
before); a revisited one keeps what it learned. Warps stay global and untouched either way (a
door-back is a specific world location, not room-relative).

Convention: world y increases DOWNWARD (verified: DOWN raises y). dir in {up,down,left,right}.
"""
from collections import deque

UNKNOWN, FREE, BLOCKED = 0, 1, 2
# cardinal unit deltas in CELL space; y-down so 'down' = +y.
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


class WalkGrid:
    def __init__(self, cell=16):
        self.cell = cell
        self.g = {}                       # (cx,cy) -> UNKNOWN/FREE/BLOCKED (default UNKNOWN)
        self.warps = set()                # door-back tiles into VISITED rooms: permanently impassable to the planner (survives mark_free -- the agent SPAWNS on the door-back tile, so FREE must not clear it)
        self.bad_edges = set()            # (from_cell, (dx,dy)) directional moves proven to fail (2026-08-19, see mark_edge_blocked)

    def cell_of(self, pos):
        return (pos[0] // self.cell, pos[1] // self.cell)

    def center(self, c):
        return (c[0] * self.cell + self.cell // 2, c[1] * self.cell + self.cell // 2)

    def state(self, c):
        if c in self.warps:               # a learned door-back warp tile is never a frontier / never steered toward
            return BLOCKED
        return self.g.get(c, UNKNOWN)

    def mark_free(self, c):
        self.g[c] = FREE

    def mark_blocked(self, c):
        if self.g.get(c) != FREE:         # a cell Link has stood on is never a wall
            self.g[c] = BLOCKED

    def mark_warp(self, c):
        if self.g.get(c) != FREE:         # a cell Link has stood on is never a wall (same guard as mark_blocked)
            self.warps.add(c)

    def mark_edge_blocked(self, c, delta):
        """Record that moving `delta` FROM cell `c` is proven to fail, independent of the target
        cell's own state. WHY (2026-08-19, live-diagnosed via debug trace): mark_blocked guards
        against overwriting a cell Link has proven FREE -- correct in general, but the grid is
        CELL-granular while real obstacles (furniture, a wall corner) can be sub-cell: Link can
        reach a cell's FAR side from one approach while a specific edge into it stays genuinely
        blocked. Without this, a target cell already marked FREE (from a different approach)
        keeps winning plan_to_frontier's BFS forever, so the SAME doomed move repeats every
        decision (confirmed live: 3900+ consecutive decisions jittering in one cell)."""
        self.bad_edges.add((c, delta))

    def neighbors(self, c):
        return [(c[0] + dx, c[1] + dy) for dx, dy in DIRS.values()]

    def plan_to_frontier(self, start):
        """BFS over non-BLOCKED cells/edges; return the path [start..first UNKNOWN cell], or None.

        Traversing UNKNOWN optimistically (only BLOCKED is impassable) lets the planner head into
        unexplored space; returning at the FIRST unknown dequeued makes it the NEAREST frontier.
        """
        came = {start: None}
        q = deque([start])
        while q:
            c = q.popleft()
            if self.state(c) == UNKNOWN:
                path = []
                while c is not None:
                    path.append(c)
                    c = came[c]
                return path[::-1]
            for delta in DIRS.values():
                n = (c[0] + delta[0], c[1] + delta[1])
                if n not in came and self.state(n) != BLOCKED and (c, delta) not in self.bad_edges:
                    came[n] = c
                    q.append(n)
        return None


class Navigator:
    """Closed-loop frontier explorer. step(pos, moved) -> a dir to press, or None if boxed in."""

    def __init__(self, cell=16, kick_after=40):
        self.grid = WalkGrid(cell)
        self.room_maps = {}               # room_key -> WalkGrid.g dict, persisted across visits (2026-08-19)
        self._room_key = None             # the room_key self.grid.g currently points at (None until the first switch)
        self.path = []                    # remaining waypoint cells (excluding Link's current cell)
        self.last_dir = None
        self.kick_after = kick_after      # decisions with no new FREE ground -> escape kick
        self._since_new = 0
        self._kick_i = 0
        self._retries = 0                 # times we've re-opened BLOCKED cells to retry a narrow exit
        self._controlled = True           # is Link accepting input? FALSE during the post-warp fade-in (a few
                                          # decisions where presses do nothing) -- marking walls then paints FALSE
                                          # walls that box Link into the doorway pocket. Set FALSE on room change,
                                          # TRUE on the first real move (control confirmed by input-response).
        self._entry_cell = None           # where THIS room-visit started -- a known way back out, for the frontier-desert fallback
        self._need_entry = False          # capture _entry_cell on the next step() (set on every room switch)

    def step(self, pos, moved):
        g = self.grid
        # 1. update the map from what just happened
        cur = g.cell_of(pos)
        if self._need_entry:
            self._entry_cell = cur
            self._need_entry = False
        new_free = g.state(cur) != FREE                      # did we reach NEW walkable ground?
        g.mark_free(cur)
        if moved:
            self._controlled = True                          # input produced motion -> Link is in control now
        if self.last_dir is not None and not moved and self._controlled:  # bumped a wall -> cell ahead is blocked.
            # GATED on control: during the post-warp fade-in Link ignores input, so a no-move is NOT a wall --
            # marking it painted false walls that boxed Link into a 2-cell doorway pocket and forced a re-warp.
            dx, dy = DIRS[self.last_dir]
            g.mark_blocked((cur[0] + dx, cur[1] + dy))
            g.mark_edge_blocked(cur, (dx, dy))                # THIS specific approach failed even if the target cell is FREE from elsewhere (sub-cell obstacles)
            self.path = []                                   # force a replan around it
        # progress watchdog: reset ONLY on new FREE ground. (Resetting on ANY new cell let the
        # escape's own wall-bumps -- new BLOCKED cells -- count as progress, so a wedged Link
        # never triggered recovery: it pressed into the same wall for ~1900 decisions.)
        self._since_new = 0 if new_free else self._since_new + 1
        if new_free:
            # _retries EARNS BACK its budget on genuine progress (2026-08-19). Without this, 3 failed
            # replans ANYWHERE in a room-visit permanently exhausted it, so a room that had already
            # covered a lot of new ground got the same trigger-happy entry-seek retreat (below) as one
            # that had never made any progress at all. Confirmed live: repeat visits to the same room
            # retreated faster each time (1518 -> 45 -> 21 decisions before giving up) while covering a
            # shrinking sliver of a room that is mostly still unexplored -- persisted state (learned
            # walls/edges near the door) made the LOCAL area look boxed-in sooner each visit, and a
            # permanently-spent retry budget meant there was nothing left to push back with.
            self._retries = 0

        # 2. drop waypoints Link has already reached/passed
        while self.path and self.path[0] == cur:
            self.path.pop(0)

        # 3. escape kick if we've stopped discovering (oscillation / trapped pocket)
        if self._since_new >= self.kick_after:
            self._since_new = 0
            self._kick_i += 1
            self.path = []
            return list(DIRS)[self._kick_i % 4]

        # 4. (re)plan to the nearest frontier
        if not self.path:
            p = g.plan_to_frontier(cur)
            if not p or len(p) < 2:
                # No reachable unknown left. A narrow exit can be false-BLOCKED by a glancing bump,
                # so give blocked cells one more chance (bounded) before a blind rotating nudge.
                if self._retries < 3:
                    self._retries += 1
                    g.g = {c: v for c, v in g.g.items() if v != BLOCKED}   # re-open BLOCKED walls (a narrow exit can be a false glancing-bump block); warps persist -- the drive_agent esc_lvl SWEEP is the wedge safety valve
                    self._since_new = 0
                    self._kick_i += 1
                    return list(DIRS)[self._kick_i % 4]
                if self._entry_cell is not None and cur != self._entry_cell:
                    # FRONTIER DESERT (2026-08-19): retries exhausted and still nothing reachable to
                    # explore -- this room is fully known from here (more likely now that persistence
                    # lets a room's map accumulate across visits: measured live, 2/4500 decisions stuck
                    # oscillating 1 cell for the rest of a run once this hit). A direction that rotates
                    # every decision can't produce sustained travel, so it never found its own way out.
                    # Head to the room's ENTRY cell instead -- a known way out -- so crossing it
                    # re-triggers a room transition and exploration resumes wherever still has a
                    # frontier (the minimal version of "no local frontier -> nearest room with one").
                    self.path = [self._entry_cell]
                else:
                    self._kick_i += 1
                    return list(DIRS)[self._kick_i % 4]
            else:
                self.path = p[1:]

        # 5. steer toward the next waypoint's centre (recomputed from live pos -> self-correcting)
        nxt = g.center(self.path[0])
        d = self._dir_toward(pos, nxt)
        self.last_dir = d
        return d

    def notify_room_change(self, room_key):
        """Call on EVERY room entry, including the very first (register the start room too, or
        returning to it later would wrongly look unexplored). SWITCHES self.grid.g to the PERSISTENT
        map for `room_key`: a never-seen key starts fresh (UNKNOWN everywhere, same as the old
        reset-every-time behavior); a REVISITED key restores exactly what was explored last time --
        this is the fix for the spatial-boundary oscillation (2026-08-19): re-entering an adjacent
        room no longer forgets it and re-treats already-walked ground as a frontier to chase.
        room_maps stores the dict by REFERENCE, so it's mutated in place by mark_free/mark_blocked --
        leaving a room needs no explicit save step. Learned warp-back blocks (self.grid.warps) are
        never swapped, so they stay global and survive every switch. Force a replan too."""
        self._room_key = room_key
        self.grid.g = self.room_maps.setdefault(room_key, {})
        self.path = []
        self.last_dir = None
        self._since_new = 0
        self._retries = 0
        self._controlled = False          # a warp is followed by a fade-in where Link ignores input; don't mark walls until he moves
        self._need_entry = True           # capture this visit's entry cell on the next step() (the frontier-desert fallback target)

    def block_warp_exit(self, last_pos, motion_dir, depth=2):
        """Call when the agent just RE-ENTERED an already-visited room (a ping-pong). `last_pos` is its
        last position in the room it just LEFT and `motion_dir` the direction it was moving there -- i.e.
        the door/warp tile back into the now-visited room. The agent is never observed standing on that
        tile (the warp fires mid-step, between decision boundaries), so it stays UNKNOWN and keeps being
        picked as the nearest frontier -- the ping-pong. Mark that tile + the next `depth` cells along
        the motion as permanent warp-blocks so the planner routes AWAY from the door into new ground.
        Directional (not a neighborhood) so it never seals a 1-cell-wide corridor exit."""
        if motion_dir not in DIRS:
            return
        dx, dy = DIRS[motion_dir]
        cx, cy = self.grid.cell_of(last_pos)
        for k in range(0, depth + 1):
            self.grid.mark_warp((cx + dx * k, cy + dy * k))

    @staticmethod
    def _dir_toward(pos, target):
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"


def _selftest():
    # Synthetic house: 12x12 walkable interior walled in, with a 1-cell door gap at the bottom.
    # Simulate Link on a fine world grid (cell=16). walls: a set of BLOCKED cells; door: one cell
    # on the bottom wall whose entry counts as "left the room".
    CELL = 16
    lo, hi = 1, 12                       # interior cells [1..11]
    door = (6, 12)                       # gap in the bottom wall (row 12)

    def is_wall(c):
        cx, cy = c
        if c == door:
            return False
        return cx <= 0 or cx >= 12 or cy <= 0 or cy >= 12

    nav = Navigator(cell=CELL, kick_after=30)
    pos = [6 * CELL + 8, 2 * CELL + 8]   # start near top-centre
    left = False
    for t in range(4000):
        cur_cell = (pos[0] // CELL, pos[1] // CELL)
        if cur_cell == door:
            left = True
            break
        moved_prev = getattr(_selftest, "_moved", True)
        d = nav.step((pos[0], pos[1]), moved_prev)
        if d is None:
            break
        dx, dy = DIRS[d]
        nxt = (pos[0] + dx * 6, pos[1] + dy * 6)          # ~6 units/decision (measured)
        ncell = (nxt[0] // CELL, nxt[1] // CELL)
        if is_wall(ncell) and ncell != cur_cell:
            _selftest._moved = False                       # bump
        else:
            pos[0], pos[1] = nxt
            _selftest._moved = True
    assert left, f"navigator failed to reach the door (explored {len(nav.grid.g)} cells)"
    # and it should have discovered a good chunk of the interior en route
    free = sum(1 for v in nav.grid.g.values() if v == FREE)
    assert free >= 15, f"too little explored: {free}"
    # notify_room_change SWITCHES to a per-room PERSISTENT map (2026-08-19 fix for the spatial-
    # boundary oscillation): a never-seen room starts fresh; a REVISITED room restores exactly what
    # was explored last time. Global warp-backs survive every switch, in and out.
    n2 = Navigator(cell=CELL)
    n2.block_warp_exit((5 * CELL + 8, 5 * CELL + 8), "up", depth=2)     # warp tile (5,5), motion up -> warp (5,5),(5,4),(5,3)
    n2.notify_room_change("A")
    n2.grid.mark_free((7, 7)); n2.path = [(9, 9)]
    n2.notify_room_change("B")                                          # switch to a NEW room -> fresh occupancy
    assert n2.grid.state((7, 7)) == UNKNOWN and n2.path == [], "an unseen room must start fresh and force a replan"
    assert all(n2.grid.state((5, 5 - k)) == BLOCKED for k in range(3)), "warp-backs must SURVIVE the switch"
    n2.grid.mark_free((5, 5))                                            # re-spawning on the door-back tile must NOT clear the warp
    assert n2.grid.state((5, 5)) == BLOCKED, "warp block must survive mark_free"
    n2.grid.mark_free((3, 3))
    n2.notify_room_change("A")                                          # REVISIT room A -> must RESTORE its map, not reset
    assert n2.grid.state((7, 7)) == FREE, "revisiting a room must RESTORE its persisted occupancy (the plateau fix)"
    assert n2.grid.state((3, 3)) == UNKNOWN, "room B's exploration must stay in room B's own map, not bleed into A"

    # Directional edge-blocking (2026-08-19): a cell already proven FREE from one approach can still
    # have ONE specific incoming edge genuinely blocked (sub-cell obstacles the grid can't otherwise
    # represent -- confirmed live via debug trace: a FREE neighbor kept winning plan_to_frontier's
    # BFS despite being unreachable from Link's actual direction, so he jittered in one cell for
    # 3900+ consecutive decisions). A blocked edge must be impassable even when the target cell itself
    # reads FREE, not just when it reads BLOCKED.
    g3 = WalkGrid(cell=CELL)
    g3.mark_free((5, 5)); g3.mark_free((6, 5))                            # (6,5) reached FREE from some OTHER approach
    g3.mark_blocked((4, 5)); g3.mark_blocked((5, 4)); g3.mark_blocked((5, 6))   # (5,5)'s only other neighbors sealed
    p_open = g3.plan_to_frontier((5, 5))
    assert p_open is not None and p_open[1] == (6, 5), f"control: must route through the (still-open) edge, got {p_open}"
    g3.mark_edge_blocked((5, 5), (1, 0))                                  # now seal the SPECIFIC (5,5)->(6,5) edge (moving right)
    assert g3.plan_to_frontier((5, 5)) is None, "a blocked edge must be impassable even though the target cell is FREE"
    print(f"navigator selftest OK -- reached door in {t} decisions, explored {free} free cells; transition+warp+persistence+edges OK")


if __name__ == "__main__":
    _selftest()
