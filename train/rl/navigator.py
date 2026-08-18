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

    def cell_of(self, pos):
        return (pos[0] // self.cell, pos[1] // self.cell)

    def center(self, c):
        return (c[0] * self.cell + self.cell // 2, c[1] * self.cell + self.cell // 2)

    def state(self, c):
        return self.g.get(c, UNKNOWN)

    def mark_free(self, c):
        self.g[c] = FREE

    def mark_blocked(self, c):
        if self.g.get(c) != FREE:         # a cell Link has stood on is never a wall
            self.g[c] = BLOCKED

    def neighbors(self, c):
        return [(c[0] + dx, c[1] + dy) for dx, dy in DIRS.values()]

    def plan_to_frontier(self, start):
        """BFS over non-BLOCKED cells; return the path [start..first UNKNOWN cell], or None.

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
            for n in self.neighbors(c):
                if n not in came and self.state(n) != BLOCKED:
                    came[n] = c
                    q.append(n)
        return None


class Navigator:
    """Closed-loop frontier explorer. step(pos, moved) -> a dir to press, or None if boxed in."""

    def __init__(self, cell=16, kick_after=40):
        self.grid = WalkGrid(cell)
        self.path = []                    # remaining waypoint cells (excluding Link's current cell)
        self.last_dir = None
        self.kick_after = kick_after      # decisions with no new FREE ground -> escape kick
        self._since_new = 0
        self._kick_i = 0
        self._retries = 0                 # times we've re-opened BLOCKED cells to retry a narrow exit

    def step(self, pos, moved):
        g = self.grid
        # 1. update the map from what just happened
        cur = g.cell_of(pos)
        new_free = g.state(cur) != FREE                      # did we reach NEW walkable ground?
        g.mark_free(cur)
        if self.last_dir is not None and not moved:          # bumped a wall -> cell ahead is blocked
            dx, dy = DIRS[self.last_dir]
            g.mark_blocked((cur[0] + dx, cur[1] + dy))
            self.path = []                                   # force a replan around it
        # progress watchdog: reset ONLY on new FREE ground. (Resetting on ANY new cell let the
        # escape's own wall-bumps -- new BLOCKED cells -- count as progress, so a wedged Link
        # never triggered recovery: it pressed into the same wall for ~1900 decisions.)
        self._since_new = 0 if new_free else self._since_new + 1

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
                    g.g = {c: v for c, v in g.g.items() if v != BLOCKED}
                    self._since_new = 0
                self._kick_i += 1
                return list(DIRS)[self._kick_i % 4]
            self.path = p[1:]

        # 5. steer toward the next waypoint's centre (recomputed from live pos -> self-correcting)
        nxt = g.center(self.path[0])
        d = self._dir_toward(pos, nxt)
        self.last_dir = d
        return d

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
    print(f"navigator selftest OK -- reached door in {t} decisions, explored {free} free cells")


if __name__ == "__main__":
    _selftest()
