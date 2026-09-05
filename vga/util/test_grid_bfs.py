"""Hidden acceptance test for g18_grid_bfs. The subordinate never sees this file.

Independent BFS reference + deterministic biting cases (reject classic mistakes every run):
  1. open grid            -> Manhattan distance;
  2. wall forces a detour -> strictly longer than Manhattan (rejects Manhattan/greedy impls);
  3. walled-off goal      -> -1 (rejects impls that ignore reachability);
  4. start == goal        -> 0;
  5. start/goal on a wall or out of bounds -> -1;
  6. path exists ONLY via a diagonal -> -1 (rejects 8-connected impls).
Randomized grids (30/run) compare exactly against the BFS reference and check no mutation.
"""
import copy
import random
from collections import deque

from grid_bfs import shortest_path_len


def expected(grid, start, goal):
    if not grid or not grid[0]:
        return -1
    R, C = len(grid), len(grid[0])
    sr, sc = start
    gr, gc = goal

    def blocked(r, c):
        return not (0 <= r < R and 0 <= c < C) or grid[r][c] != 0

    if blocked(sr, sc) or blocked(gr, gc):
        return -1
    if (sr, sc) == (gr, gc):
        return 0
    seen = {(sr, sc)}
    q = deque([(sr, sc, 0)])
    while q:
        r, c, d = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 0 and (nr, nc) not in seen:
                if (nr, nc) == (gr, gc):
                    return d + 1
                seen.add((nr, nc))
                q.append((nr, nc, d + 1))
    return -1


def main():
    random.seed()
    problems = []

    # 1. open grid -> Manhattan
    g = [[0, 0, 0], [0, 0, 0]]
    if shortest_path_len(g, (0, 0), (1, 2)) != 3:
        problems.append("open grid: expected 3, got %r" % shortest_path_len(g, (0, 0), (1, 2)))
    # 2. wall forces detour
    g2 = [[0, 1, 0],
          [0, 1, 0],
          [0, 0, 0]]
    if shortest_path_len(g2, (0, 0), (0, 2)) != 6:
        problems.append("detour: expected 6, got %r" % shortest_path_len(g2, (0, 0), (0, 2)))
    # 3. walled-off goal
    g3 = [[0, 1, 0],
          [1, 1, 0],
          [0, 1, 0]]
    if shortest_path_len(g3, (0, 0), (0, 2)) != -1:
        problems.append("unreachable: expected -1, got %r" % shortest_path_len(g3, (0, 0), (0, 2)))
    # 4. start == goal
    if shortest_path_len([[0]], (0, 0), (0, 0)) != 0:
        problems.append("start==goal must be 0")
    # 5. start/goal on wall
    if shortest_path_len([[0, 9]], (0, 0), (0, 1)) != -1:
        problems.append("goal on wall must be -1")
    if shortest_path_len([[0, 0]], (0, 5), (0, 1)) != -1:
        problems.append("start out of bounds must be -1")
    # 6. only-diagonal path is NOT connected under 4-connectivity
    gd = [[0, 1],
          [1, 0]]
    if shortest_path_len(gd, (0, 0), (1, 1)) != -1:
        problems.append("diagonal-only: expected -1 (4-connected), got %r"
                        % shortest_path_len(gd, (0, 0), (1, 1)))

    # randomized
    for _ in range(30):
        R = random.randint(1, 6)
        C = random.randint(1, 6)
        grid = [[1 if random.random() < 0.3 else 0 for _ in range(C)] for _ in range(R)]
        start = (random.randint(0, R - 1), random.randint(0, C - 1))
        goal = (random.randint(0, R - 1), random.randint(0, C - 1))
        snap = copy.deepcopy(grid)
        got = shortest_path_len(grid, start, goal)
        exp = expected(grid, start, goal)
        if got != exp:
            problems.append("RAND MISMATCH grid=%r start=%r goal=%r exp=%r got=%r"
                            % (grid, start, goal, exp, got)); break
        if grid != snap:
            problems.append("MUTATED INPUT"); break

    if problems:
        print("FAIL")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
