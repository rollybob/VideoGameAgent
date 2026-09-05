"""Hidden acceptance test for g15_stuck_detector. The subordinate never sees this file.

Independent per-axis reference + deterministic biting cases that reject the classic mistakes
on EVERY run:
  1. fewer than `window` positions            -> False (rejects "judge on whatever we have");
  2. recent window is stuck but EARLIER points -> True  (rejects using the whole history
     roam far                                          instead of only the last `window`);
  3. recent window moves > tol in ONE axis     -> False (rejects single-axis impls);
  4. diagonal move, per-axis spread <= tol but -> True  (rejects Euclidean/hypot distance
     Euclidean distance > tol                          impls -- axes must be independent).
Randomized cases (40/run) compare exactly against the independent reference and confirm the
input list is not mutated.
"""
import copy
import random

from stuck_detector import is_stuck


def expected(positions, window, tol):
    """Independent reference: per-axis bounding box over the last `window` positions."""
    if window < 1 or len(positions) < window:
        return False
    recent = positions[-window:]
    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    return (max(xs) - min(xs)) <= tol and (max(ys) - min(ys)) <= tol


def main():
    random.seed()
    problems = []

    # 1. not enough history
    if is_stuck([[0, 0], [1, 1]], 3, 5) is not False:
        problems.append("len<window must be False")
    # 2. recent stuck, earlier far -> must ignore the far earlier points
    d2 = [[0, 0], [100, 100], [7, 7], [7, 8], [8, 7]]
    if is_stuck(d2, 3, 2) is not True:
        problems.append("recent-stuck/earlier-far: expected True (only last window matters)")
    # 3a. recent moves > tol in x only
    if is_stuck([[0, 0], [0, 0], [5, 0]], 3, 2) is not False:
        problems.append("x-axis exceeds tol -> must be False")
    # 3b. recent moves > tol in y only (bites x-only impls)
    if is_stuck([[0, 0], [0, 0], [0, 5]], 3, 2) is not False:
        problems.append("y-axis exceeds tol -> must be False")
    # 4. diagonal: per-axis spread 2<=2 but Euclidean sqrt(8)=2.83 > 2 -> True
    if is_stuck([[0, 0], [2, 2]], 2, 2) is not True:
        problems.append("diagonal within per-axis tol -> True (axes independent, not Euclidean)")
    # exact-boundary: spread == tol -> True (rejects `< tol`)
    if is_stuck([[0, 0], [3, 3]], 2, 3) is not True:
        problems.append("spread == tol must count as stuck (<=, not <)")

    for _ in range(40):
        n = random.randint(1, 10)
        pts = [[random.randint(-20, 20), random.randint(-20, 20)] for _ in range(n)]
        window = random.randint(1, n)
        tol = random.randint(0, 8)
        snap = copy.deepcopy(pts)
        got = is_stuck(pts, window, tol)
        exp = expected(pts, window, tol)
        if not isinstance(got, bool):
            problems.append("did not return a bool; got %r" % type(got)); break
        if got != exp:
            problems.append("MISMATCH pts=%r window=%d tol=%d exp=%r got=%r"
                            % (pts, window, tol, exp, got)); break
        if pts != snap:
            problems.append("MUTATED INPUT"); break

    if problems:
        print("FAIL")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
