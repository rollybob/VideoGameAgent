"""Hidden acceptance test for g25_nms. The subordinate never sees this file.

Independent greedy-NMS reference + deterministic biting cases (reject classic mistakes):
  1. identical box, lower score          -> suppressed (baseline);
  2. IoU exactly == threshold            -> NOT suppressed (rejects `>=` impls);
  3. selection order = descending score  -> kept order reflects score (rejects index-order out);
  4. tie score                           -> lower index first;
  5. wrong IoU denominator (min-area vs union) is caught by case 2's exact 0.5.
Randomized integer-box cases (40/run) compare exactly; any case with an IoU within 1e-9 of the
threshold is regenerated so float-boundary ties never make the gate flaky.
"""
import random


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def expected(boxes, scores, thr):
    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))
    removed = set()
    kept = []
    for i in order:
        if i in removed:
            continue
        kept.append(i)
        for j in order:
            if j == i or j in removed:
                continue
            if _iou(boxes[i], boxes[j]) > thr:
                removed.add(j)
    return kept


def near_boundary(boxes, thr):
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i != j and abs(_iou(boxes[i], boxes[j]) - thr) < 1e-9:
                return True
    return False


def main():
    from nms import nms
    problems = []

    det = [
        ([[0, 0, 2, 2], [0, 0, 2, 2]], [0.9, 0.8], 0.5, [0]),
        ([[0, 0, 2, 2], [0, 0, 2, 1]], [0.9, 0.8], 0.5, [0, 1]),   # IoU exactly 0.5 -> keep both
        ([[0, 0, 1, 1], [5, 5, 6, 6]], [0.3, 0.9], 0.5, [1, 0]),   # selection by score
        ([[0, 0, 1, 1], [5, 5, 6, 6]], [0.7, 0.7], 0.5, [0, 1]),   # tie -> lower index first
    ]
    for boxes, scores, thr, exp in det:
        got = nms(boxes, scores, thr)
        if got != exp:
            problems.append("nms(%r,%r,%s) -> %r, expected %r" % (boxes, scores, thr, got, exp))

    random.seed()
    made = 0
    guard = 0
    while made < 40 and guard < 4000:
        guard += 1
        n = random.randint(1, 6)
        boxes = []
        for _ in range(n):
            x1 = random.randint(0, 6); y1 = random.randint(0, 6)
            boxes.append([x1, y1, x1 + random.randint(1, 4), y1 + random.randint(1, 4)])
        thr = random.choice([0.3, 0.5, 0.7])
        if near_boundary(boxes, thr):
            continue
        made += 1
        scores = [random.randint(0, 4) for _ in range(n)]  # ties exercise the index tie-break
        got = nms(boxes, scores, thr)
        exp = expected(boxes, scores, thr)
        if got != exp:
            problems.append("RAND boxes=%r scores=%r thr=%s -> %r exp %r"
                            % (boxes, scores, thr, got, exp)); break

    if problems:
        print("FAIL")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
