"""Hidden acceptance test for g12_hud_delta. The subordinate never sees this file.

Independent oracle + deterministic biting cases that reject the classic mistakes every run:
  1. no change            -> {} / []            (rejects impls that invent events);
  2. item field DECREASES -> no "pickup"        (rejects "pickup on any change / on decrease");
  3. health INCREASES     -> no "damage"        (rejects "damage on any health change");
  4. health -> 0 from >0  -> ["damage","death"] (rejects impls that omit the death special-case).
Randomized cases force a death + a pickup + a decrease together and compare exactly, so an impl
that reports changes as [new, old] (reversed) is rejected on every run.
"""
import random

from hud_delta import hud_delta

ITEMS = ["rupees", "keys", "bombs", "arrows", "magic"]


def expected(prev, cur):
    changes = {}
    for k in prev:
        if prev[k] != cur[k]:
            changes[k] = [prev[k], cur[k]]
    ev = set()
    if cur["health"] < prev["health"]:
        ev.add("damage")
    if prev["health"] > 0 and cur["health"] == 0:
        ev.add("death")
    for k in ITEMS:
        if k in prev and cur[k] > prev[k]:
            ev.add("pickup")
    return {"changes": changes, "events": sorted(ev)}


def check(prev, cur, problems, label):
    exp = expected(prev, cur)
    got = hud_delta(prev, cur)
    if got != exp:
        problems.append("%s MISMATCH\n    prev=%r cur=%r\n    exp=%r\n    got=%r"
                        % (label, prev, cur, exp, got))


def main():
    random.seed()
    problems = []

    check({"health": 8, "keys": 2}, {"health": 8, "keys": 2}, problems, "no-change")
    check({"health": 5, "rupees": 10}, {"health": 5, "rupees": 3}, problems, "item-decrease")
    check({"health": 3}, {"health": 6}, problems, "heal")
    check({"health": 4, "bombs": 2}, {"health": 0, "bombs": 2}, problems, "death")

    for _ in range(30):
        fields = random.sample(ITEMS, random.randint(2, len(ITEMS)))
        prev = {"health": random.randint(1, 9)}
        for f in fields:
            prev[f] = random.randint(5, 20)
        cur = dict(prev)
        cur["health"] = 0                                   # death (=> damage + death)
        up, down = fields[0], fields[1]
        cur[up] = prev[up] + random.randint(1, 5)           # a pickup
        cur[down] = prev[down] - random.randint(1, prev[down])   # a decrease, no pickup
        check(prev, cur, problems, "rand")
        if problems:
            break

    if problems:
        print("FAIL")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
