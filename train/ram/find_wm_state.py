#!/usr/bin/env python3
"""Find world-map RAM signals for the extended Herb Picking ladder (C4/C5) by
multi-dump EWRAM diffing, find_mission.py style: bytes must be CONSTANT within a
group of same-situation dumps and DIFFER across groups, which kills scroll/anim/
timer noise that a single pairwise diff drowns in.

Groups (dumps produced by explore_pub.py runs on 2026-07-03):
  cyril_field: on the world map, clan standing at Cyril (4 dumps)
  giza_field : on the world map, clan standing at Giza Plains (1 dump)
  town       : inside town scenes - pub menu / missions list / pub interior /
               town exterior (4 dumps)

Outputs:
  C5 (clan node id): bytes equal across ALL wm-field dumps at Cyril, different at
      Giza. Prior: the value at Cyril should be 97 (matches worldmap_cursor=97 =
      Cyril), so print those first.
  C4 (on world map): bytes with one constant value across ALL 5 wm-field dumps and
      a DIFFERENT constant value across ALL town dumps.
"""
import os

EWRAM_BASE = 0x02000000

CYRIL = ["/tmp/wm_save/ewram_final.bin", "/tmp/wm2/ewram_final.bin",
         "/tmp/wm3/ewram_final.bin", "/tmp/wm4/ewram_final.bin"]
# wm14 = at Giza, field (node menu closed); giza2 = just arrived (node menu open).
# A true clan-position byte must agree across both.
GIZA = ["/tmp/wm14/ewram_final.bin", "/tmp/giza2/ewram_final.bin"]
TOWN = ["/tmp/pubmenu_dump/ewram_final.bin", "/tmp/chain1/ewram_final.bin",
        "/tmp/chain2/ewram_final.bin", "/tmp/chain6/ewram_final.bin"]
# bat1 = battle-map intro cutscene; bat2 = dispatch/placement phase.
BATTLE = ["/tmp/bat1/ewram_final.bin", "/tmp/bat2/ewram_final.bin"]


def load(paths):
    out = []
    for p in paths:
        with open(p, "rb") as f:
            out.append(f.read())
    return out


def main():
    cyril, giza, town, battle = load(CYRIL), load(GIZA), load(TOWN), load(BATTLE)
    n = min(len(b) for b in cyril + giza + town + battle)

    def const(group, i):
        v = group[0][i]
        return v if all(b[i] == v for b in group) else None

    # --- C5: clan node/position (constant within Cyril AND within Giza, differing) ---
    c5 = []
    for i in range(n):
        v1, v2 = const(cyril, i), const(giza, i)
        if v1 is not None and v2 is not None and v1 != v2:
            c5.append((i, v1, v2))
    print(f"[C5] {len(c5)} bytes constant@Cyril and constant@Giza, differing:")
    for i, v1, v2 in c5[:80]:
        print(f"  0x{EWRAM_BASE + i:08x}: cyril={v1} giza={v2}")

    # --- C4: on-world-map flag (one constant on wm, another constant in town) ---
    wm = cyril + giza
    c4 = []
    for i in range(n):
        v, t = const(wm, i), const(town, i)
        if v is not None and t is not None and v != t:
            c4.append((i, v, t, const(battle, i)))
    print(f"\n[C4] {len(c4)} bytes constant-on-wm vs constant-in-town (battle value last):")
    for i, v, t, bb in c4[:60]:
        print(f"  0x{EWRAM_BASE + i:08x}: wm={v} town={t} battle={bb}")

    # --- C6: on-battle-map flag (constant in battle, different constant elsewhere) ---
    other = cyril + giza + town
    c6 = []
    for i in range(n):
        vb, vo = const(battle, i), const(other, i)
        if vb is not None and vo is not None and vb != vo:
            c6.append((i, vb, vo))
    print(f"\n[C6] {len(c6)} bytes constant-in-battle vs constant-everywhere-else:")
    for i, vb, vo in c6[:60]:
        print(f"  0x{EWRAM_BASE + i:08x}: battle={vb} other={vo}")


if __name__ == "__main__":
    main()
