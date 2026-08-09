"""_ring_diff.py -- rigorous forensic: where do overworld pot/bush items live in RAM?

For each of today's rings (300 IWRAM snapshots @ 2-tick spacing), find u16 addresses in the
object region (0x2000-0x3FFF) whose on-screen projection (value - camera) transitions from
absent/off-screen EARLY to a STABLE on-screen point LATE = an object SPAWNING mid-window
(a bush cut / pot smashed dropping an item). Report, per ring:
  (A) spawns landing in the KNOWN sprite-slot x-table (0x03846+4i) -> items reuse the enemy table;
  (B) spawns at OTHER addresses -> a different item table (report the address).
Numbers, not squinting. Read-only. thor-rl:cu130.
"""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
ROOT = "/work"; RINGS = os.path.join(ROOT, "link/sessions/ramhits_solo")
IWRAM = 32 * 1024; SNAP = (256 + 32) * 1024
CAMX, CAMY = 0x02B82, 0x02B86
SPRITE_X = {0x03846 + 4 * i for i in range(16)}


def load_iw(d):
    meta = json.load(open(os.path.join(d, "meta.json"))); n = meta["n"]
    a = np.empty((n, IWRAM), np.uint8)
    with open(os.path.join(d, "ring.bin"), "rb") as f:
        for t in range(n):
            f.seek(t * SNAP); a[t] = np.frombuffer(f.read(IWRAM), np.uint8)
    return a


def spawns_in_ring(a):
    n = a.shape[0]
    v = a[:, :-1].astype(np.int32) | (a[:, 1:].astype(np.int32) << 8)   # (n, 32767) u16 LE
    cx = v[:, CAMX]; cy = v[:, CAMY]
    sx = v - cx[:, None]                       # screen-x projection per address
    onx = (sx >= 0) & (sx < 240)
    q = max(4, n // 4)
    early_absent = ~onx[:q].any(axis=0)        # never on-screen in first quarter
    late_on = onx[-q:].mean(axis=0) > 0.85     # on-screen almost all of last quarter
    late_stable = v[-q:].std(axis=0) < 2.0     # value ~constant late (at-rest object)
    cand = np.where(early_absent & late_on & late_stable)[0]
    out = []
    for c in cand:
        if not (0x2000 <= c < 0x4000):
            continue
        cy_ = int(cy[-1]); sx_ = int(v[-1, c] - cx[-1]); sy_ = int(v[-1, c + 2] - cy_) if c + 2 < v.shape[1] else -1
        if not (0 <= sy_ < 200):               # paired y also plausible-ish
            continue
        out.append((c, sx_, sy_))
    # collapse addresses within 3 bytes (same object) keeping the lowest
    out.sort()
    merged = []
    for c, sx_, sy_ in out:
        if merged and c - merged[-1][0] <= 3:
            continue
        merged.append((c, sx_, sy_))
    return merged


def main():
    tallies = {"sprite_table": 0, "other": {}}
    for d in sorted(glob.glob(os.path.join(RINGS, "20260808-*-hit"))):
        a = load_iw(d)
        sp = spawns_in_ring(a)
        if not sp:
            continue
        tag = os.path.basename(d)[9:15]
        for (c, sx_, sy_) in sp:
            where = "SPRITE-TABLE" if c in SPRITE_X else "OTHER"
            if c in SPRITE_X:
                tallies["sprite_table"] += 1
            else:
                tallies["other"][c] = tallies["other"].get(c, 0) + 1
            print(f"{tag}: spawn @ 0x{c:04X} ({where}) -> screen ({sx_},{sy_})", flush=True)
    print("\n=== SUMMARY ===")
    print(f"spawns in known sprite-slot x-table (0x03846+4i): {tallies['sprite_table']}")
    if tallies["other"]:
        print("spawns at OTHER addresses (candidate new item table), addr -> #rings:")
        for c in sorted(tallies["other"], key=lambda k: -tallies["other"][k]):
            print(f"  0x{c:04X}: {tallies['other'][c]}")
    else:
        print("no non-sprite-table spawns found")
    print("DONE")


if __name__ == "__main__":
    main()
