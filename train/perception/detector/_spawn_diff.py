"""_spawn_diff.py -- nail the exact item bytes in ring 205603, assumption-free.
Find the spawn frame (byte 0x33F0 flips when the pot item appears), diff snapshot[spawn-1] vs
[spawn+1] over IWRAM, and report every changed byte whose NEW value is a plausible screen coord
(the visible drop is at ~x=117,y=127). Adjacent x~117 / y~127 bytes = the item's real position
fields. Also list the full changed-byte cluster so we see the object's structure. thor-rl:cu130.
"""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
RINGS = "/work/link/sessions/ramhits_solo"
SNAP = (256 + 32) * 1024; IWRAM = 32 * 1024
d = glob.glob(os.path.join(RINGS, "*205603-hit"))[0]
n = json.load(open(os.path.join(d, "meta.json")))["n"]
A = np.empty((n, SNAP), np.uint8)
with open(os.path.join(d, "ring.bin"), "rb") as f:
    for t in range(n):
        f.seek(t * SNAP); A[t] = np.frombuffer(f.read(SNAP), np.uint8)

# spawn frame: first t where byte 0x33F0 differs from its early value
base = A[0, 0x33F0]
spawn = next((t for t in range(n) if A[t, 0x33F0] != base), None)
print(f"spawn frame t={spawn} (byte0x33F0 {base}->{A[spawn,0x33F0]}); item visible ~x=117 y=127", flush=True)

pre, post = A[spawn - 1], A[spawn + 1]
iw_changed = np.where(pre[:IWRAM] != post[:IWRAM])[0]
print(f"\n{len(iw_changed)} IWRAM bytes changed across the spawn (2-frame window):", flush=True)
# bytes whose new value is a plausible screen X (110-125) or Y (120-134)
print("\n-- changed bytes with new value near the drop (x~117 / y~127) --", flush=True)
for a in iw_changed:
    v = int(post[a])
    if 108 <= v <= 136:
        print(f"  0x{a:04X}: {int(pre[a]):3d} -> {v:3d}", flush=True)

print("\n-- full changed cluster in 0x3300-0x3900 (object region) --", flush=True)
for a in iw_changed:
    if 0x3300 <= a < 0x3900:
        print(f"  0x{a:04X}: {int(pre[a]):3d} -> {int(post[a]):3d}", flush=True)
print("DONE", flush=True)
