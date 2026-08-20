"""Detector v5: item-class data pass (composer arc, 2026-08-20).

Changes vs baseline train_detector.py (recipe otherwise IDENTICAL: same net,
30 ep, Adam 1e-3, bs32, weight 1+10t, no aug):
  1. item labels: item_train/item_held REGENERATED slot-3-only (statues no
     longer positives); room*.npz item targets ZEROED (plain walks drop nothing
     -> statues/junk become true background there too).
  2. NEW fresh-drop files (gen_drops.py): drops_room4 + drops_room2_train in
     TRAIN, drops_room2_held = the held bar.
  3. CHANNEL-MASKED loss: drops files train link+item only (enemy mask 0) --
     the v3 lesson (partially-labeled enemies in new rooms poison the enemy
     head) stays honored without fixing the enemy labeler today.
  4. Eval: per-file groups (NEVER pooled -- the 2026-08-08 lesson), and every
     number is computed for BOTH v5 and baseline detector.pt in THIS harness so
     no-regress bars compare like with like.

PRE-REGISTERED BARS (json "bars"): drops_room2_held item@4 >= 80; item FP-frame
rate (max item hm >= 0.6, the router's gate) <= 5% on room4.npz AND room2.npz;
enemy@4/link@4 on held rooms{2,4} >= baseline-in-this-harness minus 2pts;
item_held (bank keys) item@4 >= 95.

  docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/vga \\
      -w /vga/train/perception/detector thor-rl:cu130 python3 -u train_detector_v5.py
"""
import os
import json
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
from train_detector import Net, to_in, make_targets, decode_peak, heatmap  # noqa: E402,F401

DEV = "cuda"
NCH = 3

# (file, (link,enemy,item) loss mask, zero_item)
# paste_train is ITEM-ONLY (0,0,1): its room2-half backgrounds must not leak
# link/enemy supervision into the held-room evals.
TRAIN_SPEC = [
    ("room0.npz",       (1, 1, 1), True),
    ("room1.npz",       (1, 1, 1), True),
    ("room3.npz",       (1, 1, 1), True),
    ("item_train.npz",  (1, 1, 1), False),
    ("drops_room4.npz", (1, 0, 1), False),
    ("paste_train.npz", (0, 0, 1), False),
    ("room4_negs.npz",  (0, 0, 1), False),   # v5b: explicit statue negatives, seed-
                                             # disjoint from the room4.npz FP eval walk
]
ROOM_HELD = ["room2.npz", "room4.npz"]          # link+enemy eval (item labels polluted -> ignored)
ITEM_HELD = ["item_held.npz"]                    # bank keys (regenerated slot3-only)
DROP_HELD = ["paste_room2_held.npz"]             # room2-context bar (SYNTHETIC: pasted real
                                                 # sprite on disjoint real room2 frames; real
                                                 # captures structurally contaminated -- see
                                                 # gen_paste.py. Live judge = composer re-fire.)
FP_FILES = ["room4.npz", "room2.npz"]            # plain walks: any item peak >=0.6 = FP frame


def load_one(f, zero_item):
    d = np.load(os.path.join(DATA, f))
    item = d["item"].copy()
    if zero_item:
        item[:] = -1
    return d["frames"], d["link"], d["enem"], item


def loc_group(net, files, zero_item=False):
    """Per-group localization (single-entity frames for enemy/item, as baseline)."""
    F, L, E, I = [], [], [], []
    for f in files:
        if not os.path.exists(os.path.join(DATA, f)):
            return None
        a, b, c, d = load_one(f, zero_item)
        F.append(a); L.append(b); E.append(c); I.append(d)
    F = np.concatenate(F); L = np.concatenate(L); E = np.concatenate(E); I = np.concatenate(I)
    net.eval()
    N = len(F)
    lk4 = 0; e4 = en = 0; i4 = it = 0
    with torch.no_grad():
        for k in range(0, N, 64):
            pr = net(to_in(F[k:k + 64]).to(DEV)).cpu().numpy()
            for j in range(len(pr)):
                px, py = decode_peak(pr[j, 0])
                gx, gy = L[k + j]
                lk4 += max(abs(px - gx), abs(py - gy)) <= 4
                es = [e for e in E[k + j] if e[0] >= 0]
                if len(es) == 1:
                    ex, ey = decode_peak(pr[j, 1])
                    en += 1
                    e4 += max(abs(ex - es[0][0]), abs(ey - es[0][1])) <= 4
                its = [e for e in I[k + j] if e[0] >= 0]
                if len(its) == 1:
                    ix, iy = decode_peak(pr[j, 2])
                    it += 1
                    i4 += max(abs(ix - its[0][0]), abs(iy - its[0][1])) <= 4
    f_ = lambda h, n_: float(round(100.0 * h / n_, 1)) if n_ else None  # plain float: np.float64 breaks json.dumps
    return {"link@4": f_(lk4, N), "enemy@4": f_(e4, en), "enemy_n": en,
            "item@4": f_(i4, it), "item_n": it, "frames": N}


def fp_rate(net, f, thr=0.6):
    d = np.load(os.path.join(DATA, f))
    F = d["frames"]
    net.eval()
    hits = 0
    with torch.no_grad():
        for k in range(0, len(F), 64):
            pr = net(to_in(F[k:k + 64]).to(DEV)).cpu().numpy()
            hits += int((pr[:, 2].max(axis=(1, 2)) >= thr).sum())
    return float(round(100.0 * hits / len(F), 1))


def main():
    F_, L_, E_, I_, M_ = [], [], [], [], []
    for f, mask, zi in TRAIN_SPEC:
        p = os.path.join(DATA, f)
        if not os.path.exists(p):
            print(f"  (skip missing {f})", flush=True)
            continue
        a, b, c, d = load_one(f, zi)
        F_.append(a); L_.append(b); E_.append(c); I_.append(d)
        M_.append(np.tile(np.array(mask, np.float32), (len(a), 1)))
        print(f"  {f}: {len(a)} frames mask={mask} zero_item={zi}", flush=True)
    F = np.concatenate(F_); L = np.concatenate(L_)
    E = np.concatenate(E_); I = np.concatenate(I_); M = np.concatenate(M_)
    print(f"train {len(F)} frames", flush=True)
    X = to_in(F)
    T = torch.from_numpy(make_targets(L, E, I))
    Mt = torch.from_numpy(M)
    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    n = len(X)
    bs = 32
    t0 = time.time()
    for ep in range(30):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x = X[idx].to(DEV); t = T[idx].to(DEV)
            m = Mt[idx].to(DEV)[:, :, None, None]
            p = net(x)
            w = 1 + 10 * t
            loss = (m * w * (p - t) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % 5 == 0 or ep == 29:
            print(f"ep{ep} loss={tot / n:.4f}", flush=True)
    print(f"trained in {time.time() - t0:.0f}s", flush=True)
    torch.save(net.state_dict(), os.path.join(HERE, "detector_v5.pt"))

    base = Net().to(DEV)
    base.load_state_dict(torch.load(os.path.join(HERE, "detector.pt"), map_location=DEV))
    base.eval()
    res = {}
    for name, model in (("v5", net), ("baseline", base)):
        res[name] = {
            "rooms_held": loc_group(model, ROOM_HELD, zero_item=True),
            "bank_item_held": loc_group(model, ITEM_HELD),
            "drops_room2_held": loc_group(model, DROP_HELD),
            "fp": {f: fp_rate(model, f) for f in FP_FILES},
        }
    b_rooms = res["baseline"]["rooms_held"]
    v_rooms = res["v5"]["rooms_held"]
    dh = res["v5"]["drops_room2_held"]
    bars = {
        "paste_room2_held_item@4_ge80": bool(dh and dh["item@4"] is not None and dh["item@4"] >= 80),
        "fp_room4_le5": bool(res["v5"]["fp"]["room4.npz"] <= 5),
        "fp_room2_le5": bool(res["v5"]["fp"]["room2.npz"] <= 5),
        "enemy_noregress": bool(v_rooms["enemy@4"] >= b_rooms["enemy@4"] - 2),
        "link_noregress": bool(v_rooms["link@4"] >= b_rooms["link@4"] - 2),
        "bank_item_ge95": bool(res["v5"]["bank_item_held"]["item@4"] is not None
                               and res["v5"]["bank_item_held"]["item@4"] >= 95),
    }
    res["bars"] = bars
    res["all_pass"] = bool(all(bars.values()))
    print("RESULT", json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(HERE, "_detector_v5_result.json"), "w"), indent=2)
    print("BARS " + " ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in bars.items()), flush=True)
    print("ALL_PASS" if res["all_pass"] else "SOME_FAIL", flush=True)


if __name__ == "__main__":
    main()
