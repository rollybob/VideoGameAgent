"""train_detector_v3.py -- DATA-DRIVEN Link retrain: bigger U-Net, NO augmentation, EXPANDED
room set (13 rooms incl. new dungeon rooms) to close the held-out Link generalization gap.

Rationale: v2/v2b showed compute (net size + aug) only SEE-SAWS Link vs enemy on 3 train rooms;
the binding constraint is DATA (background/appearance diversity). This adds rooms {5,6,7,8,11,13,14}
to the {0,1,3} train set (10 rooms + item_train), holds out the original {2,4} (directly comparable
to baseline 82.7 / v2b 84.8) plus a brand-new dungeon room {12} as a generalization probe. NO color
aug (it hurt enemy). Seeded. Preserves detector.pt / detector_v2.pt / detector_v2b.pt; writes
detector_v3.pt + _eval_v3_result.json, checkpointing every 5 epochs. Success: held Link@4 >= 90
while enemy stays >= 80. thor-rl:cu130.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from train_detector import make_targets, to_in                 # identical labels/geometry
from train_detector_v2 import UNet                             # bigger net (v2b arch)
from eval_detector import eval_split, bench_fps                # identical honest eval
DATA = os.path.join(HERE, "data"); DEV = "cuda"
TRAIN_ROOMS = [0, 1, 3, 5, 7, 8, 11, 13, 14]   # room6 excluded: degenerate (Link pinned 900 frames)
HELD_ROOMS = [2, 4]      # original held-out -- directly comparable to baseline/v2/v2b
HELD_NEW = [12]          # brand-new dungeon room -- generalization probe
torch.manual_seed(0); np.random.seed(0)


def load_rooms(rooms):
    F, L, E, I = [], [], [], []
    for rm in rooms:
        d = np.load(os.path.join(DATA, f"room{rm}.npz"))
        F.append(d["frames"]); L.append(d["link"]); E.append(d["enem"]); I.append(d["item"])
    return [np.concatenate(x) for x in (F, L, E, I)]


def main():
    Ftr, Ltr, Etr, Itr = load_rooms(TRAIN_ROOMS)
    dit = np.load(os.path.join(DATA, "item_train.npz"))
    F = np.concatenate([Ftr, dit["frames"]]); L = np.concatenate([Ltr, dit["link"]])
    E = np.concatenate([Etr, dit["enem"]]); I = np.concatenate([Itr, dit["item"]])
    print(f"train frames={len(F)}  rooms {TRAIN_ROOMS} + item_train ({len(dit['frames'])})", flush=True)

    X = to_in(F); T = torch.from_numpy(make_targets(L, E, I))
    net = UNet().to(DEV); npar = sum(p.numel() for p in net.parameters())
    print(f"params={npar/1e6:.3f}M | NO-AUG | held={HELD_ROOMS} held_new={HELD_NEW}", flush=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50)
    bs, n = 32, len(X)
    ckpt = os.path.join(HERE, "detector_v3.pt")
    for ep in range(50):
        net.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x = X[idx].to(DEV); t = T[idx].to(DEV)              # NO augmentation
            p = net(x); w = 1 + 10 * t; loss = (w * (p - t) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        if ep % 5 == 0 or ep == 49:
            torch.save(net.state_dict(), ckpt)                  # checkpoint often
            print(f"ep{ep} loss={tot/n:.4f} lr={sched.get_last_lr()[0]:.2e} (ckpt)", flush=True)
    torch.save(net.state_dict(), ckpt)

    net.eval()
    res = {"params_M": round(npar / 1e6, 3), "train_rooms": TRAIN_ROOMS, "aug": False}
    res["fps_forward"], res["fps_end2end"] = bench_fps(net, np.load(os.path.join(DATA, "room2.npz"))["frames"][0])
    Fh, Lh, Eh, Ih = load_rooms(HELD_ROOMS); res["heldout_2_4"] = eval_split(net, Fh, Lh, Eh, Ih)
    Fn, Ln, En, In = load_rooms(HELD_NEW); res["heldnew_12"] = eval_split(net, Fn, Ln, En, In)
    dih = np.load(os.path.join(DATA, "item_held.npz"))
    res["item_held"] = eval_split(net, dih["frames"], dih["link"], dih["enem"], dih["item"], want_link=False)
    json.dump(res, open(os.path.join(HERE, "_eval_v3_result.json"), "w"), indent=2)

    h = res["heldout_2_4"]
    print("\n=== v3 HELD-OUT {2,4} vs GATES (baseline 82.7 / v2b 84.8) ===", flush=True)
    print(f"  LINK  @4={h['link@4']}% (bar 90)  central={h['link@4_ctr']} edge={h['link@4_edge']}", flush=True)
    print(f"  ENEMY @4={h['enemy@4']}% (bar 80)  fp={h['enemy_fp']}", flush=True)
    print(f"  new-dungeon room12: link@4={res['heldnew_12']['link@4']} enemy@4={res['heldnew_12']['enemy@4']}", flush=True)
    print(f"  item(keydrop held)@4={res['item_held']['item@4']}", flush=True)
    print("RESULT", json.dumps(res), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
