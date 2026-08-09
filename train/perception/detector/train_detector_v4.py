"""train_detector_v4.py -- v3 recipe + ENEMY-BALANCED sampling. The one-line diagnosis
of v3: expanding to 13 rooms fixed Link (87.3, central/edge balanced) but crashed enemy
(68.6) because the NEW rooms are largely enemy-LESS -- the enemy signal got diluted, not
harder. So v4 keeps v3's data/net/no-aug exactly and only reweights the SAMPLER: frames
containing at least one enemy are duplicated in each epoch's index so enemy-present
frames make up ~half of what the net trains on. Labels, geometry, eval identical.
Success: held-out {2,4} Link >= 87 AND enemy >= 80 (the clean probe pass v3 missed).
Preserves all prior weights; writes detector_v4.pt + _eval_v4_result.json. thor-rl:cu130.
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
HELD_ROOMS = [2, 4]      # original held-out -- directly comparable to baseline/v2b/v3
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

    # ENEMY-BALANCED index: duplicate enemy-present frames until they are ~half of
    # each epoch. v3's train mix was the problem, not the net or the task.
    has_e = (E[:, :, 0] >= 0).any(axis=1)
    n_e, n_o = int(has_e.sum()), int((~has_e).sum())
    dup = max(0, int(round(n_o / max(1, n_e))) - 1)
    idx_all = np.arange(len(F))
    epoch_idx = np.concatenate([idx_all] + [idx_all[has_e]] * dup)
    print(f"train frames={len(F)} (enemy-present {n_e}, dup x{dup} -> epoch size "
          f"{len(epoch_idx)}, enemy share {(n_e*(dup+1))/len(epoch_idx):.2f})", flush=True)

    X = to_in(F); T = torch.from_numpy(make_targets(L, E, I))
    net = UNet().to(DEV); npar = sum(p.numel() for p in net.parameters())
    print(f"params={npar/1e6:.3f}M | NO-AUG | enemy-balanced | held={HELD_ROOMS} held_new={HELD_NEW}", flush=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50)
    bs, n = 32, len(epoch_idx)
    ckpt = os.path.join(HERE, "detector_v4.pt")
    for ep in range(50):
        net.train(); perm = torch.from_numpy(np.random.permutation(epoch_idx)); tot = 0.0
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
    res = {"params_M": round(npar / 1e6, 3), "train_rooms": TRAIN_ROOMS, "aug": False,
           "enemy_balanced": True, "dup": dup}
    res["fps_forward"], res["fps_end2end"] = bench_fps(net, np.load(os.path.join(DATA, "room2.npz"))["frames"][0])
    Fh, Lh, Eh, Ih = load_rooms(HELD_ROOMS); res["heldout_2_4"] = eval_split(net, Fh, Lh, Eh, Ih)
    Fn, Ln, En, In = load_rooms(HELD_NEW); res["heldnew_12"] = eval_split(net, Fn, Ln, En, In)
    dih = np.load(os.path.join(DATA, "item_held.npz"))
    res["item_held"] = eval_split(net, dih["frames"], dih["link"], dih["enem"], dih["item"], want_link=False)
    json.dump(res, open(os.path.join(HERE, "_eval_v4_result.json"), "w"), indent=2)

    h = res["heldout_2_4"]
    print("\n=== v4 HELD-OUT {2,4} vs GATES (v3: link 87.3 / enemy 68.6) ===", flush=True)
    print(f"  LINK  @4={h['link@4']}% (bar 87)  central={h['link@4_ctr']} edge={h['link@4_edge']}", flush=True)
    print(f"  ENEMY @4={h['enemy@4']}% (bar 80)  fp={h['enemy_fp']}", flush=True)
    print(f"  new-dungeon room12: link@4={res['heldnew_12']['link@4']} enemy@4={res['heldnew_12']['enemy@4']}", flush=True)
    print(f"  item(keydrop held)@4={res['item_held']['item@4']}", flush=True)
    print("RESULT", json.dumps(res), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
