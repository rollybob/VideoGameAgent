"""train_detector_v2.py -- Link-generalization retrain.

Diagnosis (eval_detector.py, 2026-08-08): held-out Link@4=82.7% (bar 90). NOT bias
(dx0.5/dy0.9), NOT edges (edge 89.5% > central 82.3%). It is a capacity/room-generalization
gap: train-room Link@4=92.3 -> held-room 82.7 (Link's sprite animates -- walk/sword -- so his
visual centroid wobbles off the fixed RAM anchor, and the tiny 0.11M net memorizes per-room
background cues). Speed has ~20x headroom, so capacity is nearly free.

This run changes exactly TWO things vs the baseline train_detector.py, holding the DATA
pipeline identical (same TRAIN/HELD files, same labels incl. the current item channel -- the
item-label transient is a SEPARATE forensic, not entangled here):
  (1) bigger net: ~0.7M-param U-Net WITH SKIP CONNECTIONS (finer spatial features -> sharper
      peaks) instead of the 0.11M plain encoder-decoder;
  (2) color augmentation (brightness/contrast jitter) for cross-room appearance generalization.
Baseline detector.pt is preserved; this writes detector_v2.pt + _eval_v2_result.json.
Success test (same honest metric): held-out Link@4 >= 90 while enemy stays >= 80.
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from train_detector import make_targets, to_in, HH, HW, STRIDE, NCH  # identical label/geometry
from eval_detector import eval_split, bench_fps, load, LINK_ENEMY, ITEM  # identical honest eval
DATA = os.path.join(HERE, "data"); DEV = "cuda"
AUG = os.environ.get("AUG", "1") == "1"   # v2b ablation: AUG=0 isolates whether color aug caused the enemy drop
TAG = os.environ.get("TAG", "v2")          # output tag -> detector_<TAG>.pt / _eval_<TAG>_result.json


class UNet(nn.Module):
    """~0.7M-param U-Net. Same I/O as baseline Net (in (3,160,240) -> out (3,80,120), stride 2),
    so eval_detector's decode/geometry apply unchanged. Skips e2->u2, e3->u1 sharpen peaks."""
    def __init__(s):
        super().__init__()
        def cbr(i, o, st=1): return nn.Sequential(nn.Conv2d(i, o, 3, st, 1), nn.BatchNorm2d(o), nn.ReLU(True))
        s.e1 = cbr(3, 32, 1)      # 160x240
        s.e2 = cbr(32, 64, 2)     # 80x120
        s.e3 = cbr(64, 128, 2)    # 40x60
        s.e4 = cbr(128, 128, 2)   # 20x30
        s.up1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True))   # ->40x60
        s.f1 = cbr(64 + 128, 64)  # fuse with e3
        s.up2 = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True))    # ->80x120
        s.f2 = cbr(64 + 64, 64)   # fuse with e2
        s.head = nn.Conv2d(64, NCH, 1)

    def forward(s, x):
        a = s.e1(x); b = s.e2(a); c = s.e3(b); d = s.e4(c)
        u = s.f1(torch.cat([s.up1(d), c], 1))
        u = s.f2(torch.cat([s.up2(u), b], 1))
        return s.head(u)


def augment(x):
    """Color-only aug on the [0,1] input tensor (B,3,H,W). Geometry untouched -> labels valid.
    HUD chrome at screen top rules out translation aug; brightness/contrast is the safe lever."""
    B = x.shape[0]
    bright = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
    contr = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
    x = x * bright
    m = x.mean(dim=(2, 3), keepdim=True)
    x = (x - m) * contr + m
    return x.clamp_(0.0, 1.0)


def main():
    # identical training data to baseline: rooms{0,1,3} (link/enemy) + item_train (item)
    Ftr, Ltr, Etr, Itr = load(LINK_ENEMY["train"])
    dit = np.load(os.path.join(DATA, ITEM["train"]))
    F = np.concatenate([Ftr, dit["frames"]]); L = np.concatenate([Ltr, dit["link"]])
    E = np.concatenate([Etr, dit["enem"]]); I = np.concatenate([Itr, dit["item"]])
    print(f"train frames={len(F)} (rooms{{0,1,3}}={len(Ftr)} + item_train={len(dit['frames'])})", flush=True)

    X = to_in(F); T = torch.from_numpy(make_targets(L, E, I))
    net = UNet().to(DEV); npar = sum(p.numel() for p in net.parameters())
    print(f"params={npar/1e6:.3f}M (baseline 0.11M) | TAG={TAG} AUG={AUG}", flush=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50)
    bs, n = 32, len(X)
    for ep in range(50):
        net.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x = augment(X[idx].to(DEV)) if AUG else X[idx].to(DEV); t = T[idx].to(DEV)
            p = net(x); w = 1 + 10 * t; loss = (w * (p - t) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        if ep % 5 == 0 or ep == 49:
            print(f"ep{ep} loss={tot/n:.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
    torch.save(net.state_dict(), os.path.join(HERE, f"detector_{TAG}.pt"))

    # honest eval, identical harness to the baseline
    net.eval()
    Fhe, Lhe, Ehe, Ihe = load(LINK_ENEMY["held"])
    dih = np.load(os.path.join(DATA, ITEM["held"]))
    res = {"params_M": round(npar / 1e6, 3)}
    res["fps_forward"], res["fps_end2end"] = bench_fps(net, Fhe[0])
    res["heldout"] = eval_split(net, Fhe, Lhe, Ehe, Ihe)
    res["item_held"] = eval_split(net, dih["frames"], dih["link"], dih["enem"], dih["item"], want_link=False)
    res["per_room_held"] = {}
    for f in LINK_ENEMY["held"]:
        d = np.load(os.path.join(DATA, f)); res["per_room_held"][f] = eval_split(net, d["frames"], d["link"], d["enem"], d["item"])
    json.dump(res, open(os.path.join(HERE, f"_eval_{TAG}_result.json"), "w"), indent=2)
    h = res["heldout"]
    print(f"\n=== {TAG} HELD-OUT vs GATES (baseline in parens) ===", flush=True)
    print(f"  SPEED  fwd={res['fps_forward']} e2e={res['fps_end2end']} FPS (bars 100/60)", flush=True)
    print(f"  LINK   @4={h['link@4']}% (bar 90; baseline 82.7)  central={h['link@4_ctr']} edge={h['link@4_edge']}", flush=True)
    print(f"  ENEMY  @4={h['enemy@4']}% (bar 80; baseline 91.4)  fp={h['enemy_fp']}", flush=True)
    print(f"  ITEM   @4(keydrop held)={res['item_held']['item@4']}% (baseline 100)", flush=True)
    print("RESULT", json.dumps(res), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
