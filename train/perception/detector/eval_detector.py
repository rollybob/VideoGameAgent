"""eval_detector.py -- honest, pre-registered evaluation of the RAM-taught heatmap detector.

Reproduces the headline metrics AND closes two gaps in the in-train eval:
  (1) MULTI-ENTITY frames are scored (multi-peak decode + greedy nearest-match),
      not silently skipped -- the spec's "nearest-match for multiple enemies".
  (2) END-TO-END FPS (uint8 frame -> preprocess -> forward -> decode) measured,
      not just the forward pass.
Also DIAGNOSES the held-out Link@4 near-miss (87.4 vs 90 bar): splits Link error by
central vs off-center (camera-clamped edge) frames, and reports the systematic (dx,dy)
bias -- a constant offset would be a trivial label fix, edge failures would not.

Read-only: loads the existing detector.pt, no retraining. Prints a compact JSON + summary.
Run: docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/work -w /work \
        thor-rl:cu130 python3 -u train/perception/detector/eval_detector.py
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
from train_detector import Net, to_in, HH, HW, STRIDE, NCH, DEV  # reuse exact architecture

# room-level splits (same as train_detector); item files scored as the bonus collectible class
LINK_ENEMY = {"train": ["room0.npz", "room1.npz", "room3.npz"],
              "held":  ["room2.npz", "room4.npz"]}
ITEM = {"train": "item_train.npz", "held": "item_held.npz"}
# a Link screen pos in the outer margin => camera clamped / Link off-center (the hard case)
MARGIN = dict(xlo=60, xhi=180, ylo=45, yhi=120)


def peak_single(hm):
    """Single sub-pixel peak (Link): argmax + 5x5 centroid. Mirrors train_detector.decode_peak."""
    idx = int(hm.argmax()); hy, hx = divmod(idx, HW)
    y0, y1 = max(0, hy - 2), min(HH, hy + 3); x0, x1 = max(0, hx - 2), min(HW, hx + 3)
    win = np.clip(hm[y0:y1, x0:x1], 0, None); s = win.sum()
    if s > 1e-6:
        ys, xs = np.mgrid[y0:y1, x0:x1]; hy = (win * ys).sum() / s; hx = (win * xs).sum() / s
    return hx * STRIDE, hy * STRIDE


def peaks_multi(hm, thr=0.3, radius=3, maxn=8):
    """Multiple sub-pixel local maxima (enemies/items): iterative argmax + NMS suppression."""
    H, W = hm.shape; flat = hm.astype(np.float32).copy(); out = []
    for _ in range(maxn):
        idx = int(flat.argmax()); v = float(flat.flat[idx])
        if v < thr: break
        hy, hx = divmod(idx, W)
        y0, y1 = max(0, hy - 2), min(H, hy + 3); x0, x1 = max(0, hx - 2), min(W, hx + 3)
        win = np.clip(hm[y0:y1, x0:x1], 0, None); s = win.sum()
        if s > 1e-6:
            ys, xs = np.mgrid[y0:y1, x0:x1]; cy = (win * ys).sum() / s; cx = (win * xs).sum() / s
        else:
            cy, cx = hy, hx
        out.append((cx * STRIDE, cy * STRIDE))
        yy0, yy1 = max(0, hy - radius), min(H, hy + radius + 1)
        xx0, xx1 = max(0, hx - radius), min(W, hx + radius + 1)
        flat[yy0:yy1, xx0:xx1] = -1.0
    return out


def match(preds, gts, r):
    """Greedy nearest-match (L-inf). Returns (#gt matched within r, #gt, #unmatched preds=FP)."""
    used = [False] * len(preds); hit = 0
    for gx, gy in gts:
        best, bestd = -1, 1e9
        for k, (px, py) in enumerate(preds):
            if used[k]: continue
            d = max(abs(px - gx), abs(py - gy))
            if d < bestd: bestd, best = d, k
        if best >= 0 and bestd <= r: used[best] = True; hit += 1
    return hit, len(gts), used.count(False)


def load(files):
    F, L, E, I = [], [], [], []
    for f in files:
        d = np.load(os.path.join(DATA, f))
        F.append(d["frames"]); L.append(d["link"]); E.append(d["enem"]); I.append(d["item"])
    return np.concatenate(F), np.concatenate(L), np.concatenate(E), np.concatenate(I)


def predict(net, frames):
    """Batched forward; returns raw heatmaps (N,NCH,HH,HW) numpy."""
    outs = []
    with torch.no_grad():
        for k in range(0, len(frames), 64):
            outs.append(net(to_in(frames[k:k + 64]).to(DEV)).cpu().numpy())
    return np.concatenate(outs)


def eval_split(net, frames, link, enem, item, want_link=True):
    """Full honest metrics on one split. Multi-peak nearest-match for enemy/item."""
    hm = predict(net, frames)
    N = len(frames)
    # --- Link: single peak, with central/edge diagnosis + signed bias ---
    lk = {"h4": 0, "h8": 0, "n": 0, "h4_edge": 0, "n_edge": 0, "h4_ctr": 0, "n_ctr": 0}
    dxs, dys = [], []
    if want_link:
        for j in range(N):
            px, py = peak_single(hm[j, 0]); gx, gy = link[j]
            d = max(abs(px - gx), abs(py - gy)); lk["n"] += 1
            lk["h4"] += d <= 4; lk["h8"] += d <= 8
            dxs.append(px - gx); dys.append(py - gy)
            edge = not (MARGIN["xlo"] <= gx <= MARGIN["xhi"] and MARGIN["ylo"] <= gy <= MARGIN["yhi"])
            if edge: lk["n_edge"] += 1; lk["h4_edge"] += d <= 4
            else:    lk["n_ctr"] += 1;  lk["h4_ctr"] += d <= 4
    # --- Enemy(ch1)/Item(ch2): multi-peak, nearest-match recall over ALL frames ---
    def score(ch, gt_arr):
        h4 = h8 = ngt = fp = 0
        for j in range(N):
            gts = [tuple(e) for e in gt_arr[j] if e[0] >= 0]
            if not gts: continue
            pk = peaks_multi(hm[j, ch])
            a, g, f = match(pk, gts, 4); h4 += a; ngt += g; fp += f
            b, _, _ = match(pk, gts, 8); h8 += b
        return {"r4": h4, "r8": h8, "n": ngt, "fp": fp}
    en = score(1, enem); it = score(2, item)
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    out = {
        "enemy@4": pct(en["r4"], en["n"]), "enemy@8": pct(en["r8"], en["n"]),
        "enemy_gt": en["n"], "enemy_fp": en["fp"],
        "item@4": pct(it["r4"], it["n"]), "item@8": pct(it["r8"], it["n"]),
        "item_gt": it["n"], "item_fp": it["fp"],
    }
    if want_link:
        out.update({
            "link@4": pct(lk["h4"], lk["n"]), "link@8": pct(lk["h8"], lk["n"]), "link_n": lk["n"],
            "link@4_edge": pct(lk["h4_edge"], lk["n_edge"]), "link_n_edge": lk["n_edge"],
            "link@4_ctr": pct(lk["h4_ctr"], lk["n_ctr"]), "link_n_ctr": lk["n_ctr"],
            "link_bias_dx": round(float(np.mean(dxs)), 2), "link_bias_dy": round(float(np.mean(dys)), 2),
            "link_std_dx": round(float(np.std(dxs)), 2), "link_std_dy": round(float(np.std(dys)), 2),
        })
    return out


def bench_fps(net, frame):
    """forward-only vs true end-to-end (uint8 numpy -> GPU -> forward -> decode), bs=1."""
    x1 = to_in(frame[None]).to(DEV)
    with torch.no_grad():
        for _ in range(20): net(x1)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(300): net(x1)
        torch.cuda.synchronize(); fwd = 300 / (time.time() - t0)

    def e2e(fr):
        x = torch.from_numpy(fr[None]).to(DEV).float().permute(0, 3, 1, 2).div_(255.)
        with torch.no_grad():
            p = net(x)[0].cpu().numpy()
        return peak_single(p[0]), peaks_multi(p[1]), peaks_multi(p[2])
    for _ in range(20): e2e(frame)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(300): e2e(frame)
    torch.cuda.synchronize(); full = 300 / (time.time() - t0)
    return round(fwd, 1), round(full, 1)


def main():
    net = Net().to(DEV); net.load_state_dict(torch.load(os.path.join(HERE, "detector.pt"), map_location=DEV))
    net.eval()
    npar = sum(p.numel() for p in net.parameters())

    Fhe, Lhe, Ehe, Ihe = load(LINK_ENEMY["held"]); Ftr, Ltr, Etr, Itr = load(LINK_ENEMY["train"])
    dih = np.load(os.path.join(DATA, ITEM["held"])); dit = np.load(os.path.join(DATA, ITEM["train"]))

    res = {"params_M": round(npar / 1e6, 3)}
    res["fps_forward"], res["fps_end2end"] = bench_fps(net, Fhe[0])
    res["heldout"] = eval_split(net, Fhe, Lhe, Ehe, Ihe)
    res["train"] = eval_split(net, Ftr, Ltr, Etr, Itr)
    res["item_held"] = eval_split(net, dih["frames"], dih["link"], dih["enem"], dih["item"], want_link=False)
    res["item_train"] = eval_split(net, dit["frames"], dit["link"], dit["enem"], dit["item"], want_link=False)
    # per-room breakdown (held-out rooms individually)
    res["per_room_held"] = {}
    for f in LINK_ENEMY["held"]:
        d = np.load(os.path.join(DATA, f))
        res["per_room_held"][f] = eval_split(net, d["frames"], d["link"], d["enem"], d["item"])

    print("RESULT", json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(HERE, "_eval_result.json"), "w"), indent=2)

    h = res["heldout"]
    print("\n=== HELD-OUT (rooms 2,4) vs PRE-REGISTERED GATES ===", flush=True)
    print(f"  SPEED   forward={res['fps_forward']} FPS (bar>=100)  end2end={res['fps_end2end']} FPS (bar>=60)")
    print(f"  LINK    @4={h['link@4']}% (bar>=90)  @8={h['link@8']}%  | central @4={h['link@4_ctr']}% (n={h['link_n_ctr']})  EDGE @4={h['link@4_edge']}% (n={h['link_n_edge']})")
    print(f"  LINK    systematic bias dx={h['link_bias_dx']} dy={h['link_bias_dy']} (std {h['link_std_dx']}/{h['link_std_dy']})  <- constant bias = trivial fix")
    print(f"  ENEMY   @4={h['enemy@4']}% (bar>=80)  @8={h['enemy@8']}%  gt={h['enemy_gt']} fp={h['enemy_fp']}  <- MULTI-PEAK, all frames")
    print(f"  ITEM    @4(held npz)={res['item_held']['item@4']}%  gt={res['item_held']['item_gt']}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
