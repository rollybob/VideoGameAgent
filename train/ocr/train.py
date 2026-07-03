"""
F1 OCR trainer (CONTAINER side -- runs in thor-torch:cu130; torch + numpy only).

Loads the synthetic .npz produced by gen_data.py, trains the CRNN with CTC loss,
evaluates character error rate (CER) + exact-line accuracy by greedy CTC decode on
the val set, checkpoints the best model, and honours a wallclock cap so an
unattended thor-job stops cleanly. All progress is printed for `thor-job tail`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CRNN, WIDTH_DOWNSAMPLE  # noqa: E402


def load_split(path):
    d = np.load(path)
    images, widths = d["images"], d["widths"]
    flat, lengths = d["labels"], d["label_lengths"]
    labels, off = [], 0
    for n in lengths:
        labels.append(flat[off:off + n])
        off += int(n)
    return images, widths, labels


def lev(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[m]


def greedy_decode(argmax_row, blank=0):
    out, prev = [], -1
    for a in argmax_row:
        a = int(a)
        if a != prev and a != blank:
            out.append(a)
        prev = a
    return out


def batches(n, bs, shuffle, rng):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def to_tensor(images, widths, sel, device):
    batch = torch.from_numpy(images[sel]).float().div_(255.0)
    if batch.ndim == 4:                              # (B,H,W,3) RGB -> (B,3,H,W)
        x = batch.permute(0, 3, 1, 2).contiguous().to(device)
    else:                                            # (B,H,W) grayscale -> (B,1,H,W)
        x = batch.unsqueeze(1).to(device)
    in_len = np.clip(widths[sel] // WIDTH_DOWNSAMPLE, 1, images.shape[2] // WIDTH_DOWNSAMPLE)
    return x, torch.from_numpy(in_len.astype(np.int64))


@torch.no_grad()
def evaluate(model, images, widths, labels, device, limit=1000):
    model.eval()
    n = min(limit, len(labels))
    tot_cer_num = tot_cer_den = exact = 0
    for sel in batches(n, 256, False, None):
        x, in_len = to_tensor(images, widths, sel, device)
        logits = model(x)                       # (B,T,C)
        am = logits.argmax(-1).cpu().numpy()
        for k, s in enumerate(sel):
            pred = greedy_decode(am[k][:int(in_len[k])])
            gt = list(labels[s])
            tot_cer_num += lev(pred, gt)
            tot_cer_den += max(1, len(gt))
            exact += int(pred == gt)
    return tot_cer_num / max(1, tot_cer_den), exact / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--init", default="",
                    help="warm-start: load model weights from this checkpoint (.pt) before "
                         "training -- for fine-tuning a synthetic-trained CRNN on real crops")
    args = ap.parse_args()

    os.makedirs(args.ckpt, exist_ok=True)
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    n_classes = meta["n_classes"]
    in_ch = int(meta.get("channels", 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device} n_classes={n_classes} channels={in_ch} "
          f"fonts={meta.get('n_fonts', meta.get('font'))}", flush=True)

    tr_imgs, tr_w, tr_lab = load_split(os.path.join(args.data, "train.npz"))
    va_imgs, va_w, va_lab = load_split(os.path.join(args.data, "val.npz"))
    print(f"[train] train={len(tr_lab)} val={len(va_lab)} img={tr_imgs.shape[1:]}", flush=True)

    model = CRNN(n_classes, in_ch=in_ch).to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        if ck["meta"]["n_classes"] != n_classes or int(ck["meta"].get("channels", 1)) != in_ch:
            print(f"[train] WARN: --init meta mismatch (n_classes/channels); skipping warm-start")
        else:
            model.load_state_dict(ck["model"])
            print(f"[train] warm-started from {args.init} "
                  f"(its val_CER={ck.get('val_cer', '?')})", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    rng = np.random.default_rng(0)

    t0 = time.time()
    best_cer = float("inf")
    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = nb = 0
        for sel in batches(len(tr_lab), args.batch, True, rng):
            x, in_len = to_tensor(tr_imgs, tr_w, sel, device)
            tgt = torch.from_numpy(np.concatenate([tr_lab[s] for s in sel]).astype(np.int64)).to(device)
            tgt_len = torch.tensor([len(tr_lab[s]) for s in sel], dtype=torch.int64)
            logits = model(x)                                  # (B,T,C)
            logp = logits.log_softmax(-1).permute(1, 0, 2)     # (T,B,C) for CTC
            in_len_c = torch.clamp(in_len, max=logits.shape[1])
            loss = ctc(logp, tgt, in_len_c, tgt_len)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.detach())
            nb += 1
            if (time.time() - t0) / 60.0 > args.max_minutes:
                stop = True
                break
        sched.step()
        cer, acc = evaluate(model, va_imgs, va_w, va_lab, device)
        mins = (time.time() - t0) / 60.0
        print(f"[train] epoch {epoch:02d} loss={ep_loss / max(1, nb):.3f} "
              f"val_CER={cer:.4f} val_exact={acc:.3f} elapsed={mins:.1f}m", flush=True)
        if cer < best_cer:
            best_cer = cer
            torch.save({"model": model.state_dict(), "meta": meta, "val_cer": cer,
                        "val_exact": acc, "epoch": epoch},
                       os.path.join(args.ckpt, "best.pt"))
            print(f"[train] new best CER={cer:.4f} -> best.pt", flush=True)
        if stop:
            print(f"[train] hit max-minutes={args.max_minutes}; stopping.", flush=True)
            break

    print(f"[train] DONE best_val_CER={best_cer:.4f} total={(time.time() - t0) / 60.0:.1f}m", flush=True)


if __name__ == "__main__":
    main()
