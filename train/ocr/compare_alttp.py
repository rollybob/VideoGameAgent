"""compare_alttp.py -- honest before/after: baseline vs fine-tuned CRNN on the held-out REAL
ALttP val lines (data_alttp/val.npz, un-augmented). Prints per-model CER + each read."""
import sys, json, numpy as np, torch
sys.path.insert(0, "/work/train/ocr")
from model import CRNN, WIDTH_DOWNSAMPLE
D = "/work/train/ocr/data_alttp"

def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = CRNN(ck["meta"]["n_classes"], in_ch=int(ck["meta"].get("channels", 3))).eval()
    m.load_state_dict(ck["model"]); return m

def greedy(am, chars, blank=0):
    out, prev = [], -1
    for a in am:
        a = int(a)
        if a != prev and a != blank: out.append(a)
        prev = a
    return "".join(chars[i - 1] for i in out if 1 <= i <= len(chars))

def lev(a, b):
    n, m = len(a), len(b)
    if n == 0 or m == 0: return max(n, m)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]; dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = cur
    return dp[m]

meta = json.load(open(f"{D}/meta.json")); chars = meta["chars"]
d = np.load(f"{D}/val.npz"); imgs, Wd, flat, LL = d["images"], d["widths"], d["labels"], d["label_lengths"]
labs, off = [], 0
for n in LL: labs.append(flat[off:off + int(n)]); off += int(n)
gt_txt = lambda idxs: "".join(chars[i - 1] for i in idxs if 1 <= i <= len(chars))

for name, p in [("BASELINE ", "/work/train/ocr/ckpt_ft_mfreal3/best.pt"),
                ("FINETUNED", "/work/train/ocr/ckpt_alttp/best.pt")]:
    m = load(p); tn = td = 0; rows = []
    with torch.no_grad():
        for k in range(len(labs)):
            x = torch.from_numpy(imgs[k:k + 1]).float().div_(255.).permute(0, 3, 1, 2)
            il = max(1, int(Wd[k]) // WIDTH_DOWNSAMPLE)
            pred = greedy(m(x).argmax(-1)[0].numpy()[:il], chars)
            gt = gt_txt(labs[k]); tn += lev(list(pred), list(gt)); td += max(1, len(gt))
            rows.append((gt, pred))
    print(f"=== {name}  val_CER={tn / max(1, td):.3f}  (n={len(labs)}) ===", flush=True)
    for gt, pred in rows:
        print(f"   GT  {gt!r}\n   OUT {pred!r}")
