import os, sys, cv2, math, shutil, argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple

# -----------------------------
# Utilities
# -----------------------------
def list_images(img_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in img_dir.glob("*") if p.suffix.lower() in exts])

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def downscale_gray(im: np.ndarray, size: int = 32) -> np.ndarray:
    # robust grayscale + nearest-neighbor to preserve pixel edges
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = im.shape[:2]
    s = size / min(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_NEAREST)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_NEAREST)
    return im

def dct_2d(a: np.ndarray) -> np.ndarray:
    # OpenCV dct expects float32
    return cv2.dct(a.astype(np.float32))

def phash(image: np.ndarray) -> np.ndarray:
    """
    64-bit perceptual hash (8x8 from 32x32 DCT, ignoring DC).
    Returns a boolean array of length 64.
    """
    g = downscale_gray(image, 32)
    d = dct_2d(g)
    d_small = d[:8, :8]
    # ignore top-left DC? classic pHash compares to median of 8x8
    med = np.median(d_small)
    bits = d_small > med
    return bits.flatten()

def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))

def compute_hsv_hist(img: np.ndarray, bins: int = 32) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv],[0],None,[bins],[0,180]).flatten()
    h_hist = h_hist / (np.sum(h_hist) + 1e-6)
    return h_hist

def try_import_ssim():
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim
    except Exception:
        return None

def try_import_kmeans():
    try:
        from sklearn.cluster import KMeans
        return KMeans
    except Exception:
        return None

# -----------------------------
# Main pruning pipeline
# -----------------------------
def uniform_thin(paths: List[Path], target_keep: int) -> List[Path]:
    if target_keep <= 0 or target_keep >= len(paths):
        return paths
    step = max(1, len(paths) // target_keep)
    return paths[::step][:target_keep * 2]  # keep a little extra for dedupe

def dedupe_phash(paths: List[Path], phash_thresh: int) -> Tuple[List[Path], List[Path]]:
    kept, dropped = [], []
    hashes = []
    # simple bucketing by first 12 bits to avoid O(N^2) full compare
    buckets = {}

    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            dropped.append(p); continue
        h = phash(img)
        key = "".join(['1' if x else '0' for x in h[:12]])
        bucket = buckets.setdefault(key, [])
        # compare only within bucket
        is_dup = False
        for i, (kh, kp) in enumerate(bucket):
            if hamming(h, kh) <= phash_thresh:
                dropped.append(p); is_dup = True; break
        if not is_dup:
            kept.append(p)
            bucket.append((h, p))
            hashes.append(h)

    return kept, dropped

def refine_ssim(kept_paths: List[Path], ssim_thresh: float) -> Tuple[List[Path], List[Path]]:
    ssim_fn = try_import_ssim()
    if ssim_fn is None:
        # SSIM not available
        return kept_paths, []
    final_keep, dropped = [], []
    # sliding window dedupe to avoid O(N^2)
    window = max(20, len(kept_paths)//20)  # compare with last few kept
    gray_cache = []

    for p in kept_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            dropped.append(p); continue
        # downscale for speed
        img_small = downscale_gray(img, 96)
        dup = False
        # compare with last few kept
        for g2 in gray_cache[-window:]:
            s = ssim_fn(img_small, g2, data_range=255)
            if s >= ssim_thresh:
                dup = True; break
        if dup:
            dropped.append(p)
        else:
            final_keep.append(p)
            gray_cache.append(img_small)
    return final_keep, dropped

def rebalance_diversity(kept_paths: List[Path], target_keep: int, clusters: int = 12) -> List[Path]:
    if target_keep <= 0 or len(kept_paths) <= target_keep:
        return kept_paths
    KMeans = try_import_kmeans()
    if KMeans is None:
        # fall back to uniform if sklearn not present
        step = max(1, len(kept_paths)//target_keep)
        return kept_paths[::step][:target_keep]

    # compute simple hue histogram vectors
    feats = []
    for p in kept_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            feats.append(np.zeros(32, dtype=np.float32))
        else:
            feats.append(compute_hsv_hist(img, 32).astype(np.float32))
    feats = np.stack(feats, axis=0)

    k = min(clusters, len(kept_paths))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(feats)
    labels = kmeans.labels_

    # sample evenly from clusters
    per_cluster = max(1, target_keep // k)
    selected = []
    for c in range(k):
        idxs = [i for i, lab in enumerate(labels) if lab == c]
        if not idxs: 
            continue
        step = max(1, len(idxs)//per_cluster)
        selected.extend([kept_paths[i] for i in idxs[::step]][:per_cluster])

    # fill if short
    if len(selected) < target_keep:
        remaining = [p for p in kept_paths if p not in selected]
        selected.extend(remaining[:target_keep - len(selected)])
    return selected[:target_keep]

def move_files(paths: List[Path], out_dir: Path):
    ensure_dir(out_dir)
    for p in paths:
        dst = out_dir / p.name
        try:
            shutil.move(str(p), str(dst))
        except Exception:
            # if name collision, append numeric suffix
            stem, ext = p.stem, p.suffix
            i = 1
            while (out_dir / f"{stem}_{i}{ext}").exists():
                i += 1
            shutil.move(str(p), str(out_dir / f"{stem}_{i}{ext}"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", type=str, required=True, help="Folder with captured frames")
    ap.add_argument("--target_keep", type=int, default=500, help="Final images to keep")
    ap.add_argument("--phash_thresh", type=int, default=8, help="Hamming distance threshold for pHash")
    ap.add_argument("--use_ssim", action="store_true", help="Enable SSIM refinement if scikit-image is available")
    ap.add_argument("--ssim_thresh", type=float, default=0.985, help="SSIM threshold for near-duplicates")
    ap.add_argument("--balance_clusters", type=int, default=0, help="KMeans clusters for diversity (0=off)")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    all_imgs = list_images(img_dir)
    if not all_imgs:
        print("No images found."); return

    print(f"Found {len(all_imgs)} images")

    # 1) uniform thinning (coarse)
    coarse = uniform_thin(all_imgs, max(args.target_keep * 2, args.target_keep + 200))
    print(f"After uniform thinning: {len(coarse)}")

    # 2) pHash near-duplicate removal
    kept, dropped1 = dedupe_phash(coarse, args.phash_thresh)
    print(f"After pHash dedupe: kept {len(kept)}, dropped {len(dropped1)}")

    # 3) SSIM refinement (optional)
    dropped2 = []
    if args.use_ssim:
        kept, dropped2 = refine_ssim(kept, args.ssim_thresh)
        print(f"After SSIM refinement: kept {len(kept)}, dropped {len(dropped2)}")

    # 4) Diversity balancing (optional)
    if args.balance_clusters > 0:
        kept_bal = rebalance_diversity(kept, args.target_keep, clusters=args.balance_clusters)
    else:
        # final uniform select if still too many
        step = max(1, len(kept)//args.target_keep) if len(kept) > args.target_keep else 1
        kept_bal = kept[::step][:args.target_keep]

    # Create output dirs
    clean_dir = img_dir.parent / "clean"
    dupes_dir = img_dir.parent / "dupes"
    ensure_dir(clean_dir); ensure_dir(dupes_dir)

    # Move selected keeps to clean/, everything else to dupes/
    keep_set = set(kept_bal)
    to_dupes = [p for p in all_imgs if p not in keep_set]

    print(f"Moving {len(keep_set)} to {clean_dir}")
    move_files(list(keep_set), clean_dir)
    print(f"Moving {len(to_dupes)} to {dupes_dir}")
    move_files(to_dupes, dupes_dir)

    print("Done.")
    print(f"Final keep: {len(list_images(clean_dir))} | Moved to dupes: {len(list_images(dupes_dir))}")

if __name__ == "__main__":
    main()