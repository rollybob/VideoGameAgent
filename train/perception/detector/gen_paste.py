"""Paste-augmented item data (detector v5 fallback, 2026-08-20).

Real room2 key captures are structurally contaminated: the floor key renders
only when Link is within ~40px (object-system culling) and the knight -- whose
trim shares the strict-gold palette, sitting ~10px away -- is always converged
there at that moment (three capture designs starved/mislabeled; probes in
gen_drops.py). GBA rendering IS sprite blitting, so a pasted key sprite is
pixel-identical to a rendered one: paste the REAL key sprite (cropped from a
drops_room4 frame at its verified label) onto REAL room frames at random floor
positions -> unlimited, perfectly-labeled keys in each room's visual context.

Files: paste_train.npz (room0/1/3 + room2 train-half backgrounds),
paste_room2_held.npz (DISJOINT room2 frames -- a SYNTHETIC held bar, stated as
such; the composer re-fire on the live bench is the real judge). Trainer masks
these (0,0,1): item channel only, so held-room link/enemy evals stay clean.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga \\
      -w /vga/train/perception/detector thor-rl:cu130 python3 -u gen_paste.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RNG = np.random.RandomState(11)
PER_FRAME_TRIES = 20


def key_sprite():
    """Crop the key sprite + binary mask from a verified drops_room4 label."""
    d = np.load(os.path.join(DATA, "drops_room4.npz"))
    fr, (ix, iy) = d["frames"][0], d["item"][0][0]
    x0, y0 = int(ix) - 7, int(iy) - 10
    patch = fr[y0:y0 + 20, x0:x0 + 14].copy()
    p = patch.astype(int)
    gold = (p[..., 0] >= 180) & (p[..., 1] >= 120) & (p[..., 2] <= 120)
    dark = (p[..., 0] < 70) & (p[..., 1] < 70) & (p[..., 2] < 70)
    mask = gold | dark
    # keep only the connected-ish core: drop mask rows/cols with <2 px (noise)
    keep_r = mask.sum(axis=1) >= 2
    keep_c = mask.sum(axis=0) >= 2
    ys = np.where(keep_r)[0]
    xs = np.where(keep_c)[0]
    patch = patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return patch, mask


def paste(frame, spr, msk, x, y):
    h, w = msk.shape
    out = frame.copy()
    y0, x0 = y - h // 2, x - w // 2
    if y0 < 24 or x0 < 0 or y0 + h > 160 or x0 + w > 240:
        return None
    region = out[y0:y0 + h, x0:x0 + w]
    region[msk] = spr[msk]
    return out


def far(px, py, pts, r):
    return all(abs(px - qx) > r or abs(py - qy) > r for qx, qy in pts)


def gen(bg_frames, bg_link, bg_enem, n_out, spr, msk):
    F, L, E, I = [], [], [], []
    idxs = RNG.permutation(len(bg_frames))
    for i in idxs:
        if len(F) >= n_out:
            break
        lx, ly = bg_link[i]
        en = [(ex, ey) for ex, ey in bg_enem[i] if ex >= 0]
        for _ in range(PER_FRAME_TRIES):
            x = int(RNG.randint(24, 216))
            y = int(RNG.randint(40, 140))
            if far(x, y, [(lx, ly)], 24) and far(x, y, en, 20):
                out = paste(bg_frames[i], spr, msk, x, y)
                if out is None:
                    continue
                F.append(out)
                L.append((lx, ly))
                e = np.full((8, 2), -1, np.int16)
                for j, (ex, ey) in enumerate(en[:8]):
                    e[j] = (ex, ey)
                E.append(e)
                it = np.full((8, 2), -1, np.int16)
                it[0] = (x, y)
                I.append(it)
                break
    return F, L, E, I


def main():
    spr, msk = key_sprite()
    print(f"key sprite {spr.shape[:2]}, mask px={int(msk.sum())}", flush=True)
    from PIL import Image
    Image.fromarray(np.kron(spr, np.ones((6, 6, 1), np.uint8))).save(
        os.path.join(HERE, "_dbg_sprite.png"))

    r2 = np.load(os.path.join(DATA, "room2.npz"))
    n2 = len(r2["frames"])
    split = n2 // 2
    sets = {
        "paste_train": [],
        "paste_room2_held": [],
    }
    # train backgrounds: rooms 0/1/3 + first half of room2
    for f in ("room0.npz", "room1.npz", "room3.npz"):
        d = np.load(os.path.join(DATA, f))
        sets["paste_train"].append(gen(d["frames"], d["link"], d["enem"], 200, spr, msk))
    sets["paste_train"].append(
        gen(r2["frames"][:split], r2["link"][:split], r2["enem"][:split], 250, spr, msk))
    sets["paste_room2_held"].append(
        gen(r2["frames"][split:], r2["link"][split:], r2["enem"][split:], 120, spr, msk))
    for name, parts in sets.items():
        F = [f for p in parts for f in p[0]]
        L = [x for p in parts for x in p[1]]
        E = [x for p in parts for x in p[2]]
        I = [x for p in parts for x in p[3]]
        np.savez_compressed(os.path.join(DATA, name + ".npz"),
                            frames=np.stack(F), link=np.array(L, np.int16),
                            enem=np.stack(E), item=np.stack(I))
        print(f"{name}: {len(F)} frames", flush=True)


if __name__ == "__main__":
    main()
