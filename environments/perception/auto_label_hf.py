
import os, json, glob, argparse, cv2, torch
import numpy as np
from PIL import Image
from typing import List
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def xyxy_to_yolo(box, img_w, img_h):
    x1,y1,x2,y2 = box
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    cx = x1 + w/2.0; cy = y1 + h/2.0
    return cx/img_w, cy/img_h, w/img_w, h/img_h

def nn_resize_short(img, short_side: int):
    if short_side <= 0:
        return img, 1.0, 1.0
    H, W = img.shape[:2]
    s = short_side / float(min(H, W))
    newW, newH = int(round(W*s)), int(round(H*s))
    if newW == W and newH == H:
        return img, 1.0, 1.0
    out = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_NEAREST)
    return out, (W / newW), (H / newH)

def draw_legend(canvas):
    lines = [
        "Keys: [/] prev/next  SPACE toggle  x none  k all",
        "      a save&next  s save  n next  r recompute  q quit",
        "Colors: green=selected kept, red=selected off, cyan=kept, gray=off"
    ]
    y = 20
    for line in lines:
        cv2.putText(canvas, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        y += 22

def draw_dets(img, dets, keep_mask, highlight_idx=-1):
    out = img.copy()
    draw_legend(out)
    for i,(cls,score,box) in enumerate(dets):
        x1,y1,x2,y2 = map(int, box)
        # Colors: kept=cyan, disabled=gray; selected kept=green, selected disabled=red
        if i == highlight_idx:
            color = (0,255,0) if keep_mask[i] else (0,0,255)
        else:
            color = (0,200,255) if keep_mask[i] else (80,80,80)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        label = f"{i}:{cls} {score:.2f}"
        ytxt = max(40, y1-5)  # keep above legend
        cv2.putText(out, label, (x1, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if not keep_mask[i]:
            # strikethrough
            lx2 = x1 + max(40, len(label)*8)
            cv2.line(out, (x1, ytxt+3), (lx2, ytxt+3), color, 2)
    return out

def save_yolo(label_path, img_w, img_h, dets, keep_mask, class_to_id):
    lines = []
    for keep,(cls,score,box) in zip(keep_mask,dets):
        if not keep:
            continue
        if cls not in class_to_id:
            continue
        cls_id = class_to_id[cls]
        cx,cy,w,h = xyxy_to_yolo(box, img_w, img_h)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(label_path, "w") as f:
        f.write("\n".join(lines))

def load_image_paths(img_dir: str, stride: int = 1) -> List[str]:
    exts = (".png",".jpg",".jpeg",".bmp",".webp")
    paths = sorted([p for p in glob.glob(os.path.join(img_dir, "*")) if os.path.splitext(p)[1].lower() in exts])
    if stride > 1:
        paths = paths[::stride]
    return paths

class HFGroundingDINO:
    def __init__(self, model_id: str, device: str, threads: int = 0):
        self.device = device
        if threads > 0:
            torch.set_num_threads(threads)
        print(f"[Model] Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        print("[Model] Ready.")

    def run(self, img_bgr, prompt_phrases: List[str], box_thresh: float, text_thresh: float):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        inputs = self.processor(images=pil_img, text=[prompt_phrases], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=[pil_img.size[::-1]]
        )[0]
        dets = []
        for box, score, lbl in zip(res["boxes"], res["scores"], res["labels"]):
            dets.append((str(lbl), float(score), [float(x) for x in box.tolist()]))
        return dets

def maybe_load_sam(use_sam: bool, sam_impl: str, sam_ckpt: str, device: str):
    if not use_sam:
        return None, None, False
    try:
        if sam_impl.lower() == "mobile":
            from mobile_sam import sam_model_registry as mobile_registry, SamPredictor as MobilePredictor
            sam = mobile_registry["vit_t"]()
            if sam_ckpt and os.path.exists(sam_ckpt):
                state = torch.load(sam_ckpt, map_location=device)
                sam.load_state_dict(state)
            sam = sam.to(device).eval()
            return MobilePredictor(sam), "mobile", True
        else:
            from segment_anything import sam_model_registry, SamPredictor
            model_type = "vit_b"
            if not sam_ckpt or not os.path.exists(sam_ckpt):
                print("[SAM] Warning: checkpoint not provided/found; SAM disabled.")
                return None, None, False
            sam = sam_model_registry[model_type](checkpoint=sam_ckpt).to(device).eval()
            return SamPredictor(sam), "sam", True
    except Exception as e:
        print(f"[SAM] Failed to load SAM: {e}")
        return None, None, False

def refine_with_sam(predictor, img_bgr, dets):
    if predictor is None or len(dets) == 0:
        return [None]*len(dets)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    masks = []
    for _,_,box in dets:
        box_arr = np.array(box, dtype=np.float32)
        try:
            mask, _, _ = predictor.predict(box=box_arr, point_coords=None, point_labels=None, multimask_output=False)
            masks.append(mask[0])
        except Exception:
            masks.append(None)
    return masks

def load_cache(path: str):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_cache(path: str, cache: dict):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser(description="Interactive reviewer with persistent cache & explicit toggles.")
    ap.add_argument("--img_dir", default="dataset/images/clean", type=str)
    ap.add_argument("--lbl_dir", default="dataset/labels/train", type=str)
    ap.add_argument("--synonyms_json", default="synonyms.json", type=str)
    ap.add_argument("--classes", default="", type=str)
    ap.add_argument("--model_id", default="IDEA-Research/grounding-dino-tiny", type=str)
    ap.add_argument("--box_thresh", default=0.35, type=float)
    ap.add_argument("--text_thresh", default=0.25, type=float)
    ap.add_argument("--stride", default=1, type=int)
    ap.add_argument("--img_short", default=512, type=int)
    ap.add_argument("--threads", default=0, type=int)
    ap.add_argument("--phrase_limit", default=0, type=int)
    ap.add_argument("--cache_json", default="detections_cache.json", type=str)
    ap.add_argument("--no_cache_write", action="store_true")
    ap.add_argument("--warm_cache", action="store_true")
    ap.add_argument("--use_sam", default=0, type=int)
    ap.add_argument("--sam_impl", default="sam", type=str)
    ap.add_argument("--sam_ckpt", default="", type=str)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using {device}")
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    ensure_dir(args.lbl_dir)

    if not os.path.exists(args.synonyms_json):
        raise FileNotFoundError(f"Synonyms file not found: {args.synonyms_json}")
    with open(args.synonyms_json, "r", encoding="utf-8") as f:
        canon2syn = json.load(f)

    if args.classes.strip():
        classes = [c.strip().lower().replace(" ","_") for c in args.classes.split(",") if c.strip()]
    else:
        classes = list(canon2syn.keys())
    class_to_id = {c:i for i,c in enumerate(classes)}

    phrase2canon = {}
    for canon, phrases in canon2syn.items():
        for p in phrases:
            phrase2canon[p.strip().lower()] = canon
    prompt_phrases = list(phrase2canon.keys())
    if args.phrase_limit and args.phrase_limit > 0:
        prompt_phrases = prompt_phrases[:args.phrase_limit]
        print(f"[Prompts] Using first {len(prompt_phrases)} phrases")

    with open("class_map.json","w", encoding="utf-8") as f:
        json.dump(class_to_id, f, indent=2)
    print("[Classes]", class_to_id)

    gdino = HFGroundingDINO(args.model_id, device, threads=args.threads)
    sam_predictor, sam_kind, sam_ok = maybe_load_sam(bool(args.use_sam), args.sam_impl, args.sam_ckpt, device)
    if sam_ok:
        print(f"[SAM] Loaded ({sam_kind})")

    images = load_image_paths(args.img_dir, args.stride)
    if not images:
        print("No images found in", args.img_dir); return
    cache = load_cache(args.cache_json)
    print(f"[Cache] Loaded {len(cache)} entries from {args.cache_json}")

    # Optional warm-cache pass
    if args.warm_cache:
        print(f"[Warm] Precomputing detections for {len(images)} images...")
        for path in tqdm(images, desc="Precompute", unit="img"):
            img = cv2.imread(path)
            if img is None:
                continue
            infer_img, sx, sy = nn_resize_short(img, args.img_short)
            key = os.path.relpath(path, start=args.img_dir)
            if key in cache:
                continue
            dets_raw = gdino.run(infer_img, prompt_phrases, args.box_thresh, args.text_thresh)
            mapped = []
            for lbl, score, box in dets_raw:
                x1,y1,x2,y2 = box
                mapped.append([lbl, score, [x1*sx, y1*sy, x2*sx, y2*sy]])
            cache[key] = mapped
        save_cache(args.cache_json, cache)
        print("[Warm] Done. Opening GUI...")

    # GUI loop
    i = 0
    print("[GUI] Opening window...")
    while i < len(images):
        path = images[i]
        img = cv2.imread(path)
        if img is None:
            print("Failed to read", path)
            i += 1
            continue
        H,W = img.shape[:2]

        infer_img, sx, sy = nn_resize_short(img, args.img_short)
        key = os.path.relpath(path, start=args.img_dir)
        if key in cache:
            dets_raw = cache[key]
        else:
            dets_raw = gdino.run(infer_img, prompt_phrases, args.box_thresh, args.text_thresh)
            mapped = []
            for lbl, score, box in dets_raw:
                x1,y1,x2,y2 = box
                mapped.append([lbl, score, [x1*sx, y1*sy, x2*sx, y2*sy]])
            dets_raw = mapped
            if not args.no_cache_write:
                cache[key] = dets_raw
                if (i % 10) == 0:
                    save_cache(args.cache_json, cache)

        # Map to canonical
        dets = []
        for lbl, score, box in dets_raw:
            key_lbl = str(lbl).strip().lower()
            if key_lbl in phrase2canon:
                canon = phrase2canon[key_lbl]
                if canon in class_to_id:
                    dets.append((canon, float(score), [float(x) for x in box]))

        if sam_ok:
            _masks = refine_with_sam(sam_predictor, img, dets)

        # Interactive state
        keep = [True]*len(dets)
        # Always select the first detection if present, so SPACE works even with 1 box
        hi = 0 if len(dets) > 0 else -1
        vis = draw_dets(img, dets, keep, hi)
        win_title = "Auto-label Reviewer"
        cv2.imshow(win_title, vis)

        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (ord('['), ord(',')):  # prev selection
                if len(dets) == 0: continue
                hi = (hi - 1) % len(dets)
                vis = draw_dets(img, dets, keep, hi); cv2.imshow(win_title, vis)
            elif k in (ord(']'), ord('.')):  # next selection
                if len(dets) == 0: continue
                hi = (hi + 1) % len(dets)
                vis = draw_dets(img, dets, keep, hi); cv2.imshow(win_title, vis)
            elif k == ord(' '):  # toggle keep for current selection
                if hi == -1 and len(dets) > 0:
                    hi = 0  # ensure a selection exists
                if hi >= 0:
                    keep[hi] = not keep[hi]
                    vis = draw_dets(img, dets, keep, hi); cv2.imshow(win_title, vis)
            elif k == ord('x'):  # disable all
                keep = [False]*len(dets)
                vis = draw_dets(img, dets, keep, hi); cv2.imshow(win_title, vis)
            elif k == ord('k'):  # keep all
                keep = [True]*len(dets)
                vis = draw_dets(img, dets, keep, hi); cv2.imshow(win_title, vis)
            elif k == ord('r'):  # recompute this image (ignore cache)
                if key in cache: del cache[key]; save_cache(args.cache_json, cache)
                dets_raw = gdino.run(infer_img, prompt_phrases, args.box_thresh, args.text_thresh)
                mapped = []
                for lbl, score, box in dets_raw:
                    x1,y1,x2,y2 = box
                    mapped.append([lbl, score, [x1*sx, y1*sy, x2*sx, y2*sy]])
                cache[key] = mapped; save_cache(args.cache_json, cache)
                dets = []
                for lbl, score, box in mapped:
                    key_lbl = str(lbl).strip().lower()
                    if key_lbl in phrase2canon and phrase2canon[key_lbl] in class_to_id:
                        dets.append((phrase2canon[key_lbl], float(score), [float(x) for x in box]))
                keep = [True]*len(dets)
                hi = 0 if len(dets)>0 else -1
                vis = draw_dets(img, dets, keep, hi); cv2.imshow(win_title, vis)
            elif k == ord('a'):  # save & next
                base = os.path.splitext(os.path.basename(path))[0]
                out_lbl = os.path.join(args.lbl_dir, base + ".txt")
                save_yolo(out_lbl, W, H, dets, keep, class_to_id)
                print("Saved", out_lbl)
                break
            elif k == ord('s'):  # save (stay)
                base = os.path.splitext(os.path.basename(path))[0]
                out_lbl = os.path.join(args.lbl_dir, base + ".txt")
                save_yolo(out_lbl, W, H, dets, keep, class_to_id)
                print("Saved", out_lbl)
                vis = draw_dets(img, dets, keep, hi); cv2.imshow(win_title, vis)
            elif k == ord('n'):  # next (no save)
                break
            elif k == ord('q'):
                save_cache(args.cache_json, cache)
                cv2.destroyAllWindows()
                return

        if (i % 10) == 0:
            save_cache(args.cache_json, cache)
        i += 1

    save_cache(args.cache_json, cache)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
