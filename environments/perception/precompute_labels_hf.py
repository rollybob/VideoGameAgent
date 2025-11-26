
import os, json, glob, argparse, cv2, torch, sys, time
import numpy as np
from PIL import Image
from typing import List
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

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

class HFGroundingDINO:
    def __init__(self, model_id: str, device: str, threads: int = 0):
        self.device = device
        if threads > 0:
            torch.set_num_threads(threads)
        print(f"[Model] Loading {model_id} (this may take a minute the first time)...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        print("[Model] Ready.")

    def run(self, img_bgr, phrases: List[str], box_thresh: float, text_thresh: float):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        inputs = self.processor(images=pil_img, text=[phrases], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=box_thresh, text_threshold=text_thresh,
            target_sizes=[pil_img.size[::-1]]
        )[0]
        dets = []
        for box, score, lbl in zip(res["boxes"], res["scores"], res["labels"]):
            dets.append((str(lbl), float(score), [float(x) for x in box.tolist()]))
        return dets

def main():
    ap = argparse.ArgumentParser(description="Batch auto-label with HF Grounding DINO (no GUI).")
    ap.add_argument("--img_dir", required=True, type=str)
    ap.add_argument("--lbl_dir", required=True, type=str)
    ap.add_argument("--synonyms_json", required=True, type=str)
    ap.add_argument("--model_id", default="IDEA-Research/grounding-dino-tiny", type=str)
    ap.add_argument("--box_thresh", default=0.30, type=float)
    ap.add_argument("--text_thresh", default=0.22, type=float)
    ap.add_argument("--img_short", default=512, type=int)
    ap.add_argument("--phrase_limit", default=24, type=int)
    ap.add_argument("--threads", default=0, type=int)
    ap.add_argument("--preview_dir", default="", type=str, help="If set, write preview images with boxes")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    ensure_dir(args.lbl_dir)
    if args.preview_dir:
        ensure_dir(args.preview_dir)

    # Load synonyms
    with open(args.synonyms_json, "r", encoding="utf-8") as f:
        canon2syn = json.load(f)

    # Canonical classes + ids
    classes = list(canon2syn.keys())
    class_to_id = {c:i for i,c in enumerate(classes)}
    with open("class_map.json","w", encoding="utf-8") as f:
        json.dump(class_to_id, f, indent=2)

    # Phrase -> canon map
    phrase2canon = {}
    for canon, phrases in canon2syn.items():
        for p in phrases:
            phrase2canon[p.strip().lower()] = canon
    phrases = list(phrase2canon.keys())
    if args.phrase_limit > 0:
        phrases = phrases[:args.phrase_limit]
    print(f"[Prompts] {len(phrases)} phrases; classes={classes}")

    # Images
    exts = (".png",".jpg",".jpeg",".bmp",".webp")
    images = sorted([p for p in glob.glob(os.path.join(args.img_dir, "*")) if os.path.splitext(p)[1].lower() in exts])
    total = len(images)
    if not images:
        print(f"[Images] 0 found under {args.img_dir}. Nothing to do.")
        return
    print(f"[Images] {total} found in {args.img_dir}")

    model = HFGroundingDINO(args.model_id, device, args.threads)

    saved = 0
    for idx, path in enumerate(tqdm(images, desc="Labeling", unit="img"), 1):
        img = cv2.imread(path)
        if img is None:
            tqdm.write(f"[Skip] unreadable {path}")
            continue
        H, W = img.shape[:2]
        infer_img, sx, sy = nn_resize_short(img, args.img_short)

        dets_raw = model.run(infer_img, phrases, args.box_thresh, args.text_thresh)
        mapped = []
        for lbl, score, box in dets_raw:
            x1,y1,x2,y2 = box
            mapped.append((lbl, score, [x1*sx, y1*sy, x2*sx, y2*sy]))

        # map to canonical + save YOLO
        dets = []
        for lbl, score, box in mapped:
            key = str(lbl).strip().lower()
            if key in phrase2canon:
                canon = phrase2canon[key]
                if canon in class_to_id:
                    dets.append((canon, score, box))

        base = os.path.splitext(os.path.basename(path))[0]
        out_lbl = os.path.join(args.lbl_dir, base + ".txt")

        # write YOLO
        lines = []
        for (cls, score, box) in dets:
            cls_id = class_to_id[cls]
            cx,cy,w,h = xyxy_to_yolo(box, W, H)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(out_lbl, "w") as f:
            f.write("\n".join(lines))
        saved += 1

        # optional preview
        if args.preview_dir:
            vis = img.copy()
            for (cls, score, box) in dets:
                x1,y1,x2,y2 = map(int, box)
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,200,255),2)
                cv2.putText(vis, f"{cls} {score:.2f}", (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(args.preview_dir, base + ".jpg"), vis)

    print(f"[Done] Labels written: {saved} in {args.lbl_dir}")
    print("[Tip] Use the interactive labeler next to spot-fix any mistakes.")
if __name__ == "__main__":
    main()
