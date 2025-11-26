
import json, os, argparse, glob

def to_yolo(x1, y1, x2, y2, W, H):
    # Convert absolute xyxy to normalized cx, cy, w, h
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / W, cy / H, w / W, h / H

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder containing *_labels.json files")
    ap.add_argument("--dst", required=True, help="Folder to write YOLO .txt labels")
    ap.add_argument("--classes", required=True, help="ontology_v1.json with fixed ordered classes")
    ap.add_argument("--skip-unknown", action="store_true", help="Skip labels whose class is not in ontology")
    args = ap.parse_args()

    with open(args.classes, "r") as f:
        ontology = json.load(f)
    classes = ontology["classes"]
    class_to_id = {name: i for i, name in enumerate(classes)}

    os.makedirs(args.dst, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(args.src, "*_labels.json")))
    total, written = 0, 0
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        # image_size is [H, W] based on provided samples
        H, W = data["image_size"]
        labels = data.get("labels", [])
        lines = []
        for lab in labels:
            cname = lab["type"].lower().replace("/", "_").replace(" ", "_")
            # Small normalization to match ontology naming
            if cname == "exp_bar" or cname == "xp_bar":
                cname = "exp_bar"
            if cname == "hpbar":
                cname = "hp_bar"
            if cname == "pokeball/item" or cname == "pokeball_item":
                cname = "pokeball_item"

            if cname not in class_to_id:
                if args.skip_unknown:
                    continue
                else:
                    raise SystemExit(f"Class '{cname}' not in ontology. Add it or use --skip-unknown.")
            cid = class_to_id[cname]

            x1, y1, x2, y2 = lab["bbox"]
            cx, cy, w, h = to_yolo(x1, y1, x2, y2, W, H)
            # clamp to [0,1]
            cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0);   h = min(max(h, 0.0), 1.0)
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Write .txt with same basename as screenshot path
        base = os.path.basename(data["screenshot_path"]).replace(".png", ".txt").replace(".jpg",".txt")
        out_path = os.path.join(args.dst, base)
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        total += len(labels); written += len(lines)

    print(f"Processed {len(json_files)} files. Wrote {written}/{total} labels to {args.dst}.")
    # Also export classes.txt in YOLO order
    with open(os.path.join(args.dst, "classes.txt"), "w") as f:
        f.write("\n".join(classes))

if __name__ == "__main__":
    main()
