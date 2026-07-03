"""Pre-label uniq_crops with the warm-mix CRNN -> seed labels for VLM spot-check
(self-labeling). Writes a 'pred' field into uniq_manifest.json. Host onnxruntime."""
import argparse, json, os, sys
import cv2
sys.path.insert(0, os.path.expanduser("~/projects/VGA"))
from vga.core.textrecog import CrnnRecognizer

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--onnx", default=os.path.expanduser("~/projects/VGA/train/ocr/ckpt_ft_warm-mix/crnn.onnx"))
a = ap.parse_args()
rec = CrnnRecognizer(onnx_path=a.onnx)
assert rec.enabled, "CRNN disabled"
man = json.load(open(os.path.join(a.dir, "uniq_manifest.json")))
for e in man:
    im = cv2.imread(os.path.join(a.dir, "uniq_crops", e["id"] + ".png"))
    e["pred"] = rec.read(im) if im is not None else ""
json.dump(man, open(os.path.join(a.dir, "uniq_manifest.json"), "w"), ensure_ascii=False, indent=1)
print(f"pre-labeled {len(man)} crops with warm-mix CRNN")
