"""Text-reading tier: can the VLM transcribe on-screen text? Loads a VLM, asks it to
transcribe each hand-labeled intro frame (upscaled -> enough image tokens for Qwen-VL
grounding), and scores vs ground truth by word-recall + char-similarity. Prints every
GT-vs-prediction pair for eyeball spot-check. Runs in thor-vlm:cu130, --network none.

  python score_text.py --model-dir /model --tag qwen8b [--upscale 3]
"""
import argparse
import difflib
import json
import os
import re
import statistics

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

BD = "/work/train/perception"
Q = "Transcribe the text shown on the screen, exactly as written. Reply with only that text."


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gt", default=os.path.join(BD, "text_gt.jsonl"))
    ap.add_argument("--upscale", type=int, default=3)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.gt)]
    print("loading %s ..." % args.model_dir, flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(args.model_dir)

    res = []
    for r in rows:
        img = Image.open(os.path.join(BD, r["file"])).convert("RGB")
        if args.upscale > 1:
            img = img.resize((img.width * args.upscale, img.height * args.upscale), Image.LANCZOS)
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": Q}]}]
        chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[chat], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=80, do_sample=False)
        out = [o[len(i):] for i, o in zip(inp.input_ids, gen)]
        pred = proc.batch_decode(out, skip_special_tokens=True)[0].strip().replace("\n", " ")
        gt = r["text"]
        cs = difflib.SequenceMatcher(None, gt.lower(), pred.lower()).ratio()
        gw, pw = set(words(gt)), set(words(pred))
        wr = len(gw & pw) / max(len(gw), 1)
        res.append(dict(file=r["file"], gt=gt, pred=pred, char_sim=cs, word_recall=wr))
        print("[%s] %s  word_recall=%.2f char_sim=%.2f\n   GT : %s\n   VLM: %s"
              % (args.tag, r["file"], wr, cs, gt, pred), flush=True)

    print("=" * 60, flush=True)
    print("ARM %s: word_recall %.2f  char_sim %.2f  (n=%d, upscale=%dx)" % (
        args.tag, statistics.mean(r["word_recall"] for r in res),
        statistics.mean(r["char_sim"] for r in res), len(res), args.upscale), flush=True)
    json.dump(res, open(os.path.join(BD, "text_results_%s.json" % args.tag), "w"), indent=1)


main()
