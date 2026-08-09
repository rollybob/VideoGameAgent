"""Batch VQA scorer for one perception-bench arm.

Loads a local VLM, answers every question in questions.jsonl from the frame pixels,
and scores vs the RAM-oracle ground truth (hearts within +/-0.5, rupees exact).
Reports per-type accuracy + latency + peak GPU memory. Meant to run in the thor-vlm
GPU container with --network none (no data leaves the box; models load from a
read-only mount). Same generate path as serve_vlm.py / bench_pixels.py.

  python score_arm.py --model-dir /model --tag qwen8b [--limit N]
"""
import argparse
import json
import os
import re
import time

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

BD = "/work/train/perception/bench_data"
SYSTEM = ("You are looking at a single screenshot from the Game Boy Advance game "
          "The Legend of Zelda: A Link to the Past. Answer the question about what is "
          "visible on the screen right now, based only on the pixels.")
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_num(txt):
    m = NUM.search(txt.replace(",", ""))
    return float(m.group(0)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--questions", default=os.path.join(BD, "questions.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--upscale", type=int, default=1,
                    help="resize the frame NxN before feeding (tests whether tiny-HUD reads are resolution-bound)")
    args = ap.parse_args()

    qs = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        qs = qs[:args.limit]

    print("loading %s ..." % args.model_dir, flush=True)
    t_load = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(args.model_dir)
    print("loaded in %.1fs" % (time.time() - t_load), flush=True)

    torch.cuda.reset_peak_memory_stats()
    results, lat = [], []
    for i, q in enumerate(qs):
        img = Image.open(os.path.join(BD, q["file"])).convert("RGB")
        if args.upscale > 1:
            img = img.resize((img.width * args.upscale, img.height * args.upscale), Image.LANCZOS)
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": q["question"]}]}]
        chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[chat], images=[img], return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize()
        lat.append(time.time() - t0)
        out = [o[len(ii):] for ii, o in zip(inp.input_ids, gen)]
        txt = proc.batch_decode(out, skip_special_tokens=True)[0].strip()
        pred = parse_num(txt)
        gt, tol = float(q["answer"]), float(q.get("tol", 0))
        ok = pred is not None and abs(pred - gt) <= tol
        results.append(dict(id=q["id"], type=q["type"], gt=gt, pred=pred, raw=txt[:40], correct=bool(ok)))
        if (i + 1) % 25 == 0:
            print("  %d/%d done" % (i + 1, len(qs)), flush=True)

    out_path = os.path.join(BD, "results_%s.jsonl" % args.tag)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    def acc(t):
        sub = [r for r in results if r["type"] == t]
        return sum(r["correct"] for r in sub), len(sub)

    hc, hn = acc("hearts")
    rc, rn = acc("rupees")
    print("=" * 52, flush=True)
    print("ARM %s: hearts %d/%d=%.0f%%  rupees %d/%d=%.0f%%" % (
        args.tag, hc, hn, 100 * hc / max(hn, 1), rc, rn, 100 * rc / max(rn, 1)), flush=True)
    print("latency mean %.2fs (n=%d)  peak GPU mem %.1f GB" % (
        sum(lat) / len(lat), len(lat), peak_gb), flush=True)
    print("wrote", out_path, flush=True)


main()
