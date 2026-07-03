"""
Benchmark the vLLM FP8 server against the transformers bf16 baseline (~3.6s/decision).

Sends the same game frame through vLLM's OpenAI-compatible /v1/chat/completions N times
and reports per-request latency + the parsed action. Run on the host (stdlib only).

    # against a live game (captures one frame):
    DISPLAY=:99 .venv/bin/python serve/bench_vllm.py --n 6
    # against a saved frame:
    .venv/bin/python serve/bench_vllm.py --frame /tmp/frame.png --n 6
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
import urllib.request

import cv2

BUTTONS = ["up", "down", "left", "right", "A", "B", "start", "select", "L", "R", "wait"]

SYSTEM = (
    "You are VGA, an agent that plays video games from the screen alone, like a human. "
    "You are given one screenshot of a Game Boy Advance game and must choose the single "
    "best controller input to make progress right now. Read on-screen text and menus and "
    "follow them. Advance dialogue/confirm with A; cancel with B; move with the D-pad; "
    "menu with start; if an animation is playing, wait. Pick exactly ONE input."
)
INSTRUCTION = (
    "This is the current screen. Respond with ONLY a compact JSON object of the form: "
    '{"reason": "<one short sentence>", "button": "<one of: ' + " ".join(BUTTONS)
    + '>", "repeats": <1, 2, or 3>}'
)


def parse_action(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            o = json.loads(m.group(0))
            b = str(o.get("button", "wait"))
            return {"button": b if b in BUTTONS else "wait", "reason": str(o.get("reason", ""))[:70]}
        except json.JSONDecodeError:
            pass
    return {"button": "?", "reason": text[:70]}


def load_frame(path: str | None, title: str):
    if path:
        return cv2.imread(path)
    from vga.core.emulator import GbaEmulator
    return GbaEmulator(title_keyword=title).capture()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8078")
    ap.add_argument("--frame", default=None, help="PNG path; if omitted, capture from emulator")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--title", default="mGBA")
    args = ap.parse_args()

    frame = load_frame(args.frame, args.title)
    ok, buf = cv2.imencode(".png", frame)
    b64 = base64.standard_b64encode(buf.tobytes()).decode()
    data_uri = "data:image/png;base64," + b64

    body = {
        "model": "qwen3vl-fp8",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": INSTRUCTION},
            ]},
        ],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    payload = json.dumps(body).encode()

    lat = []
    for i in range(args.n):
        req = urllib.request.Request(
            args.url + "/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read().decode())
        dt = time.time() - t0
        text = resp["choices"][0]["message"]["content"]
        a = parse_action(text)
        lat.append(dt)
        print(f"req {i}: {dt:.2f}s  -> {a['button']:6s}  {a['reason']!r}")

    warm = lat[1:] if len(lat) > 1 else lat
    print(f"\nlatency: first {lat[0]:.2f}s | warm min {min(warm):.2f}s "
          f"mean {sum(warm)/len(warm):.2f}s max {max(warm):.2f}s")
    print(f"baseline (transformers bf16) was ~3.6s/decision.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
