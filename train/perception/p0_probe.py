#!/usr/bin/env python3
"""P0 (System-2 in the loop): does the served Qwen3-VL-8B /act path emit SENSIBLE, grounded
actions on a title/menu frame and an overworld frame -- and does the image-token upscale
lever (bake-off-proven; LANCZOS resize, == train/perception/score_arm.py --upscale) change
the behavior in the served path?

Client-side upscale so one server load covers both arms and it mirrors what P1's
vga/reason/local_reasoner would bake in. No RAM oracle here -- this is a perception/action
sanity probe judged from the model's own /read scene-label vs its /act choice.
"""
import base64, io, json, sys, time, urllib.request
from PIL import Image

URL = "http://127.0.0.1:8077"
FRAMES = [
    # Ground-truthed re-probe (run 1 used mislabeled/duplicate frames): distinct md5s,
    # labels cross-checked to true game/screen. ALttP thumbs are model-verified Zelda
    # gameplay; the FireRed menus are genuine selectable-option menus with a cursor.
    ("overworld A (ALttP)",    "sessions/state_thumbs/alttp_human-01.png"),
    ("overworld B (ALttP)",    "sessions/state_thumbs/alttp_human-04.png"),
    ("menu: party (FireRed)",  "captures/tour/013_party_submenu.png"),
    ("menu: battle (FireRed)", "captures/route2_battle/54_startmenu.png"),
    ("title (FireRed)",        "captures/tour/000_title.png"),
]
UPSCALES = [1]  # native only: run 1 showed the upscale lever does NOT change action choice
                # (it helps tiny HUD digits only, per the bake-off) -> drop the redundant arm.


def b64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def post(path, payload, timeout=120):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(path, timeout=30):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print("HEALTH:", json.dumps(get("/health")))
    print("=" * 72)
    results = []
    for label, rel in FRAMES:
        base = Image.open(f"{root}/{rel}").convert("RGB")
        print(f"\n### {label}  [{rel}]  native={base.size}")
        for u in UPSCALES:
            img = base if u == 1 else base.resize(
                (base.width * u, base.height * u), Image.LANCZOS)
            try:
                im64 = b64(img)
                t = time.time()
                rd = post("/read", {"image_b64": im64,
                    "prompt": "In ONE sentence: what game screen is this and what is happening? "
                              "Name the screen type (title/menu/dialogue/overworld/battle/shop)."})
                ac = post("/act", {"image_b64": im64,
                    "goal": "Play the game and make progress.", "step": 0})
                dt = round(time.time() - t, 2)
                row = {"label": label, "file": rel, "upscale": u, "img_size": list(img.size),
                       "read": (rd.get("text") or "").strip()[:220],
                       "button": ac.get("button"), "mode": ac.get("mode"),
                       "reason": (ac.get("reason") or "").strip()[:180],
                       "raw": (ac.get("raw") or "").strip()[:220], "dt": dt}
            except Exception as e:
                row = {"label": label, "file": rel, "upscale": u, "error": repr(e)[:200]}
                print(f"  up{u} ERROR: {row['error']}")
                results.append(row); continue
            results.append(row)
            print(f"  up{u} img{tuple(img.size)} -> BUTTON={row['button']!r} "
                  f"mode={row['mode']!r} ({dt}s)")
            print(f"     READ: {row['read']}")
            print(f"     ACT:  {row['reason']}")
    out = f"{root}/sessions/p0_probe_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSAVED", out)
    # crude auto-verdict: fraction of arms that returned a real button (not None/wait-only)
    acted = [r for r in results if r.get("button") and r.get("button") != "wait"]
    print(f"VERDICT: {len(acted)}/{len(results)} arms returned a non-wait grounded button")


if __name__ == "__main__":
    main()
