"""Record an annotated MP4 of Composer v0 playing one episode (2026-08-23).

Banks the see-saw counter-example as a WATCHABLE artifact: it re-runs the exact
frozen stack from composer_v0 (r2c COMBAT + scripted COLLECT + 2-mode router,
detector_v5) on one (room, seed) and writes a video with the router mode and the
detector's Link/key markers overlaid on every decision frame. Nothing new is
trained or changed -- this only visualizes composer_v0's control loop, so the
picture is faithful to the eval (same Composer class, same detector, same env).

Run (in the thor-rl container, GPU needed for detector + policy):
  docker run --rm -i --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/vga \\
      -w /vga/train/rl -e DET=/vga/train/perception/detector/detector_v5.pt \\
      -e ROOM=4 -e SEED=7001 -e OUT=/vga/sessions/composer_video/room4_seed7001.mp4 \\
      thor-rl:cu130 python3 -u record_composer.py

Env: ROOM (4|2|0), SEED, OUT (mp4 path), SCALE (px zoom, def 4), FPS (def 12),
     HOLD (frames to linger after key/EOP, def 24). DET override is read by
     composer_v0 at import (defaults to detector.pt -- pass detector_v5.pt).
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

# composer_v0 sets up sys.path for the detector pkg and pulls the env/net/oracle
# symbols into its namespace; import it FIRST, then borrow those exact objects so
# the recording uses the identical frozen stack (no drift vs the eval).
import composer_v0 as C
from stable_baselines3 import PPO
import torch
from mgba._pylib import ffi

ROOM = int(os.environ.get("ROOM", "4"))
SEED = int(os.environ.get("SEED", "7001"))
OUT = os.environ.get("OUT", os.path.join(C.HERE, f"_composer_video_room{ROOM}_seed{SEED}.mp4"))
SCALE = int(os.environ.get("SCALE", "4"))
FPS = int(os.environ.get("FPS", "12"))
HOLD = int(os.environ.get("HOLD", "24"))
HORIZON = int(os.environ.get("HORIZON", "1500"))

BANNER = 64
GAME_W, GAME_H = 240, 160
W, H = GAME_W * SCALE, GAME_H * SCALE + BANNER
MODE_COL = {"COMBAT": (232, 74, 74), "S1": (232, 74, 74),
            "COLLECT": (70, 150, 255), "EXPLORE": (240, 200, 64)}
DRIVER = {"COMBAT": "S1 PPO (frozen)", "S1": "S1 PPO (frozen)",
          "COLLECT": "scripted walk to key", "EXPLORE": "random"}


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


FONT = _font(22)
FONT_S = _font(15)


def annotate(frame, comp, mode, step, kills, flash):
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    game = Image.fromarray(arr.astype("uint8")).convert("RGB").resize(
        (GAME_W * SCALE, GAME_H * SCALE), Image.NEAREST)
    canvas = Image.new("RGB", (W, H), (18, 18, 22))
    canvas.paste(game, (0, BANNER))
    d = ImageDraw.Draw(canvas)

    # detector markers (what the router actually sees): Link green, key blue.
    lx, ly = comp.last_link
    d.ellipse([lx * SCALE - 12, ly * SCALE + BANNER - 12,
               lx * SCALE + 12, ly * SCALE + BANNER + 12], outline=(60, 255, 96), width=3)
    if mode == "COLLECT" and comp.last_item is not None:
        ix, iy = comp.last_item
        d.line([lx * SCALE, ly * SCALE + BANNER, ix * SCALE, iy * SCALE + BANNER],
               fill=(70, 150, 255), width=2)
        d.ellipse([ix * SCALE - 12, iy * SCALE + BANNER - 12,
                   ix * SCALE + 12, iy * SCALE + BANNER + 12], outline=(70, 150, 255), width=3)

    # banner
    col = MODE_COL.get(mode, (210, 210, 210))
    d.text((10, 7), f"COMPOSER v0   room{ROOM}", font=FONT, fill=(235, 235, 235))
    d.text((10, 38), "green = Link (detector)    blue = key (detector)",
           font=FONT_S, fill=(150, 150, 156))
    box_x = W - 232
    d.rectangle([box_x, 8, box_x + 150, 40], outline=col, width=3)
    d.text((box_x + 12, 11), mode, font=FONT, fill=col)
    d.text((box_x + 12, 43), f"{DRIVER.get(mode, '')}", font=FONT_S, fill=col)
    d.text((box_x + 168, 11), f"step {step}", font=FONT_S, fill=(200, 200, 206))
    d.text((box_x + 168, 30), f"kills {kills}", font=FONT_S, fill=(200, 200, 206))
    if flash:
        fcol = (90, 255, 120) if "KEY" in flash else (255, 210, 70)
        d.text((W // 2 - 90, 40), flash, font=FONT, fill=fcol)
    return np.asarray(canvas)


def render_one(model, net, room, seed, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    state = os.path.join(C.HERE, "states", f"alttp_human-0{room}.state")
    print(f"record_composer room{room} seed{seed} det={os.path.basename(C.DET_PT)} -> {out}",
          flush=True)

    env = C.AlttpPpoEnv(state_path=state, horizon=HORIZON)
    obs, _ = env.reset(seed=seed)
    iw = ffi.cast("uint8_t *", env.core._native.memory.iwram)
    model.set_random_seed(seed)
    comp = C.Composer(model, net, np.random.RandomState(seed))

    frames = []
    first_kill = None
    kills = 0
    got_key = False
    flash = ""
    flash_t = 0
    step = 0
    done = False
    while not done:
        frame = env._frames[-1]                 # the frame the composer acts on
        act, mode = comp.act(obs, frame)
        frames.append(annotate(frame, comp, mode, step, kills, flash))
        ex, ey = C.u16(iw, C.ENEMY_X), C.u16(iw, C.ENEMY_Y)
        obs, _r, term, trunc, info = env.step(act)
        done = term or trunc
        step += 1
        for _t, ch, _d, val in info["events"]:
            if ch == "enemy_dmg" and val == 0 and (ex, ey) != (0, 0):
                kills += 1
                if first_kill is None:
                    first_kill = step
                flash, flash_t = "ENEMY KILLED", FPS
            elif ch == "key":
                if first_kill is not None and step - first_kill <= C.WINDOW:
                    got_key = True
                    flash, flash_t = "KEY COLLECTED", 3 * FPS
        flash_t -= 1
        if flash_t <= 0:
            flash = ""
        if got_key:
            for _ in range(HOLD):
                frames.append(annotate(env._frames[-1], comp, mode, step, kills, "KEY COLLECTED"))
            break

    imageio.mimwrite(out, frames, fps=FPS, codec="libx264", quality=8,
                     macro_block_size=None)
    print(f"WROTE {out}  ({len(frames)} frames, {len(frames)/FPS:.1f}s, "
          f"kills={kills} key={'YES' if got_key else 'no'})", flush=True)
    env.close()


def main():
    # Batch: SEEDS=comma,list + OUTDIR renders room{ROOM}_seed{sd}.mp4 for each,
    # loading the model/detector ONCE. Falls back to single SEED/OUT.
    seeds_env = os.environ.get("SEEDS")
    seeds = [int(s) for s in seeds_env.split(",") if s.strip()] if seeds_env else [SEED]
    outdir = os.environ.get("OUTDIR")
    model = PPO.load(C.S1_CKPT, device=C.DEV)
    net = C.Net().to(C.DEV)
    net.load_state_dict(torch.load(C.DET_PT, map_location=C.DEV))
    net.eval()
    for sd in seeds:
        out = os.path.join(outdir, f"room{ROOM}_seed{sd}.mp4") if outdir else OUT
        render_one(model, net, ROOM, sd, out)


if __name__ == "__main__":
    main()
