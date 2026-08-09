"""Arbiter v1: System-1 reflex policy + bounded low-frequency System-2 key interject.

THE CLAIM UNDER TEST (2026-08-06 reframe, after reward shaping was exhausted): the sparse
multi-step kill->collect-key sequence is System-2's job. S1 keeps its reflexes; a served
VLM is consulted at low frequency, and when it SEES a dropped key it steers Link onto it
with short directed bursts. Inference is PIXELS-ONLY (the VLM sees the raw frame; S1 sees
its stacked frames); RAM is read for EVAL METRICS ONLY (keys, damage, time-to-key).

EVAL DESIGN: the keydrop state bank (train/rl/harvest_key_frames.py --save-states) --
episodes that START with a key already on the floor -- so every episode carries signal
and attribution is unambiguous: does the agent collect it, and how fast? Arms:
  s1       the 12ch backbone alone (sampling mode -- det is passive at these ckpts)
  arbiter  same backbone + S2 interject
  random   uniform-action floor
Per-episode we also log whether the key event landed DURING/just after an interject
burst ("collected_by_interject"), which is the mechanism receipt.

Run (needs --network host to reach the VLM server on 127.0.0.1:8077):
  docker run --rm --runtime nvidia --network host --user 1000:1000 \
      -v ~/projects/VGA:/work -w /work thor-rl:cu130 python3 -u train/rl/arbiter_v1.py \
      --arm arbiter --states 'train/perception/keyprobe_v2/states/keydrop-*.state'
"""
import argparse
import base64
import glob
import io
import json
import math
import os
import re
import sys
import time
import urllib.request
import warnings

warnings.filterwarnings("ignore")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from alttp_ppo_env import AlttpPpoEnv  # noqa: E402

# ---- S2 client (prompt + parse mirror train/perception/key_probe.py, the probe that
# ---- gated this build; keep the two in sync) ----
DIRS = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]
DIR_IDX = {"up": 1, "down": 2, "left": 3, "right": 4,
           "up-right": 5, "up-left": 6, "down-right": 7, "down-left": 8}
PROMPT = (
    "This is a frame from The Legend of Zelda: A Link to the Past on Game Boy Advance. "
    "Link is the small character in the green tunic. QUESTION: is there a small dungeon "
    "KEY item lying on the floor anywhere in this frame? A dropped key is a small "
    "golden/yellow key-shaped sprite lying on the ground (not in the HUD at the top). "
    "Answer with ONLY a JSON object and no other text, exactly like "
    '{"key_visible": true, "direction": "up-left"} . '
    '"direction" is where the key is RELATIVE TO LINK, one of: up, down, left, right, '
    'up-left, up-right, down-left, down-right, "on-link" if Link is standing on it, '
    'or "none" if there is no key.'
)


def ask_s2(url, frame, upscale=1, timeout=180):
    """frame (H,W,3 uint8) -> (key_visible, direction, seconds). Raises on transport error."""
    from PIL import Image
    img = Image.fromarray(frame)
    if upscale != 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    payload = {"image_b64": base64.b64encode(buf.getvalue()).decode(), "prompt": PROMPT}
    req = urllib.request.Request(url + "/read", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = (json.loads(r.read().decode()).get("text") or "")
    dt = time.time() - t0
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return bool(d.get("key_visible")), str(d.get("direction") or "none").lower(), dt
        except Exception:
            pass
    low = text.lower()
    vis = "true" in low or ("yes" in low and "no key" not in low)
    for cand in ["up-left", "up-right", "down-left", "down-right", "on-link",
                 "up", "down", "left", "right"]:
        if cand in low:
            return vis, cand, dt
    return vis, "none", dt


def run_episode(env, arm, model, seed, a):
    obs, _ = env.reset(seed=seed)
    if model is not None:
        model.set_random_seed(seed)
    env.action_space.seed(seed)

    stats = dict(keys=0, first_key_step=None, damage=0.0, s2_calls=0, s2_secs=0.0,
                 interjects=0, collected_by_interject=False, consults=[])
    # Distill recording (2026-08-06): raw frames + EXECUTED actions + source flag.
    # frames[0] = reset frame; frames[i] = frame after decision i. The env obs at
    # decision i is exactly frames[max(0,i-3)..i] left-padded with frames[0] (reset
    # fills the stack with 4 copies of frame 0), so 12ch stacks rebuild losslessly
    # from raw frames at train time -- 4x smaller on disk than storing stacks.
    rec = None
    if getattr(a, "record_dir", None):
        rec = {"frames": [env._frames[-1].copy()], "actions": [], "source": []}
    burst_left = 0
    burst_dir = 0
    consult_countdown = a.first_consult   # first consult early: the bank states start
                                          # with the key already on the floor
    interject_active = 0                  # >0 while bursting; decays after
    step = 0
    done = False
    while not done:
        if arm == "arbiter" and burst_left <= 0:
            consult_countdown -= 1
        if (arm == "arbiter" and burst_left <= 0 and consult_countdown <= 0
                and stats["s2_calls"] < a.max_consults):
            try:
                vis, direc, dt = ask_s2(a.s2_url, env._frames[-1], a.upscale)
            except Exception as e:
                print("  S2 ERROR:", repr(e)[:120], flush=True)
                vis, direc, dt = False, "none", 0.0
            stats["s2_calls"] += 1
            stats["s2_secs"] += dt
            stats["consults"].append({"step": step, "vis": vis, "dir": direc})
            if vis and direc in DIR_IDX:
                burst_dir = DIR_IDX[direc]
                burst_left = a.burst
                stats["interjects"] += 1
            elif vis and direc == "on-link":
                # nudge through the sprite: keep last burst direction 2 decisions
                burst_dir = burst_dir or 2
                burst_left = 2
                stats["interjects"] += 1
            else:
                consult_countdown = a.period

        if burst_left > 0:
            act = np.array([burst_dir, 0, 0, 0, 0], dtype=np.int64)
            src = 1
            burst_left -= 1
            interject_active = a.attr_window
            if burst_left == 0:
                consult_countdown = 1   # re-consult right after a burst
        elif arm == "random":
            act = env.action_space.sample()
            src = 0
        else:
            act, _ = model.predict(obs, deterministic=a.det)
            src = 0

        obs, r, term, trunc, info = env.step(act)
        if rec is not None:
            rec["frames"].append(env._frames[-1].copy())
            rec["actions"].append(np.asarray(act, dtype=np.int64))
            rec["source"].append(src)
        done = term or trunc
        step += 1
        if interject_active > 0:
            interject_active -= 1

        for _t, ch, d, _v in info["events"]:
            if ch == "key":
                stats["keys"] += 1
                if stats["first_key_step"] is None:
                    stats["first_key_step"] = step
                if interject_active > 0 or burst_left > 0:
                    stats["collected_by_interject"] = True
            elif ch == "damage":
                stats["damage"] += -d
    return stats, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["s1", "arbiter", "random"], required=True)
    ap.add_argument("--states", default=os.path.join(os.path.dirname(HERE), "perception",
                                                     "keyprobe_v2", "states", "keydrop-*.state"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "alttp_s1v2_scratch04",
                                                   "ppo_alttp_600000_steps.zip"))
    ap.add_argument("--horizon", type=int, default=1200, help="emulator frames (300 decisions)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s2-url", default="http://127.0.0.1:8077")
    ap.add_argument("--upscale", type=int, default=1)
    ap.add_argument("--period", type=int, default=40, help="decisions between consults when no key seen")
    ap.add_argument("--first-consult", type=int, default=5)
    ap.add_argument("--burst", type=int, default=10, help="decisions per directed burst")
    ap.add_argument("--max-consults", type=int, default=12)
    ap.add_argument("--attr-window", type=int, default=6,
                    help="key events within this many decisions after a burst count as interject-collected")
    ap.add_argument("--det", action="store_true",
                    help="deterministic S1 predictions (default: sampling; det is a "
                         "known-divergent secondary report, not the primary metric)")
    ap.add_argument("--record-dir", default=None,
                    help="save SUCCESSFUL episodes (keys>0), trimmed to pickup+2, as "
                         "npz {frames,actions,source,first_key_step} for distillation")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    states = sorted(glob.glob(a.states))
    if not states:
        print("NO STATES match", a.states)
        return 2

    model = None
    if a.arm in ("s1", "arbiter"):
        from stable_baselines3 import PPO
        model = PPO.load(a.ckpt, device="cuda")

    print("arm=%s states=%d horizon=%d ckpt=%s" % (a.arm, len(states), a.horizon,
                                                   os.path.basename(a.ckpt)), flush=True)
    all_stats = []
    t0 = time.time()
    if a.record_dir:
        os.makedirs(a.record_dir, exist_ok=True)
    for i, sp in enumerate(states):
        env = AlttpPpoEnv(state_path=sp, horizon=a.horizon)
        st, rec = run_episode(env, a.arm, model, a.seed + i, a)
        env.close()
        st["state"] = os.path.basename(sp)
        if a.record_dir and rec is not None and st["keys"] > 0:
            # success only, trimmed to pickup+2 (the seek behavior is the label;
            # post-pickup wandering is not). first_key_step is 1-based; action k
            # (0-based) = decision k+1; frames has one extra leading reset frame.
            end = min(st["first_key_step"] + 2, len(rec["actions"]))
            path = os.path.join(a.record_dir, "%s_seed%d.npz"
                                % (st["state"].replace(".state", ""), a.seed + i))
            np.savez_compressed(
                path,
                frames=np.stack(rec["frames"][:end + 1]).astype(np.uint8),
                actions=np.stack(rec["actions"][:end]).astype(np.int8),
                source=np.asarray(rec["source"][:end], dtype=np.uint8),
                first_key_step=np.int64(st["first_key_step"]))
            st["recorded"] = os.path.basename(path)
        all_stats.append(st)
        print("%-18s keys=%d first@%s dmg=%.0f s2=%d calls/%.0fs interjects=%d byS2=%s"
              % (st["state"], st["keys"], st["first_key_step"], st["damage"],
                 st["s2_calls"], st["s2_secs"], st["interjects"],
                 st["collected_by_interject"]), flush=True)

    n = len(all_stats)
    collected = sum(1 for s in all_stats if s["keys"] > 0)
    by_s2 = sum(1 for s in all_stats if s["collected_by_interject"])
    tks = [s["first_key_step"] for s in all_stats if s["first_key_step"] is not None]
    summary = dict(arm=a.arm, episodes=n,
                   collect_rate=collected / n,
                   keys_per_ep=float(np.mean([s["keys"] for s in all_stats])),
                   collected_by_interject=by_s2,
                   median_first_key=float(np.median(tks)) if tks else None,
                   mean_damage=float(np.mean([s["damage"] for s in all_stats])),
                   mean_s2_calls=float(np.mean([s["s2_calls"] for s in all_stats])),
                   wall_secs=round(time.time() - t0, 1))
    print("SUMMARY:", json.dumps(summary), flush=True)
    out = a.out or os.path.join(HERE, "arbiter_v1_%s_results.json" % a.arm)
    with open(out, "w") as f:
        json.dump({"summary": summary, "episodes": all_stats}, f, indent=2)
    print("SAVED", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
