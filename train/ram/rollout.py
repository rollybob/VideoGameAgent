#!/usr/bin/env python3
"""Oracle-labeled rollout: drive the headless emulator with a policy, and at every step
log the frame, the action, the RAM-truth STATE, and the oracle REWARD. This is the data
engine the learning loop needs - the existing trajectory logger records frame+action+meta
but NO state and NO reward, so there is nothing to tell a good step from a bad one. Here
every step is scored by the RAM oracle (train/ram/oracle.py).

Output (sessions/oracle-<ts>/):
    frames/000000.png ...            one framebuffer PNG per step (clean 240x160)
    steps.jsonl                      one oracle-labeled step per line
    meta.json                        run-level metadata + reward summary

A step line:
    {step, t, frame, action:{button,repeats}, state:{...ram...}, reward, cum_reward, mode}

Policies:
    random   - uniform over buttons (no GPU, no server) - use to prove the pipeline
    vlm      - the REAL agent: the full VlmPlugin + LocalVlmReasoner, skills and all, so
               the oracle scores exactly what the pixels-only policy does (needs the VLM
               server up; the resulting labeled data is what a filtered-BC pass consumes).

This is the FOUNDATION, not the whole loop: it produces (frame, action, state, reward)
tuples. The next links are (a) a filtered-BC / self-imitation reader over high-reward
steps, and (b) a reflection pass that turns oracle-detected stuck->escape transitions into
SkillStore entries. Both are deliberately left for the next session; this makes their input.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

# The headless harness lives here; the vga package (for the vlm policy) is at the repo root.
sys.path.insert(0, os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from emu import Emu
from oracle import for_rom

BUTTONS = ["up", "down", "left", "right", "A", "B", "start", "select", "L", "R"]


def _ahash_pil(pil) -> int:
    """Dependency-free 16x16 average hash of a PIL frame -> 256-bit int. Same idea as the
    plugin's loop detector: near-identical screens hash close (small Hamming distance)."""
    g = pil.convert("L").resize((16, 16))
    px = list(g.getdata())
    m = sum(px) / len(px)
    bits = 0
    for v in px:
        bits = (bits << 1) | (1 if v >= m else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def act(self, pil_frame, state: dict) -> tuple[str, int, dict]:
        return self._rng.choice(BUTTONS), 1, {}

    def reset(self):
        pass


class VlmAgentPolicy:
    """The real pixels-only agent (VlmPlugin + LocalVlmReasoner) run inside the headless
    loop, so the oracle scores the actual policy - skills, working memory and all. Needs
    the VLM server (serve/run_server.sh) up. Imports are local so the random policy has no
    dependency on cv2/vga."""
    name = "vlm"

    def __init__(self, goal: str = "", url: str = "http://127.0.0.1:8077", upscale: int = 3,
                 game: str = "", enable_tutor: bool = False):
        import cv2
        import numpy as np
        from vga.core.contract import GameState
        from vga.reason.local_reasoner import LocalVlmReasoner
        from vga.reason.plugin import VlmPlugin
        self._cv2 = cv2
        self._np = np
        self._GameState = GameState
        knowledge = tutor = None
        if enable_tutor:
            # Tutorial-learning ON: persistent KnowledgeStore + TutorReader. upscale=1 because
            # act() already upscales the frame before it reaches the plugin, so the frames the
            # plugin accumulates are ALREADY display-resolution - upscaling again would distort.
            from vga.reason.knowledge import KnowledgeStore
            from vga.reason.tutor import TutorReader
            knowledge = KnowledgeStore()
            tutor = TutorReader(url=url, knowledge=knowledge, upscale=1)
        self.plugin = VlmPlugin(LocalVlmReasoner(url=url), goal=goal, game=game,
                                knowledge=knowledge, tutor=tutor)
        self._upscale = upscale
        self._i = 0

    def act(self, pil_frame, state: dict) -> tuple[str, int, dict]:
        # PIL RGB -> BGR ndarray (what capture()/the plugin expect).
        rgb = self._np.array(pil_frame)
        bgr = rgb[:, :, ::-1].copy()
        # Upscale native 240x160 with NEAREST (crisp pixel art) so the VLM can read text -
        # the on-screen path fed it a large scaled window; native is too small to read.
        if self._upscale and self._upscale != 1:
            bgr = self._cv2.resize(bgr, None, fx=self._upscale, fy=self._upscale,
                                   interpolation=self._cv2.INTER_NEAREST)
        h, w = bgr.shape[:2]
        gs = self._GameState(frame_index=self._i, timestamp=time.time(),
                             width=w, height=h, frame=bgr)
        action = self.plugin.decide(gs)
        self._i += 1
        btn = action.button.value if action.button is not None else "wait"
        meta = self.plugin.last_decision.meta if self.plugin.last_decision else {}
        # Feed the working-memory outcome for the NEXT decision: did the last action change
        # the screen? We approximate "changed" with reward!=0 here; the pixel diff is the
        # loop's job in the live runner. Keep the mode + tutorial signals for the step record.
        return btn, int(getattr(action, "repeats", 1) or 1), {
            "mode": meta.get("mode", ""), "reason": meta.get("reason", ""),
            "is_tutorial": bool(meta.get("is_tutorial")),
            "tutorial_learned": meta.get("tutorial_learned")}

    def note_changed(self, changed: bool, button: str, reason: str):
        self.plugin.note_outcome(button, reason, changed)

    def reset(self):
        self.plugin.reset()


def make_policy(kind: str, goal: str, url: str, game: str = "", enable_tutor: bool = False):
    if kind == "random":
        return RandomPolicy()
    if kind == "vlm":
        return VlmAgentPolicy(goal=goal, url=url, game=game, enable_tutor=enable_tutor)
    raise ValueError(f"unknown policy '{kind}'")


def rollout(rom: str, steps: int, policy_kind: str, out_dir: str,
            goal: str = "", url: str = "http://127.0.0.1:8077",
            hold: int = 6, then: int = 30, load_state: str = "",
            enable_tutor: bool = False, task: str = "") -> dict:
    # `then` (settle frames after each press) was 8 until 2026-07-03: FFTA UI boxes
    # ignore input for ~30 frames while animating in, so at then=8 an agent's press on a
    # freshly opened menu/confirm was EATEN (verified deterministically: the golden
    # naming commit start,left,A fails at then=8 and succeeds at then>=30). That both
    # blocked commits outright and taught the model that its presses "do nothing".
    # 30 frames adds ~0.5s emulated time per step -- noise next to the ~8s VLM latency.
    oracle = for_rom(rom)
    if task:
        oracle.task = task
    emu = Emu(rom)
    emu.boot(load_save=True)
    if load_state:
        with open(load_state, "rb") as f:
            emu.core.load_raw_state(f.read())
    else:
        oracle.boot_to_play(emu)

    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    steps_path = os.path.join(out_dir, "steps.jsonl")
    policy = make_policy(policy_kind, goal, url, game=os.path.basename(rom),
                         enable_tutor=enable_tutor)

    prev_state = None
    cum = 0.0
    with open(steps_path, "w") as f:
        for i in range(steps):
            frame_rel = os.path.join("frames", f"{i:06d}.png")
            emu.save_png(os.path.join(out_dir, frame_rel))
            state = oracle.read_state(emu)
            pil = emu.image.to_pil().convert("RGB")
            h_before = _ahash_pil(pil)

            button, repeats, pmeta = policy.act(pil, state)

            # Apply the action, then read the resulting state to score the transition.
            if button != "wait":
                for _ in range(max(1, repeats)):
                    emu.tap(button, hold=hold, then=then)
            else:
                emu.run(hold + then)
            new_state = oracle.read_state(emu)
            r = oracle.reward(prev_state, new_state)
            cum += r
            # Did the SCREEN actually change? This is the real looping signal (RAM barely
            # moves inside a menu). Threshold mirrors the plugin's revisit tolerance.
            h_after = _ahash_pil(emu.image.to_pil().convert("RGB"))
            changed = _hamming(h_before, h_after) > 10

            # Feed the TRUE screen-change outcome to the stateful policy's working memory.
            if hasattr(policy, "note_changed"):
                policy.note_changed(changed, button, pmeta.get("reason", ""))

            # Checkpoint ladder (train/eval only): instantaneous milestone predicates over the
            # resulting state, so analyze_rollout can score the FURTHEST rung the episode reached.
            cps = {name: bool(ok) for name, ok in oracle.checkpoints(new_state)}
            f.write(json.dumps({
                "step": i, "t": round(time.time(), 3), "frame": frame_rel,
                "action": {"button": button, "repeats": repeats},
                "state": state, "next_state": new_state,
                "reward": round(r, 3), "cum_reward": round(cum, 3),
                "mode": pmeta.get("mode", ""),
                "phash": h_before, "changed": changed,
                "checkpoints": cps,
                "is_tutorial": pmeta.get("is_tutorial", False),
                "tutorial_learned": pmeta.get("tutorial_learned"),
            }) + "\n")
            f.flush()  # keep steps.jsonl analyzable mid-run and durable if libmgba segfaults
            prev_state = new_state
            if (i + 1) % 25 == 0:
                print(f"[rollout] step {i+1}/{steps}  cum_reward={cum:.2f}  "
                      f"{oracle.progress_summary(new_state)}", flush=True)

    summary = {
        "rom": os.path.basename(rom), "oracle": oracle.name, "policy": policy_kind,
        "steps": steps, "cum_reward": round(cum, 3),
        "final": oracle.progress_summary(prev_state or {}),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"started": time.time(), "goal": goal, **summary}, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rom", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--policy", choices=["random", "vlm"], default="random")
    ap.add_argument("--goal", default="")
    ap.add_argument("--goal-file", default="", help="read goal from a file (avoids shell "
                    "quoting/env-expansion bugs in job launchers); overrides --goal")
    ap.add_argument("--url", default="http://127.0.0.1:8077")
    ap.add_argument("--out", default="")
    ap.add_argument("--hold", type=int, default=6)
    ap.add_argument("--then", type=int, default=8)
    ap.add_argument("--load-state", default="", help="raw savestate to load instead of boot_to_play")
    ap.add_argument("--learn-tutorials", action="store_true",
                    help="enable tutorial-learning (KnowledgeStore + TutorReader): detect "
                         "instructional screens, distill facts, retrieve them into the prompt")
    args = ap.parse_args()

    goal = args.goal
    if args.goal_file:
        with open(args.goal_file) as gf:
            goal = gf.read().strip()

    out = args.out or os.path.join(_REPO, "sessions", f"oracle-{int(time.time())}")
    print(f"[rollout] rom={os.path.basename(args.rom)} policy={args.policy} "
          f"steps={args.steps} goal={goal!r} -> {out}", flush=True)
    summary = rollout(args.rom, args.steps, args.policy, out, goal=goal,
                      url=args.url, hold=args.hold, then=args.then,
                      load_state=args.load_state, enable_tutor=args.learn_tutorials)
    print("[rollout] done:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
