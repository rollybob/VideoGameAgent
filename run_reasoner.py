"""
VGA reasoner loop - Phase 1: an API-VLM over the seam.

    capture -> perceive -> decide (VLM) -> act, logging every transition.

This is the autonomous loop from docs/NORTH_STAR.md Sec 8: capture -> perceive ->
decide -> act, logging every (frame_before, action, frame_after) transition as the
training data for the distilled policy and world model.

We pivoted away from a paid API as the runtime brain (Sec 6): the autonomous
reasoner will be a VLM served locally on Thor (--backend local, WIP - free, and no
frames leave the box). High-quality *seed* trajectories are collected with a
teacher in the loop via teacher.py (Claude Code under the Max plan). The `claude`
API backend remains as an option for anyone with a separately-billed key.

Usage:
    # Offline plumbing test - no X, no emulator, no model (fake frames + stub):
    .venv/bin/python run_reasoner.py --selftest --max-steps 12

    # Real emulator, deterministic stub reasoner (validates capture/action grounding):
    DISPLAY=:99 .venv/bin/python run_reasoner.py --backend stub --max-steps 20

    # Local on-Thor VLM (once built) - the free autonomous runtime:
    DISPLAY=:99 .venv/bin/python run_reasoner.py --backend local --goal "..." --max-steps 50

    # Seed data with a teacher in the loop (Max plan, no API): see teacher.py
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from vga.reason.base import ReasonContext, action_from_choice
from vga.reason.plugin import VlmPlugin
from vga.reason.stub_reasoner import StubReasoner
from vga.trajectory import TrajectoryLogger


class _FakeEmulator:
    """A stand-in emulator for --selftest: emits random GBA-sized BGR frames and
    swallows inputs. Lets us exercise the full loop and logger without X."""

    def __init__(self, w: int = 240, h: int = 160):
        self._w, self._h = w, h
        self._rng = np.random.default_rng(0)

    def capture(self) -> np.ndarray:
        return self._rng.integers(0, 256, size=(self._h, self._w, 3), dtype=np.uint8)

    def send(self, action) -> None:
        pass


def _build_reasoner(args):
    if args.selftest or args.stub or args.backend == "stub":
        return StubReasoner()
    if args.backend == "local":
        # The on-Thor VLM server (serve/serve_vlm.py); free autonomous runtime.
        from vga.reason.local_reasoner import LocalVlmReasoner
        return LocalVlmReasoner(url=args.vlm_url)
    if args.backend == "claude":
        # Optional: requires a separately-billed ANTHROPIC_API_KEY. We have pivoted
        # away from this as the default (see NORTH_STAR Sec 6); kept as a backend.
        from vga.reason.claude_reasoner import ClaudeReasoner
        return ClaudeReasoner(model=args.model)
    raise SystemExit(f"[reasoner] unknown backend {args.backend!r}")


def _build_emulator(args):
    if args.selftest:
        return _FakeEmulator()
    from vga.core.emulator import EmulatorNotFound, GbaEmulator
    try:
        return GbaEmulator(title_keyword=args.title)
    except EmulatorNotFound as e:
        print(f"[reasoner] {e}")
        sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description="VGA reasoner loop (API-VLM over the seam)")
    ap.add_argument("--title", default="mGBA", help="emulator window title keyword")
    ap.add_argument("--goal", default="", help="objective handed to the reasoner each frame")
    ap.add_argument("--model", default="claude-opus-4-8", help="VLM model id (claude backend)")
    ap.add_argument("--vlm-url", default="http://127.0.0.1:8077",
                    help="local VLM server URL (--backend local)")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N steps")
    ap.add_argument("--frame-delay", type=float, default=0.6,
                    help="seconds between decisions (System 2 cadence, not frame rate)")
    ap.add_argument("--run-dir", default=None, help="trajectory output dir (default: sessions/traj-<ts>)")
    ap.add_argument("--backend", choices=["stub", "local", "claude"], default="stub",
                    help="reasoner backend: stub (default, no model), local (on-Thor VLM, WIP), "
                         "claude (optional, needs a separately-billed API key)")
    ap.add_argument("--stub", action="store_true", help="alias for --backend stub")
    ap.add_argument("--selftest", action="store_true", help="fake emulator + stub reasoner (no X, no model)")
    ap.add_argument("--stuck-thresh", type=float, default=2.0,
                    help="mean-abs pixel diff below which a frame counts as unchanged (stuck detector)")
    ap.add_argument("--stuck-n", type=int, default=3,
                    help="consecutive unchanged frames before forcing an unstick action (0 disables)")
    ap.add_argument("--loop-thresh", type=float, default=6.0,
                    help="mean-abs diff below which a repeated-button step counts as 'no progress'")
    ap.add_argument("--loop-n", type=int, default=6,
                    help="same button repeated this many times with < loop-thresh change = stuck loop (0 disables)")
    ap.add_argument("--change-thresh", type=float, default=2.0,
                    help="mean-abs pixel diff at/above which the screen counts as CHANGED for the "
                         "working-memory signal fed back to the reasoner (Phase-A state awareness)")
    ap.add_argument("--history-len", type=int, default=6,
                    help="how many recent executed steps the reasoner is shown (working memory)")
    ap.add_argument("--revisit-window", type=int, default=30,
                    help="how many recent screen-hashes to remember (>= --loop-window)")
    ap.add_argument("--revisit-tol", type=int, default=10,
                    help="Hamming distance (of 256 hash bits) at/under which two screens count as the same")
    ap.add_argument("--loop-window", type=int, default=12,
                    help="recent screens examined for loop density (the 'going in circles' signal)")
    ap.add_argument("--loop-density", type=float, default=0.6,
                    help="fraction of the loop window that must be recurring screens to flag a loop")
    ap.add_argument("--no-dedup", action="store_true",
                    help="disable OCR dialogue-dedup (the 'you already read this' signal)")
    ap.add_argument("--learn-tutorials", action="store_true",
                    help="enable tutorial-learning: detect instructional screens, distill facts "
                         "into a persistent per-game KnowledgeStore, and feed them back into the "
                         "prompt. Ideal for dropping the agent into a NEW game (needs --backend local)")
    ap.add_argument("--game", default="",
                    help="game key for the KnowledgeStore (defaults to --title); learned facts "
                         "persist per game across runs")
    args = ap.parse_args()

    reasoner = _build_reasoner(args)
    emu = _build_emulator(args)
    # Tutorial-learning wiring. Frames from emu.capture() are already display-resolution, so the
    # TutorReader must NOT upscale again (upscale=1). Only active with --learn-tutorials.
    knowledge = tutor = None
    game = args.game or args.title
    if args.learn_tutorials:
        from vga.reason.knowledge import KnowledgeStore
        from vga.reason.tutor import TutorReader
        knowledge = KnowledgeStore()
        tutor = TutorReader(url=args.vlm_url, knowledge=knowledge, upscale=1)
        print(f"[reasoner] tutorial-learning ON (game='{game}', store={knowledge.path}, "
              f"{knowledge.count()} facts known)")
    plugin = VlmPlugin(reasoner, goal=args.goal, history_len=args.history_len,
                       revisit_window=args.revisit_window, revisit_tol=args.revisit_tol,
                       loop_window=args.loop_window, loop_density=args.loop_density,
                       enable_dedup=not args.no_dedup,
                       game=game, knowledge=knowledge, tutor=tutor)
    plugin.reset()

    run_dir = args.run_dir or f"sessions/traj-{int(time.time())}"
    logger = TrajectoryLogger(run_dir, run_meta={
        "reasoner": reasoner.name, "model": args.model, "goal": args.goal,
        "selftest": args.selftest,
    })
    print(f"[reasoner] driving with '{reasoner.name}' -> logging to {run_dir}. Ctrl+C to stop.")

    prev_state = None
    prev_action = None
    prev_meta = None
    prev_frame = None
    prev_executed = None  # {button, reason} actually SENT last step (post-unstick), for the
                          # working-memory outcome we report to the plugin once we can diff it
    # Stuck detector: a self-driving VLM can choose "wait" on a screen that is halted
    # waiting for input (e.g. an unadvanced "press A" text prompt) -- the frame never
    # changes, so it re-derives the same wait forever and deadlocks (observed in
    # traj-pokemon-3: 300 frozen steps on a "XYLOMON fainted!" prompt). When the frame
    # has not changed for --stuck-n steps, override the reasoner. Two stuck modes seen:
    #   (1) FROZEN: an unadvanced "press A" text box halts the emulator -> diff ~0
    #       (traj-pokemon-3: 300 steps on "XYLOMON fainted!"). Fix = press A.
    #   (2) LOOP: the VLM mashes one button in a menu; the screen only flickers (cursor
    #       blink, diff ~2, just under the frozen bar) so it never trips (1), but no real
    #       progress happens (traj-pokemon-4: 358 A-presses stuck in POKEMON LIST).
    # The unstick cycle tries a DIFFERENT button than the one the VLM is repeating -- B
    # first (backs out of menus, the loop case), then A (advances text, the frozen case),
    # then Start/directions for anything else.
    UNSTICK_CYCLE = ["B", "A", "start", "down", "right", "up"]
    stuck = 0            # consecutive frozen frames (mode 1)
    unstick_idx = 0
    repeat_button = None  # the VLM's last chosen button and how many times running (mode 2)
    repeat_count = 0
    i = 0
    try:
        while True:
            frame = emu.capture()
            state = plugin.perceive(frame, i, time.time())

            # Frame delta vs the previous decision's frame. Computed BEFORE decide() so
            # it feeds two things: (a) the reasoner's working memory - report whether the
            # last EXECUTED action changed the screen; (b) the unstick detector below.
            if prev_frame is not None:
                diff = float(np.abs(frame.astype(np.int16) - prev_frame.astype(np.int16)).mean())
            else:
                diff = 999.0
            if prev_executed is not None:
                plugin.note_outcome(prev_executed["button"], prev_executed["reason"],
                                    diff >= args.change_thresh)

            # Log the previous transition now that we have its resulting frame.
            if prev_state is not None:
                logger.log_transition(prev_state, prev_action, state, prev_meta)

            action = plugin.decide(state)
            decision = plugin.last_decision
            meta = dict(decision.meta) if decision else {}
            meta["reason"] = decision.reason if decision else ""
            if meta.get("tutorial_learned"):
                tl = meta["tutorial_learned"]
                print(f"[reasoner] LEARNED tutorial fact ({tl.get('topic')}), "
                      f"new={tl.get('added')} -> {plugin.knowledge.count()} facts known")

            # Force an unstick if either the screen is frozen (mode 1) or the VLM is
            # looping one button to no effect (mode 2). Uses the diff computed above.
            vlm_button = meta.get("button")
            if vlm_button == repeat_button:
                repeat_count += 1
            else:
                repeat_button, repeat_count = vlm_button, 1
            frozen = diff < args.stuck_thresh
            looping = args.loop_n and repeat_count >= args.loop_n and diff < args.loop_thresh
            if frozen:
                stuck += 1
            else:
                stuck = 0
            if not (frozen or looping):
                unstick_idx = 0
            if (args.stuck_n and stuck >= args.stuck_n) or looping:
                # Pick the next cycle button that differs from what the VLM is repeating,
                # so a menu A-loop is broken by B (back out), not another A.
                btn = UNSTICK_CYCLE[unstick_idx % len(UNSTICK_CYCLE)]
                tries = 0
                while btn == vlm_button and tries < len(UNSTICK_CYCLE):
                    unstick_idx += 1; tries += 1
                    btn = UNSTICK_CYCLE[unstick_idx % len(UNSTICK_CYCLE)]
                unstick_idx += 1
                mode = "frozen" if frozen else f"loop x{repeat_count}"
                note = f"unstick ({mode}), vlm chose {vlm_button}"
                action = action_from_choice(btn, note=note)
                meta["unstick"] = True
                meta["stuck_frames"] = stuck
                meta["vlm_button"] = vlm_button
                meta["button"] = btn
                meta["reason"] = note
                repeat_button, repeat_count = btn, 1  # don't immediately re-flag the override
                print(f"[reasoner] step {i}: UNSTICK -> {btn} ({mode}, vlm wanted {vlm_button})")
            else:
                print(f"[reasoner] step {i}: {action}  ({meta.get('reason', '')})")

            emu.send(action)

            # Record what was ACTUALLY sent (post-unstick) so next iteration can report
            # its on-screen effect to the reasoner's working memory.
            prev_executed = {"button": meta.get("button", "wait"), "reason": meta.get("reason", "")}
            prev_state, prev_action, prev_meta = state, action, meta
            prev_frame = frame
            i += 1
            if args.max_steps is not None and i >= args.max_steps:
                # Capture the final resulting frame so the last transition is complete.
                final = plugin.perceive(emu.capture(), i, time.time())
                logger.log_transition(prev_state, prev_action, final, prev_meta)
                print(f"[reasoner] reached max-steps={args.max_steps}, stopping.")
                break
            if args.frame_delay > 0:
                time.sleep(args.frame_delay)
    except KeyboardInterrupt:
        print("\n[reasoner] stopped by user.")
    finally:
        logger.close()
        print(f"[reasoner] logged {logger.count} transitions to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
