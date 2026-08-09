#!/usr/bin/env python3
"""Capture a FAITHFUL post-accept world-map savestate for the R4 travel-commit harness.

Day-2 lesson: ffta_worldmap.state is a MISLEADING proxy (no mission accepted; its Start menu
doesn't default to Area List). The real R4 context is: mission ACCEPTED, standing on the world
map. We get it by running the greedy VLM policy from ffta_pub.state with the staged goal and
snapshotting the emulator the instant worldmap_regained fires (accepted AND scene==7). Greedy
decoding -> reproducible. Output: ffta_wm_postaccept.state (+ a verification print).
"""
import os, sys, time
sys.path.insert(0, "/home/timothy/projects/VGA")
from emu import Emu
from oracle import for_rom
from rollout import make_policy, _ahash_pil, _hamming

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/Final Fantasy Tactics Advance (E)(Surplus).gba"
PUB = os.path.join(os.path.dirname(__file__), "ffta_pub.state")
OUT = os.path.join(os.path.dirname(__file__), "ffta_wm_postaccept.state")
GOAL_FILE = os.path.join(os.path.dirname(__file__), "goal_herb_stages2.txt")
URL = "http://127.0.0.1:8077"
HOLD, THEN, MAX_STEPS = 6, 30, 90

def main():
    goal = open(GOAL_FILE).read().strip()
    emu = Emu(ROM); emu.boot(load_save=True)
    with open(PUB, "rb") as f:
        emu.core.load_raw_state(f.read())
    emu.run(2)
    oracle = for_rom(ROM)
    policy = make_policy("vlm", goal, URL, game=os.path.basename(ROM))

    for i in range(MAX_STEPS):
        pil = emu.image.to_pil().convert("RGB")
        state = oracle.read_state(emu)
        cps = dict(oracle.checkpoints(state))
        if cps.get("worldmap_regained"):
            raw = emu.core.save_raw_state()
            with open(OUT, "wb") as f:
                f.write(bytes(raw))
            print(f"[capture] worldmap_regained at step {i}: scene={state.get('scene')} "
                  f"funds={state.get('clan_funds')} pos={state.get('clan_pos')} -> {OUT}")
            # verify reload
            emu2 = Emu(ROM); emu2.boot(load_save=True)
            with open(OUT, "rb") as f:
                emu2.core.load_raw_state(f.read())
            emu2.run(4)
            s2 = oracle.read_state(emu2); c2 = dict(oracle.checkpoints(s2))
            print(f"[verify] reload: scene={s2.get('scene')} funds={s2.get('clan_funds')} "
                  f"accepted={c2.get('mission_accepted')} worldmap_regained={c2.get('worldmap_regained')}")
            return 0
        h_before = _ahash_pil(pil)
        btn, reps, pmeta = policy.act(pil, state)
        if btn != "wait":
            for _ in range(max(1, reps)):
                emu.tap(btn, hold=HOLD, then=THEN)
        else:
            emu.run(HOLD + THEN)
        # Feed the change signal back to the policy -- WITHOUT this the plugin's loop-escape /
        # commit-scaffold machinery is starved and the agent dithers forever (bug that made an
        # earlier capture stall at the accept rung). Mirrors rollout.py's loop.
        h_after = _ahash_pil(emu.image.to_pil().convert("RGB"))
        changed = _hamming(h_before, h_after) > 10
        if hasattr(policy, "note_changed"):
            policy.note_changed(changed, btn, pmeta.get("reason", ""))
        if (i + 1) % 10 == 0:
            print(f"[capture] step {i+1}: {oracle.progress_summary(oracle.read_state(emu))}", flush=True)
    print(f"[capture] FAILED to reach worldmap_regained in {MAX_STEPS} greedy steps")
    return 1

if __name__ == "__main__":
    sys.exit(main())
