"""Rung-3 pivot measurement: does the room-4-trained chain skill TRANSFER to a
held-out room, with NO new training? Health-free (kills+keys from events only;
the 0x00428 health oracle is invalid in solo rooms != -04).

Policies: base reflex S1 (scratch04 600k), rung1 (collect-distilled), rung2
(chain-distilled, trained ONLY on room-4 demos). Rooms: 4 = train reference,
2 = HELD-OUT clean key-chain room (ChainHunter gets 12/12 there), 0 = no-key-drop
control (a perfect policy still can't complete a chain -> isolates room vs policy).

chain_complete(ep) = a kill (enemy_dmg==0, live pos) then a key within WINDOW decisions.

  docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/vga \\
      -w /vga/train/rl thor-rl:cu130 python3 -u transfer_chain_eval.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from alttp_ppo_env import AlttpPpoEnv          # noqa: E402
from harvest_key_frames import u16, ENEMY_X, ENEMY_Y  # noqa: E402

POLICIES = {
    "base":  os.path.join(HERE, "runs", "alttp_s1v2_scratch04", "ppo_alttp_600000_steps.zip"),
    "rung1": os.path.join(HERE, "runs", "distill_r1", "s1_distilled.zip"),
    "rung2": os.path.join(HERE, "runs", "distill_r2", "s1_joint_r2c.zip"),
    "rung3": os.path.join(HERE, "runs", "distill_r3", "s1_joint_r3.zip"),
}
# room2 & room4 are the castle's only key rooms; both are IN rung-3's training set,
# so rung-3 room2 = in-sample (was held-out for rung2). room0 = no-key control.
ROOMS = {"room4": 4, "room2": 2, "room0_ctrl": 0}
WINDOW = 250
N_EPS = 16


def eval_policy_room(model, state, n_eps, ffi, seed=7000, horizon=1500):
    complete = killed = 0
    for ep in range(n_eps):
        env = AlttpPpoEnv(state_path=state, horizon=horizon)
        obs, _ = env.reset(seed=seed + ep)
        iw = ffi.cast("uint8_t *", env.core._native.memory.iwram)
        model.set_random_seed(seed + ep)
        first_kill = None
        got_key_after = False
        got_kill = False
        step, done = 0, False
        while not done:
            act, _ = model.predict(obs, deterministic=False)
            ex, ey = u16(iw, ENEMY_X), u16(iw, ENEMY_Y)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            step += 1
            for _t, ch, _d, val in info["events"]:
                if ch == "enemy_dmg" and val == 0 and (ex, ey) != (0, 0):
                    got_kill = True
                    if first_kill is None:
                        first_kill = step
                elif ch == "key":
                    if first_kill is not None and step - first_kill <= WINDOW:
                        got_key_after = True
            if got_key_after:
                break
        killed += int(got_kill)
        complete += int(got_key_after)
        env.close()
    return killed, complete


def main():
    from mgba._pylib import ffi
    from stable_baselines3 import PPO
    out = {}
    print(f"chain-complete rate (kill->key within {WINDOW}), {N_EPS} eps each\n")
    print(f"{'policy':7} | " + " | ".join(f"{r:>14}" for r in ROOMS))
    for pname, ppath in POLICIES.items():
        if not os.path.exists(ppath):
            print(f"{pname:7} | (skipped -- {os.path.basename(ppath)} not found)", flush=True)
            continue
        model = PPO.load(ppath, device="cuda")
        row = {}
        cells = []
        for rname, ridx in ROOMS.items():
            state = os.path.join(HERE, "states", f"alttp_human-0{ridx}.state")
            killed, complete = eval_policy_room(model, state, N_EPS, ffi)
            row[rname] = {"killed": killed, "complete": complete, "n": N_EPS}
            cells.append(f"{complete:2d}/{N_EPS} (k{killed:2d})")
        out[pname] = row
        print(f"{pname:7} | " + " | ".join(f"{c:>14}" for c in cells), flush=True)
    with open(os.path.join(HERE, "_transfer_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSAVED _transfer_eval.json")


if __name__ == "__main__":
    main()
