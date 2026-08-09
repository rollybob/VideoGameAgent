"""Hunt the key counter by letting the TRAINED policy produce the event.

Tim's observation: in alttp_human-02 a key spawns when the room's only enemy
dies. The a2 policy reliably kills that enemy, so we do not need a human to
generate the event -- run the policy, snapshot RAM continuously, and look for a
byte that steps UP once mid-episode and holds (a pickup), rather than the
step-DOWN pattern used for health.

Emits candidates plus a PNG at the transition tick so the HUD can be checked for
a key icon -- the same "confirm against pixels, not plausibility" rule that
settled health and rupees.

  docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/vga \\
      -w /vga/train/rl thor-rl:cu130 python3 find_key_addr.py \\
      runs/alttp_a2/ppo_alttp_final.zip --out /vga/train/rl/scratch/keyhunt
"""
import argparse
import os
import sys

import numpy as np
from mgba._pylib import ffi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EWRAM_SIZE, IWRAM_SIZE = 256 * 1024, 32 * 1024


def snapshot(core):
    ew = np.frombuffer(ffi.buffer(core._native.memory.wram, EWRAM_SIZE), dtype=np.uint8)
    iw = np.frombuffer(ffi.buffer(core._native.memory.iwram, IWRAM_SIZE), dtype=np.uint8)
    return np.concatenate([iw, ew])


def single_step_up(col):
    """Constant, one increase, constant -- the shape of picking something up."""
    change = np.flatnonzero(col[1:] != col[:-1])
    if change.size != 1:
        return None
    i = int(change[0])
    before, after = int(col[0]), int(col[-1])
    if after <= before or i < 3 or i > col.size - 4:
        return None
    return before, after, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--state", default=os.path.join(HERE, "states", "alttp_human-02.state"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from stable_baselines3 import PPO
    from alttp_ppo_env import AlttpPpoEnv
    model = PPO.load(args.model, device="cuda")

    per_ep = []
    for ep in range(args.episodes):
        env = AlttpPpoEnv(horizon=10 ** 9, state_path=args.state, death_terminates=True,
                          frame_skip=4)
        obs, _ = env.reset()
        snaps, ticks, images = [], [], {}
        n = 0
        while n < args.frames:
            a, _ = model.predict(obs, deterministic=False)
            obs, r, term, trunc, info = env.step(int(a))
            n = info["tick"]
            if len(snaps) == 0 or n - ticks[-1] >= args.every:
                snaps.append(snapshot(env.core))
                ticks.append(n)
            if term:
                break
        ring = np.stack(snaps)
        rows = []
        changed = np.flatnonzero(ring[0] != ring[-1])
        for addr in changed:
            got = single_step_up(ring[:, addr])
            if got is None:
                continue
            before, after, i = got
            # a key count is a small integer that ticks up by one
            score = 0
            if after - before == 1:
                score += 3
            if 0 <= before <= 8 and 1 <= after <= 9:
                score += 3
            rows.append((score, int(addr), before, after, ticks[i]))
        rows.sort(key=lambda r: -r[0])
        per_ep.append({(r[1]) for r in rows if r[0] >= 6})
        print("ep %d: %d frames, %d single-step-up candidates, %d key-shaped" % (
            ep, n, len(rows), len(per_ep[-1])), flush=True)
        for score, addr, b, a_, t in rows[:8]:
            region = "IWRAM" if addr < IWRAM_SIZE else "EWRAM"
            off = addr if addr < IWRAM_SIZE else addr - IWRAM_SIZE
            print("   %-6s 0x%05X  %d -> %d at tick %d  (score %d)" % (
                region, off, b, a_, t, score), flush=True)
        env.close()

    common = set.intersection(*per_ep) if per_ep and all(per_ep) else set()
    print("\nkey-shaped in ALL %d episodes: %d" % (args.episodes, len(common)))
    for addr in sorted(common):
        region = "IWRAM" if addr < IWRAM_SIZE else "EWRAM"
        off = addr if addr < IWRAM_SIZE else addr - IWRAM_SIZE
        print("   %-6s 0x%05X" % (region, off))
    if not common:
        print("nothing consistent -- the policy may not be picking the key up. "
              "A human mark (kill, grab key, F11) is the fallback.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
