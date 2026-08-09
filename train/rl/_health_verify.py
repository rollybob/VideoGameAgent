"""Verify the REAL solo-ALttP health oracle (EWRAM 0x0234D via AlttpOracle.read_health,
bee-validated) -- correcting my earlier probes that read the wrong addr/region.
Part 1: health + max across all 7 states (should be sane, <=max, ~hearts*8).
Part 2: room-3 ChainHunter episodes with the CORRECT health trace -- confirm deaths are
REAL and show HOW Link dies (delta -2 = bee sting/quarter-heart; big deltas = hard hits).
"""
import math, os, random, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from alttp_ppo_env import AlttpPpoEnv
from harvest_key_frames import u16, ENEMY_X, ENEMY_Y, ENEMY_HP
from chain_harvest import ChainHunter, DIR_KEYS


def part1():
    print("=== PART 1: real health (EWRAM 0x0234D) at reset, all 7 states ===")
    for i in range(7):
        env = AlttpPpoEnv(state_path=os.path.join(HERE, "states", f"alttp_human-0{i}.state"), horizon=16)
        env.reset(seed=0)
        hp = env.oracle.read_health(env.core)
        mx = env.oracle.read_health_max(env.core)
        print(f"  room{i}: health={hp} (={hp/8:.2f} hearts)  max={mx} (={mx/8:.1f})  "
              f"{'OK' if 0 < hp <= mx else 'CHECK'}")
        env.close()


def part2():
    from mgba._pylib import ffi
    print("\n=== PART 2: room-3 death trace (correct health) ===")
    for ep in range(5):
        env = AlttpPpoEnv(state_path=os.path.join(HERE, "states", "alttp_human-03.state"), horizon=1500)
        env.reset(seed=5000 + ep)
        iw = ffi.cast("uint8_t *", env.core._native.memory.iwram)
        rng = random.Random(5000 * 7 + ep); _ = rng.random()
        hunter = ChainHunter(iw, env.oracle, env.core, rng, validate=True)
        preroll = rng.randint(0, 15); drops = []
        hp_prev = env.oracle.read_health(env.core)
        hits = []; kill_step = None; step = 0; done = False
        while not done:
            act = (np.array([rng.choice(DIR_KEYS),0,0,0,0]) if step < preroll
                   else np.array(hunter.act(drops), dtype=np.int64))
            ex, ey = u16(iw, ENEMY_X), u16(iw, ENEMY_Y)
            _o,_r,term,trunc,info = env.step(act); done = term or trunc; step += 1
            hp = env.oracle.read_health(env.core)
            if hp < hp_prev:
                hits.append((step, hp - hp_prev, hp))
            hp_prev = hp
            for _t, ch, _d, val in info["events"]:
                if ch == "enemy_dmg" and val == 0 and (ex,ey)!=(0,0):
                    if kill_step is None: kill_step = step
                    drops.append([ex,ey,step]); hunter.on_kill((ex,ey))
                elif ch == "key": hunter.on_pickup()
        died = env.oracle.read_health(env.core) == 0
        deltas = [d for _s,d,_h in hits]
        print(f"  ep{ep}: died={died} final_hp={hp_prev} kill_step={kill_step} "
              f"n_hits={len(hits)} deltas={deltas[:12]}")
        env.close()


if __name__ == "__main__":
    part1(); part2()
