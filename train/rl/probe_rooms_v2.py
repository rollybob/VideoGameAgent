"""Rung-3 room selection, CLEAN re-probe (health-free).

The v1 probe (_r3_room_probe.sh) was corrupted by the env's health/death oracle:
0x00428 is a FOUR-SWORDS health addr and reads garbage (222-254) in 6/7 solo-ALttP
human states (only room4=-04 valid). chain_harvest's `if died: break` then fires on
spurious 0x00428->0 events, miscounting real episodes as "death" (room3's 10 "deaths"
were this artifact -- Link does NOT actually die there).

This probe classifies rooms using ONLY health-independent signals:
  kill  = enemy_dmg event, val==0, live enemy pos (ex,ey)!=(0,0)  [death-teardown guard]
  key   = key event fires after a kill
No death-break, no health reward. Targets slot-3 enemy (0x03852/54,0x03253) as v1 did
-- room5 has slot3 dead so it will read no_kill (needs live-slot targeting, deferred).

Per room over N eps: killed (>=1 kill), keyed (kill+key = usable chain), no_kill,
and death_events_seen (how noisy the bogus health signal is, informational only).

  docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/vga \\
      -w /vga/train/rl thor-rl:cu130 python3 -u probe_rooms_v2.py
"""
import json
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from alttp_ppo_env import AlttpPpoEnv          # noqa: E402
from harvest_key_frames import u16, ENEMY_X, ENEMY_Y  # noqa: E402
from chain_harvest import ChainHunter, DIR_KEYS  # noqa: E402


def probe_room(state, n_eps=12, horizon=1500, seed=5000):
    from mgba._pylib import ffi
    killed = keyed = no_kill = death_events = 0
    for ep in range(n_eps):
        env = AlttpPpoEnv(state_path=state, horizon=horizon)
        env.reset(seed=seed + ep)
        iw = ffi.cast("uint8_t *", env.core._native.memory.iwram)
        rng = random.Random(seed * 7 + ep)
        _ = rng.random()                       # match chain_harvest rng ordering
        hunter = ChainHunter(iw, env.oracle, env.core, rng, validate=True)
        preroll = rng.randint(0, 15)
        drops, got_kill, got_key = [], False, False
        step, done = 0, False
        while not done:
            if step < preroll:
                act = np.array([rng.choice(DIR_KEYS), 0, 0, 0, 0], dtype=np.int64)
            else:
                act = np.array(hunter.act(drops), dtype=np.int64)
            ex, ey = u16(iw, ENEMY_X), u16(iw, ENEMY_Y)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            step += 1
            for _t, ch, d, v in info["events"]:
                if ch == "enemy_dmg" and v == 0 and (ex, ey) != (0, 0):
                    got_kill = True
                    drops.append([ex, ey, step]); hunter.on_kill((ex, ey))
                elif ch == "key":
                    got_key = True; hunter.on_pickup()
                elif ch.startswith("death"):
                    death_events += 1          # counted, NOT acted on
            if got_key and got_kill:
                break                          # usable chain done
        killed += int(got_kill)
        keyed += int(got_kill and got_key)
        no_kill += int(not got_kill)
        env.close()
    return {"n": n_eps, "killed": killed, "keyed": keyed,
            "no_kill": no_kill, "death_events": death_events}


def main():
    rows = {}
    for i in range(7):
        s = os.path.join(HERE, "states", f"alttp_human-0{i}.state")
        r = probe_room(s)
        rows[f"room{i}"] = r
        print(f"room{i}: killed={r['killed']}/{r['n']} keyed(usable chain)={r['keyed']}"
              f" no_kill={r['no_kill']} spurious_death_events={r['death_events']}",
              flush=True)
    with open(os.path.join(HERE, "_r3probe_v2.json"), "w") as f:
        json.dump(rows, f, indent=2)
    usable = [k for k, v in rows.items() if v["keyed"] >= 6]
    print(f"\nUSABLE key-chain rooms (keyed>=6/12): {usable}", flush=True)


if __name__ == "__main__":
    main()
