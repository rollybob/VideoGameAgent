"""Harvest RAM-labeled key-drop frames on alttp_human-04 (headless, offline labeling).

WHY (arbiter Stage 2, 2026-08-06 evening): the retrain saga ended with the reframe that
kill->collect-key is System-2's job. The planned arbiter interject is "S2 sees a dropped
key and steers Link onto it" -- which stands or falls on whether the VLM can actually SEE
a dropped key in a raw GBA frame. This script produces the labeled probe set to measure
exactly that, with zero hand labeling. RAM is TEACHER/LABELER only (offline data
collection), never an inference input -- north-star compliant.

DRIVERS
  --driver scripted (default): RAM-guided hunter state machine. Homes onto the -04
    enemy (slot 3, pos IWRAM 0x03852/54, health 0x03253), slashes it with B, records
    the drop at the death position, then WALKS AWAY a random direction/distance (that
    is what makes the dx/dy geometry diverse instead of always standing on the drop),
    loiters, and on every PICKUP_EVERY-th drop walks back onto it until the key event
    fires -- which VALIDATES that "walk onto the death position" collects the key,
    the exact primitive the arbiter burst will use.
  --driver policy --ckpt <zip>: original S1-driven harvest, kept for comparison runs.

LESSON BAKED IN (first run, ep1): when Link DIES, the sprite table tears down and every
occupied slot's health drops to 0 at once -- 11 fake "kills" with position (0,0), then
3000 decisions of the continue screen labeled key_onscreen. Kill validation now requires
Link alive + no death event this step + nonzero enemy position; all classification is
gated on Link being alive.

Label phases: control (no drop yet), death_anim (drop younger than VIS_DELAY),
key_onscreen (active drop within centered-camera margins), key_offscreen (outside).
Output: <out>/frames/*.png, <out>/labels.jsonl, <out>/summary.json.

Run:
  docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/work -w /work \
      thor-rl:cu130 python3 -u train/rl/harvest_key_frames.py
"""
import argparse
import json
import math
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from alttp_ppo_env import AlttpPpoEnv  # noqa: E402

# -04 enemy = sprite slot 3: health 0x03253 (oracle's old slot-specific addr), live
# position 0x03852/0x03854 (authoritative -- poke-tested in confirm_enemy_pos.py).
ENEMY_HP = 0x03253
ENEMY_X, ENEMY_Y = 0x03852, 0x03854

VIS_DELAY = 6            # decisions after kill before the drop is labeled visible
MARGIN_X, MARGIN_Y = 104, 64   # centered-camera on-screen margins (world units)
ONLINK_DIST = 14
ATTACK_DIST = 24         # start slashing inside this range
# The -04 enemy does NOT respawn while you stay in the room (measured: 1 kill/ep,
# then slot 3 stays 0 for 2900+ decisions), so many SHORT episodes -- fresh state
# each -- is how this harvester gets kill-geometry diversity, not respawn farming.
PICKUP_EVERY = 2         # validate every 2nd drop by walking onto it

DIRS = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]
# MultiDiscrete dir index for an 8-way world direction (env DIR_MASKS order:
# 0=none 1=U 2=D 3=L 4=R 5=U+R 6=U+L 7=D+R 8=D+L). UP = -Y, LEFT = -X
# (convention proven by confirm_enemy_pos.py's poke arithmetic).
DIR_IDX = {"up": 1, "down": 2, "left": 3, "right": 4,
           "up-right": 5, "up-left": 6, "down-right": 7, "down-left": 8}


def direction_of(dx, dy):
    ang = math.degrees(math.atan2(dy, dx)) % 360.0
    return DIRS[int(((ang + 22.5) % 360.0) // 45.0)]


def u16(iw, addr):
    return int(iw[addr]) | (int(iw[addr + 1]) << 8)


def save_png(arr, path):
    from PIL import Image
    Image.fromarray(arr).save(path)


class ScriptedHunter:
    """RAM-guided kill->scatter->loiter->(pickup) state machine over MultiDiscrete."""

    def __init__(self, iw, oracle, core, rng, validate=False, short_scatter=False):
        # validate is per-EPISODE (one kill per episode -- see PICKUP_EVERY note)
        self.validate = validate
        self.short_scatter = short_scatter
        self.iw, self.oracle, self.core, self.rng = iw, oracle, core, rng
        self.mode = "hunt"
        self.timer = 0
        self.scatter_dir = "down"
        self.pickup_target = None
        self.kills = 0
        self.stuck = 0
        self.prev_dist = None
        self.t = 0

    def act(self, drops):
        self.t += 1
        lx, ly = self.oracle.read_pos(self.core)
        ex, ey = u16(self.iw, ENEMY_X), u16(self.iw, ENEMY_Y)
        ehp = int(self.iw[ENEMY_HP])

        if self.mode == "scatter":
            self.timer -= 1
            if self.timer <= 0:
                self.mode = "loiter"
                self.timer = self.rng.randint(6, 12)
            return [DIR_IDX[self.scatter_dir], 0, 0, 0, 0]

        if self.mode == "loiter":
            self.timer -= 1
            if self.timer <= 0:
                if self.pickup_target is not None:
                    self.mode = "pickup"
                    self.timer = 120
                else:
                    self.mode = "hunt"
            return [0, 0, 0, 0, 0]

        if self.mode == "pickup":
            tx, ty = self.pickup_target
            dx, dy = tx - lx, ty - ly
            self.timer -= 1
            if self.timer <= 0:   # give up (drop may have bounced out of reach)
                self.pickup_target = None
                self.mode = "hunt"
                return [0, 0, 0, 0, 0]
            return [DIR_IDX[direction_of(dx, dy)], 0, 0, 0, 0]

        # hunt: home onto a live enemy, slash in range; wait out respawns in place
        if ehp == 0 or (ex, ey) == (0, 0):
            return [0, 0, 0, 0, 0]      # respawn wait (also saves loiter-ish frames)
        dx, dy = ex - lx, ey - ly
        dist = math.hypot(dx, dy)
        # stuck detection: not closing for 40 decisions -> jiggle sideways
        if self.prev_dist is not None and dist >= self.prev_dist - 0.5:
            self.stuck += 1
        else:
            self.stuck = 0
        self.prev_dist = dist
        if self.stuck > 40:
            self.stuck = 20
            return [DIR_IDX[self.rng.choice(DIRS)], 0, 0, 0, 0]
        # TAP B (alternate decisions = 4 frames on, 4 off): holding it would charge
        # a spin attack and never actually swing
        slash = 1 if (dist <= ATTACK_DIST and self.t % 2 == 0) else 0
        return [DIR_IDX[direction_of(dx, dy)], 0, slash, 0, 0]

    def on_kill(self, drop):
        self.kills += 1
        self.scatter_dir = self.rng.choice(DIRS)
        self.mode = "scatter"
        # 8-18 decisions (~32-72 units): drops mostly stay on-camera; the first
        # run's 8-30 range walked drops clean off screen. Bank harvests use 6-13
        # (~24-52 units) so the drop stays inside the BANK safe box more often.
        self.timer = self.rng.randint(6, 13) if self.short_scatter else self.rng.randint(8, 18)
        if self.validate:
            self.pickup_target = tuple(drop)
        self.prev_dist = None

    def on_pickup(self):
        was_deliberate = (self.mode == "pickup")
        self.pickup_target = None
        if was_deliberate:
            self.mode = "hunt"
        return was_deliberate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", choices=["scripted", "policy"], default="scripted")
    ap.add_argument("--ckpt", default=None, help="required for --driver policy")
    ap.add_argument("--state", default=os.path.join(HERE, "states", "alttp_human-04.state"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "perception",
                                                  "keyprobe_data"))
    ap.add_argument("--save-states", action="store_true",
                    help="snapshot a keydrop-NN.state per episode once the drop is "
                         "visible (pre-pickup) -> the arbiter eval's key-on-floor bank")
    ap.add_argument("--no-frames", action="store_true",
                    help="bank-only harvest: skip probe frame/label output")
    ap.add_argument("--wide-scatter", action="store_true",
                    help="bank harvest with the original 8-18 scatter (harder bank: "
                         "fuller key-distance spread; safe box still enforced)")
    a = ap.parse_args()

    # Safe onscreen box for banked states (2026-08-06 fix): the first bank snapshotted
    # at a fixed age with NO position check, and 4/12 states started with the key at or
    # past the vertical camera edge (|dy| 64-70 vs MARGIN_Y=64) -- the arbiter eval's
    # key-visible-at-t0 premise was silently violated. Snapshot only inside margins
    # minus a buffer (key sprite extent + camera settle), checked EVERY step so a
    # too-far scatter can still bank later when the drop re-enters view.
    BANK_X, BANK_Y = MARGIN_X - 20, MARGIN_Y - 14

    from mgba._pylib import ffi

    model = None
    if a.driver == "policy":
        from stable_baselines3 import PPO
        model = PPO.load(a.ckpt, device="cuda")

    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True)
    labels_f = open(os.path.join(a.out, "labels.jsonl"), "w")
    env = AlttpPpoEnv(state_path=a.state, horizon=a.horizon)

    totals = {"kills": 0, "keys": 0, "deaths": 0, "pickup_validated": 0, "frames": {}}
    for ep in range(a.episodes):
        det = (ep % 2 == 0)
        obs, _ = env.reset(seed=a.seed + ep)
        iw = ffi.cast("uint8_t *", env.core._native.memory.iwram)
        rng = random.Random(a.seed * 1000 + ep)
        hunter = ScriptedHunter(iw, env.oracle, env.core, rng, validate=(ep % 2 == 1),
                                short_scatter=a.save_states and not a.wide_scatter)
        state_saved = False
        validated_ep = False
        drops = []
        seen_kill = False
        saved = {"control": 0, "death_anim": 0, "key_onscreen": 0, "key_offscreen": 0}
        caps = {"control": 30, "death_anim": 15, "key_onscreen": 80, "key_offscreen": 25}
        every = {"control": 10, "death_anim": 3, "key_onscreen": 2, "key_offscreen": 4}
        step = 0
        done = False
        while not done:
            if a.driver == "scripted":
                act = np.array(hunter.act(drops), dtype=np.int64)
            else:
                act, _ = model.predict(obs, deterministic=det)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            step += 1
            lx, ly = env.oracle.read_pos(env.core)
            alive = env.oracle.prev_health is not None and env.oracle.prev_health > 0
            died_now = any(ch == "death" for _t, ch, _d, _v in info["events"])

            for _t, ch, _d, val in info["events"]:
                if ch == "enemy_dmg" and val == 0:
                    ex, ey = u16(iw, ENEMY_X), u16(iw, ENEMY_Y)
                    # teardown guard: a real kill needs Link alive, no death this
                    # step, and a real enemy position
                    if alive and not died_now and (ex, ey) != (0, 0):
                        drops.append([ex, ey, step])
                        totals["kills"] += 1
                        seen_kill = True
                        hunter.on_kill((ex, ey))
                elif ch == "key":
                    totals["keys"] += 1
                    if hunter.on_pickup():
                        totals["pickup_validated"] += 1
                        validated_ep = True
                    if drops:
                        drops.sort(key=lambda d: (d[0] - lx) ** 2 + (d[1] - ly) ** 2)
                        drops.pop(0)
                elif ch == "death":
                    totals["deaths"] += 1
                    drops.clear()   # death resets the room; ground drops despawn

            if not alive:
                continue            # dead/continue screen: never classify or save
            if not drops:
                phase = "control" if not seen_kill else None
            else:
                nearest = min(drops, key=lambda d: (d[0] - lx) ** 2 + (d[1] - ly) ** 2)
                age = step - nearest[2]
                dx, dy = nearest[0] - lx, nearest[1] - ly
                if age < VIS_DELAY:
                    phase = "death_anim"
                elif abs(dx) <= MARGIN_X and abs(dy) <= MARGIN_Y:
                    phase = "key_onscreen"
                else:
                    phase = "key_offscreen"

            # keydrop state bank: snapshot once per episode, as soon as the drop is
            # past the death poof, Link has scattered a little (pre-pickup), AND the
            # drop is safely onscreen (see BANK_X/BANK_Y above) -- re-checked every
            # step so a too-far scatter can still bank when the drop re-enters view
            if (a.save_states and not state_saved and drops
                    and step - drops[0][2] >= VIS_DELAY + 4
                    and abs(drops[0][0] - lx) <= BANK_X
                    and abs(drops[0][1] - ly) <= BANK_Y):
                sdir = os.path.join(a.out, "states")
                os.makedirs(sdir, exist_ok=True)
                with open(os.path.join(sdir, "keydrop-%02d.state" % ep), "wb") as f:
                    f.write(bytes(env.core.save_raw_state()))
                with open(os.path.join(sdir, "keydrop-%02d.json" % ep), "w") as f:
                    json.dump({"ep": ep, "step": step, "link": [lx, ly],
                               "drop": drops[0][:2]}, f)
                state_saved = True

            # bank harvest: the episode's job is done once banked (validate eps also
            # need their walk-back pickup receipt) -- don't burn the rest of the horizon
            if a.save_states and state_saved and (not hunter.validate or validated_ep):
                break

            if a.no_frames or phase is None or saved[phase] >= caps[phase] or step % every[phase]:
                continue

            name = "ep%02d_step%05d.png" % (ep, step)
            save_png(env._frames[-1], os.path.join(a.out, "frames", name))
            saved[phase] += 1
            rec = {"file": name, "ep": ep, "step": step,
                   "mode": a.driver if a.driver == "scripted" else ("det" if det else "samp"),
                   "phase": phase, "link": [lx, ly], "drops": [d[:2] for d in drops]}
            if drops:
                dist = math.hypot(dx, dy)
                rec.update(dx=dx, dy=dy, dist=round(dist, 1),
                           direction="on-link" if dist <= ONLINK_DIST else direction_of(dx, dy))
            labels_f.write(json.dumps(rec) + "\n")

        for k, v in saved.items():
            totals["frames"][k] = totals["frames"].get(k, 0) + v
        print("ep%d: steps=%d kills=%d keys=%d deaths=%d validated=%d saved=%s"
              % (ep, step, totals["kills"], totals["keys"], totals["deaths"],
                 totals["pickup_validated"], saved), flush=True)

    labels_f.close()
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(totals, f, indent=2)
    print("TOTALS:", json.dumps(totals))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
