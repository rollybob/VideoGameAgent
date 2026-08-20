"""Composer v0 -- the composition kill-shot (2026-08-20 recovery meeting).

HYPOTHESIS (pre-registered): FROZEN specialists + a ~40-line scripted router
clear >=70% collect|kill on BOTH room4 AND room2 in the SAME agent -- the axis
every co-trained monolith traded on (r2c: 75%/20%; r3: 33%/85%). Nothing is
trained here, so a see-saw is structurally impossible; the only question is
whether composition itself works.

Parts (ALL FROZEN):
  COMBAT  = s1_joint_r2c (chain-distilled PPO; kills generalize 15/16 held-out)
  COLLECT = scripted greedy walk toward the detector's item peak
  ROUTER  = confirmed item -> COLLECT, otherwise -> S1. TWO modes, not three:
            the first smoke (0 kills in 2 eps) proved that gating S1 on
            enemy-on-screen starves it of its HUNT phase -- approach is part
            of the combat skill, so S1 owns the room until a key drops. The
            enemy-gated 3-mode router (with free explore) is the OVERWORLD
            composer, a separate experiment.

RAM is EVAL-ONLY (kill/key scoring, identical to transfer_chain_eval.py).
Runtime perception is the detector alone -- north-star clean.

Protocol mirrors transfer_chain_eval.py EXACTLY: same states, seeds 7000+ep,
N_EPS=16, horizon=1500, WINDOW=250. An "r2c" control arm re-runs the plain
monolith through THIS script so parity with the historical numbers is checked
in the same process (guards against silent harness drift).

Run (mirrors the transfer eval; needs GPU for detector + policy):
  docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/vga \\
      -w /vga/train/rl thor-rl:cu130 python3 -u composer_v0.py
Env knobs: ARMS=composed,r2c  ROOMS=4,2,0  N_EPS=16  SMOKE=1 (2 eps, room2,
verbose mode trace).
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VGA = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(VGA, "train", "perception", "detector"))

from alttp_ppo_env import AlttpPpoEnv                   # noqa: E402
from harvest_key_frames import u16, ENEMY_X, ENEMY_Y   # noqa: E402

import torch                                            # noqa: E402
from train_detector import Net, to_in                   # noqa: E402
from eval_detector import peak_single, peaks_multi     # noqa: E402

DEV = "cuda"
S1_CKPT = os.path.join(HERE, "runs", "distill_r2", "s1_joint_r2c.zip")
DET_PT = os.path.join(VGA, "train", "perception", "detector", "detector.pt")

WINDOW = 250
N_EPS = int(os.environ.get("N_EPS", "16"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
ARMS = os.environ.get("ARMS", "composed,r2c").split(",")
ROOM_IDS = [int(r) for r in os.environ.get("ROOMS", "4,2,0").split(",")]

# MODE=bank isolates the NEW machinery (router + greedy collect) on the arbiter's
# keydrop bank: the key is already RESTING on the floor, so success needs no kill
# and no drop luck -- it measures pure see-key -> walk-to-key -> collect. External
# reference on the same fair-8 states: arbiter 79% / plain S1 58% / random 42%.
# The chain smoke forced this split: room2 kills often hit a NON-dropping enemy
# (slot3 alive or empty post-kill), so chain numbers mix target-selection with
# collection; the bank removes the confound.
MODE = os.environ.get("MODE", "chain")   # chain | bank
TRACE0 = os.environ.get("TRACE0", "0") == "1"   # chain-mode debug: track items from
# step 0 (not just post-kill) and dump a marked PNG at the FIRST COLLECT lock --
# added to get eyes on WHAT the item channel fires at in room4 (router theft).
BANK_DIR = os.path.join(VGA, "train", "perception", "keyprobe_v2", "states")
# fair-8 = the 12-state bank minus the 4 perception-dead states (00/07/08/09:
# key at/past the vertical camera edge -- a null there is CORRECT, arbiter v1).
FAIR8 = ["01", "02", "03", "04", "05", "06", "10", "11"]
BANK_SEEDS = [7000, 7100, 7200]

# Router constants. ITEM_GATE=3 consecutive detections (=12 emu frames) rides
# out the ~12-frame death-poof window where the key sprite has not rendered yet
# (kill_drop_forensic 2026-08-08) -- the detector correctly rejects poof frames,
# so gating on persistence costs nothing and kills flicker false-positives.
# ITEM_HOLD keeps COLLECT alive through brief detection dropouts (and releases
# ~4 decisions after pickup makes the item vanish for real).
ITEM_GATE = 3
ITEM_HOLD = 4
# Give-up/suppress (the ONE mechanism added by the room4 debug, 2026-08-20): the
# detector's item head fires on room4's STATUES at conf up to 1.05 (= real-key
# level, so thresholds cannot discriminate). Pickup SEMANTICS can: a real key
# VANISHES when Link steps on it; a statue is solid and just gets ground against.
# If Link has been adjacent to a still-detected target for NEAR_M decisions, or
# COLLECT has chased one target for GIVEUP decisions, it is not a collectible ->
# suppress that screen position for the episode and hand control back to S1.
NEAR_PX = 12
NEAR_M = 4
GIVEUP = 40
SUPPRESS_R = 14
LINK_THR = 0.5     # drive_agent's gate: below it the argmax "peak" is noise
STUCK_N = 6        # link-position window (decisions) for the collect unstick
STUCK_PX = 3       # span below this over STUCK_N sightings = stuck
JITTER_N = 3       # decisions of full-random burst to unstick
DIAG_MIN = 6       # both |dx|,|dy| beyond this -> walk the diagonal

# MultiDiscrete [dir, A, B, L, R]; dir: 0=none 1=U 2=D 3=L 4=R 5=UR 6=UL 7=DR 8=DL
def _greedy_dir(dx, dy):
    if abs(dx) > DIAG_MIN and abs(dy) > DIAG_MIN:
        return {(1, 1): 7, (1, -1): 5, (-1, 1): 8, (-1, -1): 6}[(int(np.sign(dx)), int(np.sign(dy)))]
    if abs(dx) >= abs(dy):
        return 4 if dx > 0 else 3
    return 2 if dy > 0 else 1


class Composer:
    """Scripted router over frozen specialists. Holds NO learned state; per-
    episode counters only. Pixels-only: sees the detector, never RAM."""

    def __init__(self, s1, net, rng):
        self.s1 = s1
        self.net = net
        self.rng = rng
        self.item_run = 0        # consecutive decisions with an item peak
        self.item_absent = 0     # decisions since last item peak (while engaged)
        self.item_lock = False   # COLLECT engaged (survives ITEM_HOLD dropouts)
        self.last_item = None
        self.last_link = (120.0, 80.0)   # frame center fallback; camera tracks Link
        self.link_hist = []
        self.jitter = 0
        self.mode = "EXPLORE"
        self.suppressed = []     # screen positions proven non-collectible this episode
        self.near_run = 0        # consecutive decisions adjacent to a live target
        self.chase = 0           # decisions spent on the current engagement

    def detect(self, frame):
        with torch.no_grad():
            hm = self.net(to_in(frame[None]).to(DEV)).cpu().numpy()[0]
        link = peak_single(hm[0]) if float(hm[0].max()) >= LINK_THR else None
        items = peaks_multi(hm[2], thr=0.6)
        self.dbg_items_n = len(items)              # smoke diagnostics only
        self.dbg_item_conf = float(hm[2].max())
        return link, items

    def act(self, obs, frame):
        link, items = self.detect(frame)
        if link is not None:
            self.last_link = (float(link[0]), float(link[1]))
            self.link_hist.append(self.last_link)
            self.link_hist = self.link_hist[-STUCK_N:]
        items = [p for p in items if all((p[0] - sx) ** 2 + (p[1] - sy) ** 2 > SUPPRESS_R ** 2
                                         for sx, sy in self.suppressed)]

        if items:
            self.item_run += 1
            self.item_absent = 0
            lx, ly = self.last_link
            self.last_item = min(items, key=lambda p: (p[0] - lx) ** 2 + (p[1] - ly) ** 2)
            if self.item_run >= ITEM_GATE:
                self.item_lock = True
        else:
            self.item_run = 0
            self.item_absent += 1
            if self.item_absent > ITEM_HOLD:
                self.item_lock = False
                self.link_hist = []   # stale stuck-window must not fire next engage

        if self.item_lock and self.last_item is not None:
            self.mode = "COLLECT"
            self.chase += 1
            dx = self.last_item[0] - self.last_link[0]
            dy = self.last_item[1] - self.last_link[1]
            self.near_run = self.near_run + 1 if (dx * dx + dy * dy) <= NEAR_PX ** 2 else 0
            if (self.near_run >= NEAR_M and self.item_run > 0) or self.chase >= GIVEUP:
                # Adjacent-and-still-there, or chased too long: not a collectible.
                self.suppressed.append((self.last_item[0], self.last_item[1]))
                self.item_lock = False
                self.item_run = 0
                self.chase = 0
                self.near_run = 0
                self.link_hist = []
                self.mode = "COMBAT"
                act, _ = self.s1.predict(obs, deterministic=False)
                return act, self.mode
            if self.jitter > 0:
                self.jitter -= 1
                return self._random_action(), self.mode
            if len(self.link_hist) >= STUCK_N:
                xs = [p[0] for p in self.link_hist]
                ys = [p[1] for p in self.link_hist]
                if max(xs) - min(xs) < STUCK_PX and max(ys) - min(ys) < STUCK_PX:
                    # Greedy is grinding against something (pot/wall/geometry).
                    # Full-random burst, per the explorer lesson: the broad
                    # action set (incl. lift/slash) is what gets un-stuck.
                    self.jitter = JITTER_N
                    self.link_hist = []
                    return self._random_action(), self.mode
            return np.array([_greedy_dir(dx, dy), 0, 0, 0, 0]), self.mode

        self.chase = 0
        self.near_run = 0
        self.mode = "COMBAT"
        act, _ = self.s1.predict(obs, deterministic=False)
        return act, self.mode

    def _random_action(self):
        return np.array([self.rng.randint(9), self.rng.randint(2),
                         self.rng.randint(2), self.rng.randint(2), self.rng.randint(2)])


def eval_room(arm, model, net, state, ffi, seeds, horizon=1500, chain=True):
    """Same shape as transfer_chain_eval.eval_policy_room; adds mode receipts.
    chain=True: complete = key within WINDOW of first kill (historical metric).
    chain=False (bank): complete = any key event (key already on the floor)."""
    complete = killed = by_collect = 0
    receipts = []
    for ep, sd in enumerate(seeds):
        env = AlttpPpoEnv(state_path=state, horizon=horizon)
        obs, _ = env.reset(seed=sd)
        iw = ffi.cast("uint8_t *", env.core._native.memory.iwram)
        model.set_random_seed(sd)
        comp = Composer(model, net, np.random.RandomState(sd)) if arm == "composed" else None
        first_kill = None
        got_key_after = False
        got_kill = False
        key_mode = None
        modes = {"COMBAT": 0, "COLLECT": 0, "EXPLORE": 0}
        pk_dec = pk_item_dec = 0          # post-kill decisions / with item seen
        pk_conf = 0.0                     # max item-channel confidence post-kill
        dumped = False
        step, done = 0, False
        while not done:
            if comp is not None:
                act, mode = comp.act(obs, env._frames[-1])
                modes[mode] += 1
            else:
                act, _ = model.predict(obs, deterministic=False)
                mode = "S1"
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
                    if not chain or (first_kill is not None and step - first_kill <= WINDOW):
                        got_key_after = True
                        key_mode = mode
            if comp is not None and TRACE0 and comp.item_lock and not dumped:
                from PIL import Image, ImageDraw
                img = Image.fromarray(env._frames[-1])
                dr = ImageDraw.Draw(img)
                ix, iy = comp.last_item[0], comp.last_item[1]
                lx, ly = comp.last_link
                dr.ellipse([ix - 5, iy - 5, ix + 5, iy + 5], outline=(0, 128, 255), width=2)
                dr.ellipse([lx - 5, ly - 5, lx + 5, ly + 5], outline=(0, 255, 0), width=2)
                os.makedirs(os.path.join(HERE, "_composer_dbg"), exist_ok=True)
                p = os.path.join(HERE, "_composer_dbg", f"lock_ep{ep}_step{step}.png")
                img.save(p)
                print(f"    [lock] ep{ep} step{step} item=({ix:.0f},{iy:.0f}) "
                      f"conf={comp.dbg_item_conf:.2f} link=({lx:.0f},{ly:.0f}) -> {p}", flush=True)
                dumped = True
            if comp is not None and (not chain or first_kill is not None or TRACE0):
                pk_dec += 1
                if comp.dbg_items_n > 0:
                    pk_item_dec += 1
                pk_conf = max(pk_conf, comp.dbg_item_conf)
                if SMOKE and step % 25 == 0:
                    # RAM truth for the DIAGNOSIS only (never feeds control):
                    # a dropped key reuses the dead enemy's slot with HP==0
                    # (kill_drop_forensic 2026-08-08), screen = world - camera.
                    kx, ky = u16(iw, ENEMY_X), u16(iw, ENEMY_Y)
                    hp = int(iw[0x03253])
                    cx, cy = u16(iw, 0x02B82), u16(iw, 0x02B86)
                    print(f"    [dbg] step{step} slot3 world({kx},{ky}) hp={hp} "
                          f"screen({kx - cx},{ky - cy}) det_items={comp.dbg_items_n} "
                          f"conf={comp.dbg_item_conf:.2f} link={comp.last_link}", flush=True)
            if got_key_after:
                break
        killed += int(got_kill)
        complete += int(got_key_after)
        by_collect += int(key_mode == "COLLECT")
        receipts.append({"ep": ep, "seed": sd, "kill_step": first_kill,
                         "complete": got_key_after, "key_mode": key_mode,
                         "modes": modes, "steps": step, "pk_dec": pk_dec,
                         "pk_item_dec": pk_item_dec, "pk_conf": round(pk_conf, 3)})
        if SMOKE:
            print(f"  ep{ep}: kill@{first_kill} complete={got_key_after} "
                  f"key_mode={key_mode} modes={modes} tracked={pk_dec}dec "
                  f"item_seen={pk_item_dec} maxconf={pk_conf:.2f}", flush=True)
        env.close()
    return {"killed": killed, "complete": complete, "by_collect": by_collect,
            "n": len(seeds), "receipts": receipts}


def main():
    global N_EPS, ARMS, ROOM_IDS
    if SMOKE:
        ARMS = ["composed"]
    from mgba._pylib import ffi
    from stable_baselines3 import PPO
    model = PPO.load(S1_CKPT, device=DEV)
    net = Net().to(DEV)
    net.load_state_dict(torch.load(DET_PT, map_location=DEV))
    net.eval()
    print(f"composer_v0 MODE={MODE}: arms={ARMS} "
          f"s1={os.path.basename(S1_CKPT)} det={os.path.basename(DET_PT)}", flush=True)

    out = {}
    if MODE == "bank":
        states = FAIR8[:2] if SMOKE else FAIR8
        seeds = BANK_SEEDS[:1] if SMOKE else BANK_SEEDS
        for arm in ARMS:
            out[arm] = {}
            coll = tot = 0
            for sx in states:
                sp = os.path.join(BANK_DIR, f"keydrop-{sx}.state")
                r = eval_room(arm, model, net, sp, ffi, seeds, horizon=1200, chain=False)
                out[arm][f"keydrop-{sx}"] = r
                coll += r["complete"]
                tot += r["n"]
                print(f"{arm:9} keydrop-{sx}: collected {r['complete']}/{r['n']} "
                      f"byCollect {r['by_collect']}", flush=True)
            print(f"{arm:9} BANK TOTAL: {coll}/{tot} ({100.0 * coll / max(1, tot):.0f}%)  "
                  f"[refs: arbiter 79 / s1 58 / random 42]", flush=True)
    else:
        rooms = ROOM_IDS if "ROOMS" in os.environ else ([2] if SMOKE else ROOM_IDS)
        n = 2 if SMOKE else N_EPS
        for arm in ARMS:
            out[arm] = {}
            for ridx in rooms:
                state = os.path.join(HERE, "states", f"alttp_human-0{ridx}.state")
                r = eval_room(arm, model, net, state, ffi, [7000 + i for i in range(n)])
                out[arm][f"room{ridx}"] = r
                ck = f"{100.0 * r['complete'] / max(1, r['killed']):.0f}%"
                print(f"{arm:9} room{ridx}: complete {r['complete']:2d}/{r['n']} "
                      f"(kills {r['killed']:2d}, collect|kill {ck}, byCollect {r['by_collect']})",
                      flush=True)
    path = os.path.join(HERE, f"_composer_v0_{MODE}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"SAVED {path}", flush=True)


if __name__ == "__main__":
    main()
