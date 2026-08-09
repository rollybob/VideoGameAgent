"""Goal-conditioned WALKER env: "walk to target X" as a learned, room-general skill.

WHY (2026-08-09, the 3-tier post-mortem): the agent has intent but no LEGS --
nothing can execute "go to position X". The VLM says "down" and the drive loop
holds DOWN into a wall; S1 is a combat reflex; collect in rung 3 failed to
generalize because reactive S1 memorizes room-specific motor paths. This env
trains the missing primitive. At inference the target comes from the live
detector (walk to the detected key = the rung-3 collect fix) or from System-2
intent (walk to a door edge); at training time targets are sampled from
gen_walk_bank.py banks (positions a random walk actually stood on, so they are
reachable by construction) and the RAM oracle supplies reward. Pixels-only at
inference holds: the policy sees frames + a target BLOB, never coordinates.

Observation = AlttpPpoEnv's 12ch frame stack + 1 GOAL channel: a bright disc at
the target's SCREEN position (screen = world - camera(0x02B82/86), the detector
label pipeline's own convention). When the target is off-screen the disc is
clamped to the screen border -- "it's that way" -- which is exactly the cue a
door-target needs on scrolling rooms. The policy's whole job: walk to the blob.

Action = Discrete(9) directions (none + 4 cardinal + 4 diagonal). No buttons:
walking needs none, and a smaller space learns faster. Combat stays S1's job.

Reward (replaces the oracle reward stack wholesale -- keys/rupees would pollute
a pure navigation skill):
  +PROG * (dist_prev - dist_cur)   potential-based progress (farm-proof: walking
                                   away and back nets zero minus step costs)
  +REACH_BONUS on arrival          (dist <= REACH_R), then RESAMPLE a new target
                                   in-episode (resets are expensive; one episode
                                   practices several targets)
  +STEP_COST per decision          mild pressure against dawdling
  +DMG_W * damage/death            navigation should route around hazards, but
                                   the walk signal stays dominant
"""
import os
import numpy as np
from gymnasium import spaces

from alttp_ppo_env import AlttpPpoEnv, STACK_K, W, H, channel_reward, FRAME_SKIP
from harvest_key_frames import u16
from mgba._pylib import ffi

CAM_X, CAM_Y = 0x02B82, 0x02B86
LINK_X, LINK_Y = 0x038F4, 0x038F0

PROG = 0.1          # per world-unit of approach; a 100-unit trek pays +10
REACH_R = 10.0      # world-units counting as "arrived" (Link is 16 wide)
REACH_BONUS = 10.0
STEP_COST = -0.02   # per decision; 600-decision episode floor is -12
DMG_W = 0.5         # damage/death at half weight -- avoid hazards, don't obsess
MIN_DIST = 40.0     # sampled targets start at least this far away ("far" per
                    # the rung-2 regime finding: near-target wandering suffices)
BLOB_R = 5          # goal-disc radius in screen pixels

_BLOB_OFF = [(dx, dy) for dy in range(-BLOB_R, BLOB_R + 1)
             for dx in range(-BLOB_R, BLOB_R + 1) if dx * dx + dy * dy <= BLOB_R * BLOB_R]


def goal_channel(sx, sy):
    """(H,W,1) uint8 goal channel with the disc at SCREEN pos (sx,sy), border-clamped
    when off-screen so direction survives. Shared by the env (training) and the drive
    agent (inference) -- one renderer, zero train/serve skew."""
    sx = min(max(int(round(sx)), BLOB_R + 1), W - BLOB_R - 2)
    sy = min(max(int(round(sy)), BLOB_R + 1), H - BLOB_R - 2)
    ch = np.zeros((H, W, 1), np.uint8)
    for dx, dy in _BLOB_OFF:
        ch[sy + dy, sx + dx, 0] = 255
    return ch


class WalkerEnv(AlttpPpoEnv):
    def __init__(self, state_path=None, horizon=2400, frame_skip=FRAME_SKIP,
                 bank_dir=None):
        super().__init__(state_path=state_path, horizon=horizon,
                         death_terminates=True, frame_skip=frame_skip)
        self.observation_space = spaces.Box(0, 255, (H, W, 3 * STACK_K + 1), np.uint8)
        self.action_space = spaces.Discrete(9)   # DIR_MASKS index: none/U/D/L/R/UR/UL/DR/DL
        self.bank_dir = bank_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "walker_banks")
        self._banks = {}
        self._iw = None
        self.goal = None
        self._d_prev = 0.0
        self._reaches = 0

    # -- helpers ------------------------------------------------------------
    def _bank(self):
        name = os.path.basename(self.state_path).replace(".state", "")
        if name not in self._banks:
            z = np.load(os.path.join(self.bank_dir, name + ".npz"))
            self._banks[name] = z["pos"].astype(np.float64)
        return self._banks[name]

    def _link(self):
        return float(u16(self._iw, LINK_X)), float(u16(self._iw, LINK_Y))

    def _dist(self):
        lx, ly = self._link()
        return float(np.hypot(self.goal[0] - lx, self.goal[1] - ly))

    def _sample_goal(self):
        bank = self._bank()
        lx, ly = self._link()
        d = np.hypot(bank[:, 0] - lx, bank[:, 1] - ly)
        far = np.flatnonzero(d >= MIN_DIST)
        cand = far if len(far) else np.flatnonzero(d >= REACH_R)
        idx = int(self.np_random.integers(len(cand)))
        self.goal = tuple(bank[cand[idx]])
        self._d_prev = self._dist()

    def _goal_channel(self):
        cx, cy = float(u16(self._iw, CAM_X)), float(u16(self._iw, CAM_Y))
        return goal_channel(self.goal[0] - cx, self.goal[1] - cy)

    def _obs13(self):
        return np.concatenate([self._stack(), self._goal_channel()], axis=-1)

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        self._iw = ffi.cast("uint8_t *", self.core._native.memory.iwram)
        self._reaches = 0
        self._sample_goal()
        info["goal"] = self.goal
        return self._obs13(), info

    def step(self, action):
        # Discrete dir index -> the MultiDiscrete [dir,A,B,L,R] path (an int would
        # hit the legacy single-button ACTIONS table, which orders keys differently)
        md = np.array([int(action), 0, 0, 0, 0], dtype=np.int64)
        _, _, terminated, truncated, info = super().step(md)

        reward = STEP_COST
        for _t, channel, delta, _v in info["events"]:
            if channel in ("damage", "death"):
                reward += DMG_W * channel_reward(channel, delta)

        d = self._dist()
        reward += PROG * (self._d_prev - d)
        self._d_prev = d
        if d <= REACH_R:
            reward += REACH_BONUS
            self._reaches += 1
            self._sample_goal()   # next target, same episode

        info["dist"] = d
        info["reaches"] = self._reaches
        info["goal"] = self.goal
        return self._obs13(), reward, terminated, truncated, info


def _smoke():
    import time
    states = os.path.join(os.path.dirname(os.path.abspath(__file__)), "states")
    env = WalkerEnv(state_path=os.path.join(states, "alttp_human-04.state"), horizon=1200)
    obs, info = env.reset(seed=3)
    assert obs.shape == (H, W, 3 * STACK_K + 1) and obs.dtype == np.uint8
    assert obs[..., -1].max() == 255, "goal blob missing"
    print("reset ok, obs %s, goal %s, d0 %.1f" % (obs.shape, info["goal"], env._d_prev))
    total = 0.0
    t0 = time.time()
    greedy_hits = 0
    for i in range(300):
        # greedy-toward-goal action mix: sanity that approach pays positive reward
        lx, ly = env._link()
        gx, gy = env.goal
        dx, dy = gx - lx, gy - ly
        if env.np_random.random() < 0.3:
            a = int(env.np_random.integers(9))
        else:
            a = {(0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4, (1, -1): 5, (-1, -1): 6,
                 (1, 1): 7, (-1, 1): 8}[(int(np.sign(dx)) if abs(dx) > 4 else 0,
                                          int(np.sign(dy)) if abs(dy) > 4 else 0)] \
                if (abs(dx) > 4 or abs(dy) > 4) else 0
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    dt = time.time() - t0
    print("300 steps in %.1fs (%.0f steps/s), total reward %.2f, reaches %d, last dist %.1f"
          % (dt, (i + 1) / dt, total, info["reaches"], info["dist"]))
    env.close()


if __name__ == "__main__":
    _smoke()
