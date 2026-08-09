"""Task 09 Phase A1: gymnasium Env wrapping solo A Link to the Past (GBA).

Single mgba core (no LinkSession -- solo mode has no SIO/link complexity at
all). Reward = damage (-) / death (- lump) / rupees (+) from AlttpOracle.

2026-07-31: the solo rupee address WAS found and validated (EWRAM 0x02340,
u16 LE -- write-tested against the on-screen HUD, see find_rupee_addr.py), so
this is no longer a pure punishment signal. That matters: the A1 baseline went
FLAT, and the leading suspect was that with only damage/death the optimal
policy is to stand still and do nothing. Rupees give it something to gain.

The default start state is still alttp_ingame.state, but that checkpoint sits
in a zone with no enemies -- pass --state / state_path one of the human-driven
train/rl/states/alttp_human-*.state captures for a start-point that actually
has something to interact with.

Run directly for a random-action smoke test:
    docker run --rm --runtime nvidia -v ~/projects/VGA:/work -w /work \\
        thor-rl:cu130 python3 train/rl/alttp_ppo_env.py
"""
import os
import sys
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
VGA = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import mgba.core
import mgba.gba
import mgba.image
import mgba.log
from mgba._pylib import ffi

from oracle import AlttpOracle

mgba.log.silence()

ROM = os.path.join(VGA, "Emulator", "mGBA", "roms",
                    "Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
STATE = os.path.join(HERE, "states", "alttp_ingame.state")
W, H = 240, 160
GBA = mgba.gba.GBA

# Discrete action table. Index 0 = no input. set_keys(raw=...) wants a
# BITMASK (1<<index), not the raw key index -- see alttp_probe.py's note.
# NO START. Tim identified 2026-07-31 by watching a replay on screen: the agent
# was not dying to enemies at all -- it was opening the menu with START and
# scrolling onto "Quit", ending the session. That is why every state died at the
# identical tick regardless of which room was loaded (the outcome depended only
# on the action stream, never on the game world).
#
# This is a self-termination exploit, not a cosmetic bug. Under A1's
# damage/death-only reward, quitting to a menu where nothing can hurt you scores
# ~0, which is the BEST achievable return -- so "press START and quit" was very
# plausibly the optimum the flat A1 run was converging to. The oracle's "damage"
# events during a quit are the health byte being read while the game tears down
# normal play; they were never real hits.
#
# Removing the button is the robust fix: an agent cannot exploit an input it
# does not have. Nothing in this task needs the menu.
# A = lift/throw/run/talk, B = sword. L and R ADDED 2026-08-03 (Tim): in the GBA
# ALttP port L/R block with the shield and pick up pots/bushes -- necessary
# mechanics, not optional. Pots/bushes hide rupees/hearts/keys, so L/R feed the
# existing rupee/heal/key channels; and blocking is a way to AVOID damage without
# fleeing, which matters for a policy prone to the avoidance attractor (it can
# defend in place instead of running). Single-button granularity (no combos) is
# kept consistent with the rest of the table.
# NB this takes N_ACTIONS 7 -> 9: policies trained before today (the -04 lr* runs)
# have a 7-wide action head, so they are NOT the same action space -- this env is
# for fresh DR training only. Old-checkpoint evals (eval_finals.py) also see a
# 9-action random baseline now, so pre-08-03 eval numbers won't reproduce here.
ACTIONS = [0, 1 << GBA.KEY_UP, 1 << GBA.KEY_DOWN, 1 << GBA.KEY_LEFT, 1 << GBA.KEY_RIGHT,
           1 << GBA.KEY_A, 1 << GBA.KEY_B, 1 << GBA.KEY_L, 1 << GBA.KEY_R]
N_ACTIONS = len(ACTIONS)

# MULTI-INPUT action space (2026-08-04, Tim). The single-button ACTIONS table above
# cannot express two things the game needs and the human tapes use constantly:
# (1) DIAGONAL movement, and (2) holding an action button WHILE moving -- e.g.
# charging a spin attack (hold B) or lifting/blocking on the move. That projection
# loss was a prime suspect for BC failing to reproduce human play (attack+move
# collapse onto one button, so the 'B' class was ~unlearnable). New action space =
# MultiDiscrete([9, 2, 2, 2, 2]) = [direction, A, B, L, R]:
#   dir: 0=none 1=U 2=D 3=L 4=R 5=U+R 6=U+L 7=D+R 8=D+L   (4 cardinal + 4 diagonal)
#   A,B,L,R: independent 0/1, OR-ed onto the direction mask.
# START/SELECT stay absent (the self-termination-exploit fix stands). A held charge =
# the policy emitting B=1 on consecutive frame_skip windows. The legacy single-button
# ACTIONS table is KEPT so the ~13 measurement/eval scripts that pass an int action
# (rng.randrange(N_ACTIONS)) keep running -- step() dispatches on the action type.
_U, _D, _LF, _RT = 1 << GBA.KEY_UP, 1 << GBA.KEY_DOWN, 1 << GBA.KEY_LEFT, 1 << GBA.KEY_RIGHT
DIR_MASKS = [0, _U, _D, _LF, _RT, _U | _RT, _U | _LF, _D | _RT, _D | _LF]
BUTTON_MASKS = [1 << GBA.KEY_A, 1 << GBA.KEY_B, 1 << GBA.KEY_L, 1 << GBA.KEY_R]  # A, B, L, R
ACTION_NVEC = [len(DIR_MASKS), 2, 2, 2, 2]


def action_to_mask(action):
    """MultiDiscrete [dir, A, B, L, R] -> GBA key bitmask (never sets START/SELECT)."""
    mask = DIR_MASKS[int(action[0])]
    for i, bm in enumerate(BUTTON_MASKS):
        if int(action[i + 1]):
            mask |= bm
    return mask

DEATH_PENALTY = -20.0  # lump penalty on top of the eighths-of-damage reward that led to 0
# Rupees: 2 points per rupee (Tim 2026-08-02), COLLECTION-only (the oracle emits
# only on a rise). So green(1)=+2, blue(5)=+10, red(20)=+40; the rare 100/300
# rupees stay linearly large (+200/+600) -- kept linear on purpose, not capped.
RUPEE_SCALE = 2.0
# Heart pickups. Deliberately BELOW 1.0 per eighth: damage costs -1 per eighth,
# so a heal worth the same would make "take a hit, grab the heart it dropped" a
# break-even loop, and anything above would make getting hurt profitable.
HEAL_SCALE = 0.5
# Keys (EWRAM 0x0234F, confirmed 2026-08-01 from a human pickup/spend round
# trip). This is the dense POSITIVE channel the reward has been missing: the
# measured heal rate is ~0.08 events/episode, which is why a3 could score a
# perfect 0.0 by walking away from everything.
# KEY_SCALE is large because a key is not just a collectible -- ALttP exposes no
# kill counter, so the key a defeated enemy drops is the only paid proxy for
# KILLING one.
#
# 120 is DERIVED, not chosen (key_breakeven.py, 24 random episodes on -04):
# random play eats 624 reward-units of damage+death to collect 11 keys, i.e.
# 57 units per key. At the original KEY_SCALE=30 -- BELOW that break-even --
# every step toward a key was punished more than the key paid, so "touch
# nothing" was genuinely optimal and the agent was right to hide. That is
# exactly what a4 did: it took one key at 20k steps while still near-random and
# never another. 120 is ~2.1x break-even, which makes random play's return
# clearly positive (+29/episode) so the gradient points at engagement from the
# very start instead of merely being non-negative.
KEY_SCALE = float(os.environ.get("VGA_KEY_SCALE") or "150.0")  # env-overridable (VGA_KEY_SCALE=250 for the anti-farm retrain 2026-08-06); 120 -> 150 (Tim 2026-08-02): keeps the key/kill above the
# common rupees (green/blue/red = 2/10/40) as the dominant progress reward.
# Spending a key on a locked door is the only observable PROGRESS signal in the
# oracle, so it pays MORE than picking one up, and is paid on a NEGATIVE delta.
KEY_USED_SCALE = -200.0
# Heart containers are rare and permanent. Paid once, generously.
HEART_CONTAINER_SCALE = 20.0
# Exploration: first visit to each room per episode, room = (X>>9, Y>>9) from
# IWRAM position. Derived the same way KEY_SCALE was, from measure_density on
# alttp_human-03 (the only state where random play changes rooms at all): random
# eats 38.0 reward-units of damage+death per episode and reaches 0.83 new rooms,
# i.e. ~46 units per room, so anything below that leaves "stay put" optimal.
#
# 60 rather than the ~100 that a 2x margin would suggest, because in -03 a room
# change IS a fall off a ledge, so new_room and damage are COUPLED there in a way
# they are not elsewhere. Over-paying exploration in that state would reward
# throwing Link off ledges. First-visit-only already caps the per-episode total,
# so a modest value is the safer side of an uncertain trade.
#
# REBALANCED 2026-08-03, 60 -> 30, against the human-tape CEILING: an 18-min
# castle run paid new_room +1680 (28 transitions), part of exploration's 67%
# over-weight vs keys+doors' 26%. The room metric is also NOISY (warps and
# position-flicker inflate the transition count), so a new room is now worth ~a
# fifth of a key rather than 40%. env-overridable (VGA_NEW_ROOM_SCALE).
NEW_ROOM_SCALE = float(os.environ.get("VGA_NEW_ROOM_SCALE") or "30.0")
# Dense exploration breadcrumb (2026-08-03). The oracle emits 'explore' on the
# first visit to each fine cell per episode (AlttpOracle.EXPLORE_SHIFT). This is
# the continuous outward gradient that makes exploration START at all -- without
# it, new_room's per-room payoff sits ~85 undirected steps away and a cold policy
# never discovers it (new_room only ever fired in -03). Kept SMALL and below the
# -8 per-heart damage cost so it nudges movement without rewarding walking into
# damage: at ~64 cells/room a traversal earns ~8-16 cells.
#
# REBALANCED 2026-08-03, 2.0 -> 0.5, against the human-tape CEILING: the fine
# breadcrumb was the single largest channel on an 18-min castle run (explore
# +1952 of a +5424 total) -- it rewards walking around, the EASIEST thing, more
# than opening doors. Cut 4x so the same run's explore drops 1952 -> 488 and
# PROGRESS (keys/doors/kills) dominates. It is still a real dense outward
# gradient (direction matters more than magnitude). env-overridable.
EXPLORE_SCALE = float(os.environ.get("VGA_EXPLORE_SCALE") or "0.5")
# Magic-jar pickups (EWRAM 0x0234E, found 2026-08-02). Wired for AVAILABILITY,
# not because it is the lever: the standing analysis is that the binding
# constraint is the avoidance attractor / discounting, NOT a missing reward
# channel, and magic was already the lowest-value of the four candidates. A jar
# is +16 units; at 0.25 that is +4 reward, the same modest magnitude as a heart
# heal, and deliberately below the -8 cost of one heart of damage so "take a hit
# to grab the jar" never pays. Do NOT launch a run on the strength of this alone.
MAGIC_SCALE = 0.25
# Enemy damage dealt (IWRAM 0x03253 in -04, found 2026-08-02). THE dense
# engagement lever against the reliability bistability: it pays Link for hitting
# the enemy -- and it is measurably better than keys on every axis that matters.
# measure_engage.py, 24 random episodes on -04:
#   2.21 enemy hits/ep vs 0.46 keys/ep     (~5x DENSER)
#   first hit at ~67 agent steps vs ~206   (fires far EARLIER, discounts less)
#   break-even 5.9 undiscounted, 6.3 discounted (gamma 0.999 -> 0.935 at 67 steps)
# DERIVED like KEY_SCALE (not chosen): ~2x the discounted break-even = 13, at
# which random play scores +84.6 so the gradient points at attacking from step 1.
# Kept below KEY_SCALE(120) so the kill+key is still the dominant prize; a full
# kill pays +78 here (6 HP), the key +120. Unfarmable (health only drains).
ENEMY_DMG_SCALE = float(os.environ.get("VGA_ENEMY_DMG_SCALE") or "13.0")
# Per-EPISODE cap on total enemy_dmg reward (2026-08-06, Tim). DIAGNOSED: the 12ch policy
# FARMS enemy_dmg (+806/ep = 96% of its reward, 0 keys) by re-damaging respawning enemies --
# the "unfarmable" note only guarded a fresh spawn (0->N), not re-killing a respawn. Dropping
# the SCALE instead would fall below the ~6 damage break-even and reinvite hiding, so we CAP
# the channel: early engagement still pays in full (bootstraps), but the farm is bounded so
# keys/doors become the dominant return. Set to 200 (just below random's ~247 legit combat) for
# the retrain. Unset = no cap (prior behavior). env-overridable.
_edc = os.environ.get("VGA_ENEMY_DMG_CAP")
ENEMY_DMG_CAP = float(_edc) if _edc else None
# enemy_dmg is now GENERALIZED to all 16 sprite slots in the oracle (2026-08-03,
# find_enemy_slots.py): it fires for whatever enemy is present and scores in EVERY
# room, so the old per-state gate (enemy_dmg valid only in -04's slot 3) is retired
# -- it is always rewarded now. _enemy_dmg_ok is kept True for channel_reward's
# signature but no longer gates anything.

# Hold each chosen button for this many frames. NOT cosmetic: measured
# 2026-07-31, random play over 6000 frames produced ZERO reward events at
# frame_skip=1 but real damage/deaths at 4/8/16. A single-frame press barely
# nudges Link, so a random policy cannot cross a room or reach an enemy -- there
# was nothing for PPO to learn from. Both A1 runs used single-frame actions.
FRAME_SKIP = 4

# Frame stacking (2026-08-04, Tim). A SINGLE frame carries no motion information, so a
# policy cannot tell which way Link is moving -- which is exactly why the BC direction
# head was near-chance and diagonals were unlearnable. Stack the last STACK_K decision
# observations (STACK_K * frame_skip = 16 game-frames of history) along the channel axis
# so movement/direction is visible. Kept as RGB (not grayscaled) to preserve the colour
# cues the button heads already use. obs = (H, W, 3*STACK_K).
STACK_K = 4


def channel_reward(channel, delta, enemy_dmg_ok=True):
    """Reward for a single oracle event. SINGLE SOURCE OF TRUTH shared by the
    training env's step() and the human-tape scorer (score_tape.py), so the human
    ceiling, the random floor and policy evals all sit on the exact same scale and
    cannot drift apart. enemy_dmg is -04-slot-specific (0x03253 = sprite slot 3),
    so callers pass enemy_dmg_ok=False anywhere that slot is not a validated enemy.
    """
    d = float(delta)
    if channel == "damage":
        return d                        # delta already negative
    if channel == "death":
        return DEATH_PENALTY
    if channel == "rupees":
        return RUPEE_SCALE * d
    if channel == "heal":
        return HEAL_SCALE * d
    if channel == "key":
        return KEY_SCALE * d
    if channel == "key_used":
        return KEY_USED_SCALE * d       # delta and scale both negative -> positive
    if channel == "heart_container":
        return HEART_CONTAINER_SCALE * d
    if channel == "new_room":
        return NEW_ROOM_SCALE * d
    if channel == "explore":
        return EXPLORE_SCALE * d
    if channel == "magic":
        return MAGIC_SCALE * d
    if channel == "enemy_dmg":
        return ENEMY_DMG_SCALE * d if enemy_dmg_ok else 0.0
    return 0.0


def _get_frame(image):
    buf = ffi.buffer(image.buffer, W * H * 4)
    return np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 4)[:, :, :3].copy()


class AlttpPpoEnv(gym.Env):
    """Single-agent gym env over solo A Link to the Past."""

    metadata = {"render_modes": []}

    def __init__(self, rom=None, state_path=None, horizon=3000, death_terminates=False,
                 frame_skip=FRAME_SKIP):
        super().__init__()
        self.rom = rom or os.environ.get("FOUR_SWORDS_ROM", ROM)
        # state_path may be a single path OR a list/tuple of paths. With more than
        # one, reset() samples uniformly per episode (domain randomization, added
        # 2026-08-03): a policy that resets into a different room every episode
        # cannot memorise one room's pixels the way the -04 lr* runs did. A single
        # path reduces to the old fixed-room behaviour exactly. self.state_path is
        # kept as "the current episode's state" for back-compat (eval/smoke print).
        sp = state_path if state_path is not None else STATE
        self.state_paths = [sp] if isinstance(sp, str) else list(sp)
        self.state_path = self.state_paths[0]
        self.horizon = horizon
        self.frame_skip = max(1, int(frame_skip))
        self.death_terminates = death_terminates
        self.observation_space = spaces.Box(low=0, high=255, shape=(H, W, 3 * STACK_K), dtype=np.uint8)
        self.action_space = spaces.MultiDiscrete(ACTION_NVEC)
        self.core = None
        self.image = None
        self.oracle = None
        self._frames = None
        self._tick = 0
        self._enemy_dmg_ok = True  # enemy_dmg is all-slots now (2026-08-03), valid every room

    def _build_core(self):
        core = mgba.core.load_path(self.rom)
        if core is None:
            raise RuntimeError("could not load ROM: %s" % self.rom)
        w, h = core.desired_video_dimensions()
        img = mgba.image.Image(w, h)
        core.set_video_buffer(img)
        core.reset()
        with open(self.state_path, "rb") as f:
            ok = core.load_raw_state(f.read())
        if not ok:
            raise RuntimeError("load_raw_state failed: %s" % self.state_path)
        # Render one frame before anyone reads the video buffer. A savestate
        # restores RAM but NOT the Image buffer, so without this the first
        # observation of every episode is the stale post-core.reset() boot
        # frame -- byte-identical across every state and every episode, which
        # is exactly what the policy would learn its initial value from.
        core.set_keys(raw=0)
        core.run_frame()
        self.core = core
        self.image = img
        self.oracle = AlttpOracle()
        self._tick = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Domain randomization: pick this episode's start state. self.np_random is
        # seeded by super().reset(seed=...) on the first call and then advances, so
        # the sequence of rooms is varied yet reproducible from the initial seed.
        if len(self.state_paths) > 1:
            idx = int(self.np_random.integers(len(self.state_paths)))
            self.state_path = self.state_paths[idx]
        self._build_core()
        self.oracle.step(0, self.core)  # prime prev_health baseline + explore/room cells
        self._enemy_dmg_rew_ep = 0.0    # per-episode enemy_dmg reward, for the anti-farm cap
        # Fill the stack history with the first frame (standard reset semantics): the policy
        # sees STACK_K copies of the start frame, then real motion accumulates over steps.
        self._frames = deque([_get_frame(self.image)] * STACK_K, maxlen=STACK_K)
        return self._stack(), {"state": os.path.basename(self.state_path)}

    def step(self, action):
        """One agent step = frame_skip emulator frames holding the same buttons.

        horizon stays in EMULATOR FRAMES so episode length means the same thing
        it always did; the agent just gets fewer, longer decisions.
        """
        # Dual dispatch: a bare int (legacy measurement/eval scripts) -> single-button
        # ACTIONS table; a MultiDiscrete array (SB3 training, action_space.sample()) ->
        # full [dir,A,B,L,R] combo. Lets both action representations share one env.
        if isinstance(action, (int, np.integer)):
            key = ACTIONS[int(action)]
        else:
            key = action_to_mask(action)
        reward = 0.0
        died = False
        events = []
        for _ in range(self.frame_skip):
            self.core.set_keys(raw=key)
            self.core.run_frame()
            self._tick += 1
            evs = self.oracle.step(self._tick, self.core)
            events.extend(evs)
            for _tick_no, channel, delta, _value in evs:
                cr = channel_reward(channel, delta, self._enemy_dmg_ok)
                if channel == "enemy_dmg" and ENEMY_DMG_CAP is not None:
                    # Anti-farm cap (2026-08-06): bound total enemy_dmg reward per episode so
                    # camping respawns cannot dominate; early engagement still pays in full.
                    cr = max(0.0, min(cr, ENEMY_DMG_CAP - self._enemy_dmg_rew_ep))
                    self._enemy_dmg_rew_ep += cr
                reward += cr
                if channel == "death":
                    died = True
            if (died and self.death_terminates) or self._tick >= self.horizon:
                break

        self._frames.append(_get_frame(self.image))
        obs = self._stack()
        terminated = died and self.death_terminates
        truncated = (not terminated) and self._tick >= self.horizon
        info = {"tick": self._tick, "events": events}
        return obs, reward, terminated, truncated, info

    def _stack(self):
        # Concatenate the last STACK_K frames along the channel axis -> (H, W, 3*STACK_K),
        # oldest-to-newest. Motion shows up as the offset between channel groups.
        return np.concatenate(list(self._frames), axis=-1)

    def close(self):
        self.core = None
        self.image = None


def _smoke_test():
    import time
    import random
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=None,
                    help="savestate to reset from (default alttp_ingame.state, "
                         "which has no enemies -- use a states/alttp_human-*.state)")
    ap.add_argument("--ticks", type=int, default=1500)
    a = ap.parse_args()

    env = AlttpPpoEnv(horizon=a.ticks, state_path=a.state)
    print("state: %s" % os.path.basename(env.state_path))
    obs, info = env.reset()
    print("reset OK, obs shape=%s dtype=%s" % (obs.shape, obs.dtype))
    assert obs.shape == (H, W, 3 * STACK_K) and obs.dtype == np.uint8

    total_reward = 0.0
    all_events = []
    t0 = time.time()
    for i in range(a.ticks):
        action = env.action_space.sample()  # exercises the MultiDiscrete [dir,A,B,L,R] path
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        all_events.extend(info["events"])
        if terminated or truncated:
            break
    dt = time.time() - t0
    print("ran %d ticks in %.2fs (%.1f ticks/s)" % (i + 1, dt, (i + 1) / dt))
    print("total_reward=%.2f  n_events=%d  events=%s" % (total_reward, len(all_events), all_events[:10]))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(_smoke_test())
