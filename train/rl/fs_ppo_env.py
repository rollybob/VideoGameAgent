"""Task 09 Phase A1: gymnasium Env wrapping a 4-core Four Swords LinkSession.

Player 0 is RL-controlled (one Discrete action -> one button press per tick);
players 1-3 are held idle for this initial single-agent baseline (see
sessions/SESSION_2026-07-28_task09_A0_closeout.md: "Player 0 only for initial
run; expand to multi-agent in A2"). Reward comes straight from FsOracle
(rupee gain +, health_p0 damage -, death_p0 penalty on top of the damage that
brought it there). Episodes reset from the existing
link/sessions/checkpoints/p*_coop.state savestates and run a fixed tick
horizon -- Four Swords co-op has no natural terminal state at this point in
the game, so death is a reward event, not (by default) episode-ending.

Run directly for a random-action smoke test (see _smoke_test at bottom):
    docker run --rm --runtime nvidia -v ~/projects/VGA:/work -w /work \\
        thor-rl:cu130 python3 train/rl/fs_ppo_env.py
"""
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
VGA = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(VGA, "link"))
sys.path.insert(0, HERE)

import mgba.log
from mgba._pylib import ffi
from mgba.gba import GBA
from link_engine import LinkSession

from oracle import FsOracle

mgba.log.silence()

ROM = os.path.join(VGA, "Emulator", "mGBA", "roms",
                    "Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
CKPT = os.path.join(VGA, "link", "sessions", "checkpoints")


def resolve_ckpt_dir(name):
    """Per-stage banks live in checkpoints_<name>/, the originals in checkpoints/.

    Without this, checkpoint_name="taluscave" silently looked in checkpoints/
    and blew up on a missing file (the same trap link_viewer.py had).
    """
    per_stage = os.path.join(VGA, "link", "sessions", "checkpoints_" + name)
    return per_stage if os.path.isdir(per_stage) else CKPT
W, H = 240, 160

# Discrete action table for player 0. Index 0 = no input.
# set_keys(raw=...) wants a BITMASK (1<<index), not the raw key index -- see
# mgba core._keys_to_int and train/rl/alttp_probe.py's own note on this same bug.
# NO START -- same self-termination exploit found in solo ALttP on 2026-07-31
# (Tim spotted it by watching a replay: the agent menus out via START -> Quit
# rather than dying to anything). Four Swords is if anything MORE exposed: the
# pause -> quit -> stage-select sequence is exactly START, wait ~90-100 frames,
# then A -- see fs_goto_stage.py, which automates that very path. A random or
# reward-seeking policy holding both START and A can walk straight out of the
# stage, and under a damage-avoidance reward it is rewarded for doing so.
ACTIONS = [0, 1 << GBA.KEY_UP, 1 << GBA.KEY_DOWN, 1 << GBA.KEY_LEFT, 1 << GBA.KEY_RIGHT,
           1 << GBA.KEY_A, 1 << GBA.KEY_B]
N_ACTIONS = len(ACTIONS)

DEATH_PENALTY = -20.0  # lump penalty on top of the eighths-of-damage reward that led to 0
# Rupees: 2 points per rupee (Tim 2026-08-02). Four Swords keeps the SIGNED delta
# (FsOracle emits both directions), so SPENDING/losing rupees is penalised --
# deliberately unlike ALttP, because in FS rupees are contested and losable.
RUPEE_SCALE = 2.0

# Hold each chosen button for this many ticks. Measured on solo ALttP
# 2026-07-31: random play produced ZERO reward events at frame_skip=1 but real
# damage/deaths at 4/8/16 -- a single-frame press barely nudges Link, so a random
# policy never reaches anything. Same reasoning applies here.
FRAME_SKIP = 4

# Per-bank oracle overrides, keyed by checkpoint_name (matches --checkpoint-name).
# 2026-07-31: health turned out to be GLOBAL (IWRAM 0x00428), so every bank now
# just uses FsOracle's defaults. The old per-checkpoint override for seaoftrees
# (0x004A1) was wrong -- that byte is constant through a real death and never
# tracked health, which means fs-ppo-a1's health/death channels never fired.
# See the FsOracle docstring for the full correction.
ORACLE_CONFIG = {
    "coop": {},
    "seaoftrees": {},
    "taluscave": {},
    "deathmountain": {},
}


def _get_frame(image):
    buf = ffi.buffer(image.buffer, W * H * 4)
    return np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 4)[:, :, :3].copy()


class FsPpoEnv(gym.Env):
    """Single-agent (player 0) gym env over a 4P Four Swords LinkSession."""

    metadata = {"render_modes": []}

    def __init__(self, rom=None, horizon=3000, death_terminates=False,
                 checkpoint_dir=None, checkpoint_name="coop", frame_skip=FRAME_SKIP):
        super().__init__()
        self.rom = rom or os.environ.get("FOUR_SWORDS_ROM", ROM)
        self.horizon = horizon
        self.frame_skip = max(1, int(frame_skip))
        self.death_terminates = death_terminates
        self.observation_space = spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(N_ACTIONS)
        ckpt_dir = checkpoint_dir or resolve_ckpt_dir(checkpoint_name)
        self._states = [os.path.join(ckpt_dir, "p%d_%s.state" % (i, checkpoint_name)) for i in range(4)]
        self._oracle_kwargs = ORACLE_CONFIG.get(checkpoint_name, {})
        self.sess = None
        self.oracle = None
        self._tick = 0

    def _build_session(self):
        if self.sess is not None:
            self.sess.shutdown()  # documented-safe manual teardown before rebuilding, see link_engine.LinkSession.shutdown
        self.sess = LinkSession(self.rom, n=4, state_path=self._states, trace=False)
        self.oracle = FsOracle(**self._oracle_kwargs)
        self._tick = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._build_session()
        # Tick once before reading the video buffer: a savestate restores RAM
        # but NOT the rendered Image, so without this every episode's first
        # observation is the stale post-reset boot frame -- byte-identical
        # across every bank and every episode (verified 2026-07-31).
        for nd in self.sess.nodes:
            nd.core.set_keys(raw=0)
        self.sess.tick()
        core0 = self.sess.nodes[0].core
        self.oracle.step(0, core0)  # prime prev_* baseline, discard the (empty) first read
        obs = _get_frame(self.sess.nodes[0].image)
        return obs, {}

    def step(self, action):
        """One agent step = frame_skip ticks holding the same button.

        horizon stays in EMULATOR TICKS so episode length keeps its old meaning.
        """
        key = ACTIONS[int(action)]
        reward = 0.0
        died = False
        events = []
        for _ in range(self.frame_skip):
            self.sess.nodes[0].core.set_keys(raw=key)
            for nd in self.sess.nodes[1:]:
                nd.core.set_keys(raw=0)
            self.sess.tick()
            self._tick += 1
            evs = self.oracle.step(self._tick, self.sess.nodes[0].core)
            events.extend(evs)
            for _tick_no, channel, delta, _value in evs:
                if channel == "rupees":
                    reward += RUPEE_SCALE * float(delta)
                elif channel == "health_p0":
                    reward += float(delta)  # delta already negative
                elif channel == "death_p0":
                    reward += DEATH_PENALTY
                    died = True
            if (died and self.death_terminates) or self._tick >= self.horizon:
                break

        obs = _get_frame(self.sess.nodes[0].image)
        terminated = died and self.death_terminates
        truncated = (not terminated) and self._tick >= self.horizon
        info = {"tick": self._tick, "events": events}
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.sess is not None:
            self.sess.shutdown()
            self.sess = None


def _smoke_test():
    import time
    import random
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-name", default="coop",
                    help="which bank to run (coop / seaoftrees / taluscave / "
                         "deathmountain). NOTE: coop has no hazards, so it "
                         "legitimately produces zero reward events.")
    ap.add_argument("--ticks", type=int, default=300)
    a = ap.parse_args()

    env = FsPpoEnv(horizon=a.ticks, checkpoint_name=a.checkpoint_name)
    print("bank: %s" % a.checkpoint_name)
    obs, info = env.reset()
    print("reset OK, obs shape=%s dtype=%s" % (obs.shape, obs.dtype))
    assert obs.shape == (H, W, 3) and obs.dtype == np.uint8

    total_reward = 0.0
    all_events = []
    t0 = time.time()
    for i in range(a.ticks):
        action = random.randrange(N_ACTIONS)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        all_events.extend(info["events"])
        if terminated or truncated:
            break
    dt = time.time() - t0
    print("ran %d ticks in %.2fs (%.1f ticks/s)" % (i + 1, dt, (i + 1) / dt))
    print("total_reward=%.2f  n_events=%d" % (total_reward, len(all_events)))
    by_channel = {}
    for _t, ch, delta, _v in all_events:
        by_channel[ch] = by_channel.get(ch, 0) + 1
    print("events by channel:", by_channel)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(_smoke_test())
