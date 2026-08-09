"""Task 09 Phase A1: PPO baseline trainer, shared across envs (fs / alttp).

Runs N parallel env worker PROCESSES (SB3 SubprocVecEnv) by default, spreading
env-stepping CPU load across cores instead of pegging one -- standing practice
on this box, see feedback-balance-load-across-cores memory (a single-threaded
job pegging one core at 100% correlated with a concerning noise on 2026-07-30).
The GPU still centrally handles policy inference/updates across all envs' batched
experience, so this also improves throughput, not just load spread.

Auto-resumes from the latest checkpoint in --checkpoint-dir on restart (a
thor-job survives disconnect but does NOT resume a killed process itself --
see CLAUDE.md's thor-job discipline -- so this script must find its own way
back in). Reward is logged per-episode via SB3's Monitor wrapper to
<checkpoint-dir>/<rank>.monitor.csv (one file per env worker); see
check_progress.py for the IMPROVING/FLAT verdict used for the kill-if-flat
call instead of eyeballing a curve -- it merges all workers' csvs.

Run:
    docker run --rm --runtime nvidia -v ~/projects/VGA:/work -w /work \\
        thor-rl:cu130 python3 train/rl/train_ppo.py --env alttp \\
        --checkpoint-dir train/rl/runs/alttp_a1 --total-timesteps 2000000
"""
import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

ENV_BUILDERS = {}
# Each FS env internally runs a 4-core LinkSession vs. alttp's single core, so
# the safe default env-count (cores used = n_envs * cores-per-env) differs a
# lot per type. --n-envs always overrides these.
DEFAULT_N_ENVS = {"fs": 3, "alttp": 6, "walker": 7}


# Chambers of Insight (the "coop" bank) has no rupees or hazards, so it cannot
# produce a reward signal -- always train on a real stage. Selectable now that
# health is a single global address (2026-07-31) rather than per-bank.
FS_BANK = os.environ.get("VGA_FS_BANK", "taluscave")
# Human-driven start-point. alttp_ingame.state sits in an enemy-free zone, which
# is the leading suspect for alttp-ppo-a1 going flat -- a cold random policy
# never reaches anything that moves the reward.
# -00 (the room with several soldiers), NOT the densest-reward room.
# measure_density.py ranks by damage+deaths, i.e. how hard a room PUNISHES random
# play -- which is not the same as how much there is to learn. That metric picked
# -02, and watching the trained a2 policy on screen showed why that was wrong:
# -02 holds a single enemy, so the optimal strategy is "kill the one thing, then
# jitter in place with nothing left to do". -02 scores more damage per episode
# only because its lone enemy is aggressive in a small room. -00 has several
# soldiers, so combat is a repeatable skill rather than a one-off event.
def _resolve_state(tok):
    """A DR-mix token -> a .state path. Accepts an absolute path, or a bare name
    (alttp_human-04, with or without the .state extension) resolved under states/."""
    tok = tok.strip()
    if not tok:
        return None
    if os.path.isabs(tok):
        return tok
    name = tok if tok.endswith(".state") else tok + ".state"
    return os.path.join(HERE, "states", name)


# Domain-randomization mix (2026-08-03): VGA_ALTTP_STATES (comma-separated names
# or paths) takes precedence over the single VGA_ALTTP_STATE and makes the env
# sample one start state per episode. Single-state VGA_ALTTP_STATE still works
# exactly as before (the old run scripts pass it).
_states_env = os.environ.get("VGA_ALTTP_STATES")
if _states_env:
    ALTTP_STATE = [p for p in (_resolve_state(t) for t in _states_env.split(",")) if p]
else:
    ALTTP_STATE = os.environ.get(
        "VGA_ALTTP_STATE", os.path.join(HERE, "states", "alttp_human-00.state"))


def _build_fs(horizon):
    from fs_ppo_env import FsPpoEnv
    return FsPpoEnv(horizon=horizon, checkpoint_name=FS_BANK)


def _build_alttp(horizon):
    from alttp_ppo_env import AlttpPpoEnv
    # death_terminates=True for two reasons: (1) after death the game leaves
    # normal play and 0x00C93 stops being health -- leaving the episode running
    # feeds the oracle garbage (observed ramping 111->94->78 "damage" events);
    # (2) A1's fixed-horizon no-terminate design was itself a suspect for the
    # flat result, since post-death ticks dilute every real signal.
    return AlttpPpoEnv(horizon=horizon, state_path=ALTTP_STATE,
                       death_terminates=True)


def _build_walker(horizon):
    from walker_env import WalkerEnv
    # Goal-conditioned walker (2026-08-09). DR mix comes from VGA_WALKER_STATES
    # (same token format as VGA_ALTTP_STATES); every listed state must have a
    # gen_walk_bank.py bank or the env fails loudly at first reset.
    names = (os.environ.get("VGA_WALKER_STATES") or "").split(",")
    paths = [p for p in (_resolve_state(t) for t in names) if p]
    if not paths:
        raise SystemExit("walker env needs VGA_WALKER_STATES (comma-separated state names)")
    return WalkerEnv(horizon=horizon, state_path=paths if len(paths) > 1 else paths[0])


ENV_BUILDERS["fs"] = _build_fs
ENV_BUILDERS["alttp"] = _build_alttp
ENV_BUILDERS["walker"] = _build_walker


def latest_checkpoint(checkpoint_dir, name_prefix):
    pattern = os.path.join(checkpoint_dir, "%s_*_steps.zip" % name_prefix)
    candidates = glob.glob(pattern)
    if not candidates:
        return None, 0
    def steps_of(path):
        m = re.search(r"_(\d+)_steps\.zip$", path)
        return int(m.group(1)) if m else -1
    best = max(candidates, key=steps_of)
    return best, steps_of(best)


def linear_schedule(initial):
    """SB3 lr schedule: progress_remaining goes 1 -> 0 over the learn() call, so
    this anneals the learning rate from `initial` down to 0. The hypothesis
    (2026-08-02) is that the best!=final oscillation is fixed-lr PPO never
    settling -- clip_fraction sat at 0.3-0.5 (healthy ~0.1-0.2) the whole run and
    the crashes were LOW-KL cumulative drift, not single destructive updates
    (so target_kl missed them). Shrinking the step size to ~0 by the end lets the
    policy settle into wherever it is rather than jittering out of a good region.
    NOTE on resume: SB3 restores the schedule from the checkpoint but re-bases
    progress on the NEW learn() horizon, so a crashed-and-resumed run gets a
    discontinuous lr. Fine for a single uninterrupted run; do not resume these.
    """
    return lambda progress_remaining: initial * progress_remaining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=list(ENV_BUILDERS), required=True)
    ap.add_argument("--horizon", type=int, default=3000, help="ticks per episode")
    ap.add_argument("--total-timesteps", type=int, default=2_000_000)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--checkpoint-freq", type=int, default=20_000,
                     help="total env-steps between checkpoints (across all workers)")
    ap.add_argument("--n-envs", type=int, default=None,
                     help="parallel env worker processes (default depends on --env, see DEFAULT_N_ENVS)")
    ap.add_argument("--n-steps", type=int, default=512, help="PPO rollout length per env")
    ap.add_argument("--batch-size", type=int, default=64)
    # gamma 0.999, NOT SB3's 0.99 default. Measured 2026-08-01: at gamma=0.99 the
    # effective horizon is ~100 agent steps while episodes run 750 and keys
    # arrive at a mean of 206, so a key is discounted to 12.6% of face value.
    # The KEY_SCALE=120 break-even was derived in UNDISCOUNTED units; in the
    # discounted units PPO actually optimises, break-even is ~452, so 120 was
    # still below it. 0.999 gives an effective horizon of ~1000 steps, which
    # matches the 750-step episode, and lifts the key's discounted value to ~98.
    ap.add_argument("--gamma", type=float, default=0.999)
    # ent_coef 0.01, NOT SB3's 0.0 default. Every run so far used 0.0, i.e. NO
    # entropy bonus at all, so nothing resisted premature convergence onto a
    # deterministic policy -- and with dense negatives the greedy answer is
    # "touch nothing". All seven runs collapsed to damage-avoidance; 0.01 is the
    # standard value for image-observation PPO.
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for PPO and the envs. REQUIRED to tell a real "
                         "config difference from run-to-run luck: a5 (no "
                         "target_kl) peaked at +75.0 and a6 (target_kl) at +7.2, "
                         "but they differed in seed as well as config, so that "
                         "gap is currently unattributable. n=1 is not a result.")
    ap.add_argument("--target-kl", type=float, default=0.03,
                    help="early-stop an update epoch once the policy has moved "
                         "this far in KL. 0.03 is deliberately loose relative to "
                         "the usual 0.01-0.02, because a5 ran at 0.15-0.18 and "
                         "the aim is to stop the collapse without over-damping "
                         "learning that was otherwise working.")
    ap.add_argument("--learning-rate", type=float, default=3e-4,
                    help="PPO learning rate (SB3 default 3e-4)")
    ap.add_argument("--lr-decay", action=argparse.BooleanOptionalAction, default=True,
                    help="linearly anneal the learning rate to 0 over training, to "
                         "make the policy SETTLE at the end instead of oscillating "
                         "past its best (best!=final). STANDARD/default-on as of "
                         "2026-08-03: a 4-seed confirm banked lr-decay as the "
                         "stability fix -- it settles the final (mechanism "
                         "confirmed: finals flat over their last rollouts) and gave "
                         "3/4 clean sampled finals plus the project's two "
                         "both-mode-perfect seeds (lr11/lr12), after target_kl and "
                         "ent_coef both failed as stability levers. Pass "
                         "--no-lr-decay for constant lr (needed to reproduce the "
                         "pre-08-03 e/h/s runs). See linear_schedule + RUN_LOG "
                         "2026-08-03. NB it settles WHATEVER policy is late, so it "
                         "cannot rescue a run that already collapsed mid-training "
                         "(lr13).")
    ap.add_argument("--init-policy", default=None,
                    help="path to a BC-pretrained policy state_dict (bc_pretrain_sb3.py) to "
                         "WARM-START a fresh run -- Stage 2. Loaded after the PPO policy is "
                         "built; ignored on resume-from-checkpoint.")
    args = ap.parse_args()

    n_envs = args.n_envs or DEFAULT_N_ENVS[args.env]
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    name_prefix = "ppo_%s" % args.env

    def _make():
        return ENV_BUILDERS[args.env](args.horizon)

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = make_vec_env(_make, n_envs=n_envs, monitor_dir=args.checkpoint_dir,
                       vec_env_cls=vec_cls, seed=args.seed)
    print("Using %d parallel env worker(s) (%s)" % (n_envs, vec_cls.__name__), flush=True)

    ckpt_path, done_steps = latest_checkpoint(args.checkpoint_dir, name_prefix)
    if ckpt_path is not None:
        print("RESUMING from %s (%d steps already done)" % (ckpt_path, done_steps), flush=True)
        model = PPO.load(ckpt_path, env=env)
        remaining = max(0, args.total_timesteps - done_steps)
    else:
        print("Starting FRESH PPO run (env=%s)" % args.env, flush=True)
        # target_kl added 2026-08-01 after a MEASURED policy collapse. a5 reached
        # an excellent policy at 899910 steps (+75.0 vs random +29.0; 15 keys,
        # ZERO damage, ZERO deaths over 24 episodes) and then destroyed itself
        # over the last 10% of training: 939906 -> +6.0, final -> -7.3 with no
        # keys at all. Its update stats were approx_kl 0.15-0.18 and
        # clip_fraction 0.37, against healthy PPO values of ~0.01-0.02 and
        # ~0.1-0.2 -- roughly 10x too aggressive. target_kl early-stops the
        # epoch once an update moves the policy further than this, the standard
        # remedy for this exact failure. a4's mid-run oscillation (-4.0 at 200k,
        # -32.5 at 400k, -8.0 at 800k) is very likely the same cause.
        # 0 disables it (SB3 wants None), which is the a5 configuration.
        tkl = args.target_kl if args.target_kl and args.target_kl > 0 else None
        lr = linear_schedule(args.learning_rate) if args.lr_decay else args.learning_rate
        print("learning_rate=%s (%s)" % (args.learning_rate,
              "linear decay -> 0" if args.lr_decay else "constant"), flush=True)
        model = PPO("CnnPolicy", env, n_steps=args.n_steps, batch_size=args.batch_size,
                     target_kl=tkl, seed=args.seed, learning_rate=lr,
                     gamma=args.gamma, ent_coef=args.ent_coef,
                     verbose=1, tensorboard_log=os.path.join(args.checkpoint_dir, "tb"))
        if args.init_policy:
            model.policy.load_state_dict(torch.load(args.init_policy, map_location=model.device))
            print("WARM-START: loaded BC policy weights from %s" % args.init_policy, flush=True)
        remaining = args.total_timesteps

    if remaining <= 0:
        print("Already at/past total_timesteps=%d, nothing to do." % args.total_timesteps, flush=True)
        return

    # save_freq is per-callback-call, which fires once per rollout step across
    # ALL n_envs at once -- divide so --checkpoint-freq keeps meaning "total
    # env-steps between checkpoints" regardless of n_envs (documented SB3 gotcha).
    effective_save_freq = max(1, args.checkpoint_freq // n_envs)
    callback = CheckpointCallback(save_freq=effective_save_freq, save_path=args.checkpoint_dir,
                                   name_prefix=name_prefix)
    model.learn(total_timesteps=remaining, callback=callback,
                reset_num_timesteps=(ckpt_path is None))
    final_path = os.path.join(args.checkpoint_dir, "%s_final.zip" % name_prefix)
    model.save(final_path)
    print("done, saved %s" % final_path, flush=True)


if __name__ == "__main__":
    main()
