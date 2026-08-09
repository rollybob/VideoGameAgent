"""Offline validation for the 2026-08-03 domain-randomization + exploration
changes, BEFORE any GPU. CPU only, parallel across cores.

  RESET DR  -- a MIX env, reset with a CONTINUING rng (as SB3 does: seed once,
               then reset() advances), samples every state, each with a DISTINCT
               fresh first observation (guards the stale-boot-frame bug).
  PLAY      -- per state {-02,-03,-04}, under RANDOM play (the signal floor) and
               DIRECTED play (hold one direction, seed%4, = the reachability a
               learned policy approximates):
                 * EXPLORE fires, and cells covered at shift 4/5/6 (32/64/128-unit
                   cells) so EXPLORE_SHIFT can be TUNED from data, not guessed;
                 * balance -- explore reward share vs keys/enemy (nudge not dominate);
                 * ENEMY GATE -- enemy_dmg fires in -04, SILENT in -02/-03, plus the
                   would-be false fires the id-gate suppressed (per env-step probe,
                   a lower bound);
                 * rooms reached (is the horizon enough to leave a room).

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga -w /vga/train/rl \\
      thor-rl:cu130 python3 measure_dr.py --seeds 8
"""
import argparse
import collections
import hashlib
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MIX = ("02", "03", "04")
SHIFTS = (4, 5, 6, 9)          # 16/32/64-unit cells, and 512-unit rooms
DIRS = (1, 2, 3, 4)            # action indices: up, down, left, right


def _state(s):
    return os.path.join(HERE, "states", "alttp_human-%s.state" % s)


def _run(job):
    from alttp_ppo_env import AlttpPpoEnv, N_ACTIONS
    if job[0] == "RESET":
        _, horizon, n = job
        env = AlttpPpoEnv(horizon=horizon, death_terminates=True,
                          state_path=[_state(s) for s in MIX])
        dist = collections.Counter()
        md5s = collections.defaultdict(set)
        obs, info = env.reset(seed=12345)           # seed once
        for _ in range(n):
            dist[info["state"]] += 1
            md5s[info["state"]].add(hashlib.md5(obs.tobytes()).hexdigest()[:8])
            obs, info = env.reset()                 # no seed -> rng advances (as SB3)
        env.close()
        return ("RESET", dict(dist), {k: sorted(v) for k, v in md5s.items()})

    _, mode, seed, horizon, policy = job
    env = AlttpPpoEnv(horizon=horizon, death_terminates=True, state_path=_state(mode))
    obs, info = env.reset(seed=seed)
    rng = random.Random(seed)
    fixed = DIRS[seed % 4]
    ev = collections.Counter()
    total = 0.0
    positions = []
    SANE = env.oracle.ENEMY_HP_SANE_MAX
    EXPECT = env.oracle.ENEMY_ID_EXPECT
    prev_hp = env.oracle.read_enemy_hp(env.core)
    prev_id = env.oracle.read_enemy_id(env.core)
    wf = wfb = 0
    while True:
        a = rng.randrange(N_ACTIONS) if policy == "random" else fixed
        obs, r, term, trunc, info = env.step(a)
        total += r
        for _t, ch, _d, _v in info["events"]:
            ev[ch] += 1
        positions.append(env.oracle.read_pos(env.core))
        hp = env.oracle.read_enemy_hp(env.core)
        eid = env.oracle.read_enemy_id(env.core)
        if hp < prev_hp and 0 < prev_hp <= SANE and hp <= SANE:
            wf += 1
            if prev_id != EXPECT:
                wfb += 1
        prev_hp, prev_id = hp, eid
        if term or trunc:
            break
    cov = {sh: len({(x >> sh, y >> sh) for (x, y) in positions}) for sh in SHIFTS}
    enemy_ok = env._enemy_dmg_ok
    env.close()
    return ("play", mode, policy, total, info["tick"], dict(ev), cov, wf, wfb, enemy_ok)


def main():
    import alttp_ppo_env as E
    from oracle import AlttpOracle
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    jobs = [("RESET", args.horizon, 30)]
    for mode in MIX:
        for policy in ("random", "directed"):
            jobs += [("play", mode, s, args.horizon, policy) for s in range(args.seeds)]

    reset_res = None
    play = collections.defaultdict(lambda: {
        "rew": [], "len": [], "ev": collections.Counter(),
        "cov": collections.defaultdict(list), "wf": 0, "wfb": 0})
    with mp.Pool(args.workers) as pool:
        for res in pool.imap_unordered(_run, jobs):
            if res[0] == "RESET":
                reset_res = res
                continue
            _, mode, policy, total, length, ev, cov, wf, wfb, enemy_ok = res
            a = play[(mode, policy)]
            a["rew"].append(total); a["len"].append(length); a["ev"].update(ev)
            for sh in SHIFTS:
                a["cov"][sh].append(cov[sh])
            a["wf"] += wf; a["wfb"] += wfb; a["enemy_ok"] = enemy_ok

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0

    print("\n=== 1. RESET DR (30 continuing-rng resets of the MIX env) ===")
    _, dist, md5s = reset_res
    print("  states sampled: %s" % dist)
    print("  first-obs md5 per state: %s" % md5s)
    ok_all3 = len(dist) == len(MIX)
    ok_distinct = all(len(v) == 1 for v in md5s.values()) and \
        len({v[0] for v in md5s.values()}) == len(md5s)
    print("  -> all %d states sampled: %s ; each one distinct fresh first-obs: %s"
          % (len(MIX), ok_all3, ok_distinct))

    print("\n=== 2. PLAY: coverage + events (%d seeds x %d frames) ===" % (args.seeds, args.horizon))
    print("%-3s %-9s %8s %6s %s  %s" % ("st", "policy", "mean_rew", "len", "cells@sh4/5/6 rooms@9", "events"))
    for mode in MIX:
        for policy in ("random", "directed"):
            a = play[(mode, policy)]
            cov = "  %2.0f/%2.0f/%2.0f  %3.1f" % (
                mean(a["cov"][4]), mean(a["cov"][5]), mean(a["cov"][6]), mean(a["cov"][9]))
            print("%-3s %-9s %8.1f %6.0f %s  %s" % (
                mode, policy, mean(a["rew"]), mean(a["len"]), cov, dict(a["ev"]) or "NONE"))

    print("\n=== 3. EXPLORE balance @ shift %d, scale %.2f (directed play) ==="
          % (AlttpOracle.EXPLORE_SHIFT, E.EXPLORE_SCALE))
    for mode in MIX:
        a = play[(mode, "directed")]; n = max(1, len(a["rew"]))
        exp = a["ev"].get("explore", 0) / n
        key = a["ev"].get("key", 0) / n
        edm = a["ev"].get("enemy_dmg", 0) / n
        print("  %-3s explore ~%4.1f/ep -> +%6.1f | key %.2f -> +%6.1f | enemy_dmg %.2f -> +%.1f"
              % (mode, exp, exp * E.EXPLORE_SCALE, key, key * E.KEY_SCALE,
                 edm, edm * E.ENEMY_DMG_SCALE))

    print("\n=== 4. ENEMY GATE (per-state reward gate; enemy_dmg valid only in -04) ===")
    for mode in MIX:
        ed = sum(play[(mode, p)]["ev"].get("enemy_dmg", 0) for p in ("random", "directed"))
        enabled = play[(mode, "random")].get("enemy_ok")
        note = "" if enabled else "  (emitted but NOT rewarded -- per-state gate)"
        print("  %-3s reward-enabled=%s | enemy_dmg events emitted=%d%s" % (mode, enabled, ed, note))
    print("  -> want -04 enabled=True with events>0 (engagement restored); -02/-03 enabled=False")
    return 0


if __name__ == "__main__":
    sys.exit(main())
