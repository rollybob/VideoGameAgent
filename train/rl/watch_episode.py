"""Replay a training episode ON SCREEN, exactly as the env produces it.

Built 2026-07-31 to settle an anomaly no amount of RAM analysis explained: all
three alttp_human-* states give identical episode lengths and identical
damage/death ticks under the same seed, despite loading visibly different rooms.
Rather than keep guessing, put it on the monitor and let a human say what
happens.

Reproduces AlttpPpoEnv's dynamics step-for-step: same start state, same one
settle frame in reset(), same discrete action table, same seeded action stream.
Runs on the real desktop :1 via LINK_PY -- NOT the container (needs an X
display, and the host venv has no gymnasium/numpy, hence the small duplications
below rather than importing the env).

  source ~/projects/VGA/link/env.sh
  DISPLAY=:1 "$LINK_PY" train/rl/watch_episode.py --state states/alttp_human-00.state
"""
import argparse
import os
import random
import sys
import time

import mgba.core
import mgba.gba as gba
import mgba.image
import mgba.log
from mgba._pylib import ffi

mgba.log.silence()
K = gba.GBA
HERE = os.path.dirname(os.path.abspath(__file__))
GBA_W, GBA_H = 240, 160
HUD_H = 76

# MUST match alttp_ppo_env.ACTIONS (index order matters -- the seeded action
# stream is only reproducible if this list is identical).
ACTIONS = [0, 1 << K.KEY_UP, 1 << K.KEY_DOWN, 1 << K.KEY_LEFT, 1 << K.KEY_RIGHT,
           1 << K.KEY_A, 1 << K.KEY_B, 1 << K.KEY_START]
NAMES = ["-", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"]

HEALTH_ADDR = 0x00C93        # EWRAM, eighths
RUPEE_ADDR = 0x02340         # EWRAM, u16 LE


def read_u8(core, addr):
    return int(ffi.cast("uint8_t *", core._native.memory.wram)[addr])


def read_u16(core, addr):
    ew = ffi.cast("uint8_t *", core._native.memory.wram)
    return int(ew[addr]) | (int(ew[addr + 1]) << 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.path.join(HERE, "states", "alttp_human-00.state"))
    ap.add_argument("--rom", default=os.environ.get("FOUR_SWORDS_ROM"))
    ap.add_argument("--seed", type=int, default=3, help="action-stream seed")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=120.0, help="total wallclock budget")
    ap.add_argument("--slow-from", type=int, default=330,
                    help="tick to start slow-motion (the interesting window)")
    ap.add_argument("--slow-to", type=int, default=380)
    ap.add_argument("--slow-fps", type=float, default=12.0)
    args = ap.parse_args()

    import pygame
    pygame.init()
    win_w, win_h = GBA_W * args.scale, GBA_H * args.scale + HUD_H
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("ALttP episode replay - seed %d" % args.seed)
    font = pygame.font.SysFont("monospace", 16, bold=True)
    big = pygame.font.SysFont("monospace", 26, bold=True)

    core = mgba.core.load_path(args.rom)
    img = mgba.image.Image(*core.desired_video_dimensions())
    core.set_video_buffer(img)
    core.reset()
    with open(args.state, "rb") as f:
        state = f.read()

    t_end = time.perf_counter() + args.seconds
    run = 0
    running = True
    while running and time.perf_counter() < t_end:
        run += 1
        assert core.load_raw_state(state)
        core.set_keys(raw=0)
        core.run_frame()                 # the one settle frame AlttpPpoEnv.reset does
        rng = random.Random(args.seed)
        prev_hp = read_u8(core, HEALTH_ADDR)
        start_hp = prev_hp
        banner, banner_until = "", 0.0
        log = []
        tick = 0
        dead = False

        while running and not dead and time.perf_counter() < t_end:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (
                        ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    running = False
            ai = rng.randrange(len(ACTIONS))
            core.set_keys(raw=ACTIONS[ai])
            core.run_frame()
            tick += 1

            hp = read_u8(core, HEALTH_ADDR)
            rup = read_u16(core, RUPEE_ADDR)
            if hp < prev_hp:
                msg = "tick %d  DAMAGE %d -> %d" % (tick, prev_hp, hp)
                log.append(msg)
                print("run %d: %s" % (run, msg), flush=True)
                banner, banner_until = msg, time.perf_counter() + 2.0
            if hp == 0 and prev_hp > 0:
                log.append("tick %d  DEATH" % tick)
                print("run %d: tick %d DEATH" % (run, tick), flush=True)
                banner, banner_until = "DIED at tick %d" % tick, time.perf_counter() + 4.0
                dead = True
            prev_hp = hp

            # draw
            raw = ffi.buffer(img.buffer)
            surf = pygame.image.frombuffer(raw, (GBA_W, GBA_H), "RGBX")
            surf = pygame.transform.scale(surf, (GBA_W * args.scale, GBA_H * args.scale))
            screen.blit(surf, (0, 0))
            bar = pygame.Rect(0, GBA_H * args.scale, win_w, HUD_H)
            pygame.draw.rect(screen, (20, 20, 20), bar)
            y = GBA_H * args.scale + 6
            lines = [
                "run %d   tick %4d   action %-5s" % (run, tick, NAMES[ai]),
                "HP %3d (%.2f hearts, start %d)   rupees %d" % (
                    hp, hp / 8.0, start_hp, rup),
            ]
            slow = args.slow_from <= tick <= args.slow_to
            lines.append("SLOW-MO (watch what hits Link)" if slow else
                         "60Hz   ESC quits")
            for i, ln in enumerate(lines):
                screen.blit(font.render(ln, True, (220, 220, 220)), (8, y + i * 20))
            if banner and time.perf_counter() < banner_until:
                s = big.render(banner, True, (255, 80, 80))
                screen.blit(s, ((win_w - s.get_width()) // 2, 8))
            pygame.display.flip()

            time.sleep(1.0 / (args.slow_fps if slow else 60.0))

        # hold the death frame briefly so it can be read, then replay
        hold = time.perf_counter() + 3.0
        while running and dead and time.perf_counter() < min(hold, t_end):
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (
                        ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    running = False
            time.sleep(0.05)
        if log:
            print("run %d summary: %s" % (run, " | ".join(log)), flush=True)

    pygame.quit()
    print("watch_episode done after %d run(s)" % run, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
