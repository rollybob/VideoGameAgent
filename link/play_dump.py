"""Play trained-policy vs random frame dumps side by side on the :1 desktop.

Companion to train/rl/dump_policy_run.py. The policy needs SB3+numpy (container
only); the display needs X (host only). So the container dumps raw frames and
this plays them back at true 60Hz.

Left = trained policy, right = random, same start-state and seed, so the two
panels are directly comparable. Running damage/death tallies make the difference
visible without having to count hearts.

  source ~/projects/VGA/link/env.sh
  DISPLAY=:1 "$LINK_PY" link/play_dump.py --dir link/sessions/policy_dumps
"""
import argparse
import json
import os
import time

W, H = 240, 160
NAMES = ["-", "UP", "DOWN", "LEFT", "RIGHT", "A", "B"]


class Track:
    def __init__(self, d, tag, label):
        with open(os.path.join(d, tag + ".json")) as f:
            self.meta = json.load(f)
        self.path = os.path.join(d, tag + ".rgb")
        self.fh = open(self.path, "rb")
        self.label = label
        self.n = self.meta["frames"]
        self.recs = self.meta["records"]
        self.dmg = self.deaths = 0

    def rewind(self):
        self.fh.seek(0)
        self.dmg = self.deaths = 0

    def frame(self, i):
        self.fh.seek(i * W * H * 3)
        data = self.fh.read(W * H * 3)
        act, hp, rup, dmg, death = self.recs[i]
        self.dmg += dmg
        self.deaths += death
        return data, act, hp, rup, dmg, death


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--fps", type=float, default=60.0)
    args = ap.parse_args()

    import pygame
    pygame.init()
    tracks = [Track(args.dir, "policy", "TRAINED POLICY"),
              Track(args.dir, "random", "RANDOM")]
    s = args.scale
    pw, ph = W * s, H * s
    gap, top, bot = 16, 30, 64
    win_w = pw * 2 + gap * 3
    win_h = ph + top + bot
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("ALttP: trained policy vs random")
    font = pygame.font.SysFont("monospace", 15, bold=True)
    head = pygame.font.SysFont("monospace", 19, bold=True)

    n = min(t.n for t in tracks)
    t_end = time.perf_counter() + args.seconds
    running = True
    while running and time.perf_counter() < t_end:
        for t in tracks:
            t.rewind()
        for i in range(n):
            if not running or time.perf_counter() > t_end:
                break
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (
                        ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    running = False
            screen.fill((16, 16, 20))
            for k, t in enumerate(tracks):
                data, act, hp, rup, dmg, death = t.frame(i)
                surf = pygame.image.frombuffer(data, (W, H), "RGB")
                surf = pygame.transform.scale(surf, (pw, ph))
                x = gap + k * (pw + gap)
                screen.blit(surf, (x, top))
                colour = (255, 90, 90) if (dmg or death) else (90, 90, 100)
                pygame.draw.rect(screen, colour, (x - 2, top - 2, pw + 4, ph + 4), 3)
                screen.blit(head.render(t.label, True, (235, 235, 120)), (x, 4))
                y = top + ph + 8
                for j, ln in enumerate([
                        "HP %3d (%.2f hearts)   rupees %d" % (hp, hp / 8.0, rup),
                        "action %-6s" % NAMES[act % len(NAMES)],
                        "damage %2d   deaths %d" % (t.dmg, t.deaths)]):
                    screen.blit(font.render(ln, True, (225, 225, 225)), (x, y + j * 18))
            screen.blit(font.render(
                "frame %4d/%d   same start + seed   ESC quits" % (i + 1, n),
                True, (150, 150, 160)), (gap, win_h - 16))
            pygame.display.flip()
            time.sleep(1.0 / args.fps)

    pygame.quit()
    print("play_dump done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
