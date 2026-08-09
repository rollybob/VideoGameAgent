"""Parse SB3 pipe-table training logs and show the early-training divergence.

The 2026-08-01 g11-g14 result was 2/4 seeds succeeding, and the failures are
decided EARLY (key rate high in the first fifth, then collapses). The monitor
only logs reward, so the PPO optimisation signals that would explain the
collapse (approx_kl, clip_fraction, entropy_loss) live only in the job-log
tables. This pulls them out, paired with timesteps and ep_rew_mean, so a
collapse can be read against its optimisation signature.

ep_rew_mean is the key-collection proxy: with KEY_SCALE=120 and only small
negatives otherwise, a positive rolling mean means keys are being collected;
when it turns negative the run has fallen into damage-avoidance.

No deps -- runs on host python3. Each SB3 table is a block delimited by dashed
lines; total_timesteps sits in the time/ block BEFORE the train/ block, so a
block is parsed whole and emitted once complete.
"""
import re
import sys

KEYS = ("total_timesteps", "ep_rew_mean", "ep_len_mean", "approx_kl",
        "clip_fraction", "entropy_loss", "explained_variance")


def parse(path):
    rows, cur = [], {}
    line_re = re.compile(r"\|\s*([a-z_]+)\s*\|\s*([-0-9.e+]+)\s*\|")
    with open(path) as f:
        for line in f:
            m = line_re.search(line)
            if m and m.group(1) in KEYS:
                cur[m.group(1)] = float(m.group(2))
            elif line.startswith("---") and "total_timesteps" in cur:
                rows.append(cur)
                cur = {}
    if "total_timesteps" in cur:
        rows.append(cur)
    return rows


def show(label, path, upto=None, stride=1):
    rows = parse(path)
    if upto is not None:
        rows = [r for r in rows if r.get("total_timesteps", 0) <= upto]
    print("\n=== %s (%s) ===" % (label, path.split("/")[-2]))
    print("  %9s %9s %8s %9s %9s %8s" %
          ("steps", "ep_rew", "kl", "clipfrac", "ent_loss", "explvar"))
    for r in rows[::stride]:
        print("  %9d %9.1f %8.4f %9.3f %9.2f %8.2f" % (
            r.get("total_timesteps", 0), r.get("ep_rew_mean", 0),
            r.get("approx_kl", 0), r.get("clip_fraction", 0),
            r.get("entropy_loss", 0), r.get("explained_variance", 0)))


def main(argv):
    # argv: upto stride label=path label=path ...
    upto = int(argv[0]) if argv and argv[0].isdigit() else None
    stride = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 1
    pairs = [a for a in argv if "=" in a]
    for p in pairs:
        label, path = p.split("=", 1)
        show(label, path, upto, stride)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
