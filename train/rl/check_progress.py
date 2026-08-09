"""Task 09 Phase A1: numeric IMPROVING/FLAT verdict from a training run's
monitor CSVs, so a kill-if-flat call is a number to read, not a curve to eyeball.

With parallel env workers (see train_ppo.py's --n-envs), each worker writes its
own <rank>.monitor.csv -- this merges all of them, sorted by elapsed time, before
computing the verdict (a single worker's file alone would be a biased sample).

Usage: python3 check_progress.py <checkpoint-dir> [--window-frac 0.3]
       (also accepts a single monitor.csv path directly, for backward compat)
"""
import argparse
import csv
import glob
import os
import sys


def _load_rows(path):
    with open(path) as f:
        f.readline()  # SB3 Monitor header comment line (a JSON blob), not CSV
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("monitor_path", help="checkpoint dir (globs *.monitor.csv) or a single csv file")
    ap.add_argument("--window-frac", type=float, default=0.3,
                     help="compare mean reward of the trailing/leading fraction of episodes")
    ap.add_argument("--min-episodes", type=int, default=20)
    args = ap.parse_args()

    if os.path.isdir(args.monitor_path):
        csv_paths = sorted(glob.glob(os.path.join(args.monitor_path, "*.monitor.csv")))
    else:
        csv_paths = [args.monitor_path]

    if not csv_paths:
        print("NO_DATA: no monitor csv found at %s" % args.monitor_path)
        return

    rows = []
    for p in csv_paths:
        rows.extend(_load_rows(p))
    rows.sort(key=lambda r: float(r["t"]))

    n = len(rows)
    if n < args.min_episodes:
        print("TOO_EARLY: only %d episodes logged across %d worker(s) (need >=%d)"
              % (n, len(csv_paths), args.min_episodes))
        return

    rewards = [float(r["r"]) for r in rows]
    k = max(1, int(n * args.window_frac))
    early = rewards[:k]
    late = rewards[-k:]
    mean_early = sum(early) / len(early)
    mean_late = sum(late) / len(late)
    delta = mean_late - mean_early
    verdict = "IMPROVING" if delta > 0.05 * abs(mean_early or 1.0) else "FLAT"

    print("workers=%d  episodes=%d  window=%d  mean_reward[first]=%.2f  mean_reward[last]=%.2f  delta=%+.2f"
          % (len(csv_paths), n, k, mean_early, mean_late, delta))
    print("VERDICT: %s" % verdict)


if __name__ == "__main__":
    sys.exit(main())
