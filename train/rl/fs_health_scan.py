"""Task 09 A0: systematic write-scan to find FS per-player health address.

Loads p*_coop.state (5 hearts). For each address in the scan region:
  1. Reload saved raw state on all 4 cores  (fast in-memory restore)
  2. Write 0 to that address on all 4 cores (force HP=0 = death)
  3. Run N frames
  4. Compare full-screen pixels to baseline
  5. If changed: record candidate + save a PNG

This finds the health address without needing to manufacture actual in-game damage.
Health addresses should show heart/death HUD change; crashes show full-screen garbage.

PARALLEL across --workers processes (default 8): the address range is split into
chunks, each worker boots its OWN LinkSession from the same checkpoint and scans
only its slice. Standing practice on this box is to spread heavy CPU work across
cores rather than pegging one -- see feedback-balance-load-across-cores memory.

HANG PROTECTION: some addresses corrupt the emulated 4P link/SIO state badly
enough that sess.tick() spins forever (known hazard, see the IWRAM 0x03228+ note
in sessions/SESSION_2026-07-28_task09_A0_closeout.md). An in-process SIGALRM was
tried first and does NOT work here -- the hang happens inside a cffi native
callback, and cffi silently swallows exceptions raised inside callbacks instead
of propagating them, so the timeout never actually interrupted anything (proven
live 2026-07-30: the "exception ignored from cffi callback" message printed, but
the worker kept spinning at 99% CPU afterward, unchanged). Fix: an EXTERNAL
watchdog in the main process kills a stuck worker with SIGKILL, which no
userspace code (cffi included) can block or swallow. Each worker reports its
current address via a shared multiprocessing.Value the watchdog polls; if it
stalls on the same address for --addr-timeout seconds, the worker is killed and
a fresh one is spawned to continue the rest of its chunk (skipping only the
address that hung).

Usage:
  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga -w /vga \\
      thor-torch:cu130 python3 /vga/train/rl/fs_health_scan.py \\
      [--region ewram|iwram] [--lo 0x0] [--hi 0x8000] [--frames 5] [--workers 8]

Typical runs:
  # IWRAM full (32KB):
  python3 fs_health_scan.py --region iwram --lo 0 --hi 0x8000
  # EWRAM first 32KB:
  python3 fs_health_scan.py --region ewram --lo 0 --hi 0x8000
"""
import argparse
import json
import multiprocessing
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VGA = os.path.dirname(os.path.dirname(HERE))
LINK_DIR = os.path.join(VGA, "link")

ROM = os.path.join(VGA, "Emulator", "mGBA", "roms",
                   "Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
CKPT = os.path.join(VGA, "link", "sessions", "checkpoints")
OUT_BASE = os.path.join(HERE, "scratch", "probe")

W, H = 240, 160


def get_screen(img):
    from mgba._pylib import ffi
    buf = ffi.buffer(img.buffer, W * H * 4)
    return np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 4).copy()


def write_byte(core, is_ewram, offset, val):
    from mgba._pylib import ffi
    ptr = core._native.memory.wram if is_ewram else core._native.memory.iwram
    ffi.cast("uint8_t *", ptr)[offset] = val & 0xFF


def read_byte(core, is_ewram, offset):
    from mgba._pylib import ffi
    ptr = core._native.memory.wram if is_ewram else core._native.memory.iwram
    return int(ffi.cast("uint8_t *", ptr)[offset])


def read_region(core, is_ewram, size):
    from mgba._pylib import ffi
    ptr = core._native.memory.wram if is_ewram else core._native.memory.iwram
    return np.frombuffer(ffi.buffer(ptr, size), dtype=np.uint8).copy()


def _boot_session(rom, states, warmup):
    sys.path.insert(0, LINK_DIR)
    import mgba.log
    from link_engine import LinkSession
    mgba.log.silence()

    sess = LinkSession(rom, n=4, state_path=states, trace=False)
    for _ in range(warmup):
        for nd in sess.nodes:
            nd.core.set_keys(raw=0)
        sess.tick()
    return sess


def _scan_worker(worker_args, current_addr, result_path):
    """Runs in its own Process. Reports progress via current_addr (a shared
    Value the watchdog polls) BEFORE each risky operation, so the watchdog can
    tell exactly which address a killed worker was stuck on. Writes its final
    hits to result_path as JSON on normal completion (nothing is written if
    the watchdog kills this process first)."""
    (rom, states, region, addr_chunk, frames, warmup, val, out_dir) = worker_args
    is_ewram = region == "ewram"
    sess = _boot_session(rom, states, warmup)
    img = sess.nodes[0].image
    raw_states = [nd.core.save_raw_state() for nd in sess.nodes]

    for nd, s in zip(sess.nodes, raw_states):
        nd.core.load_raw_state(s)
    for _ in range(frames):
        for nd in sess.nodes:
            nd.core.set_keys(raw=0)
        sess.tick()
    reference = get_screen(img)

    hits = []
    for addr in addr_chunk:
        current_addr.value = addr  # watchdog checkpoint -- set BEFORE the risky part
        for nd, s in zip(sess.nodes, raw_states):
            nd.core.load_raw_state(s)
        orig_val = read_byte(sess.nodes[0].core, is_ewram, addr)
        for nd in sess.nodes:
            write_byte(nd.core, is_ewram, addr, val)
        for _ in range(frames):
            for nd in sess.nodes:
                nd.core.set_keys(raw=0)
            sess.tick()
        after = get_screen(img)

        if not np.array_equal(after, reference):
            changed_pct = 100.0 * np.mean(after != reference)
            hud_changed = not np.array_equal(after[:20], reference[:20])
            hits.append((addr, orig_val, changed_pct, hud_changed))
            png_path = os.path.join(out_dir, "%s_%05X_v%d_f%d.png" % (region, addr, orig_val, frames))
            with open(png_path, "wb") as f:
                img.save_png(f)

    with open(result_path, "w") as f:
        json.dump(hits, f)


def _run_chunk_with_watchdog(ctx, worker_args_base, chunk, addr_timeout, out_dir, tag):
    """Runs one chunk to completion, restarting past any address that hangs.
    Returns (hits, hung_addrs)."""
    (rom, states, region, _chunk, frames, warmup, val, _out_dir) = worker_args_base
    remaining = list(chunk)
    all_hits, hung = [], []
    attempt = 0

    while remaining:
        attempt += 1
        current_addr = ctx.Value("l", -1)
        result_path = os.path.join(out_dir, ".result_%s_attempt%d.json" % (tag, attempt))
        worker_args = (rom, states, region, remaining, frames, warmup, val, out_dir)
        p = ctx.Process(target=_scan_worker, args=(worker_args, current_addr, result_path))
        p.start()

        last_seen, last_change_t = -1, time.time()
        while p.is_alive():
            time.sleep(1.0)
            seen = current_addr.value
            if seen != last_seen:
                last_seen, last_change_t = seen, time.time()
            elif seen != -1 and time.time() - last_change_t > addr_timeout:
                print("HUNG %s 0x%05X  (worker %s stalled >%ds, killing)"
                      % (region, seen, tag, addr_timeout), flush=True)
                p.kill()
                p.join()
                hung.append(seen)
                idx = remaining.index(seen)
                remaining = remaining[idx + 1:]  # skip the hung address, retry the rest
                break
        else:
            p.join()
            if os.path.exists(result_path):
                with open(result_path) as f:
                    all_hits.extend(tuple(h) for h in json.load(f))
                os.remove(result_path)
            remaining = []  # normal completion, nothing left to retry

    return all_hits, hung


def _chunk(seq, n):
    k, m = divmod(len(seq), n)
    chunks, start = [], 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        if size:
            chunks.append(seq[start:start + size])
            start += size
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=["ewram", "iwram"], default="ewram")
    ap.add_argument("--lo", type=lambda x: int(x, 0), default=0x0000)
    ap.add_argument("--hi", type=lambda x: int(x, 0), default=0x8000)
    ap.add_argument("--frames", type=int, default=8,
                    help="frames to run after write before pixel check")
    ap.add_argument("--warmup", type=int, default=10,
                    help="frames to run at startup for stable baseline")
    ap.add_argument("--val", type=int, default=0,
                    help="value to write (0=kill test, 4=low-health test)")
    ap.add_argument("--min-val", type=int, default=1,
                    help="only test addresses whose current value is >= this")
    ap.add_argument("--max-val", type=int, default=80,
                    help="only test addresses whose current value is <= this")
    ap.add_argument("--rom", default=os.environ.get("FOUR_SWORDS_ROM", ROM))
    ap.add_argument("--checkpoint-dir", default=CKPT,
                    help="dir with p{0..3}_<name>.state (default: the Chambers of Insight coop checkpoint)")
    ap.add_argument("--checkpoint-name", default="coop",
                    help="state filename stem: p{i}_<name>.state")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel worker processes (spreads CPU load across cores)")
    ap.add_argument("--addr-timeout", type=int, default=10,
                    help="seconds a single address may stall before its worker is killed")
    args = ap.parse_args()

    is_ewram = args.region == "ewram"
    region_size = (256 if is_ewram else 32) * 1024
    OUT = os.path.join(OUT_BASE, "health_scan_%s" % args.checkpoint_name)
    os.makedirs(OUT, exist_ok=True)

    states = [os.path.join(args.checkpoint_dir, "p%d_%s.state" % (i, args.checkpoint_name))
              for i in range(4)]

    # Single lightweight session just to read the pre-filter baseline + save a
    # reference PNG for manual sanity-checking -- not part of the parallel scan.
    sess = _boot_session(args.rom, states, args.warmup)
    base_mem = read_region(sess.nodes[0].core, is_ewram, region_size)
    with open(os.path.join(OUT, "baseline.png"), "wb") as f:
        sess.nodes[0].image.save_png(f)
    sess.shutdown()

    scan_range = range(args.lo, min(args.hi, region_size))
    filtered = [a for a in scan_range
                if args.min_val <= int(base_mem[a]) <= args.max_val]
    n = len(filtered)
    n_workers = max(1, min(args.workers, n))
    print("Scan: %s, range 0x%X..0x%X, %d addrs pre-filtered to %d (val in [%d..%d]), %d workers"
          % (args.region, args.lo, args.hi, len(scan_range), n, args.min_val, args.max_val, n_workers))

    chunks = _chunk(filtered, n_workers)
    ctx = multiprocessing.get_context("spawn")
    worker_args_base = (args.rom, states, args.region, None, args.frames, args.warmup, args.val, OUT)

    t0 = time.time()
    # One thread per chunk, each running _run_chunk_with_watchdog (which itself
    # manages a real Process + watchdog loop) -- ThreadPoolExecutor here is just
    # for fanning out the blocking watchdog loops, the actual CPU work is all in
    # the spawned Processes.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_run_chunk_with_watchdog, ctx, worker_args_base, chunk,
                                args.addr_timeout, OUT, "w%d" % i)
                   for i, chunk in enumerate(chunks)]
        results = [f.result() for f in futures]
    elapsed = time.time() - t0

    candidates = sorted((hit for hits, _ in results for hit in hits), key=lambda c: c[0])
    hung_addrs = sorted(a for _, hung in results for a in hung)

    for addr, orig_val, changed_pct, hud_changed in candidates:
        print("HIT %s 0x%05X  orig=%d  changed=%.1f%%  hud=%s" % (
            args.region, addr, orig_val, changed_pct, hud_changed))

    print("\nScan done: %d addrs in %.1fs (%.0f addr/s, %d workers)" % (n, elapsed, n / elapsed, n_workers))
    print("\n=== CANDIDATES ===")
    for addr, orig, pct, hud in candidates:
        print("  %s 0x%05X  orig=%3d  screen_changed=%.1f%%  hud_changed=%s"
              % (args.region, addr, orig, pct, hud))
    if not candidates:
        print("  (none)")
    print("\n=== HUNG (killed after stalling, skipped -- known hazard e.g. IWRAM 0x03228+) ===")
    for addr in hung_addrs:
        print("  %s 0x%05X" % (args.region, addr))
    if not hung_addrs:
        print("  (none)")
    print("\nPNGs in: %s" % OUT)


if __name__ == "__main__":
    main()
