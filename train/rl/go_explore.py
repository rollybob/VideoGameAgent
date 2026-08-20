#!/usr/bin/env python3
"""go_explore.py -- Go-Explore-style archive explorer for ALttP-FS (2026-08-20).

WHY THIS EXISTS
  A full-action RANDOM policy out-explores our hand-built classical navigator. Two reasons:
  (1) the navigator can only emit cardinal moves -- it structurally cannot lift a bush, so the
      one action that unblocks the map is not even in its vocabulary; and
  (2) every hardcoded assumption (mark-cell-BLOCKED, frontier BFS, edge-blocking) becomes a trap
      it fixates on, while random -- having no model -- has no wrong model to get stuck in.
  So the fix is to do LESS hand-engineering, not more. This keeps random's assumption-free breadth
  and adds the ONE thing random lacks: MEMORY of reachable states. That is Go-Explore.

NO GAME KNOWLEDGE IN HERE. There is no bush/pot/NPC/door/lift code anywhere. The archive is keyed
  purely by the RAM oracle (room + coarse world position). Lifting a bush, talking to a soldier,
  crossing a door -- all get DISCOVERED because they unlock new cells, and new cells get remembered
  and resumed from. The action vocabulary is general (move, and move-while-pressing-A/L/R), so a
  contextual affordance is *reachable* by chance, but never named.

Reads emulator state IDENTICALLY to drive_agent.py so coverage compares apples-to-apples.
  Baselines on the same alttp_ingame.state, 5-min episode: classical navigator = 2 rooms;
  best full-cardinal-random = 3 rooms. Target: beat that with zero game-specific code.
"""
import os, sys, time, random, pickle, json, zlib
import numpy as np
import mgba.core, mgba.image, mgba.gba as gba, mgba.log
from mgba._pylib import ffi

mgba.log.silence()

ROOT = os.environ.get("ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROM = os.path.join(ROOT, "Emulator/mGBA/roms/Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
STATE = os.environ.get("STATE", os.path.join(ROOT, "train/rl/states/alttp_ingame.state"))
OUTDIR = os.environ.get("OUTDIR", os.path.join(ROOT, "sessions/agent_run/goexplore"))

SEED         = int(os.environ.get("SEED", "0"))
BUDGET       = int(os.environ.get("BUDGET", "150000"))   # total decisions (each = FRAME_SKIP frames)
ROLLOUT      = int(os.environ.get("ROLLOUT", "50"))      # random-action decisions per resume from the archive
FRAME_SKIP   = int(os.environ.get("FRAME_SKIP", "4"))
ACT_REPEAT   = int(os.environ.get("ACT_REPEAT", "3"))    # hold a sampled action this many decisions -> drift, not in-place jitter
CELL         = int(os.environ.get("CELL", "24"))         # world-units per archive bucket (sub-room resolution, so we can inch past an obstacle)
CORE_HYGIENE = int(os.environ.get("CORE_HYGIENE", "500"))# recreate the core every N decisions -- drive_agent's proven SEG segfault dodge
MAXCELLS     = int(os.environ.get("MAXCELLS", "15000")) # RAM guard: stop archiving new cells past this (rooms metric stays honest regardless)
LOG_EVERY    = int(os.environ.get("LOG_EVERY", "5000"))
MAX_SECONDS  = int(os.environ.get("MAX_SECONDS", "2400"))# wallclock safety valve if the emulator ever wedges
NO_ARCHIVE   = os.environ.get("NO_ARCHIVE", "0") == "1"  # CONTROL: single-episode full-action random (resume from START only, no frontier archive) -- isolates what the archive adds
SELECT_MODE  = os.environ.get("SELECT_MODE", "visit")    # "visit" = the validated default (1/(1+n_chosen)); "frontier" = also bias toward under-explored ROOMS

os.makedirs(OUTDIR, exist_ok=True)
rng = random.Random(SEED)

K = gba.GBA
BIT = {n: 1 << getattr(K, "KEY_" + n) for n in ("A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "R", "L")}

# General action vocabulary -- NOT game-specific. 4 dpad + A/B/L/R singles, plus each dpad crossed
# with A/L/R (so "press the interact/lift button while facing something" is reachable by chance).
# START/SELECT are excluded: they are UI (pause/map), not world-actions, and would only trap the search.
_DIRS = (("up", BIT["UP"]), ("down", BIT["DOWN"]), ("left", BIT["LEFT"]), ("right", BIT["RIGHT"]))
_ACTS = (("A", BIT["A"]), ("B", BIT["B"]), ("L", BIT["L"]), ("R", BIT["R"]))
ACTIONS = [(n, m) for n, m in _DIRS] + [(n, m) for n, m in _ACTS]
for dn, dm in _DIRS:
    for an, am in (("A", BIT["A"]), ("L", BIT["L"]), ("R", BIT["R"])):
        ACTIONS.append((dn + "+" + an, dm | am))   # -> 4 + 4 + 12 = 20 actions


def new_core(state_bytes=None):
    core = mgba.core.load_path(ROM); w, h = core.desired_video_dimensions()
    img = mgba.image.Image(w, h); core.set_video_buffer(img); core.reset()
    with open(STATE, "rb") as f:
        core.load_raw_state(f.read())
    if state_bytes is not None:
        core.load_raw_state(state_bytes)
    return core, img, w, h


def link_world(core):
    iw = ffi.cast("uint8_t *", core._native.memory.iwram)   # movement ORACLE (control signal, not perception)
    return (iw[0x038F4] | (iw[0x038F5] << 8), iw[0x038F0] | (iw[0x038F1] << 8))


def health(core):
    return int(ffi.cast("uint8_t *", core._native.memory.wram)[0x0234D])


def room_cell(core):
    x, y = link_world(core)
    return (x >> 9, y >> 9)


def cell_key(core):
    x, y = link_world(core)
    return (x >> 9, y >> 9, x // CELL, y // CELL)


def apply(core, mask):
    for _ in range(FRAME_SKIP):
        core.set_keys(raw=mask); core.run_frame()   # match drive_agent: keys re-asserted each frame


def snap(core):
    return zlib.compress(bytes(core.save_raw_state()))   # states are many + compress well; keeps the archive RAM-safe


def restore(core, blob):
    core.load_raw_state(zlib.decompress(blob))


class Entry:
    __slots__ = ("state", "n_chosen", "seen_at", "room")
    def __init__(self, state, seen_at, room):
        self.state = state; self.n_chosen = 0; self.seen_at = seen_at; self.room = room


def dump_summary(archive, rooms, steps, t0, final=False):
    room_cell_counts = {}
    for e in archive.values():
        room_cell_counts[str(e.room)] = room_cell_counts.get(str(e.room), 0) + 1
    summ = {"seed": SEED, "steps": steps, "cells": len(archive),
            "rooms": sorted(map(list, rooms)), "room_cell_counts": room_cell_counts,
            "t_s": int(time.time() - t0)}
    with open(os.path.join(OUTDIR, "coverage.json"), "w") as f:
        json.dump(summ, f, indent=2)
    if final and os.environ.get("SAVE_ARCHIVE", "0") == "1":
        # persist archive STATES (zlib-compressed) so the frontier (e.g. the first state PAST the
        # bush) can be resumed / rendered / handed to a later stage without re-searching. Off by
        # default -- states are large; enable only to reuse a specific run's frontier.
        with open(os.path.join(OUTDIR, "archive.pkl"), "wb") as f:
            pickle.dump({str(k): {"state": e.state, "room": list(e.room), "seen_at": e.seen_at,
                                  "n_chosen": e.n_chosen} for k, e in archive.items()}, f)


def select(archive, room_ncells):
    # Prefer under-chosen cells (newly-found cells start at n_chosen=0 -> highest weight). In
    # "frontier" mode, additionally divide by sqrt(cells already archived in that room), so resumes
    # push into UNDER-EXPLORED rooms instead of re-filling the well-mapped start. Weighted-random, O(N).
    keys = list(archive.keys())
    if SELECT_MODE == "frontier":
        weights = [1.0 / (1.0 + archive[k].n_chosen) / (room_ncells.get(archive[k].room, 1) ** 0.5) for k in keys]
    else:
        weights = [1.0 / (1.0 + archive[k].n_chosen) for k in keys]
    r = rng.random() * sum(weights)
    acc = 0.0
    for k, wt in zip(keys, weights):
        acc += wt
        if acc >= r:
            return k
    return keys[-1]


def main():
    t0 = time.time()
    core, img, w, h = new_core()
    start_key = cell_key(core)
    archive = {start_key: Entry(snap(core), 0, room_cell(core))}
    rooms = {room_cell(core)}
    room_ncells = {room_cell(core): 1}                   # per-room archived-cell count, for frontier-biased selection
    eff_rollout = BUDGET if NO_ARCHIVE else ROLLOUT      # CONTROL = one long episode from start (never resume the frontier)
    steps = 0; since_hygiene = 0; last_log = 0
    print(f"[goexplore] seed={SEED} budget={BUDGET} rollout={ROLLOUT} cell={CELL} act_repeat={ACT_REPEAT} "
          f"actions={len(ACTIONS)} start_room={room_cell(core)} start_pos={link_world(core)}", flush=True)

    while steps < BUDGET:
        k = start_key if NO_ARCHIVE else select(archive, room_ncells); e = archive[k]; e.n_chosen += 1
        restore(core, e.state)                             # resume from the frontier (or START, in CONTROL mode)
        rc = None
        for _ in range(eff_rollout):
            _, mask = ACTIONS[rng.randrange(len(ACTIONS))]
            for _rep in range(ACT_REPEAT):
                apply(core, mask)
                steps += 1; since_hygiene += 1
                if since_hygiene >= CORE_HYGIENE:          # periodic fresh core (state preserved) dodges the accumulating-state segfault
                    cur = snap(core); core, img, w, h = new_core(); restore(core, cur); since_hygiene = 0
                if health(core) == 0:                      # dead/game-over is non-spatial -- useless as a resume point
                    rc = "dead"; break
                room = room_cell(core); rooms.add(room)
                ck = cell_key(core)
                if ck not in archive and len(archive) < MAXCELLS and not NO_ARCHIVE:
                    archive[ck] = Entry(snap(core), steps, room)
                    room_ncells[room] = room_ncells.get(room, 0) + 1
                if steps >= BUDGET:
                    break
            if rc == "dead" or steps >= BUDGET:
                break
        if steps - last_log >= LOG_EVERY:
            last_log = steps
            print(f"[goexplore] steps={steps} cells={len(archive)} rooms={len(rooms)} "
                  f"seen={sorted(rooms)} t={int(time.time()-t0)}s", flush=True)
            dump_summary(archive, rooms, steps, t0)
        if time.time() - t0 > MAX_SECONDS:
            print(f"[goexplore] MAX_SECONDS hit at steps={steps}", flush=True); break

    dump_summary(archive, rooms, steps, t0, final=True)
    rl = sorted(rooms)
    print(f"COVERAGE_DONE seed={SEED} steps={steps} cells={len(archive)} rooms={len(rl)} t={int(time.time()-t0)}s", flush=True)
    print(f"ROOMS {rl}", flush=True)


if __name__ == "__main__":
    main()
