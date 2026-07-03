# F21 RAM-oracle spike — findings (2026-06-30)

Goal: confirm a headless mGBA memory-peek path on this box, then locate state
addresses (player x/y, map, party HP/PP, money, battle flag) for our ROM(s).
RAM is a TRAINING+EVAL crutch ONLY; runtime perception stays pixels-only.

## What this box has
- mgba-qt 0.10.2 (Qt frontend, /usr/games/mgba-qt). NO Lua/`--script` CLI flag.
  CLI exposes only `-g` (GDB stub :2345), `-d` (CLI debugger), savestates.
- libmgba.so.0.10.2 present (/usr/lib/aarch64-linux-gnu/), full C API exported
  (mCoreFind, mCoreLoadFile, GBACreate, VFileOpen, ...). No prebuilt python bindings.

## Path 1 — GDB stub (tried first, REJECTED)
- A tiny pure-python GDB-RSP client (train/ram/gdbrsp.py) connects to :2345 and
  reads memory: `$m<addr>,<len>#csum` works. PROVEN: read EWRAM/IWRAM, got valid
  hex back, `?`/Ctrl-C return `S02` stop replies. So memory reads themselves work.
- FATAL FLAW: `mgba-qt -g` HALTS the core at boot waiting for the debugger, and the
  RSP `c` (continue) does NOT resume emulation headless — CPU stays at 0 jiffies/s
  after continue (measured via /proc/<pid>/stat deltas, not the unreliable
  `top -n1`), and a second Ctrl-C gets no stop reply. The Qt run-loop is entangled
  with the (non-exposed) GL window. So you can read RAM but cannot run+sample the
  game. Dead end for a data engine that needs thousands of (frame, RAM) pairs.

## Path 2 — headless libmgba bindings (CHOSEN)
- Drive libmgba directly (no Qt, no window, no GL) so the Xvfb/exposure problems
  vanish, and we get the FRAMEBUFFER and RAM from ONE synced process with explicit
  run_frame() control. This is the correct foundation for the auto-labeling data
  engine (run frames fast, deterministic, headless) AND the eval oracle.
- Options: (A) build mGBA's official CFFI python bindings from the 0.10.2 source
  (BUILD_PYTHON=ON), (B) ctypes wrapper over the exported mCore C API. Prefer (A).

## RESULT (2026-06-30) -- SPIKE SUCCEEDED
- Built mGBA 0.10.2 CFFI python bindings (train/ram/build_pybindings.sh; needed
  USE_FFMPEG=ON because the e-reader EReaderScan* symbols the CFFI wrapper
  references are guarded by it; USE_LIBZIP=OFF -- the distro's libzip-dev cmake
  target is broken). Module: ~/src/mgba/build-py/python/lib.linux-aarch64-cpython-312.
  Runtime deps in VGA venv: cffi, cached_property, Pillow.
- Harness proven headless (NO window/X): load ROM, run_frame(), framebuffer via
  Image.to_pil(), scripted input via core.set_keys/clear_keys, RAM via
  core.memory.wram(=EWRAM)/iwram/u8/u16/u32, savestates via save/load_raw_state.
  Deterministic (reload -> identical RAM). Wrapper: train/ram/emu.py.
- ADDRESSES FOUND (Pokemon AI Red, by movement diff, zero known offsets):
    player_x = 0x02025554 (u16), player_y = 0x02025556 (u16)
    object-event copy at 0x02036e48/4a reads player+7 (FRLG MAP_OFFSET) -> a
    second independent structure confirming the find. Verified tile-by-tile.
  Registry: train/ram/addresses.py. Clean overworld savestate: ow_clean.state.

## Cross-game check -- FFTA (2026-06-30)
- Harness generalized cleanly: loaded the commercial FFTA ROM headless, drove
  language -> Square Enix logos -> title -> main menu -> "Saved Game" -> "Load"
  -> slot 1 (Marche) to the WORLD MAP, all via scripted input (ffta_reach.py).
  (Deliberately did NOT touch "Resume Battle" -- FFTA consumes the suspend-save
  on resume; won't disturb Tim's saved battle.)
- Movement-diff surfaced a position struct at 0x02002c__ (worldmap_cursor
  0x02002c10). KEY NUANCE for F22 schema: the world map is NODE-GRAPH navigation,
  not free tile x/y -- every direction snaps to the single reachable neighbor
  (0x02002c10: 97->65 for R/L/D/U alike). So "player tile x/y" is Pokemon-specific;
  FFTA's overworld analog is a location/cursor index. Clean grid coords would come
  from a BATTLE map (needs story advance). => the METHOD is game-agnostic; the
  STATE SCHEMA is per-game (overworld-tile vs node-graph vs battle-grid).

## Method notes (for finishing the address map / other games)
- mgba -g GDB stub is a dead end (continue won't resume headless) -- IGNORE it;
  use the python bindings.
- Coordinate diffing recipe that worked (find_coords6.py): reach an INTERACTIVE
  overworld (this hack shows a multi-page "Previously on your quest" recap on
  CONTINUE that eats dpad input -- clear it with A presses + ~600 idle frames,
  then verify RIGHT-snap != DOWN-snap before trusting), save a clean savestate,
  then require SYMMETRIC small deltas (X:+k RIGHT/-k LEFT, unchanged vertical)
  and skip the low-EWRAM (<0x1000) OAM/DMA graphics scratch. Avoid tall grass
  (wild encounters replace the RAM you are diffing).

## Next
- Finish the Pokemon field map: map_id, party HP/PP, money, battle flag, menu
  cursor (same diff method: change the field, snapshot, intersect).
- FFTA (task 3): point the harness at the FFTA ROM; find cursor/coords the same way.
- Then wire the oracle into S1/S2: auto-label frames (frame + true state) for the
  mode classifier + detector, and stand up the perceive-only eval (vision vs RAM).
