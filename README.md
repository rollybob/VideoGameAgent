# VGA - Vision-Based Game Agent

A game-playing agent that sees the screen and presses buttons - no emulator
memory hooks, no save-state inspection. It captures the emulator window, decides
on an input from pixels alone, and sends a keystroke. The architecture is
deliberately game-agnostic: a tiny core moves frames and buttons, and all
game-specific knowledge lives in a plugin.

```
capture (frame) -> perceive (GameState) -> decide (Action) -> act (button)
```

## Design

The core knows nothing about any specific game. It defines a hard contract
(`vga/core/contract.py`) and runs the loop (`vga/core/loop.py`); a plugin
supplies perception and control behind one interface (`vga/core/plugin.py`).

- **`GameState`** - everything control may see about a frame: detected objects,
  validated OCR text, named region aggregates, and the raw frame (read-only).
  It carries **no** game-semantic label like "battle" or "overworld" - that
  interpretation is the plugin's job.
- **`Action`** - the only thing control may emit: a button press (optionally a
  short mash) or a wait. Console-level, not game-level.

A new game is a new plugin registered in `run_vga.py`; the core stays untouched.

## Layout

```
vga/
  core/
    contract.py        GameState / Action / Button - the perception<->control seam
    loop.py            capture -> perceive -> decide -> act
    plugin.py          Plugin interface + registry / selection
    emulator.py        mGBA window capture + focused keystroke output
    stability.py       multi-frame confirmation / hysteresis helpers
    state_discovery.py  unsupervised state clustering (perception support)
    embedding.py       lightweight frame embeddings
  plugins/
    pokemon/           first plugin: perception, strategy, optional OCR
run_vga.py             entry point: pick a plugin for the on-screen game, run loop
docs/DIAGNOSIS.md      architecture/diagnosis write-up (Phase 0)
```

## Requirements

Lightweight by design (no torch / ultralytics / LLM / RL stack):

```
pip install -r requirements.txt
```

`numpy`, `opencv-python`, `pillow`, `pyautogui`, `pygetwindow`. OCR
(`pytesseract`) is **optional** - it additionally needs the Tesseract-OCR binary
on PATH; if absent, OCR self-disables and the loop still runs.

## Run

1. Start mGBA with a ROM loaded (window title contains "mGBA").
2. Run the agent:

   ```
   python run_vga.py                  # run until Ctrl+C
   python run_vga.py --max-frames 50  # bounded run, useful for a smoke test
   ```

The agent probes one frame, selects a matching plugin, and drives the loop.

## ROM disclaimer

This project includes **no** ROMs, BIOS files, or emulator binaries, and does not
encourage piracy. Use only ROMs you are legally entitled to. ROMs, save files,
emulator binaries, and model weights are excluded via `.gitignore`.

## Status

Active rebuild. The `vga/` package is the current architecture; an earlier
single-file prototype and its experiments live on the archived `VideoGameAgent`
branch. See `docs/DIAGNOSIS.md` for the analysis behind this design.
