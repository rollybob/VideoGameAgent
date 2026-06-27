# VGA — Phase 0 Diagnosis

**Date:** 2026-06-08
**Scope:** Diagnosis only. No code was changed. Goal: close one game's loop end-to-end.

---

## 0. TL;DR

The good news is bigger than the brief feared: **a clean single-process loop already exists** in
`GameAgentUSB/agent/proto_agent_v2.py`. It is ~90 lines, ignores the entire "Universal AI /
reasoning engine / ML detector" tangle, and is already shaped like the target architecture
(capture -> detect state -> battle-or-explore -> send input). The rebuild is therefore **small**:
formalize the perception/control seam around this file, fix three concrete integration breaks, and
quarantine the dead weight. We do **not** need to touch `VGA.py` (175 KB monolith),
`VGA_wrapped.py` (4 MB, generated), the Universal AI manager, the reasoning engines, the neural
agents, or the ML/sklearn state detectors.

---

## 1. Inventory & data flow

### The live path (what proto_agent_v2 actually uses)
```
proto_agent_v2.PrototypeAgentV2.run()                      [control loop]
  -> screen_reader.find_emulator_window("mGBA")            [I/O: locate window]
  -> screen_reader.capture_window(window)                  [perception: grab frame]
  -> screen_reader.detect_game_state(frame) -> str         [perception: pixel-threshold state]
  -> if "Battle":  battle_logic_pokemon_v2.take_turn()     [control: battle strategy, no-OCR]
     else:         self.explore() + _sim() stuck-check     [control: exploration]
  -> controllers/gba_controller.press()/mash()             [I/O: pyautogui keystrokes]
```
This path imports cleanly under Python 3.12.10 (verified). Dependencies for it are light:
`numpy, opencv-python, pillow, pyautogui, pygetwindow` (+ `pytesseract` only if OCR is used).

### The dead weight (present in repo, NOT on the live path)
- `VGA.py` (175 KB) — the original monolith GUI app. Stale; README describes a different file
  layout (`main.py`, `input_controller.py`) that no longer exists.
- `VGA_wrapped.py` (4 MB) + `vga_wrapper_*.py` — generated wrapper + launch shims.
- `universal_ai_manager.py`, `vga_integration.py`, `reasoning_engine.py`,
  `rule_based_reasoning.py`, `simple_reasoning_engine.py`, `unified_agent_controller.py`,
  `neural_game_agent.py`, `simple_neural_agent.py`, `ml_state_detector.py` (TensorFlow CNN),
  `sklearn_state_detector.py`, `text_analyzer.py`, `parallel_decision_manager.py` — competing,
  overlapping "brains." Several carry heavy deps (torch, ultralytics, easyocr, spacy,
  sentence-transformers, langchain, llama-cpp, stable-baselines3, wandb, faiss).
- `environments/perception/` — a training graveyard: dozens of trainer scripts, ~25
  `competition_session_*.json` dumps, multiple model managers. This is the "training rabbit hole"
  the brief explicitly rules out of scope.
- `PROJECT_REPORT.txt` — an old optimistic write-up calling a TF CNN system "PRODUCTION READY."
  It contradicts the honest brief and should be disregarded as a status source.

### `requirements.txt` red flag
The pinned dependency set (torch, ultralytics, easyocr, spacy, sentence-transformers, langchain,
llama-cpp-python, transformers, stable-baselines3, tensorboard, wandb, optuna, faiss) is far
heavier than the "lightweight Python, 8-12 GB RAM" constraint. The live path needs almost none of
it. A trimmed requirements file is part of the ship target.

---

## 2. What works

- **Window capture** (`capture_window`) — straightforward mss/PIL grab. Fine.
- **Battle strategy** (`battle_logic_pokemon_v2.PokemonBattleManagerV2`) — genuinely the best code
  in the repo. It is **deliberately OCR-free**: it cycles moves 1-4, tracks PP-empty by frame
  similarity, switches party members, falls back to Struggle then Run. FR/LG-engine compatible.
  This is a real strategy module and should be kept as-is (wrapped, not rewritten).
- **Controller mapping** (`gba_controller`) — clean button->key map for mGBA. Logic is fine; see
  the focus bug below.
- **Loop skeleton** (`proto_agent_v2`) — correct shape, already decoupled from the tangle.

## 3. What's broken (the integration layer)

### BREAK 1 — OCR points at a nonexistent binary (hard fail)
`screen_reader.py:10`:
```python
pytesseract.pytesseract.tesseract_cmd = r"F:\\GameAgentUSB\\agent\\tools\\tesseract\\tesseract.exe"
```
- The repo is at `F:\VGA\GameAgentUSB\...` — the path is **missing the `VGA\` segment**.
- `tools/` contains only `download_mistral.py`; **no tesseract is bundled**.
- Tesseract is **not on PATH** either (verified: `shutil.which('tesseract') -> None`).
- Setting `tesseract_cmd` to a missing file overrides PATH, so **every OCR call raises**.

Net: OCR is currently 100% non-functional through this module. *However* — the live battle path is
intentionally OCR-free, so OCR may not even be on the critical path for the ship target. Decision
needed (see Section 6).

### BREAK 2 — Controller never focuses the emulator (silent no-op)
`gba_controller.press()` calls `pyautogui.keyDown/keyUp` against **whatever window currently has
focus**. Nothing in the loop brings mGBA to the foreground. If focus drifts (or the agent's own
console/GUI has it), inputs go nowhere and the loop "runs" while the game does nothing — a prime
suspect for "doesn't reliably run end-to-end."

### BREAK 3 — Pixel-threshold state detection is brittle and over-eager (primary suspect)
`detect_game_state` (`screen_reader.py:368`) is a single-frame cascade of HSV color-mask pixel
counts with hard cutoffs (`green_pct > 0.25 -> Overworld`, `black_pct > 0.3 -> Dialogue`, etc.).
No temporal confirmation, no hysteresis. Worse, the **battle check runs first and false-fires**:

- Verified live: `detect_game_state(solid_green_frame) -> "Battle"`.
- Root cause: `detect_actual_battle_state` -> `detect_health_bars_detailed` treats **any wide green
  block in the top 25%** (aspect ratio > 3, width > 30 px) as a "health bar." Real overworld grass
  across the top of the screen satisfies this. So walking through grass can read as a battle.

This is exactly the "state transitions on pixel thresholds fire unreliably" symptom. It flaps frame
to frame and mistakes environment color for UI.

---

## 4. Where perception and control are entangled

Honestly, **less than feared on the live path** — proto_agent_v2 already separates them informally.
The entanglement that remains:

1. **State is a bare string**, not a contract. `detect_game_state` returns `"Battle"` /
   `"Overworld"` / etc. Control branches on substring matching (`if "Battle" in state`). There is
   no structured object carrying detections, OCR fields, confidence, or frame metadata — so control
   cannot reason about *how sure* perception is, and cannot apply confirmation/hysteresis centrally.
2. **`screen_reader.py` mixes three concerns**: window I/O (capture), perception (state
   classification), and OCR — all in one 800-line file, with battle-UI heuristics interleaved with
   environment heuristics. The seam to cut is between "produce a GameState" and "decide an Action."
3. **The wider repo** entangles everything (e.g. `vga_integration.py` bolts a "Universal AI" brain
   onto the legacy app with keyword-string action translation and silent `except -> fallback`).
   This is not on the live path and should stay quarantined.

---

## 5. Proposed perception <-> control boundary

A hard data contract, satisfiable by the current weak detector OR deterministic CV (per brief).

```python
# contract.py  (new, ~30 lines, no heavy deps)

@dataclass(frozen=True)
class GameState:
    frame_index: int
    timestamp: float
    label: str            # stabilized state, e.g. "battle" | "overworld" | "dialogue"
    confidence: float
    regions: dict         # optional: named color/UI aggregates used by perception
    ocr: dict             # optional OCR fields, validated; empty if OCR disabled

class Action(Enum):       # the only things control may emit
    UP, DOWN, LEFT, RIGHT, A, B, START, SELECT, WAIT, MASH_A
```

- **Perception layer** (`perception.py`): wraps `capture_window` + `detect_game_state` and returns
  a `GameState`. Internally runs a **StateStabilizer** (Section 5 fixes). The raw detector stays
  swappable behind this — weak YOLO, the current color masks, or template matching all fit.
- **Control layer** (`control.py`): consumes `GameState`, owns a small state machine
  (Explore <-> Battle), emits `Action`. Knows nothing about pixels.
- **Strategy modules**: `battle_logic_pokemon_v2` plugs into Control for the Battle state, wrapped
  to emit `Action`s instead of calling `press()` directly (or, minimally, keep its internal presses
  but invoke it from Control).
- **Emulator I/O** (`emulator.py`): `gba_controller` + a `focus_window()` guarantee (fix BREAK 2).

`proto_agent_v2.run()` becomes the thin top-level wiring of these four pieces.

### Specific fixes applied at this boundary (from brief Section 5)
- **StateStabilizer** in Perception: a transition only commits after the raw label holds **N
  consecutive frames** (multi-frame confirmation); add **hysteresis** so leaving a state needs a
  stronger/longer signal than entering it; replace per-pixel cutoffs with **region-aggregate**
  decisions (already partly there — tighten and require *two corroborating* signals for Battle,
  e.g. health-bar shape AND a bottom-region menu grid, to kill the grass-as-health-bar false
  positive).
- **OCR validation** (only if OCR is kept): fix the path first, then gate any OCR-driven transition
  behind expected-format + range checks + cross-frame confirmation.

---

## 6. Open decisions (need your call before Phase 1)

1. **Demo game & checkpoint.** Recommend a FireRed-engine Pokémon ROM (e.g. `Pokemon AI Red.gba`,
   which already has a `.sav`) since `battle_logic_pokemon_v2` targets that UI. Define the
   "meaningful checkpoint" — proposal: *from a battle encounter, autonomously win the battle and
   return to overworld*, looped a few times. That exercises the full perception->control->action
   path and the primary suspect (state transitions) without needing OCR.
2. **OCR in or out for v1?** The battle path is OCR-free and works. I recommend **shipping the loop
   with OCR disabled** (perception emits empty `ocr`), and fixing/validating OCR as a later
   swappable add-on. Avoids the tesseract-bundling yak-shave on the critical path.
3. **Quarantine confirmation.** OK to treat `proto_agent_v2` as the foundation and leave
   `VGA.py`, the Universal AI stack, the ML/sklearn detectors, and `environments/` untouched
   (moved aside / ignored, not deleted)?

---

## 7. Proposed Phase 1 step plan (pending go-ahead)

1. Add `contract.py` (`GameState`, `Action`). No behavior change.
2. Add `emulator.py`: wrap `gba_controller` + enforce mGBA foreground before each input (fix BREAK 2).
3. Add `perception.py`: wrap capture + `detect_game_state`; add `StateStabilizer`
   (N-frame confirm + hysteresis); tighten battle detection to require two corroborating signals
   (fix BREAK 3). Decide OCR on/off per Section 6.
4. Add `control.py`: Explore/Battle state machine over `GameState`, emitting `Action`; delegate
   Battle to `battle_logic_pokemon_v2`.
5. Rewire `proto_agent_v2.run()` to compose the four modules.
6. Trim a `requirements-min.txt` to just the live-path deps.
7. Verify end-to-end against mGBA + a FireRed-engine ROM to the defined checkpoint; capture a demo
   gif; write the perception-agnostic README.

Estimated change surface: ~4 new small files + edits to `proto_agent_v2.py` and a localized fix in
`screen_reader.py`. No rewrite of working modules. If this starts ballooning, stop and flag.
