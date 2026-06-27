# VGA - Action Plan toward the long-term goal

Status: draft v1 (2026-06-26). This is the overhaul plan, not a rewrite. The v1
`vga/` package closes one game's loop; this document is how it grows into the
real goal.

## 0. The goal, stated concretely

> An agent that plays 2D games from rendered frames alone, gets meaningfully
> better at a game as it plays, and can pick up a *new* game with progressively
> less hand-holding - authoring/learning a per-game-family plugin behind the same
> core.

"Plays any game" is an open research problem; we do not pretend otherwise. We
measure progress on a **generalization ladder**, each rung a real acceptance bar:

1. **Same game, robust.** Pokemon FRLG-family hack: survive an arbitrary session
   (battles, menus, dialogue, overworld) for N minutes without wedging, win a
   majority of wild battles, and pursue a coarse objective (reach grass, level
   up) rather than wander.
2. **Same family, transfer.** A *different* Pokemon hack with no per-ROM retuning.
3. **Same console, new genre.** A non-Pokemon GBA title (e.g. Fire Emblem, a
   pinball game) reaches a basic competent loop via a new plugin that reuses the
   shared perception/memory/policy machinery - days of bring-up, not a rewrite.
4. **Cold new game, assisted bring-up.** The agent itself proposes a plugin
   scaffold for an unrecognized game (discovered states + suggested controls),
   human confirms, and it runs.

Rungs 1-2 are engineering. Rung 3 is hard but tractable. Rung 4 is the research
frontier and where the bigger GPUs earn their keep.

## 1. Honest diagnosis of v1

What is load-bearing and **stays** (this is why it's not a rewrite):

- **The seam.** `vga/core/contract.py` (GameState / Action / Button), the loop
  (`core/loop.py`), the plugin Protocol + registry (`core/plugin.py`), the
  emulator I/O (`core/emulator.py`), and the stabilizer (`core/stability.py`).
  These are clean and game-agnostic. Everything below plugs in behind them.
- **The state-discovery spine.** `core/state_discovery.py` + `core/embedding.py`
  already do online clustering + persistence + stabilization. The *embedder* is
  weak; the *spine* is reusable.
- **The FSM insight.** `plugins/pokemon/strategy.py`'s multi-frame confirmation +
  hysteresis is the right pattern for stable state and should generalize.

What is at its **ceiling** and gets replaced/extended behind those seams:

- **Perception is hand-crafted single-frame CV.** `plugins/pokemon/perception.py`
  emits only `{battle, overworld, other}` from HSV masks + an HP-bar contour
  heuristic + a menu-box edge-density heuristic. No notion of dialogue vs menu vs
  shop, no object/sprite detection, no player position, no learned features.
  Brittle thresholds, per-ROM palette sensitivity.
- **`matches()` is a stub** (returns a constant 0.75). The agent literally cannot
  tell one game from another, so plugin *selection* - the whole basis of
  generalization - does not work yet.
- **Policies are placeholders.** Battle = mash A (+ B if stuck); explore = walk a
  direction, rotate when stuck, tap A every 7 steps. No move/PP/type reasoning,
  no goal-direction, no objective signal.
- **No memory anywhere.** The loop has no memory hook; nothing persists within an
  episode beyond the tiny FSM, and nothing at all across sessions.
- **No learning.** Perception and policy are fixed code. The clustering in core is
  an observer; it does not drive behavior.

These four gaps map exactly onto the four things we want: a real vision model,
reliable OCR, learning new games, and memory.

## 2. Architectural changes (the seams to add)

Small, surgical additions that keep the core game-agnostic:

- **`PerceptionBackend` interface** (new, `core/perception.py`): `encode(frame) ->
  vector` and `detect(frame) -> list[Detection]`. Today's CV becomes the default
  backend; a learned encoder/detector swaps in behind it with no plugin change.
  Perception populates `GameState.objects` / `regions` from whichever backend is
  active. (`Detection` already exists in the contract.)
- **`Memory` interface + a loop hook** (new, `core/memory.py`). The loop gains one
  call: after acting, `plugin.observe(state, action, outcome)`; the plugin owns a
  `Memory` it reads in `decide`. Add `observe()` to the Plugin Protocol (optional,
  default no-op so existing plugins still satisfy it).
- **Objective/reward signal in `GameState`.** Add optional `progress: dict`
  (score/level/HP/text-event deltas) that perception extracts and policy/memory
  consume - the substrate for goal-directed behavior and any learning.
- **Real `matches()` / game fingerprinting.** A learned or template-based family
  classifier so `select_plugin` actually routes. This is the lever for rungs 2-4.
- **Headless emulator support** (`core/emulator.py`): a capture/input path that
  works against an off-screen mGBA (Xvfb / frame-grab) so runs are unattended on
  the Thor GPU box, not tied to a visible Windows desktop.

Contract stays backward-compatible: new fields are optional, `observe` defaults
to no-op. v1 plugins keep working while v2 capabilities land.

## 3. Workstreams

### WS1 - Vision model (replace hand-crafted perception)

Problem: HSV/contour heuristics over-split by palette, miss UI semantics, and
don't transfer. Plan, in increasing ambition:

1. **Detector for UI/sprites.** Fine-tune a small object detector (YOLO/RT-DETR
   class) on GBA frames to find HP bars, menu/dialogue boxes, the player sprite,
   NPCs, and text regions. Replaces the brittle contour heuristics with
   calibrated detections feeding `GameState.objects`. Trainable on the Thor.
2. **Self-supervised frame encoder.** Train a compact visual encoder (SimCLR/DINO-
   style) on unlabeled gameplay frames; swap it in behind `Embedder` so
   state-discovery clusters on learned features instead of hand-crafted ones.
   This is the documented "real unlock" for clean, transferable state discovery.
   Small encoder on the Thor; a larger one is 5090-class work.
3. **(Later) VLM-assisted scene understanding.** A vision-language model for
   high-level reads ("this is a shop menu", "a Pokemon fainted") and for bootstrap
   labeling. Heavy; reserve for the big GPUs and use offline (label/distill), not
   in the hot loop.

Keep deterministic CV as a always-available fallback backend.

### WS2 - OCR reliability

Problem: whole-bottom-strip Tesseract on a low-res pixel font is noisy; overworld
text is garbage. Plan:

1. **Region-gate OCR.** Only OCR detected text-box regions (from WS1), never the
   whole frame. Kills overworld noise immediately.
2. **Per-region preprocessing** tuned for the GBA font: integer upscale, palette
   normalization, binarization. Cheap, large accuracy win.
3. **Font-specific recognizer.** Train a tiny CRNN/char classifier on the GBA font
   using **synthetic data** rendered from the font itself (the old repo had a
   `gba_font_renderer`) - effectively unlimited labeled samples, no manual
   labeling. This replaces Tesseract for in-game text and is GPU-trivial on Thor.
4. **Validation.** Expected-format + range checks + cross-frame voting before any
   OCR field is trusted (the contract already routes OCR as advisory `text`).

### WS3 - Learning new games (generalization)

The frontier. Sequenced so each step ships value:

1. **Make `matches()` real.** Family fingerprinting (title-screen / UI-template /
   learned classifier) so `select_plugin` routes correctly. Prereq for everything
   multi-game. (Rung 2.)
2. **Plugin scaffold + bring-up checklist.** A template plugin and a documented
   procedure: run the state-discovery observer on a new game, auto-propose
   modes/controls, human confirms, seed a plugin. Turns "new plugin from scratch"
   into "fill in a scaffold". (Rung 3.)
3. **Game-agnostic exploration/objective policy.** A shared policy that seeks
   novelty/progress using the learned encoder (intrinsic motivation) and any
   `progress` signal perception can read (score/level/text events). Replaces
   random-walk explore across all games. (Rungs 3-4.)
4. **(Research) Meta-learning across the game library.** With several plugins, a
   population/curriculum that distills shared structure (UI grammar, menu
   navigation, "advance text") so a *new* game bootstraps from priors instead of
   zero. This is the principled path to rung 4 and squarely 5090-class work.

### WS4 - Memory

Four kinds, all behind the `Memory` interface:

1. **Spatial.** Per-area maps built from frame-to-frame motion + landmarks, to fix
   aimless exploration (revisit-avoidance, "grass is north"). Reuse pathfinding
   ideas from the old repo.
2. **Episodic.** A store of `(state-embedding, action, outcome)` for retrieval and
   off-line learning; vector index over WS1 embeddings.
3. **Per-game persistent knowledge.** Discovered modes/centroids (already
   persisted), learned control maps, what-worked policies - a per-game profile
   that survives sessions and seeds future runs.
4. **Semantic.** A per-game knowledge base (for Pokemon: type chart, moves, item
   effects) - authored to start, learned/extracted later. Lets the battle policy
   reason instead of mashing A.

Storage: start simple (SQLite + numpy/FAISS for embeddings) on the Thor; the
loop hook (`observe`) is the only core change required.

### WS5 - Training & eval infrastructure (on the Thor)

This is where VGA and the Thor node converge - it *is* the Thor's "eval harness"
roadmap item, specialized to VGA.

1. **Headless mGBA on the Thor** (aarch64 Linux build + Xvfb/offscreen capture) so
   the agent runs unattended on the GPU. (The Windows `mGBA.exe` does not apply.)
2. **Data pipeline.** Capture frames -> datasets -> train (encoder/detector/OCR) ->
   versioned checkpoints. Runs as durable `thor-job`s with `notify` on completion.
3. **Benchmark/eval harness.** Concrete metrics: battle win-rate, session
   survival time, exploration coverage, OCR character-accuracy, and
   new-game-bring-up time. This is where we finally set the **unattended compute
   budget ceiling**. A run reports "should we train more?" with evidence.
4. **Checkpoint/resume discipline** so multi-hour training survives disconnect
   (already have durable jobs + the Telegram bridge to drive/monitor from phone).

## 4. Phased sequencing

- **Phase A - Robust single game (Thor-runnable now).** Headless mGBA on Thor
  (WS5.1) + region-gated OCR + synthetic-font recognizer (WS2) + UI/sprite
  detector (WS1.1) + real battle/explore objective using detections and a coarse
  `progress` signal (WS3.3 lite) + spatial memory (WS4.1). Milestone: clears
  ladder rung 1 unattended on the Thor, reported via the eval harness.
- **Phase B - Transfer + memory.** Real `matches()` (WS3.1) + episodic/persistent
  memory (WS4.2-3) + self-supervised encoder behind state-discovery (WS1.2).
  Milestone: rung 2 (a second Pokemon hack, no retuning) and measurable
  in-session improvement.
- **Phase C - New genre via scaffold.** Plugin scaffold + bring-up checklist
  (WS3.2) + semantic memory (WS4.4). Milestone: rung 3 (a non-Pokemon GBA title).
- **Phase D - Assisted cold bring-up (research, big-GPU).** Meta-learning across
  the library (WS3.4) + VLM-assisted labeling/scaffolding (WS1.3). Milestone:
  rung 4. Gated on the Threadripper/5090 workstation.

Dependencies: WS1.1 (detector) precedes WS2.1 (region-gating) and WS3.3
(objective). WS1.2 (encoder) precedes WS3.4. Memory hook (Section 2) precedes all
of WS4. `matches()` (WS3.1) precedes rungs >=2.

## 5. Data & labeling (the real bottleneck)

Manual labeling does not scale; lean on cheaper sources, in order of preference:
synthetic (render the GBA font -> free OCR labels; scripted scenarios ->
state labels), self-supervision (encoder needs no labels), auto-labeling
(heuristics/VLM propose, human spot-checks), then targeted human labeling only
where the above fall short. The state-discovery observer is already a passive
data collector; wire it to dump curated frames.

## 6. Risks & non-goals

- **General game-play is unsolved.** Hold the ladder; do not chase rung 4 before 1-3
  are solid. If a workstream starts ballooning, stop and re-scope (this killed the
  old repo - see `docs/DIAGNOSIS.md`).
- **Compute reality.** The Thor (~126 TFLOPS BF16) handles detectors, the small
  encoder, OCR, and eval. Large self-supervised encoders / VLM work wait for the
  5090s. Plan around that, don't block on it.
- **Keep the core thin.** Every addition lands behind an interface; the loop stays
  ~20 lines and game-agnostic. Resist re-growing a monolith.
- **Scope rule.** VGA is a standalone testbed; meta-learning here stays VGA-internal.

## 7. Immediate next actions

1. Stand up headless mGBA + frame capture on the Thor (WS5.1); confirm `run_vga.py`
   drives it offscreen (`--max-frames` smoke).
2. Add the `Memory` interface + `observe()` loop hook and the `PerceptionBackend`
   seam (Section 2) - no behavior change, just the sockets.
3. Build the synthetic-font OCR recognizer (WS2.3) - high ROI, fully self-labeled.
4. Collect a first labeled detection set and train the UI/sprite detector (WS1.1).
5. Stand up the eval harness skeleton + set the unattended budget ceiling (WS5.3).
