# VGA - Vision-Based Game Agent

A research agent that plays Game Boy Advance games from screen pixels. It captures
the emulator frame, perceives the scene with small learned models, decides on an
input, and presses a button - with no access to game memory at runtime.

Games used so far: *The Legend of Zelda: A Link to the Past* (GBA), including
4-player *Four Swords* link-cable sessions; *Final Fantasy Tactics Advance*;
*Pokemon* (the original plugin); and *Fire Emblem* (text-reading benchmarks).

## Core idea: memory teaches, pixels act

Emulator RAM is used only **offline**, as an automatic labeler and as a ground-truth
scorer for evaluations. Every model the agent runs is trained on pixels and sees
only pixels at inference time. That keeps the agent game-agnostic in principle while
giving it free, perfect training labels wherever the game's memory layout is known.

## Architecture - three tiers on three clocks

```
frame --> fast perception (CNN detector, CRNN text reader)   every frame
            |
            v
          reflex policy (PPO)                                every step
            ^
            |  goals / interjections, low frequency
          reasoner (vision-language model, Qwen3-VL-8B)      every few seconds
```

- **Perception** (`train/perception/`, `train/ocr/`): a 0.11M-parameter
  CenterNet-style heatmap detector for entities, trained on RAM-labeled frames,
  and CRNN readers for on-screen text (trained on synthetic plus real crops; the
  Zelda reader's real labels come from the RAM text buffer).
- **Reflex policy** (`train/rl/`): PPO policies in Gymnasium environments wrapping
  the emulator (`alttp_ppo_env.py`, `fs_ppo_env.py`).
- **Reasoner** (`serve/`, `vga/reason/`): a locally served vision-language model
  consulted at low frequency to set goals or break the policy out of stuck states.
  It is too slow (~13 s per call) to steer frame by frame, so it decides *what to
  want*, and the fast tiers decide *how to move*.

## Results

Measured on an NVIDIA Jetson AGX Thor. Success is always scored from game memory,
never from the agent's own perception.

| Component | Result |
|---|---|
| Entity detector | 2,830 FPS at batch size 1; locates 89.6% of enemies within 4 px in rooms held out from training |
| Text reader, RAM-taught (Zelda) | Character error rate cut from 43.7% to 5.3% on a small held-out set (21 real text lines), across all dialogue-box styles |
| Text reader, multi-game CRNN | Lower character error rate than Tesseract on real text from all three games tested: Pokemon 10.6% vs 21.2%, Fire Emblem 14.7% vs 17.1%, FFTA 18.0% vs 26.4% |
| Reasoner over reflex policy | Key collection 69%, vs. 50% for the PPO policy alone and 28% for random play (3 seeds x 12 start states, paired p ~ 0.04; `train/rl/arbiter_v1.py`) |
| Evaluation harness | Found that menus ignore input while animating in; fixing the press timing took a menu task from 0/8 to 8/8 |

## Layout

```
vga/
  core/        contract (GameState / Action), capture->perceive->decide->act loop,
               emulator I/O (Linux: mss + xdotool; Windows: pyautogui)
  plugins/     per-game plugins (pokemon)
  reason/      reasoner plugin, staged goals, retrieval, knowledge store
train/
  perception/  RAM-labeled detector data generation + training
  ocr/         CRNN text reader: data generation, training, evaluation, ONNX export
  rl/          PPO environments, policies, arbiter experiments
  ram/         RAM address maps, oracles, and task "ladder" benchmarks
serve/         vision-language model server (vLLM / transformers) + benchmarks
link/          headless 4-player link-cable sessions (Four Swords)
tests/         unit tests
docs/          DIAGNOSIS.md (original architecture analysis), ROADMAP.md
```

Design notes and experiment logs referenced in some code comments are kept private;
the code, results, and the two documents above are what this repository contains.

## Running

The core loop needs only `pip install -r requirements.txt` and a running mGBA
window (`python run_vga.py`). On a headless Linux box, run mGBA inside Xvfb with a
window manager. Training and the model server need PyTorch with CUDA; they were
developed in Docker containers on the Jetson Thor (`serve/Dockerfile.vlm`).

## ROM disclaimer

This project includes **no** ROMs, BIOS files, or emulator binaries, and does not
encourage piracy. Use only ROMs you are legally entitled to. ROMs, save files,
emulator binaries, and model weights are excluded via `.gitignore`.

## Status

Paused (September 2026). Perception and short-horizon tasks work; the open problem
is reliable long-horizon navigation across rooms and screens.

## Acknowledgments

Developed with Claude Code (Anthropic) as an AI pair-programmer and experiment
runner.
