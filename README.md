# Universal Game Agent

> **STATUS: WORK IN PROGRESS**
>
> This project is under active development. Architecture and modules are implemented,
> but full integration testing requires GPU hardware. Contributions welcome!

An AI agent designed to play **any video game** autonomously by reading the screen, understanding game state, and making intelligent decisions.

---

## Features

### Multi-Platform Support
- **Console Emulators**: GBA, GBC, NES, SNES, N64, Genesis, PS1 (via mGBA, RetroArch, etc.)
- **PC Games**: Native Windows games, Steam, GOG, Epic
- **Mobile Games**: Android via BlueStacks, LDPlayer, NoxPlayer, MEmu

### Intelligent Decision Making
- **Tiered Architecture**: Fast reactions (<20ms), medium analysis (500ms), strategic planning (5s+)
- **Behavior Trees**: Deterministic, fast decision-making for real-time gameplay
- **LLM Integration**: Strategic reasoning when hardware permits (Mistral 7B, DialoGPT)
- **Heuristic Fallbacks**: Works on CPU-only systems

### Perception System
- **Object Detection**: YOLOv8 for game element recognition
- **OCR**: EasyOCR + Tesseract for text reading
- **State Detection**: Automatic game state classification (battle, dialogue, exploration)

### Learning & Adaptation
- **Tutorial Learning**: Parses in-game tutorials to learn mechanics
- **Knowledge Transfer**: Applies genre knowledge to new games
- **User Assistance**: Can ask for help when stuck
- **Web Search Ready**: Framework for looking up guides online

---

## Project Structure

```
GameAgentUSB/
├── agent/
│   ├── realtime_agent.py       # Main real-time agent
│   ├── tiered_decision_system.py # Fast/medium/slow layers
│   ├── behavior_tree.py        # Behavior tree framework
│   ├── pokemon_behaviors.py    # Pokemon-specific trees
│   │
│   ├── perception_engine.py    # YOLO + OCR perception
│   ├── screen_reader.py        # Screen capture & state detection
│   ├── text_analyzer.py        # NLP for game text
│   │
│   ├── platform_adapter.py     # Multi-emulator support
│   ├── pc_game_adapter.py      # PC games (Steam, etc.)
│   ├── mobile_game_adapter.py  # Android emulators
│   │
│   ├── strategy_engine.py      # LLM-ready strategic reasoning
│   ├── knowledge_acquisition.py # Multi-source learning
│   ├── memory_system.py        # Persistent world state
│   │
│   ├── config.py               # Centralized configuration
│   ├── debug_system.py         # Logging & performance monitoring
│   └── ROADMAP.md              # Development roadmap
│
├── environments/               # Training environments
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

### External Tools

- **Tesseract OCR**: [Download](https://github.com/tesseract-ocr/tesseract)
  - Add to system PATH or configure path in `config.py`

- **Emulator(s)** (as needed):
  - mGBA for GBA games
  - RetroArch for multi-platform
  - BlueStacks/LDPlayer for Android games

### Optional (for full features)

- **CUDA-capable GPU**: For YOLO and LLM acceleration
- **vgamepad**: For virtual Xbox controller support
- **ADB**: For reliable Android emulator input

---

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GameAgentUSB.git
   cd GameAgentUSB
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure paths** (optional):
   ```python
   # Edit agent/config.py or let it auto-detect
   ```

4. **Start your game** (emulator or PC game)

5. **Run the agent**:
   ```bash
   cd agent
   python realtime_agent.py
   ```

   Or use auto-detection:
   ```python
   from platform_adapter import PlatformAdapterFactory
   adapter = PlatformAdapterFactory.auto_detect()
   ```

---

## Supported Platforms

| Platform | Status | Adapter |
|----------|--------|---------|
| GBA (mGBA) | Tested | `platform_adapter.py` |
| RetroArch | Ready | `RetroArchAdapter` |
| SNES/NES/N64 | Ready | `platform_adapter.py` |
| PC Games | Ready | `pc_game_adapter.py` |
| Steam Games | Ready | `SteamGameAdapter` |
| Android (BlueStacks) | Ready | `mobile_game_adapter.py` |
| Android (LDPlayer) | Ready | `mobile_game_adapter.py` |

---

## Architecture

```
                    Game Screen
                         |
                         v
              ┌──────────────────┐
              │  Screen Capture  │
              └────────┬─────────┘
                       |
                       v
              ┌──────────────────┐
              │ Perception Engine │  (YOLO + OCR)
              └────────┬─────────┘
                       |
          ┌────────────┼────────────┐
          v            v            v
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │   Fast   │ │  Medium  │ │   Slow   │
    │  <20ms   │ │  500ms   │ │   5s+    │
    │ Behavior │ │  State   │ │ Strategy │
    │  Trees   │ │ Analysis │ │   LLM    │
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         |            |            |
         v            v            v
              ┌──────────────────┐
              │ Platform Adapter │
              └────────┬─────────┘
                       |
                       v
                  Game Input
```

---

## ROM Disclaimer

This project does not include any ROM files. If using with emulators:

- Ensure you own physical copies of any games you emulate
- Do not commit ROM files to version control
- Respect all copyright and licensing laws

---

## Contributing

Contributions are welcome! Areas that need work:

- [ ] Testing on more games/platforms
- [ ] Additional game profiles
- [ ] Web search implementation for guides
- [ ] Behavior trees for other genres
- [ ] iOS support research

See `agent/ROADMAP.md` for detailed development plans.

---

## License

This repository is open for academic, research, or personal development. Do not use for unauthorized automation in commercial software or online multiplayer games.
