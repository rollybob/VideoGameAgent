# Universal Game Agent - Development Roadmap

> **STATUS: WORK IN PROGRESS**
>
> This project is under active development. The architecture and modules are
> designed and implemented, but full integration testing requires hardware
> that supports GPU acceleration. Contributions and testing are welcome!

## Vision
An AI agent that can play **any** video game by:
1. Learning mechanics from tutorials, experimentation, and external sources
2. Working across different platforms and emulators
3. Asking for human help when truly stuck
4. Building up knowledge over time

---

## Current Status (as of refactoring)

### Completed
- [x] Core behavior tree system for real-time decisions
- [x] Tiered decision architecture (fast/medium/slow layers)
- [x] Pokemon-specific behavior trees
- [x] Centralized configuration system
- [x] Debug and logging system
- [x] Strategy engine with LLM abstraction
- [x] Knowledge acquisition framework
- [x] Platform adapter system (multi-emulator support)
- [x] Battle logic for Pokemon games
- [x] Screen capture and state detection

### In Progress
- [ ] Integration of knowledge acquisition with main agent
- [ ] Web search for game guides
- [ ] User assistance UI integration

### Planned
- [ ] Game genre auto-detection
- [ ] Behavior tree library for different genres
- [ ] Multi-game training data collection
- [ ] Transfer learning between similar games

---

## Architecture Overview

```
+------------------------------------------------------------------+
|                     UNIVERSAL GAME AGENT                          |
+------------------------------------------------------------------+
|                                                                    |
|  USER INTERFACE LAYER                                              |
|  +------------------------------------------------------------+   |
|  | VGA.py / GUI           - Control panel, monitoring          |   |
|  | User Assistant UI      - Q&A when agent needs help          |   |
|  | Debug Visualizer       - Real-time perception display       |   |
|  +------------------------------------------------------------+   |
|                                                                    |
|  KNOWLEDGE LAYER                                                   |
|  +------------------------------------------------------------+   |
|  | Tutorial Learning      - Parse in-game instructions         |   |
|  | User Assistance        - Ask human when stuck               |   |
|  | Web Search             - Find walkthroughs online           |   |
|  | Experimentation        - Trial and error learning           |   |
|  | Transfer Learning      - Apply knowledge from similar games |   |
|  | Knowledge Base         - Persistent learned information     |   |
|  +------------------------------------------------------------+   |
|                                                                    |
|  DECISION LAYER (Tiered)                                           |
|  +------------------------------------------------------------+   |
|  | FAST (<20ms)   - Behavior trees, immediate actions          |   |
|  | MEDIUM (500ms) - State analysis, goal evaluation            |   |
|  | SLOW (async)   - Strategic reasoning, LLM planning          |   |
|  +------------------------------------------------------------+   |
|                                                                    |
|  PERCEPTION LAYER                                                  |
|  +------------------------------------------------------------+   |
|  | Screen Capture         - Platform-independent capture       |   |
|  | Object Detection       - YOLOv8 for game elements           |   |
|  | Text Recognition       - EasyOCR/Tesseract for text         |   |
|  | State Classification   - Battle, menu, overworld, etc.      |   |
|  +------------------------------------------------------------+   |
|                                                                    |
|  PLATFORM LAYER                                                    |
|  +------------------------------------------------------------+   |
|  | Platform Adapters      - GBA, SNES, N64, PC, etc.           |   |
|  | Emulator Adapters      - mGBA, RetroArch, Dolphin, etc.     |   |
|  | Input Translation      - Generic buttons to platform keys   |   |
|  | Game Profiles          - Per-game configuration             |   |
|  +------------------------------------------------------------+   |
|                                                                    |
|  MEMORY LAYER                                                      |
|  +------------------------------------------------------------+   |
|  | World State            - Current game state                 |   |
|  | Goal System            - Objectives and progress            |   |
|  | Spatial Memory         - Maps and visited locations         |   |
|  | Experience Memory      - What worked before                 |   |
|  +------------------------------------------------------------+   |
|                                                                    |
+------------------------------------------------------------------+
```

---

## Knowledge Acquisition Priority

When the agent doesn't know what to do:

1. **Check Memory** - Have we seen this before?
2. **Read Screen** - Is there tutorial/help text visible?
3. **Try Safe Actions** - Press A, B, direction buttons
4. **Apply Genre Knowledge** - What do similar games do?
5. **Ask User** - If available and enabled
6. **Search Web** - Find a walkthrough
7. **Random Exploration** - Last resort

---

## Platform Support Roadmap

### Phase 1: Retro Consoles (Current Focus)
- [x] GBA (mGBA)
- [ ] GBC (mGBA, BGB)
- [ ] NES (FCEUX, Mesen)
- [ ] SNES (SNES9x, BSNES)
- [ ] Genesis (Kega Fusion)

### Phase 2: 3D Consoles
- [ ] N64 (Project64, Mupen64)
- [ ] PS1 (DuckStation)
- [ ] GameCube/Wii (Dolphin)

### Phase 3: Modern Platforms
- [x] PC Games (native) - pc_game_adapter.py
- [x] Steam games - SteamGameAdapter with overlay detection
- [ ] Web games (selenium ready in requirements)

### Phase 4: Mobile Platforms
- [x] Android via BlueStacks - mobile_game_adapter.py
- [x] Android via LDPlayer
- [x] Android via NoxPlayer
- [x] Android via MEmu
- [x] ADB input support for reliable touch simulation
- [ ] iOS (not feasible on Windows)

### Universal Support
- [x] RetroArch (multi-core)
- [x] Virtual controller (vgamepad) ready

---

## Genre Support Roadmap

### Phase 1: RPGs (Current Focus)
- [x] Pokemon-style turn-based
- [ ] Final Fantasy-style menu combat
- [ ] Action RPGs

### Phase 2: Action/Platformers
- [ ] Side-scrolling platformers
- [ ] Top-down action (Zelda-style)
- [ ] Metroidvania

### Phase 3: Other Genres
- [ ] Puzzle games
- [ ] Racing games
- [ ] Fighting games
- [ ] Strategy games

---

## Hardware Upgrade Path

### Current (CPU-only)
- Heuristic strategy engine
- Basic behavior trees
- Tesseract/EasyOCR on CPU
- No YOLO (or YOLOv8n)

### With GPU (4GB+)
```python
# In config:
perception.ocr_gpu = True
perception.yolo_device = "cuda"
reasoning.strategy_use_gpu = True
```
- EasyOCR on GPU (10x faster)
- YOLOv8s/m for better detection
- Local LLM for strategy (Mistral 7B)

### With High-End GPU (12GB+)
- Larger YOLO models
- Larger LLMs (13B+)
- Real-time perception + reasoning
- Multi-game training

### With Cloud/API Access
```python
# Enable API-based strategy:
strategy_engine.set_backend(StrategyBackend.API_LLM)
strategy_engine.set_api_key("your-key")
```
- Claude/GPT-4 for strategic reasoning
- Web search integration
- Unlimited context for complex games

---

## Key Files Reference

### Core Agent
- `realtime_agent.py` - Main real-time agent
- `tiered_decision_system.py` - Fast/medium/slow layers
- `behavior_tree.py` - Behavior tree framework
- `pokemon_behaviors.py` - Pokemon-specific trees

### Knowledge & Learning
- `knowledge_acquisition.py` - Multi-source learning
- `strategy_engine.py` - LLM-ready strategic reasoning
- `memory_system.py` - Persistent world state
- `goal_system.py` - Objective tracking

### Perception
- `perception_engine.py` - YOLO + OCR
- `screen_reader.py` - Screen capture and state detection
- `text_analyzer.py` - NLP for game text

### Platform
- `platform_adapter.py` - Multi-platform support (emulators)
- `pc_game_adapter.py` - PC games (Steam, GOG, Epic, native)
- `mobile_game_adapter.py` - Android games (BlueStacks, LDPlayer, etc.)
- `controllers/` - Button input systems
- `config.py` - Centralized configuration

### Training
- `environments/` - Training systems for each layer

---

## Next Development Priorities

### Short Term (Next Session)
1. Integrate `KnowledgeAcquisitionSystem` into `RealTimeAgent`
2. Add user assistance UI to VGA.py
3. Test with a second game (non-Pokemon)

### Medium Term
1. Implement web search for walkthroughs
2. Add genre detection
3. Create behavior trees for platformers
4. SNES adapter testing

### Long Term
1. Transfer learning between games
2. Multi-game knowledge base
3. API integration for cloud LLM
4. Self-improvement through play

---

## Contributing

### Adding a New Platform
1. Create profile in `PLATFORM_PROFILES` dict
2. Create emulator profile in `EMULATOR_PROFILES`
3. Implement specific adapter if needed
4. Add to auto-detect list

### Adding a New Game Genre
1. Create behavior trees in new file (e.g., `platformer_behaviors.py`)
2. Add genre detection rules in `TransferLearner`
3. Add genre-specific knowledge to `GENRE_KNOWLEDGE`
4. Test with representative game

### Adding Knowledge Sources
1. Inherit from `KnowledgeProvider`
2. Implement `can_help()` and `get_knowledge()`
3. Register in `KnowledgeAcquisitionSystem`
4. Add to priority order in `get_help()`

---

## Testing Checklist

### Before Merging
- [ ] Agent runs without errors on Pokemon Fire Red
- [ ] Config system loads/saves correctly
- [ ] Debug system logs without crashing
- [ ] Behavior trees execute without timeout
- [ ] No hardcoded paths remain

### Integration Tests
- [ ] Tiered decision system processes frames < 20ms
- [ ] Context switching works (overworld -> battle -> menu)
- [ ] Battle manager handles full battles
- [ ] Memory persists across restarts

---

*Last updated: After architecture refactoring*
