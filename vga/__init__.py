"""
VGA - Vision-based Game-playing Agent.

A game-agnostic core (capture -> GameState -> decide -> Action -> emulator) with
all game-specific knowledge isolated in swappable plugins. The core never imports
a game module; plugins register themselves against the registry.

v1 ships a single plugin (Pokemon, FireRed/LeafGreen-engine family). The seam is
built so that a future, unfamiliar game becomes a *new* plugin rather than a change
to the core. See vga/core/plugin.py for the boundary.
"""
