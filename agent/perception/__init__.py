"""Perception layer of the agent.

This package contains utilities for capturing the screen and
extracting structured information such as objects and text.
"""

from .perception_engine import PerceptionEngine, GamePerception, DetectedObject, DetectedText
from .screen_reader import (
    find_emulator_window,
    capture_window,
    capture_emulator_game_area,
    capture_emulator_without_overlay,
    detect_game_state,
    analyze_screen_regions,
    extract_text_from_screen,
)

__all__ = [
    "PerceptionEngine",
    "GamePerception",
    "DetectedObject",
    "DetectedText",
    "find_emulator_window",
    "capture_window",
    "capture_emulator_game_area",
    "capture_emulator_without_overlay",
    "detect_game_state",
    "analyze_screen_regions",
    "extract_text_from_screen",
]
