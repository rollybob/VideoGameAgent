"""
Configuration System - Centralized configuration for the AI Game Agent
======================================================================

All configurable paths, settings, and thresholds in one place.
Supports environment variables, config files, and sensible defaults.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent


def get_agent_dir() -> Path:
    """Get the agent directory"""
    return Path(__file__).parent


@dataclass
class PathConfig:
    """File and directory paths"""
    # Tesseract OCR
    tesseract_cmd: str = ""

    # Model paths
    yolo_model: str = ""
    perception_model: str = ""

    # Data directories
    maps_folder: str = ""
    training_data: str = ""
    models_output: str = ""

    # Logs
    log_dir: str = ""

    def __post_init__(self):
        agent_dir = get_agent_dir()
        project_root = get_project_root()

        # Set defaults based on project structure
        if not self.tesseract_cmd:
            # Try common locations
            candidates = [
                os.environ.get('TESSERACT_CMD', ''),
                str(agent_dir / "tools" / "tesseract" / "tesseract.exe"),
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
            ]
            for candidate in candidates:
                if candidate and Path(candidate).exists():
                    self.tesseract_cmd = candidate
                    break

        if not self.yolo_model:
            # Look for custom model first, fall back to default
            custom_model = agent_dir / "models" / "game_detector.pt"
            if custom_model.exists():
                self.yolo_model = str(custom_model)
            else:
                self.yolo_model = "yolov8n.pt"  # Will download if needed

        if not self.maps_folder:
            self.maps_folder = str(agent_dir / "maps")

        if not self.training_data:
            self.training_data = str(agent_dir / "environments" / "perception" / "data")

        if not self.models_output:
            self.models_output = str(agent_dir / "models")

        if not self.log_dir:
            self.log_dir = str(agent_dir / "logs")


@dataclass
class EmulatorConfig:
    """Emulator-specific settings"""
    window_title: str = "mGBA"
    platform: str = "gba"

    # Key mappings for mGBA (can be overridden)
    keymap: Dict[str, str] = field(default_factory=lambda: {
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
        "A": "x",
        "B": "z",
        "start": "enter",
        "select": "backspace",
        "L": "a",
        "R": "s"
    })


@dataclass
class PerceptionConfig:
    """Perception layer settings"""
    # OCR settings
    ocr_engine: str = "easyocr"  # 'easyocr', 'tesseract', or 'both'
    ocr_gpu: bool = False  # Set True when you have GPU
    ocr_confidence_threshold: float = 0.3
    ocr_languages: list = field(default_factory=lambda: ["en"])

    # YOLO settings
    yolo_confidence: float = 0.5
    yolo_device: str = "cpu"  # 'cpu', 'cuda', 'mps'

    # Frame processing
    target_fps: int = 30
    frame_skip: int = 1  # Process every Nth frame for heavy operations


@dataclass
class ReasoningConfig:
    """Reasoning layer settings"""
    # Strategy LLM (for slow strategic decisions)
    strategy_model: str = "microsoft/DialoGPT-medium"
    strategy_max_tokens: int = 512
    strategy_temperature: float = 0.7
    strategy_use_gpu: bool = False

    # Fast decision settings
    fast_decision_timeout_ms: float = 20.0  # Max time for fast decisions
    medium_decision_timeout_ms: float = 200.0  # Max time for medium decisions

    # Decision intervals
    fast_layer_interval_ms: float = 33.0  # ~30 FPS
    medium_layer_interval_ms: float = 500.0  # 2x per second
    slow_layer_interval_ms: float = 5000.0  # Every 5 seconds


@dataclass
class MemoryConfig:
    """Memory system settings"""
    grid_size: tuple = (64, 64)
    save_interval: float = 30.0  # Auto-save every N seconds
    max_visited_tiles: int = 500
    area_hash_size: int = 8  # Resize dimension for area hashing
    corruption_threshold: int = 50  # Max areas before considering corruption


@dataclass
class BattleConfig:
    """Battle-specific settings"""
    menu_settle_time: float = 0.12
    turn_pace: float = 0.10
    mash_delay: float = 0.12
    switch_settle_time: float = 2.0
    fast_return_threshold: float = 0.98
    fast_return_window: float = 1.5
    move_fail_threshold: int = 2  # Failures before marking move as empty


@dataclass
class ExplorationConfig:
    """Exploration settings"""
    stuck_similarity_threshold: float = 0.992
    stuck_patience: int = 10  # Frames before direction change
    movement_duration: float = 0.18
    interaction_delay: float = 0.05

    # Pathfinding
    pathfinding_grid_size: tuple = (32, 32)


@dataclass
class AgentConfig:
    """Complete agent configuration"""
    paths: PathConfig = field(default_factory=PathConfig)
    emulator: EmulatorConfig = field(default_factory=EmulatorConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    battle: BattleConfig = field(default_factory=BattleConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)

    # Debug settings
    debug_enabled: bool = True
    debug_log_level: str = "INFO"
    debug_save_frames: bool = False
    debug_visualize: bool = False

    def save(self, path: Optional[str] = None):
        """Save configuration to JSON file"""
        if path is None:
            path = get_agent_dir() / "agent_config.json"

        config_dict = asdict(self)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: Optional[str] = None) -> 'AgentConfig':
        """Load configuration from JSON file"""
        if path is None:
            path = get_agent_dir() / "agent_config.json"

        if not Path(path).exists():
            return cls()

        with open(path, 'r') as f:
            config_dict = json.load(f)

        # Reconstruct nested dataclasses
        config = cls()
        if 'paths' in config_dict:
            config.paths = PathConfig(**config_dict['paths'])
        if 'emulator' in config_dict:
            config.emulator = EmulatorConfig(**config_dict['emulator'])
        if 'perception' in config_dict:
            config.perception = PerceptionConfig(**config_dict['perception'])
        if 'reasoning' in config_dict:
            config.reasoning = ReasoningConfig(**config_dict['reasoning'])
        if 'memory' in config_dict:
            config.memory = MemoryConfig(**config_dict['memory'])
        if 'battle' in config_dict:
            config.battle = BattleConfig(**config_dict['battle'])
        if 'exploration' in config_dict:
            config.exploration = ExplorationConfig(**config_dict['exploration'])

        # Copy simple fields
        for key in ['debug_enabled', 'debug_log_level', 'debug_save_frames', 'debug_visualize']:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        return config


# Global configuration instance
_config: Optional[AgentConfig] = None


def get_config() -> AgentConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = AgentConfig.load()
    return _config


def reload_config(path: Optional[str] = None) -> AgentConfig:
    """Reload configuration from file"""
    global _config
    _config = AgentConfig.load(path)
    return _config


def save_config(path: Optional[str] = None):
    """Save current configuration to file"""
    config = get_config()
    config.save(path)


# Convenience accessors
def get_tesseract_cmd() -> str:
    """Get Tesseract command path"""
    return get_config().paths.tesseract_cmd


def get_yolo_model() -> str:
    """Get YOLO model path"""
    return get_config().paths.yolo_model


def get_maps_folder() -> str:
    """Get maps folder path"""
    return get_config().paths.maps_folder


if __name__ == "__main__":
    # Test configuration
    config = get_config()

    print("Agent Configuration")
    print("=" * 50)
    print(f"Tesseract: {config.paths.tesseract_cmd}")
    print(f"YOLO Model: {config.paths.yolo_model}")
    print(f"Maps Folder: {config.paths.maps_folder}")
    print(f"Emulator: {config.emulator.window_title}")
    print(f"OCR Engine: {config.perception.ocr_engine}")
    print(f"OCR GPU: {config.perception.ocr_gpu}")
    print(f"Debug Enabled: {config.debug_enabled}")

    # Save default config
    config.save()
    print(f"\nConfiguration saved to: {get_agent_dir() / 'agent_config.json'}")
