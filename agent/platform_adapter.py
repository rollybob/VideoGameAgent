"""
Platform Adapter System - Making the agent work with any game platform
=======================================================================

This module provides abstraction layers for:
1. Different gaming platforms (consoles, PC, etc.)
2. Different emulators (mGBA, RetroArch, Dolphin, etc.)
3. Different input methods (keyboard, controller, touch)

The goal: write game logic once, run on any platform.

Architecture:
    GameAgent
        |
        v
    PlatformAdapter (abstract)
        |
        +-- GBAAdapter (mGBA, VisualBoyAdvance, etc.)
        +-- SNESAdapter (SNES9x, BSNES, etc.)
        +-- N64Adapter (Project64, Mupen64, etc.)
        +-- PCAdapter (native PC games)
        +-- RetroArchAdapter (multi-platform)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path

from debug_system import info, debug, warning, error


class Platform(Enum):
    """Supported gaming platforms"""
    GBA = "gba"
    GBC = "gbc"
    NES = "nes"
    SNES = "snes"
    N64 = "n64"
    GENESIS = "genesis"
    PS1 = "ps1"
    PS2 = "ps2"
    PC = "pc"
    ANDROID = "android"
    IOS = "ios"
    CUSTOM = "custom"


class EmulatorType(Enum):
    """Supported emulator types"""
    MGBA = "mgba"
    VBA = "visualboyadvance"
    RETROARCH = "retroarch"
    SNES9X = "snes9x"
    BSNES = "bsnes"
    PROJECT64 = "project64"
    MUPEN64 = "mupen64"
    DOLPHIN = "dolphin"
    PCSX2 = "pcsx2"
    DUCKSTATION = "duckstation"
    NATIVE = "native"  # PC games, no emulator
    # Android emulators
    BLUESTACKS = "bluestacks"
    LDPLAYER = "ldplayer"
    NOXPLAYER = "nox"
    MEMU = "memu"


@dataclass
class PlatformProfile:
    """Profile defining a platform's characteristics"""
    platform: Platform
    name: str
    buttons: List[str]
    dpad: bool = True
    analog_sticks: int = 0
    triggers: int = 0
    screen_resolution: Tuple[int, int] = (240, 160)
    aspect_ratio: float = 1.5
    typical_fps: int = 60


@dataclass
class EmulatorProfile:
    """Profile defining an emulator's characteristics"""
    emulator: EmulatorType
    name: str
    platform: Platform
    window_title_pattern: str
    default_keymap: Dict[str, str]
    supports_speedup: bool = False
    supports_savestates: bool = True
    supports_screenshots: bool = True


# Pre-defined platform profiles
PLATFORM_PROFILES = {
    Platform.GBA: PlatformProfile(
        platform=Platform.GBA,
        name="Game Boy Advance",
        buttons=["A", "B", "L", "R", "start", "select"],
        dpad=True,
        screen_resolution=(240, 160),
        aspect_ratio=1.5,
        typical_fps=60
    ),
    Platform.GBC: PlatformProfile(
        platform=Platform.GBC,
        name="Game Boy Color",
        buttons=["A", "B", "start", "select"],
        dpad=True,
        screen_resolution=(160, 144),
        aspect_ratio=1.11,
        typical_fps=60
    ),
    Platform.NES: PlatformProfile(
        platform=Platform.NES,
        name="Nintendo Entertainment System",
        buttons=["A", "B", "start", "select"],
        dpad=True,
        screen_resolution=(256, 240),
        aspect_ratio=1.07,
        typical_fps=60
    ),
    Platform.SNES: PlatformProfile(
        platform=Platform.SNES,
        name="Super Nintendo",
        buttons=["A", "B", "X", "Y", "L", "R", "start", "select"],
        dpad=True,
        screen_resolution=(256, 224),
        aspect_ratio=1.14,
        typical_fps=60
    ),
    Platform.N64: PlatformProfile(
        platform=Platform.N64,
        name="Nintendo 64",
        buttons=["A", "B", "Z", "L", "R", "start", "C_up", "C_down", "C_left", "C_right"],
        dpad=True,
        analog_sticks=1,
        triggers=1,
        screen_resolution=(320, 240),
        aspect_ratio=1.33,
        typical_fps=30
    ),
    Platform.GENESIS: PlatformProfile(
        platform=Platform.GENESIS,
        name="Sega Genesis",
        buttons=["A", "B", "C", "X", "Y", "Z", "start"],  # 6-button
        dpad=True,
        screen_resolution=(320, 224),
        aspect_ratio=1.43,
        typical_fps=60
    ),
}

# Pre-defined emulator profiles
EMULATOR_PROFILES = {
    EmulatorType.MGBA: EmulatorProfile(
        emulator=EmulatorType.MGBA,
        name="mGBA",
        platform=Platform.GBA,
        window_title_pattern="mGBA",
        default_keymap={
            "A": "x",
            "B": "z",
            "L": "a",
            "R": "s",
            "start": "enter",
            "select": "backspace",
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
        },
        supports_speedup=True,
        supports_savestates=True,
    ),
    EmulatorType.RETROARCH: EmulatorProfile(
        emulator=EmulatorType.RETROARCH,
        name="RetroArch",
        platform=Platform.CUSTOM,  # Supports many
        window_title_pattern="RetroArch",
        default_keymap={
            # RetroArch unified controls
            "A": "x",
            "B": "z",
            "X": "s",
            "Y": "a",
            "L": "q",
            "R": "w",
            "start": "enter",
            "select": "rshift",
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
        },
        supports_speedup=True,
        supports_savestates=True,
    ),
    EmulatorType.SNES9X: EmulatorProfile(
        emulator=EmulatorType.SNES9X,
        name="SNES9x",
        platform=Platform.SNES,
        window_title_pattern="Snes9x",
        default_keymap={
            "A": "x",
            "B": "z",
            "X": "s",
            "Y": "a",
            "L": "d",
            "R": "c",
            "start": "enter",
            "select": "space",
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
        },
        supports_speedup=True,
    ),
}


class PlatformAdapter(ABC):
    """
    Abstract base class for platform adapters.

    Each adapter translates generic game commands to platform-specific inputs.
    """

    def __init__(self, platform: Platform, emulator: EmulatorType = None):
        self.platform = platform
        self.emulator = emulator
        self.platform_profile = PLATFORM_PROFILES.get(platform)
        self.emulator_profile = EMULATOR_PROFILES.get(emulator) if emulator else None

        # Custom keymap (can override defaults)
        self.keymap: Dict[str, str] = {}
        self._load_default_keymap()

        # Window connection
        self.window = None
        self.window_title = ""

    def _load_default_keymap(self):
        """Load default keymap from emulator profile"""
        if self.emulator_profile:
            self.keymap = dict(self.emulator_profile.default_keymap)
        else:
            # Generic keymap
            self.keymap = {
                "A": "x",
                "B": "z",
                "start": "enter",
                "select": "space",
                "up": "up",
                "down": "down",
                "left": "left",
                "right": "right",
            }

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the emulator/game window"""
        pass

    @abstractmethod
    def capture_screen(self):
        """Capture the current game screen"""
        pass

    @abstractmethod
    def press_button(self, button: str, duration: float = 0.1) -> bool:
        """Press a game button"""
        pass

    @abstractmethod
    def release_button(self, button: str) -> bool:
        """Release a game button"""
        pass

    def get_available_buttons(self) -> List[str]:
        """Get list of available buttons for this platform"""
        if self.platform_profile:
            buttons = list(self.platform_profile.buttons)
            if self.platform_profile.dpad:
                buttons.extend(["up", "down", "left", "right"])
            return buttons
        return list(self.keymap.keys())

    def get_screen_resolution(self) -> Tuple[int, int]:
        """Get expected screen resolution"""
        if self.platform_profile:
            return self.platform_profile.screen_resolution
        return (256, 224)  # Default

    def set_keymap(self, button: str, key: str):
        """Override a key mapping"""
        self.keymap[button] = key
        debug("platform", f"Remapped {button} -> {key}")

    def load_keymap_from_file(self, path: Path):
        """Load custom keymap from file"""
        try:
            with open(path, 'r') as f:
                custom_keymap = json.load(f)
                self.keymap.update(custom_keymap)
                info("platform", f"Loaded custom keymap from {path}")
        except Exception as e:
            warning("platform", f"Failed to load keymap: {e}")

    def save_keymap_to_file(self, path: Path):
        """Save current keymap to file"""
        try:
            with open(path, 'w') as f:
                json.dump(self.keymap, f, indent=2)
                info("platform", f"Saved keymap to {path}")
        except Exception as e:
            error("platform", f"Failed to save keymap: {e}")


class WindowsPlatformAdapter(PlatformAdapter):
    """
    Platform adapter for Windows using pyautogui/pygetwindow.
    Works with most Windows-based emulators.
    """

    def __init__(self, platform: Platform, emulator: EmulatorType = None):
        super().__init__(platform, emulator)
        self._pyautogui = None
        self._pygetwindow = None

    def _ensure_imports(self):
        """Lazy import Windows-specific modules"""
        if self._pyautogui is None:
            import pyautogui
            import pygetwindow as gw
            self._pyautogui = pyautogui
            self._pygetwindow = gw

    def connect(self) -> bool:
        """Connect to emulator window"""
        self._ensure_imports()

        # Determine window title pattern
        if self.emulator_profile:
            pattern = self.emulator_profile.window_title_pattern
        else:
            pattern = self.window_title or "Game"

        try:
            windows = self._pygetwindow.getWindowsWithTitle(pattern)
            if not windows:
                error("platform", f"No window found matching '{pattern}'")
                return False

            self.window = windows[0]
            self.window_title = self.window.title
            info("platform", f"Connected to: {self.window_title}")
            return True

        except Exception as e:
            error("platform", f"Failed to connect: {e}")
            return False

    def capture_screen(self):
        """Capture emulator screen"""
        self._ensure_imports()
        import cv2
        import numpy as np

        if not self.window:
            return None

        try:
            # Get window bounds
            left, top = self.window.left, self.window.top
            width, height = self.window.width, self.window.height

            if width <= 0 or height <= 0:
                return None

            # Capture
            screenshot = self._pyautogui.screenshot(region=(left, top, width, height))
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            return frame

        except Exception as e:
            debug("platform", f"Screen capture error: {e}")
            return None

    def press_button(self, button: str, duration: float = 0.1) -> bool:
        """Press a button"""
        self._ensure_imports()

        if button not in self.keymap:
            warning("platform", f"Unknown button: {button}")
            return False

        key = self.keymap[button]

        try:
            self._pyautogui.keyDown(key)
            time.sleep(duration)
            self._pyautogui.keyUp(key)
            return True
        except Exception as e:
            error("platform", f"Button press error: {e}")
            return False

    def release_button(self, button: str) -> bool:
        """Release a button"""
        self._ensure_imports()

        if button not in self.keymap:
            return False

        key = self.keymap[button]

        try:
            self._pyautogui.keyUp(key)
            return True
        except Exception as e:
            return False

    def mash_button(self, button: str, count: int = 3, delay: float = 0.1) -> bool:
        """Press button repeatedly"""
        for _ in range(count):
            if not self.press_button(button, 0.05):
                return False
            time.sleep(delay)
        return True


class RetroArchAdapter(WindowsPlatformAdapter):
    """
    Specialized adapter for RetroArch.

    RetroArch is special because:
    - It runs many different cores (platforms)
    - Has unified control scheme
    - Supports hotkeys for savestates, speed, etc.
    """

    # RetroArch hotkeys
    HOTKEYS = {
        'savestate': 'F2',
        'loadstate': 'F4',
        'fast_forward': 'space',
        'pause': 'p',
        'reset': 'h',
        'screenshot': 'F8',
        'next_slot': 'F7',
        'prev_slot': 'F6',
    }

    def __init__(self, core_platform: Platform = Platform.GBA):
        super().__init__(core_platform, EmulatorType.RETROARCH)
        self.core_platform = core_platform

        # Update keymap based on core platform
        if core_platform in PLATFORM_PROFILES:
            # Adjust available buttons
            pass

    def save_state(self, slot: int = None) -> bool:
        """Save a savestate"""
        self._ensure_imports()
        try:
            if slot is not None:
                # Select slot first
                current = 0  # Would need to track this
                while current != slot:
                    self._pyautogui.press(self.HOTKEYS['next_slot'])
                    time.sleep(0.1)
                    current = (current + 1) % 10

            self._pyautogui.press(self.HOTKEYS['savestate'])
            info("platform", f"Saved state to slot {slot or 'current'}")
            return True
        except Exception as e:
            error("platform", f"Save state failed: {e}")
            return False

    def load_state(self, slot: int = None) -> bool:
        """Load a savestate"""
        self._ensure_imports()
        try:
            self._pyautogui.press(self.HOTKEYS['loadstate'])
            info("platform", f"Loaded state from slot {slot or 'current'}")
            return True
        except Exception as e:
            return False

    def set_fast_forward(self, enabled: bool) -> bool:
        """Toggle fast forward"""
        self._ensure_imports()
        try:
            if enabled:
                self._pyautogui.keyDown(self.HOTKEYS['fast_forward'])
            else:
                self._pyautogui.keyUp(self.HOTKEYS['fast_forward'])
            return True
        except Exception as e:
            return False


# =============================================================================
# ADAPTER FACTORY
# =============================================================================

class PlatformAdapterFactory:
    """Factory for creating platform adapters"""

    # Android emulator types for easy checking
    ANDROID_EMULATORS = {
        EmulatorType.BLUESTACKS,
        EmulatorType.LDPLAYER,
        EmulatorType.NOXPLAYER,
        EmulatorType.MEMU,
    }

    @staticmethod
    def create_adapter(
        platform: Platform,
        emulator: EmulatorType = None,
        custom_window_title: str = None,
        game_name: str = None,
        use_adb: bool = True
    ) -> PlatformAdapter:
        """Create appropriate adapter for platform/emulator combination"""

        # Special case for Android/mobile games
        if platform == Platform.ANDROID or emulator in PlatformAdapterFactory.ANDROID_EMULATORS:
            try:
                from mobile_game_adapter import create_mobile_adapter, MobileEmulator
                # Map EmulatorType to MobileEmulator
                mobile_emu_map = {
                    EmulatorType.BLUESTACKS: MobileEmulator.BLUESTACKS,
                    EmulatorType.LDPLAYER: MobileEmulator.LDPLAYER,
                    EmulatorType.NOXPLAYER: MobileEmulator.NOXPLAYER,
                    EmulatorType.MEMU: MobileEmulator.MEMU,
                }
                mobile_emu = mobile_emu_map.get(emulator) if emulator else None
                adapter = create_mobile_adapter(
                    game_name=game_name or custom_window_title,
                    emulator=mobile_emu,
                    use_adb=use_adb
                )
                if adapter:
                    return adapter
            except ImportError:
                warning("platform", "Mobile game adapter not available")
            # Fall through to Windows adapter if mobile adapter fails

        # Special case for PC games (native)
        if platform == Platform.PC or emulator == EmulatorType.NATIVE:
            try:
                from pc_game_adapter import create_pc_adapter
                adapter = create_pc_adapter(
                    game_name=game_name or custom_window_title,
                    auto_detect=True
                )
                if adapter:
                    return adapter
            except ImportError:
                warning("platform", "PC game adapter not available")
            # Fall through to Windows adapter if PC adapter fails

        # Special case for RetroArch
        if emulator == EmulatorType.RETROARCH:
            adapter = RetroArchAdapter(platform)
        else:
            # Default to Windows adapter
            adapter = WindowsPlatformAdapter(platform, emulator)

        # Set custom window title if provided
        if custom_window_title:
            adapter.window_title = custom_window_title

        return adapter

    @staticmethod
    def auto_detect() -> Optional[PlatformAdapter]:
        """Try to auto-detect running emulator, PC game, or mobile emulator"""
        try:
            import pygetwindow as gw

            # First check for known console emulators
            for emulator_type, profile in EMULATOR_PROFILES.items():
                windows = gw.getWindowsWithTitle(profile.window_title_pattern)
                if windows:
                    info("platform", f"Auto-detected emulator: {profile.name}")
                    adapter = PlatformAdapterFactory.create_adapter(
                        profile.platform,
                        emulator_type
                    )
                    return adapter

            # Check for Android emulators
            try:
                from mobile_game_adapter import MobileGameProfileManager, create_mobile_adapter
                mobile_manager = MobileGameProfileManager()
                detected_emu = mobile_manager.detect_running_emulator()
                if detected_emu:
                    info("platform", f"Auto-detected Android emulator: {detected_emu.value}")
                    adapter = create_mobile_adapter(emulator=detected_emu)
                    if adapter and adapter.connect():
                        return adapter
            except ImportError:
                debug("platform", "Mobile game adapter not available for auto-detect")

            # Then check for known PC games
            try:
                from pc_game_adapter import PCGameProfileManager, create_pc_adapter
                pc_manager = PCGameProfileManager()
                detected_game = pc_manager.detect_running_game()
                if detected_game:
                    info("platform", f"Auto-detected PC game: {detected_game.game_name}")
                    adapter = create_pc_adapter(detected_game.game_name)
                    return adapter
            except ImportError:
                debug("platform", "PC game adapter not available for auto-detect")

            warning("platform", "No known emulator, PC game, or mobile emulator detected")
            return None

        except Exception as e:
            error("platform", f"Auto-detect failed: {e}")
            return None

    @staticmethod
    def list_supported() -> Dict[str, Any]:
        """List all supported platforms and emulators"""
        result = {
            'platforms': [p.value for p in Platform],
            'emulators': [e.value for e in EmulatorType],
            'platform_profiles': {
                p.value: {
                    'name': profile.name,
                    'buttons': profile.buttons,
                    'resolution': profile.screen_resolution,
                }
                for p, profile in PLATFORM_PROFILES.items()
            },
            'emulator_profiles': {
                e.value: {
                    'name': profile.name,
                    'platform': profile.platform.value,
                    'window_pattern': profile.window_title_pattern,
                }
                for e, profile in EMULATOR_PROFILES.items()
            },
        }

        # Add PC games if available
        try:
            from pc_game_adapter import PCGameProfileManager
            pc_manager = PCGameProfileManager()
            result['pc_games'] = {
                game_id: {
                    'name': profile.game_name,
                    'type': profile.game_type.value,
                    'genre': profile.genre,
                    'steam_app_id': profile.steam_app_id,
                }
                for game_id, profile in pc_manager.profiles.items()
            }
        except ImportError:
            pass

        # Add mobile games if available
        try:
            from mobile_game_adapter import MobileGameProfileManager, MOBILE_EMULATOR_PROFILES
            mobile_manager = MobileGameProfileManager()
            result['mobile_games'] = {
                game_id: {
                    'name': profile.game_name,
                    'package': profile.package_name,
                    'emulator': profile.preferred_emulator.value,
                    'genre': profile.genre,
                }
                for game_id, profile in mobile_manager.profiles.items()
            }
            result['android_emulators'] = {
                emu.value: {
                    'name': config['name'],
                    'adb_port': config['adb_port'],
                }
                for emu, config in MOBILE_EMULATOR_PROFILES.items()
            }
        except ImportError:
            pass

        return result


# =============================================================================
# GAME PROFILE SYSTEM
# =============================================================================

@dataclass
class GameProfile:
    """
    Profile for a specific game, including:
    - Platform/emulator settings
    - Custom keymaps
    - Game-specific behaviors
    - Known patterns and rules
    """
    game_id: str
    game_name: str
    platform: Platform
    emulator: EmulatorType = None
    genre: str = ""
    custom_keymap: Dict[str, str] = field(default_factory=dict)
    behavior_overrides: Dict[str, Any] = field(default_factory=dict)
    known_screens: List[str] = field(default_factory=list)
    notes: str = ""


class GameProfileManager:
    """Manages game profiles for quick setup"""

    def __init__(self, profiles_dir: Path = None):
        self.profiles_dir = profiles_dir or Path("game_profiles")
        self.profiles: Dict[str, GameProfile] = {}
        self._load_profiles()

    def _load_profiles(self):
        """Load all game profiles from disk"""
        if not self.profiles_dir.exists():
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            return

        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                    profile = GameProfile(
                        game_id=data['game_id'],
                        game_name=data['game_name'],
                        platform=Platform(data['platform']),
                        emulator=EmulatorType(data['emulator']) if data.get('emulator') else None,
                        genre=data.get('genre', ''),
                        custom_keymap=data.get('custom_keymap', {}),
                        behavior_overrides=data.get('behavior_overrides', {}),
                        known_screens=data.get('known_screens', []),
                        notes=data.get('notes', ''),
                    )
                    self.profiles[profile.game_id] = profile

            except Exception as e:
                warning("platform", f"Failed to load profile {profile_file}: {e}")

        info("platform", f"Loaded {len(self.profiles)} game profiles")

    def save_profile(self, profile: GameProfile):
        """Save a game profile to disk"""
        self.profiles[profile.game_id] = profile

        profile_path = self.profiles_dir / f"{profile.game_id}.json"
        try:
            data = {
                'game_id': profile.game_id,
                'game_name': profile.game_name,
                'platform': profile.platform.value,
                'emulator': profile.emulator.value if profile.emulator else None,
                'genre': profile.genre,
                'custom_keymap': profile.custom_keymap,
                'behavior_overrides': profile.behavior_overrides,
                'known_screens': profile.known_screens,
                'notes': profile.notes,
            }
            with open(profile_path, 'w') as f:
                json.dump(data, f, indent=2)

            info("platform", f"Saved profile: {profile.game_name}")

        except Exception as e:
            error("platform", f"Failed to save profile: {e}")

    def get_profile(self, game_id: str) -> Optional[GameProfile]:
        """Get a game profile by ID"""
        return self.profiles.get(game_id)

    def create_profile(self, game_name: str, platform: Platform,
                      emulator: EmulatorType = None, genre: str = "") -> GameProfile:
        """Create a new game profile"""
        # Generate ID from name
        game_id = game_name.lower().replace(" ", "_").replace(":", "")

        profile = GameProfile(
            game_id=game_id,
            game_name=game_name,
            platform=platform,
            emulator=emulator,
            genre=genre,
        )

        self.save_profile(profile)
        return profile

    def list_profiles(self) -> List[str]:
        """List all available profile IDs"""
        return list(self.profiles.keys())


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Platform Adapter System Test")
    print("=" * 50)

    # List supported platforms
    supported = PlatformAdapterFactory.list_supported()
    print("\nSupported Platforms:")
    for platform, info in supported['platform_profiles'].items():
        print(f"  {platform}: {info['name']} ({info['buttons']})")

    print("\nSupported Emulators:")
    for emu, info in supported['emulator_profiles'].items():
        print(f"  {emu}: {info['name']} (platform: {info['platform']})")

    # Test adapter creation
    print("\nCreating GBA adapter for mGBA:")
    adapter = PlatformAdapterFactory.create_adapter(Platform.GBA, EmulatorType.MGBA)
    print(f"  Buttons: {adapter.get_available_buttons()}")
    print(f"  Resolution: {adapter.get_screen_resolution()}")
    print(f"  Keymap: {adapter.keymap}")

    # Test game profile
    print("\nTesting Game Profile Manager:")
    manager = GameProfileManager(Path("test_profiles"))
    profile = manager.create_profile(
        "Pokemon Fire Red",
        Platform.GBA,
        EmulatorType.MGBA,
        "rpg"
    )
    print(f"  Created profile: {profile.game_name} ({profile.game_id})")

    # Try auto-detect
    print("\nAttempting auto-detect:")
    detected = PlatformAdapterFactory.auto_detect()
    if detected:
        print(f"  Detected: {detected.window_title}")
    else:
        print("  No emulator detected (this is expected if none running)")

    print("\nPlatform Adapter System Test Complete!")
