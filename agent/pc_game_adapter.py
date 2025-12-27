"""
PC Game Adapter - Support for native PC games (Steam, GOG, Epic, etc.)
=======================================================================

PC games present unique challenges compared to console emulation:

1. WINDOW MANAGEMENT
   - Games can be windowed, borderless, or fullscreen
   - Window titles vary (may include FPS counter, version, etc.)
   - Some games capture mouse exclusively

2. INPUT METHODS
   - Keyboard + Mouse (most common)
   - Controller support (XInput, DirectInput)
   - Mixed input (WASD + mouse aim)
   - Rebindable keys (need to detect or configure)

3. RESOLUTION & SCALING
   - Any resolution possible
   - UI scaling varies
   - Different aspect ratios

4. ANTI-CHEAT CONSIDERATIONS
   - Some games block input simulation
   - May need hardware-level input (Arduino, etc.)
   - Single-player games generally safe

5. LAUNCHER INTEGRATION
   - Steam overlay
   - GOG Galaxy
   - Epic Games launcher
   - Direct launch vs launcher

This module provides adapters for handling these PC-specific challenges.
"""

import time
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from debug_system import info, debug, warning, error
from platform_adapter import PlatformAdapter, Platform, EmulatorType


class PCGameType(Enum):
    """Types of PC games by input method"""
    KEYBOARD_ONLY = "keyboard_only"        # Puzzle games, some RPGs
    KEYBOARD_MOUSE = "keyboard_mouse"      # FPS, RTS, most modern games
    CONTROLLER_NATIVE = "controller"       # Console ports with controller support
    CONTROLLER_PREFERRED = "controller_preferred"  # Works best with controller
    POINT_AND_CLICK = "point_and_click"    # Adventure games
    TURN_BASED = "turn_based"              # Strategy, card games


class WindowMode(Enum):
    """Window modes for PC games"""
    WINDOWED = "windowed"
    BORDERLESS = "borderless"
    FULLSCREEN = "fullscreen"
    UNKNOWN = "unknown"


@dataclass
class PCGameProfile:
    """Profile for a specific PC game"""
    game_id: str
    game_name: str
    executable: str = ""                   # e.g., "game.exe"
    window_title_pattern: str = ""         # For window detection
    launcher: str = ""                     # steam, gog, epic, direct
    steam_app_id: str = ""                 # For Steam games

    game_type: PCGameType = PCGameType.KEYBOARD_MOUSE
    genre: str = ""

    # Input configuration
    default_keymap: Dict[str, str] = field(default_factory=dict)
    uses_mouse: bool = True
    mouse_sensitivity: float = 1.0
    controller_support: bool = False

    # Window settings
    preferred_resolution: Tuple[int, int] = (1920, 1080)
    preferred_window_mode: WindowMode = WindowMode.WINDOWED

    # Game-specific settings
    ui_scale: float = 1.0
    menu_navigation: str = "keyboard"      # keyboard, mouse, or both

    # Known UI regions (for faster perception)
    ui_regions: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)

    notes: str = ""


# Common PC game control schemes
COMMON_PC_KEYMAPS = {
    'wasd_mouse': {
        'move_forward': 'w',
        'move_backward': 's',
        'move_left': 'a',
        'move_right': 'd',
        'jump': 'space',
        'crouch': 'ctrl',
        'sprint': 'shift',
        'interact': 'e',
        'inventory': 'i',
        'map': 'm',
        'pause': 'escape',
        'primary_action': 'mouse_left',
        'secondary_action': 'mouse_right',
    },
    'arrow_keys': {
        'move_up': 'up',
        'move_down': 'down',
        'move_left': 'left',
        'move_right': 'right',
        'action': 'z',
        'cancel': 'x',
        'menu': 'escape',
        'confirm': 'enter',
    },
    'rpg_standard': {
        'move_up': 'w',
        'move_down': 's',
        'move_left': 'a',
        'move_right': 'd',
        'interact': 'e',
        'attack': 'mouse_left',
        'skill_1': '1',
        'skill_2': '2',
        'skill_3': '3',
        'skill_4': '4',
        'inventory': 'i',
        'character': 'c',
        'map': 'm',
        'quest_log': 'j',
        'pause': 'escape',
    },
    'point_and_click': {
        'interact': 'mouse_left',
        'examine': 'mouse_right',
        'inventory': 'i',
        'pause': 'escape',
        'skip_dialogue': 'space',
    },
    'controller_xbox': {
        'move': 'left_stick',
        'camera': 'right_stick',
        'action_a': 'a',
        'action_b': 'b',
        'action_x': 'x',
        'action_y': 'y',
        'left_bumper': 'lb',
        'right_bumper': 'rb',
        'left_trigger': 'lt',
        'right_trigger': 'rt',
        'start': 'start',
        'back': 'back',
        'dpad_up': 'dpad_up',
        'dpad_down': 'dpad_down',
        'dpad_left': 'dpad_left',
        'dpad_right': 'dpad_right',
    },
}


class PCGameAdapter(PlatformAdapter):
    """
    Adapter for native PC games.

    Handles:
    - Window detection and management
    - Keyboard + mouse input
    - Controller input (via XInput or virtual controller)
    - Screen capture in various modes
    """

    def __init__(self, game_profile: PCGameProfile = None):
        super().__init__(Platform.PC, EmulatorType.NATIVE)

        self.game_profile = game_profile
        self.game_type = game_profile.game_type if game_profile else PCGameType.KEYBOARD_MOUSE

        # Input systems
        self._pyautogui = None
        self._pynput_keyboard = None
        self._pynput_mouse = None
        self._controller = None

        # Window state
        self.window_mode = WindowMode.UNKNOWN
        self.actual_resolution = (0, 0)

        # Mouse state
        self.mouse_captured = False
        self.last_mouse_pos = (0, 0)

        # Load appropriate keymap
        self._setup_keymap()

    def _setup_keymap(self):
        """Set up keymap based on game profile or type"""
        if self.game_profile and self.game_profile.default_keymap:
            self.keymap = dict(self.game_profile.default_keymap)
        elif self.game_type == PCGameType.KEYBOARD_MOUSE:
            self.keymap = dict(COMMON_PC_KEYMAPS['wasd_mouse'])
        elif self.game_type == PCGameType.KEYBOARD_ONLY:
            self.keymap = dict(COMMON_PC_KEYMAPS['arrow_keys'])
        elif self.game_type == PCGameType.POINT_AND_CLICK:
            self.keymap = dict(COMMON_PC_KEYMAPS['point_and_click'])
        elif self.game_type in [PCGameType.CONTROLLER_NATIVE, PCGameType.CONTROLLER_PREFERRED]:
            self.keymap = dict(COMMON_PC_KEYMAPS['controller_xbox'])
        else:
            self.keymap = dict(COMMON_PC_KEYMAPS['wasd_mouse'])

    def _ensure_imports(self):
        """Lazy import input libraries"""
        if self._pyautogui is None:
            import pyautogui
            import pygetwindow as gw
            self._pyautogui = pyautogui
            self._pygetwindow = gw

            # Disable pyautogui's fail-safe for smoother operation
            # (move mouse to corner to stop won't work, but agent is controllable)
            pyautogui.FAILSAFE = False

    def _init_pynput(self):
        """Initialize pynput for more reliable input"""
        if self._pynput_keyboard is None:
            try:
                from pynput import keyboard, mouse
                self._pynput_keyboard = keyboard.Controller()
                self._pynput_mouse = mouse.Controller()
                debug("pc_adapter", "pynput controllers initialized")
            except ImportError:
                warning("pc_adapter", "pynput not available, using pyautogui only")

    def connect(self) -> bool:
        """Connect to game window"""
        self._ensure_imports()

        # Determine window title to search for
        if self.game_profile and self.game_profile.window_title_pattern:
            pattern = self.game_profile.window_title_pattern
        elif self.game_profile:
            pattern = self.game_profile.game_name
        else:
            pattern = ""

        if not pattern:
            error("pc_adapter", "No window pattern specified")
            return False

        try:
            windows = self._pygetwindow.getWindowsWithTitle(pattern)

            if not windows:
                # Try partial match
                all_windows = self._pygetwindow.getAllWindows()
                pattern_lower = pattern.lower()
                windows = [w for w in all_windows
                          if pattern_lower in w.title.lower() and w.width > 100]

            if not windows:
                error("pc_adapter", f"No window found matching '{pattern}'")
                return False

            # Pick the largest matching window (likely the main game window)
            self.window = max(windows, key=lambda w: w.width * w.height)
            self.window_title = self.window.title

            # Detect window mode
            self._detect_window_mode()

            info("pc_adapter", f"Connected to: {self.window_title}")
            info("pc_adapter", f"Resolution: {self.window.width}x{self.window.height}")
            info("pc_adapter", f"Window mode: {self.window_mode.value}")

            return True

        except Exception as e:
            error("pc_adapter", f"Failed to connect: {e}")
            return False

    def _detect_window_mode(self):
        """Detect if game is windowed, borderless, or fullscreen"""
        if not self.window:
            return

        try:
            import ctypes
            user32 = ctypes.windll.user32

            screen_width = user32.GetSystemMetrics(0)
            screen_height = user32.GetSystemMetrics(1)

            # Check if window covers full screen
            if (self.window.width >= screen_width and
                self.window.height >= screen_height):
                # Could be fullscreen or borderless
                if self.window.left == 0 and self.window.top == 0:
                    self.window_mode = WindowMode.BORDERLESS
                else:
                    self.window_mode = WindowMode.FULLSCREEN
            else:
                self.window_mode = WindowMode.WINDOWED

            self.actual_resolution = (self.window.width, self.window.height)

        except Exception:
            self.window_mode = WindowMode.UNKNOWN

    def capture_screen(self):
        """Capture game screen"""
        self._ensure_imports()
        import cv2
        import numpy as np

        if not self.window:
            return None

        try:
            # Different capture strategies based on window mode
            if self.window_mode == WindowMode.FULLSCREEN:
                # For true fullscreen, capture entire screen
                screenshot = self._pyautogui.screenshot()
            else:
                # For windowed/borderless, capture window region
                left, top = self.window.left, self.window.top
                width, height = self.window.width, self.window.height

                # Clamp to valid coordinates
                left = max(0, left)
                top = max(0, top)

                if width <= 0 or height <= 0:
                    return None

                screenshot = self._pyautogui.screenshot(region=(left, top, width, height))

            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            return frame

        except Exception as e:
            debug("pc_adapter", f"Screen capture error: {e}")
            return None

    def press_button(self, button: str, duration: float = 0.1) -> bool:
        """Press a keyboard key or mouse button"""
        self._ensure_imports()

        # Handle mouse buttons
        if button.startswith('mouse_'):
            return self._press_mouse_button(button, duration)

        # Handle controller buttons (if controller mode)
        if button in ['a', 'b', 'x', 'y', 'lb', 'rb', 'lt', 'rt', 'start', 'back']:
            if self._controller:
                return self._press_controller_button(button, duration)
            else:
                # Map to keyboard equivalent
                controller_to_key = {
                    'a': 'space', 'b': 'escape', 'x': 'e', 'y': 'r',
                    'lb': 'q', 'rb': 'e', 'start': 'escape', 'back': 'tab'
                }
                button = controller_to_key.get(button, button)

        # Get key from keymap if it's a logical button
        key = self.keymap.get(button, button)

        try:
            self._pyautogui.keyDown(key)
            time.sleep(duration)
            self._pyautogui.keyUp(key)
            return True
        except Exception as e:
            error("pc_adapter", f"Key press error: {e}")
            return False

    def release_button(self, button: str) -> bool:
        """Release a keyboard key"""
        self._ensure_imports()

        key = self.keymap.get(button, button)

        try:
            self._pyautogui.keyUp(key)
            return True
        except Exception:
            return False

    def _press_mouse_button(self, button: str, duration: float) -> bool:
        """Press a mouse button"""
        button_map = {
            'mouse_left': 'left',
            'mouse_right': 'right',
            'mouse_middle': 'middle',
        }

        mouse_button = button_map.get(button, 'left')

        try:
            self._pyautogui.mouseDown(button=mouse_button)
            time.sleep(duration)
            self._pyautogui.mouseUp(button=mouse_button)
            return True
        except Exception as e:
            error("pc_adapter", f"Mouse button error: {e}")
            return False

    def _press_controller_button(self, button: str, duration: float) -> bool:
        """Press a controller button (requires vgamepad or similar)"""
        # This would require a virtual controller library
        # For now, return False to fall back to keyboard
        warning("pc_adapter", "Controller input not yet implemented")
        return False

    def move_mouse(self, x: int, y: int, relative: bool = False) -> bool:
        """Move mouse to position"""
        self._ensure_imports()

        try:
            if relative:
                self._pyautogui.moveRel(x, y)
            else:
                # Convert to screen coordinates if windowed
                if self.window and self.window_mode == WindowMode.WINDOWED:
                    x += self.window.left
                    y += self.window.top
                self._pyautogui.moveTo(x, y)

            self.last_mouse_pos = self._pyautogui.position()
            return True
        except Exception as e:
            error("pc_adapter", f"Mouse move error: {e}")
            return False

    def click(self, x: int = None, y: int = None, button: str = 'left') -> bool:
        """Click at position (or current position if not specified)"""
        self._ensure_imports()

        try:
            if x is not None and y is not None:
                # Convert to screen coordinates if windowed
                if self.window and self.window_mode == WindowMode.WINDOWED:
                    x += self.window.left
                    y += self.window.top
                self._pyautogui.click(x, y, button=button)
            else:
                self._pyautogui.click(button=button)
            return True
        except Exception as e:
            error("pc_adapter", f"Click error: {e}")
            return False

    def type_text(self, text: str, interval: float = 0.05) -> bool:
        """Type text (for text input fields)"""
        self._ensure_imports()

        try:
            self._pyautogui.typewrite(text, interval=interval)
            return True
        except Exception as e:
            error("pc_adapter", f"Type error: {e}")
            return False

    def scroll(self, amount: int) -> bool:
        """Scroll mouse wheel"""
        self._ensure_imports()

        try:
            self._pyautogui.scroll(amount)
            return True
        except Exception:
            return False

    def focus_window(self) -> bool:
        """Bring game window to foreground"""
        if not self.window:
            return False

        try:
            self.window.activate()
            time.sleep(0.1)  # Brief pause for window to focus
            return True
        except Exception as e:
            warning("pc_adapter", f"Could not focus window: {e}")
            return False

    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position relative to game window"""
        self._ensure_imports()

        pos = self._pyautogui.position()

        if self.window and self.window_mode == WindowMode.WINDOWED:
            # Convert to window-relative coordinates
            return (pos[0] - self.window.left, pos[1] - self.window.top)

        return pos


class SteamGameAdapter(PCGameAdapter):
    """
    Specialized adapter for Steam games.

    Additional features:
    - Steam overlay detection
    - Achievement popup handling
    - Steam input API awareness
    - Launch via Steam protocol
    """

    STEAM_OVERLAY_HOTKEY = 'shift+tab'

    def __init__(self, game_profile: PCGameProfile = None, steam_app_id: str = None):
        super().__init__(game_profile)
        self.steam_app_id = steam_app_id or (game_profile.steam_app_id if game_profile else None)
        self.overlay_open = False

    def launch_game(self) -> bool:
        """Launch game via Steam"""
        if not self.steam_app_id:
            warning("steam_adapter", "No Steam App ID specified")
            return False

        try:
            import subprocess
            steam_url = f"steam://rungameid/{self.steam_app_id}"
            subprocess.Popen(['start', steam_url], shell=True)
            info("steam_adapter", f"Launching Steam game {self.steam_app_id}")
            return True
        except Exception as e:
            error("steam_adapter", f"Failed to launch game: {e}")
            return False

    def toggle_overlay(self) -> bool:
        """Toggle Steam overlay"""
        self._ensure_imports()

        try:
            self._pyautogui.hotkey('shift', 'tab')
            self.overlay_open = not self.overlay_open
            return True
        except Exception:
            return False

    def detect_overlay(self, frame) -> bool:
        """Detect if Steam overlay is open"""
        # Steam overlay has a distinctive dark semi-transparent background
        # This is a simple heuristic - could be improved with template matching
        if frame is None:
            return False

        import cv2
        import numpy as np

        # Check for dark overlay (Steam's characteristic color)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark_pixels = np.sum(gray < 50)
        total_pixels = gray.size

        # If more than 40% is very dark, overlay might be open
        if dark_pixels / total_pixels > 0.4:
            return True

        return False

    def wait_for_game_ready(self, timeout: float = 60.0) -> bool:
        """Wait for game to be ready after launch"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.connect():
                # Wait a bit more for game to fully load
                time.sleep(2.0)
                return True
            time.sleep(1.0)

        return False


# =============================================================================
# PC GAME PROFILE MANAGER
# =============================================================================

class PCGameProfileManager:
    """Manages profiles for PC games"""

    # Pre-defined profiles for popular games
    KNOWN_GAMES = {
        'hollow_knight': PCGameProfile(
            game_id='hollow_knight',
            game_name='Hollow Knight',
            window_title_pattern='Hollow Knight',
            steam_app_id='367520',
            game_type=PCGameType.CONTROLLER_PREFERRED,
            genre='metroidvania',
            default_keymap={
                'move_left': 'left',
                'move_right': 'right',
                'jump': 'z',
                'attack': 'x',
                'dash': 'c',
                'focus': 'a',
                'dream_nail': 's',
                'quick_cast': 'd',
                'map': 'tab',
                'inventory': 'i',
                'pause': 'escape',
            },
            uses_mouse=False,
            controller_support=True,
        ),
        'stardew_valley': PCGameProfile(
            game_id='stardew_valley',
            game_name='Stardew Valley',
            window_title_pattern='Stardew Valley',
            steam_app_id='413150',
            game_type=PCGameType.KEYBOARD_MOUSE,
            genre='simulation',
            default_keymap={
                'move_up': 'w',
                'move_down': 's',
                'move_left': 'a',
                'move_right': 'd',
                'use_tool': 'mouse_left',
                'check': 'mouse_right',
                'menu': 'escape',
                'inventory': 'e',
                'journal': 'f',
                'map': 'm',
                'crafting': 'c',
            },
            controller_support=True,
        ),
        'undertale': PCGameProfile(
            game_id='undertale',
            game_name='UNDERTALE',
            window_title_pattern='UNDERTALE',
            steam_app_id='391540',
            game_type=PCGameType.KEYBOARD_ONLY,
            genre='rpg',
            default_keymap={
                'move_up': 'up',
                'move_down': 'down',
                'move_left': 'left',
                'move_right': 'right',
                'confirm': 'z',
                'cancel': 'x',
                'menu': 'c',
            },
            uses_mouse=False,
        ),
        'celeste': PCGameProfile(
            game_id='celeste',
            game_name='Celeste',
            window_title_pattern='Celeste',
            steam_app_id='504230',
            game_type=PCGameType.CONTROLLER_PREFERRED,
            genre='platformer',
            default_keymap={
                'move_left': 'left',
                'move_right': 'right',
                'move_up': 'up',
                'move_down': 'down',
                'jump': 'c',
                'dash': 'x',
                'grab': 'z',
                'pause': 'escape',
            },
            uses_mouse=False,
            controller_support=True,
        ),
    }

    def __init__(self, profiles_dir: Path = None):
        self.profiles_dir = profiles_dir or Path("pc_game_profiles")
        self.profiles: Dict[str, PCGameProfile] = dict(self.KNOWN_GAMES)
        self._load_custom_profiles()

    def _load_custom_profiles(self):
        """Load custom profiles from disk"""
        if not self.profiles_dir.exists():
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            return

        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                    profile = PCGameProfile(
                        game_id=data['game_id'],
                        game_name=data['game_name'],
                        window_title_pattern=data.get('window_title_pattern', ''),
                        steam_app_id=data.get('steam_app_id', ''),
                        game_type=PCGameType(data.get('game_type', 'keyboard_mouse')),
                        genre=data.get('genre', ''),
                        default_keymap=data.get('default_keymap', {}),
                        uses_mouse=data.get('uses_mouse', True),
                        controller_support=data.get('controller_support', False),
                    )
                    self.profiles[profile.game_id] = profile
            except Exception as e:
                warning("pc_profiles", f"Failed to load {profile_file}: {e}")

    def get_profile(self, game_id: str) -> Optional[PCGameProfile]:
        """Get a game profile"""
        return self.profiles.get(game_id)

    def create_profile(self, game_name: str, window_pattern: str = None,
                      game_type: PCGameType = PCGameType.KEYBOARD_MOUSE,
                      genre: str = "") -> PCGameProfile:
        """Create a new game profile"""
        game_id = game_name.lower().replace(" ", "_").replace(":", "")

        profile = PCGameProfile(
            game_id=game_id,
            game_name=game_name,
            window_title_pattern=window_pattern or game_name,
            game_type=game_type,
            genre=genre,
        )

        self.profiles[game_id] = profile
        self._save_profile(profile)

        return profile

    def _save_profile(self, profile: PCGameProfile):
        """Save profile to disk"""
        profile_path = self.profiles_dir / f"{profile.game_id}.json"

        try:
            data = {
                'game_id': profile.game_id,
                'game_name': profile.game_name,
                'window_title_pattern': profile.window_title_pattern,
                'steam_app_id': profile.steam_app_id,
                'game_type': profile.game_type.value,
                'genre': profile.genre,
                'default_keymap': profile.default_keymap,
                'uses_mouse': profile.uses_mouse,
                'controller_support': profile.controller_support,
            }
            with open(profile_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            error("pc_profiles", f"Failed to save profile: {e}")

    def detect_running_game(self) -> Optional[PCGameProfile]:
        """Try to detect which known game is running"""
        try:
            import pygetwindow as gw

            all_windows = gw.getAllWindows()

            for profile in self.profiles.values():
                pattern = profile.window_title_pattern.lower()
                for window in all_windows:
                    if pattern in window.title.lower() and window.width > 100:
                        info("pc_profiles", f"Detected game: {profile.game_name}")
                        return profile

            return None

        except Exception:
            return None


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_pc_adapter(game_name: str = None,
                      steam_app_id: str = None,
                      auto_detect: bool = True) -> Optional[PCGameAdapter]:
    """
    Create appropriate PC game adapter.

    Args:
        game_name: Name of game to look for
        steam_app_id: Steam App ID for Steam games
        auto_detect: Try to auto-detect running game

    Returns:
        Configured PCGameAdapter or SteamGameAdapter
    """
    manager = PCGameProfileManager()
    profile = None

    # Try to find profile
    if game_name:
        game_id = game_name.lower().replace(" ", "_")
        profile = manager.get_profile(game_id)

    if not profile and auto_detect:
        profile = manager.detect_running_game()

    if not profile and game_name:
        # Create basic profile
        profile = manager.create_profile(game_name)

    if not profile:
        warning("pc_adapter", "Could not determine game profile")
        return None

    # Create appropriate adapter
    if profile.steam_app_id or steam_app_id:
        adapter = SteamGameAdapter(profile, steam_app_id or profile.steam_app_id)
    else:
        adapter = PCGameAdapter(profile)

    return adapter


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("PC Game Adapter Test")
    print("=" * 50)

    # List known games
    manager = PCGameProfileManager()
    print("\nKnown PC Games:")
    for game_id, profile in manager.profiles.items():
        print(f"  {profile.game_name} ({profile.game_type.value})")
        print(f"    Keys: {list(profile.default_keymap.keys())[:5]}...")

    # Test auto-detect
    print("\nTrying to detect running game...")
    detected = manager.detect_running_game()
    if detected:
        print(f"  Detected: {detected.game_name}")

        # Try to connect
        adapter = create_pc_adapter(detected.game_name)
        if adapter and adapter.connect():
            print(f"  Connected to window: {adapter.window_title}")
            print(f"  Resolution: {adapter.actual_resolution}")
            print(f"  Window mode: {adapter.window_mode.value}")
    else:
        print("  No known game detected (this is expected if none running)")

    print("\nPC Game Adapter Test Complete!")
