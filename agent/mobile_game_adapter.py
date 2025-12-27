"""
Mobile Game Adapter - Support for Android/iOS games via emulators
==================================================================

Mobile games via emulators present unique challenges:

1. EMULATOR TYPES
   - BlueStacks (most popular, good performance)
   - LDPlayer (lightweight, good for older PCs)
   - NoxPlayer (feature-rich)
   - MEmu (good compatibility)
   - Android Studio Emulator (official, slow)
   - Genymotion (developer-focused)

2. INPUT METHODS
   - Touch emulation via mouse clicks
   - Virtual joystick/dpad overlays
   - Keyboard mapping (emulator-specific)
   - Gesture simulation (swipe, pinch, multi-touch)
   - ADB (Android Debug Bridge) for direct input

3. SCREEN CONSIDERATIONS
   - Variable resolutions and aspect ratios
   - Portrait vs landscape orientation
   - Notch/cutout handling
   - UI scaling differences

4. SPECIAL FEATURES
   - ADB for advanced control
   - Macro recording/playback
   - Multi-instance support
   - GPS spoofing (some games)

5. GAME TYPES
   - Gacha/RPGs (touch + drag)
   - Action games (virtual joystick)
   - Puzzle games (tap/swipe)
   - Idle games (minimal input)
   - Rhythm games (precise timing)
"""

import time
import subprocess
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from debug_system import info, debug, warning, error
from platform_adapter import PlatformAdapter, Platform, EmulatorType


class MobileEmulator(Enum):
    """Supported Android emulators"""
    BLUESTACKS = "bluestacks"
    BLUESTACKS_5 = "bluestacks5"
    LDPLAYER = "ldplayer"
    NOXPLAYER = "nox"
    MEMU = "memu"
    ANDROID_STUDIO = "android_studio"
    GENYMOTION = "genymotion"
    WAYDROID = "waydroid"  # Linux native


class MobileInputMethod(Enum):
    """Input methods for mobile games"""
    TOUCH = "touch"              # Mouse click = tap
    VIRTUAL_JOYSTICK = "joystick"  # Emulator's virtual controls
    KEYBOARD_MAP = "keyboard"    # Emulator's key mapping
    ADB = "adb"                  # Android Debug Bridge
    HYBRID = "hybrid"            # Combination


class TouchGesture(Enum):
    """Touch gesture types"""
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    LONG_PRESS = "long_press"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    PINCH_IN = "pinch_in"
    PINCH_OUT = "pinch_out"
    DRAG = "drag"


@dataclass
class MobileGameProfile:
    """Profile for a mobile game"""
    game_id: str
    game_name: str
    package_name: str = ""           # Android package (com.company.game)

    # Emulator settings
    preferred_emulator: MobileEmulator = MobileEmulator.BLUESTACKS
    preferred_resolution: Tuple[int, int] = (1280, 720)
    orientation: str = "landscape"    # portrait or landscape

    # Input configuration
    input_method: MobileInputMethod = MobileInputMethod.TOUCH
    keyboard_map: Dict[str, str] = field(default_factory=dict)
    virtual_joystick_position: Tuple[int, int] = (100, 400)  # Screen coords

    # Touch regions (for common UI elements)
    ui_regions: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)

    # Game-specific
    genre: str = ""
    has_auto_play: bool = False
    has_skip_button: bool = False

    notes: str = ""


# Emulator profiles with window patterns and default settings
MOBILE_EMULATOR_PROFILES = {
    MobileEmulator.BLUESTACKS: {
        "name": "BlueStacks",
        "window_patterns": ["BlueStacks", "BlueStacks App Player"],
        "adb_port": 5555,
        "default_resolution": (1600, 900),
        "keyboard_support": True,
        "multi_instance": True,
    },
    MobileEmulator.BLUESTACKS_5: {
        "name": "BlueStacks 5",
        "window_patterns": ["BlueStacks 5", "BlueStacks App Player"],
        "adb_port": 5555,
        "default_resolution": (1600, 900),
        "keyboard_support": True,
        "multi_instance": True,
    },
    MobileEmulator.LDPLAYER: {
        "name": "LDPlayer",
        "window_patterns": ["LDPlayer", "LDMultiPlayer"],
        "adb_port": 5555,
        "default_resolution": (1280, 720),
        "keyboard_support": True,
        "multi_instance": True,
    },
    MobileEmulator.NOXPLAYER: {
        "name": "NoxPlayer",
        "window_patterns": ["NoxPlayer", "Nox App Player"],
        "adb_port": 62001,
        "default_resolution": (1280, 720),
        "keyboard_support": True,
        "multi_instance": True,
    },
    MobileEmulator.MEMU: {
        "name": "MEmu",
        "window_patterns": ["MEmu", "MEmu Play"],
        "adb_port": 21503,
        "default_resolution": (1280, 720),
        "keyboard_support": True,
        "multi_instance": True,
    },
    MobileEmulator.ANDROID_STUDIO: {
        "name": "Android Emulator",
        "window_patterns": ["Android Emulator"],
        "adb_port": 5554,
        "default_resolution": (1080, 1920),
        "keyboard_support": True,
        "multi_instance": False,
    },
}


class ADBController:
    """
    Android Debug Bridge controller for direct device/emulator control.

    ADB provides the most reliable input method as it bypasses
    the emulator's input handling entirely.
    """

    def __init__(self, device_id: str = None, port: int = 5555):
        self.device_id = device_id
        self.port = port
        self.connected = False
        self.adb_path = self._find_adb()

    def _find_adb(self) -> Optional[str]:
        """Find ADB executable"""
        common_paths = [
            r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe",
            r"C:\Program Files\BlueStacks\HD-Adb.exe",
            r"C:\Program Files\Nox\bin\adb.exe",
            r"C:\Program Files\Microvirt\MEmu\adb.exe",
            r"C:\LDPlayer\LDPlayer4.0\adb.exe",
            # Android SDK paths
            Path.home() / "AppData" / "Local" / "Android" / "Sdk" / "platform-tools" / "adb.exe",
        ]

        # Check PATH first
        try:
            result = subprocess.run(
                ["adb", "version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "adb"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Check common paths
        for path in common_paths:
            if Path(path).exists():
                return str(path)

        return None

    def connect(self, host: str = "127.0.0.1") -> bool:
        """Connect to emulator via ADB"""
        if not self.adb_path:
            warning("adb", "ADB not found")
            return False

        try:
            # Connect to emulator
            device = self.device_id or f"{host}:{self.port}"
            result = subprocess.run(
                [self.adb_path, "connect", device],
                capture_output=True,
                text=True,
                timeout=10
            )

            if "connected" in result.stdout.lower():
                self.connected = True
                self.device_id = device
                info("adb", f"Connected to {device}")
                return True
            else:
                warning("adb", f"Failed to connect: {result.stdout}")
                return False

        except Exception as e:
            error("adb", f"ADB connect error: {e}")
            return False

    def tap(self, x: int, y: int) -> bool:
        """Tap at coordinates"""
        return self._shell(f"input tap {x} {y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int,
              duration_ms: int = 300) -> bool:
        """Swipe from (x1,y1) to (x2,y2)"""
        return self._shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> bool:
        """Long press at coordinates"""
        # Long press is a swipe to the same location
        return self._shell(f"input swipe {x} {y} {x} {y} {duration_ms}")

    def key_event(self, keycode: int) -> bool:
        """Send Android keycode"""
        return self._shell(f"input keyevent {keycode}")

    def text(self, text: str) -> bool:
        """Type text (for text fields)"""
        # Escape special characters
        escaped = text.replace(" ", "%s").replace("'", "\\'")
        return self._shell(f"input text '{escaped}'")

    def back(self) -> bool:
        """Press back button"""
        return self.key_event(4)  # KEYCODE_BACK

    def home(self) -> bool:
        """Press home button"""
        return self.key_event(3)  # KEYCODE_HOME

    def menu(self) -> bool:
        """Press menu button"""
        return self.key_event(82)  # KEYCODE_MENU

    def screenshot(self, local_path: str = None) -> Optional[bytes]:
        """Take screenshot via ADB"""
        if not self.connected:
            return None

        try:
            # Capture to device
            self._shell("screencap -p /sdcard/screen.png")

            # Pull to local
            if local_path:
                subprocess.run(
                    [self.adb_path, "-s", self.device_id,
                     "pull", "/sdcard/screen.png", local_path],
                    capture_output=True,
                    timeout=10
                )
                return None
            else:
                # Return raw bytes
                result = subprocess.run(
                    [self.adb_path, "-s", self.device_id,
                     "exec-out", "screencap", "-p"],
                    capture_output=True,
                    timeout=10
                )
                return result.stdout

        except Exception as e:
            error("adb", f"Screenshot error: {e}")
            return None

    def get_current_activity(self) -> Optional[str]:
        """Get current foreground activity"""
        try:
            result = subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell",
                 "dumpsys", "activity", "activities"],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Parse for current activity
            for line in result.stdout.split('\n'):
                if 'mResumedActivity' in line or 'mFocusedActivity' in line:
                    # Extract package/activity
                    if '/' in line:
                        parts = line.split('{')[1].split()[1] if '{' in line else ''
                        return parts
            return None

        except Exception:
            return None

    def launch_app(self, package_name: str, activity: str = None) -> bool:
        """Launch an app by package name"""
        if activity:
            return self._shell(f"am start -n {package_name}/{activity}")
        else:
            return self._shell(f"monkey -p {package_name} 1")

    def _shell(self, command: str) -> bool:
        """Execute shell command on device"""
        if not self.connected or not self.adb_path:
            return False

        try:
            result = subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", command],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0

        except Exception as e:
            debug("adb", f"Shell command failed: {e}")
            return False


class MobileGameAdapter(PlatformAdapter):
    """
    Adapter for mobile games running in Android emulators.

    Supports multiple input methods:
    - Direct touch emulation via mouse
    - ADB for more reliable input
    - Emulator keyboard mapping
    """

    def __init__(self,
                 game_profile: MobileGameProfile = None,
                 emulator: MobileEmulator = MobileEmulator.BLUESTACKS,
                 use_adb: bool = True):
        super().__init__(Platform.CUSTOM, EmulatorType.NATIVE)

        self.game_profile = game_profile
        self.emulator = emulator
        self.emulator_config = MOBILE_EMULATOR_PROFILES.get(emulator, {})

        # Input systems
        self._pyautogui = None
        self._pygetwindow = None

        # ADB controller
        self.use_adb = use_adb
        self.adb = None
        if use_adb:
            adb_port = self.emulator_config.get('adb_port', 5555)
            self.adb = ADBController(port=adb_port)

        # Screen info
        self.screen_size = (0, 0)
        self.orientation = "landscape"

        # Load keymap
        self._setup_keymap()

    def _setup_keymap(self):
        """Set up keyboard mapping"""
        if self.game_profile and self.game_profile.keyboard_map:
            self.keymap = dict(self.game_profile.keyboard_map)
        else:
            # Generic mobile game keymap (common emulator defaults)
            self.keymap = {
                'up': 'w',
                'down': 's',
                'left': 'a',
                'right': 'd',
                'action': 'space',
                'back': 'escape',
                'menu': 'tab',
                'skill_1': '1',
                'skill_2': '2',
                'skill_3': '3',
                'skill_4': '4',
            }

    def _ensure_imports(self):
        """Lazy import GUI libraries"""
        if self._pyautogui is None:
            import pyautogui
            import pygetwindow as gw
            self._pyautogui = pyautogui
            self._pygetwindow = gw
            pyautogui.FAILSAFE = False

    def connect(self) -> bool:
        """Connect to emulator window and optionally ADB"""
        self._ensure_imports()

        # Find emulator window
        window_patterns = self.emulator_config.get('window_patterns', [])

        for pattern in window_patterns:
            try:
                windows = self._pygetwindow.getWindowsWithTitle(pattern)
                if windows:
                    self.window = windows[0]
                    self.window_title = self.window.title
                    self.screen_size = (self.window.width, self.window.height)
                    info("mobile", f"Connected to emulator: {self.window_title}")
                    info("mobile", f"Screen size: {self.screen_size}")
                    break
            except Exception:
                continue

        if not self.window:
            # Try partial match
            try:
                all_windows = self._pygetwindow.getAllWindows()
                for pattern in window_patterns:
                    pattern_lower = pattern.lower()
                    for w in all_windows:
                        if pattern_lower in w.title.lower() and w.width > 200:
                            self.window = w
                            self.window_title = w.title
                            self.screen_size = (w.width, w.height)
                            info("mobile", f"Connected to emulator: {self.window_title}")
                            break
                    if self.window:
                        break
            except Exception:
                pass

        if not self.window:
            error("mobile", "No emulator window found")
            return False

        # Connect ADB if enabled
        if self.use_adb and self.adb:
            if self.adb.connect():
                info("mobile", "ADB connected - using for input")
            else:
                warning("mobile", "ADB not available - using mouse input")
                self.use_adb = False

        return True

    def capture_screen(self):
        """Capture emulator screen"""
        # Try ADB first (more reliable for some emulators)
        if self.use_adb and self.adb and self.adb.connected:
            screenshot_data = self.adb.screenshot()
            if screenshot_data:
                import cv2
                import numpy as np
                nparr = np.frombuffer(screenshot_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame

        # Fallback to window capture
        self._ensure_imports()
        import cv2
        import numpy as np

        if not self.window:
            return None

        try:
            left, top = self.window.left, self.window.top
            width, height = self.window.width, self.window.height

            left = max(0, left)
            top = max(0, top)

            if width <= 0 or height <= 0:
                return None

            screenshot = self._pyautogui.screenshot(region=(left, top, width, height))
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            return frame

        except Exception as e:
            debug("mobile", f"Screen capture error: {e}")
            return None

    def press_button(self, button: str, duration: float = 0.1) -> bool:
        """Press a mapped button"""
        self._ensure_imports()

        key = self.keymap.get(button, button)

        try:
            self._pyautogui.keyDown(key)
            time.sleep(duration)
            self._pyautogui.keyUp(key)
            return True
        except Exception as e:
            error("mobile", f"Button press error: {e}")
            return False

    def release_button(self, button: str) -> bool:
        """Release a button"""
        self._ensure_imports()
        key = self.keymap.get(button, button)

        try:
            self._pyautogui.keyUp(key)
            return True
        except Exception:
            return False

    def tap(self, x: int, y: int) -> bool:
        """Tap at screen coordinates"""
        # Use ADB if available
        if self.use_adb and self.adb and self.adb.connected:
            return self.adb.tap(x, y)

        # Otherwise use mouse
        return self._mouse_tap(x, y)

    def _mouse_tap(self, x: int, y: int) -> bool:
        """Tap using mouse click"""
        self._ensure_imports()

        if not self.window:
            return False

        try:
            # Convert to screen coordinates
            screen_x = self.window.left + x
            screen_y = self.window.top + y

            self._pyautogui.click(screen_x, screen_y)
            return True
        except Exception as e:
            error("mobile", f"Tap error: {e}")
            return False

    def double_tap(self, x: int, y: int) -> bool:
        """Double tap at coordinates"""
        if self.use_adb and self.adb and self.adb.connected:
            self.adb.tap(x, y)
            time.sleep(0.1)
            return self.adb.tap(x, y)

        self._ensure_imports()
        if not self.window:
            return False

        try:
            screen_x = self.window.left + x
            screen_y = self.window.top + y
            self._pyautogui.doubleClick(screen_x, screen_y)
            return True
        except Exception:
            return False

    def long_press(self, x: int, y: int, duration: float = 1.0) -> bool:
        """Long press at coordinates"""
        if self.use_adb and self.adb and self.adb.connected:
            return self.adb.long_press(x, y, int(duration * 1000))

        self._ensure_imports()
        if not self.window:
            return False

        try:
            screen_x = self.window.left + x
            screen_y = self.window.top + y

            self._pyautogui.mouseDown(screen_x, screen_y)
            time.sleep(duration)
            self._pyautogui.mouseUp()
            return True
        except Exception:
            return False

    def swipe(self, x1: int, y1: int, x2: int, y2: int,
              duration: float = 0.3) -> bool:
        """Swipe from (x1,y1) to (x2,y2)"""
        if self.use_adb and self.adb and self.adb.connected:
            return self.adb.swipe(x1, y1, x2, y2, int(duration * 1000))

        self._ensure_imports()
        if not self.window:
            return False

        try:
            # Convert to screen coordinates
            screen_x1 = self.window.left + x1
            screen_y1 = self.window.top + y1
            screen_x2 = self.window.left + x2
            screen_y2 = self.window.top + y2

            # Perform swipe via mouse drag
            self._pyautogui.moveTo(screen_x1, screen_y1)
            self._pyautogui.mouseDown()
            self._pyautogui.moveTo(screen_x2, screen_y2, duration=duration)
            self._pyautogui.mouseUp()
            return True
        except Exception as e:
            error("mobile", f"Swipe error: {e}")
            return False

    def gesture(self, gesture_type: TouchGesture,
                x: int = None, y: int = None,
                **kwargs) -> bool:
        """Perform a touch gesture"""
        # Default to screen center if not specified
        if x is None:
            x = self.screen_size[0] // 2
        if y is None:
            y = self.screen_size[1] // 2

        if gesture_type == TouchGesture.TAP:
            return self.tap(x, y)

        elif gesture_type == TouchGesture.DOUBLE_TAP:
            return self.double_tap(x, y)

        elif gesture_type == TouchGesture.LONG_PRESS:
            duration = kwargs.get('duration', 1.0)
            return self.long_press(x, y, duration)

        elif gesture_type == TouchGesture.SWIPE_UP:
            distance = kwargs.get('distance', 200)
            return self.swipe(x, y, x, y - distance)

        elif gesture_type == TouchGesture.SWIPE_DOWN:
            distance = kwargs.get('distance', 200)
            return self.swipe(x, y, x, y + distance)

        elif gesture_type == TouchGesture.SWIPE_LEFT:
            distance = kwargs.get('distance', 200)
            return self.swipe(x, y, x - distance, y)

        elif gesture_type == TouchGesture.SWIPE_RIGHT:
            distance = kwargs.get('distance', 200)
            return self.swipe(x, y, x + distance, y)

        elif gesture_type == TouchGesture.DRAG:
            x2 = kwargs.get('x2', x)
            y2 = kwargs.get('y2', y)
            return self.swipe(x, y, x2, y2)

        else:
            warning("mobile", f"Gesture not implemented: {gesture_type}")
            return False

    def back(self) -> bool:
        """Press back button"""
        if self.use_adb and self.adb and self.adb.connected:
            return self.adb.back()
        return self.press_button('back')

    def home(self) -> bool:
        """Press home button"""
        if self.use_adb and self.adb and self.adb.connected:
            return self.adb.home()
        # Most emulators don't have a keyboard shortcut for home
        warning("mobile", "Home button requires ADB")
        return False

    def launch_game(self) -> bool:
        """Launch the game via ADB"""
        if not self.game_profile or not self.game_profile.package_name:
            warning("mobile", "No package name specified")
            return False

        if self.use_adb and self.adb and self.adb.connected:
            return self.adb.launch_app(self.game_profile.package_name)

        warning("mobile", "Game launch requires ADB")
        return False

    def focus_window(self) -> bool:
        """Bring emulator to foreground"""
        if not self.window:
            return False

        try:
            self.window.activate()
            time.sleep(0.1)
            return True
        except Exception:
            return False


# =============================================================================
# MOBILE GAME PROFILE MANAGER
# =============================================================================

class MobileGameProfileManager:
    """Manages profiles for mobile games"""

    # Pre-defined profiles for popular mobile games
    KNOWN_GAMES = {
        'genshin_impact': MobileGameProfile(
            game_id='genshin_impact',
            game_name='Genshin Impact',
            package_name='com.miHoYo.GenshinImpact',
            preferred_emulator=MobileEmulator.BLUESTACKS_5,
            preferred_resolution=(1920, 1080),
            orientation='landscape',
            input_method=MobileInputMethod.KEYBOARD_MAP,
            genre='action_rpg',
            keyboard_map={
                'move_forward': 'w',
                'move_back': 's',
                'move_left': 'a',
                'move_right': 'd',
                'jump': 'space',
                'attack': 'mouse_left',
                'skill': 'e',
                'burst': 'q',
                'sprint': 'shift',
                'aim': 'r',
                'switch_1': '1',
                'switch_2': '2',
                'switch_3': '3',
                'switch_4': '4',
                'map': 'm',
                'inventory': 'b',
            },
        ),
        'arknights': MobileGameProfile(
            game_id='arknights',
            game_name='Arknights',
            package_name='com.YoStarEN.Arknights',
            preferred_emulator=MobileEmulator.BLUESTACKS,
            preferred_resolution=(1280, 720),
            orientation='landscape',
            input_method=MobileInputMethod.TOUCH,
            genre='tower_defense',
            has_auto_play=True,
        ),
        'azur_lane': MobileGameProfile(
            game_id='azur_lane',
            game_name='Azur Lane',
            package_name='com.YoStarEN.AzurLane',
            preferred_emulator=MobileEmulator.BLUESTACKS,
            preferred_resolution=(1280, 720),
            orientation='landscape',
            input_method=MobileInputMethod.TOUCH,
            genre='gacha_shooter',
            has_auto_play=True,
        ),
        'cookie_run_kingdom': MobileGameProfile(
            game_id='cookie_run_kingdom',
            game_name='Cookie Run: Kingdom',
            package_name='com.devsisters.ck',
            preferred_emulator=MobileEmulator.BLUESTACKS,
            preferred_resolution=(1280, 720),
            orientation='landscape',
            input_method=MobileInputMethod.TOUCH,
            genre='gacha_rpg',
            has_auto_play=True,
        ),
        'pokemon_go': MobileGameProfile(
            game_id='pokemon_go',
            game_name='Pokemon GO',
            package_name='com.nianticlabs.pokemongo',
            preferred_emulator=MobileEmulator.BLUESTACKS,
            preferred_resolution=(1080, 1920),
            orientation='portrait',
            input_method=MobileInputMethod.TOUCH,
            genre='ar_game',
            notes='Requires GPS spoofing, may violate ToS',
        ),
        'clash_royale': MobileGameProfile(
            game_id='clash_royale',
            game_name='Clash Royale',
            package_name='com.supercell.clashroyale',
            preferred_emulator=MobileEmulator.BLUESTACKS,
            preferred_resolution=(1080, 1920),
            orientation='portrait',
            input_method=MobileInputMethod.TOUCH,
            genre='strategy',
        ),
    }

    def __init__(self, profiles_dir: Path = None):
        self.profiles_dir = profiles_dir or Path("mobile_game_profiles")
        self.profiles: Dict[str, MobileGameProfile] = dict(self.KNOWN_GAMES)
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
                    profile = MobileGameProfile(
                        game_id=data['game_id'],
                        game_name=data['game_name'],
                        package_name=data.get('package_name', ''),
                        preferred_emulator=MobileEmulator(data.get('preferred_emulator', 'bluestacks')),
                        orientation=data.get('orientation', 'landscape'),
                        input_method=MobileInputMethod(data.get('input_method', 'touch')),
                        keyboard_map=data.get('keyboard_map', {}),
                        genre=data.get('genre', ''),
                        has_auto_play=data.get('has_auto_play', False),
                    )
                    self.profiles[profile.game_id] = profile
            except Exception as e:
                warning("mobile_profiles", f"Failed to load {profile_file}: {e}")

    def get_profile(self, game_id: str) -> Optional[MobileGameProfile]:
        """Get a game profile"""
        return self.profiles.get(game_id)

    def create_profile(self, game_name: str, package_name: str = "",
                      emulator: MobileEmulator = MobileEmulator.BLUESTACKS,
                      genre: str = "") -> MobileGameProfile:
        """Create a new game profile"""
        game_id = game_name.lower().replace(" ", "_").replace(":", "")

        profile = MobileGameProfile(
            game_id=game_id,
            game_name=game_name,
            package_name=package_name,
            preferred_emulator=emulator,
            genre=genre,
        )

        self.profiles[game_id] = profile
        self._save_profile(profile)

        return profile

    def _save_profile(self, profile: MobileGameProfile):
        """Save profile to disk"""
        profile_path = self.profiles_dir / f"{profile.game_id}.json"

        try:
            data = {
                'game_id': profile.game_id,
                'game_name': profile.game_name,
                'package_name': profile.package_name,
                'preferred_emulator': profile.preferred_emulator.value,
                'orientation': profile.orientation,
                'input_method': profile.input_method.value,
                'keyboard_map': profile.keyboard_map,
                'genre': profile.genre,
                'has_auto_play': profile.has_auto_play,
            }
            with open(profile_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            error("mobile_profiles", f"Failed to save profile: {e}")

    def detect_running_emulator(self) -> Optional[MobileEmulator]:
        """Detect which Android emulator is running"""
        try:
            import pygetwindow as gw
            all_windows = gw.getAllWindows()

            for emulator, config in MOBILE_EMULATOR_PROFILES.items():
                patterns = config.get('window_patterns', [])
                for pattern in patterns:
                    for window in all_windows:
                        if pattern.lower() in window.title.lower() and window.width > 200:
                            info("mobile", f"Detected emulator: {config['name']}")
                            return emulator

            return None

        except Exception:
            return None


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_mobile_adapter(
    game_name: str = None,
    package_name: str = None,
    emulator: MobileEmulator = None,
    use_adb: bool = True
) -> Optional[MobileGameAdapter]:
    """
    Create appropriate mobile game adapter.

    Args:
        game_name: Name of game to look for
        package_name: Android package name
        emulator: Specific emulator to use
        use_adb: Whether to use ADB for input

    Returns:
        Configured MobileGameAdapter
    """
    manager = MobileGameProfileManager()
    profile = None

    # Try to find profile
    if game_name:
        game_id = game_name.lower().replace(" ", "_")
        profile = manager.get_profile(game_id)

    if not profile and game_name:
        # Create basic profile
        profile = manager.create_profile(
            game_name,
            package_name=package_name or "",
            emulator=emulator or MobileEmulator.BLUESTACKS
        )

    # Detect emulator if not specified
    if not emulator:
        emulator = manager.detect_running_emulator() or MobileEmulator.BLUESTACKS

    # Create adapter
    adapter = MobileGameAdapter(
        game_profile=profile,
        emulator=emulator,
        use_adb=use_adb
    )

    return adapter


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Mobile Game Adapter Test")
    print("=" * 50)

    # List known games
    manager = MobileGameProfileManager()
    print("\nKnown Mobile Games:")
    for game_id, profile in manager.profiles.items():
        print(f"  {profile.game_name}")
        print(f"    Package: {profile.package_name}")
        print(f"    Emulator: {profile.preferred_emulator.value}")
        print(f"    Input: {profile.input_method.value}")

    # Detect running emulator
    print("\nDetecting emulator...")
    emulator = manager.detect_running_emulator()
    if emulator:
        print(f"  Found: {MOBILE_EMULATOR_PROFILES[emulator]['name']}")

        # Try to connect
        adapter = create_mobile_adapter(emulator=emulator)
        if adapter.connect():
            print(f"  Connected to: {adapter.window_title}")
            print(f"  Screen size: {adapter.screen_size}")
            print(f"  ADB: {'Connected' if adapter.adb and adapter.adb.connected else 'Not available'}")
    else:
        print("  No emulator detected (this is expected if none running)")

    print("\nMobile Game Adapter Test Complete!")
