"""
Game Boy Advance controller implementation for mGBA emulator.
"""

import pyautogui
import time
from typing import Dict, List
from .base_controller import BaseController

class GBAController(BaseController):
    """Controller for Game Boy Advance games via mGBA emulator"""
    
    def __init__(self):
        super().__init__("Game Boy Advance")
        self._setup_keymap()
        
    def _setup_keymap(self):
        """Set up the GBA button to keyboard key mapping"""
        self.keymap = {
            "up": "up",
            "down": "down", 
            "left": "left",
            "right": "right",
            "A": "x",        # Action A button -> X key
            "B": "z",        # Action B button -> Z key
            "start": "enter",
            "select": "backspace",
            "L": "a",        # Left shoulder -> A key
            "R": "s"         # Right shoulder -> S key
        }
        
        self.available_buttons = list(self.keymap.keys())
    
    def get_keymap(self) -> Dict[str, str]:
        """Return the button to key mapping"""
        return self.keymap.copy()
    
    def get_available_buttons(self) -> List[str]:
        """Return list of available GBA buttons"""
        return self.available_buttons.copy()
    
    def press(self, button: str, duration: float = 0.2) -> bool:
        """Press a GBA button"""
        if not self.validate_button(button):
            print(f"[GBA Controller] Warning: Unknown button '{button}'")
            return False
            
        try:
            key = self.keymap[button]
            pyautogui.keyDown(key)
            time.sleep(duration)
            pyautogui.keyUp(key)
            return True
        except Exception as e:
            print(f"[GBA Controller] Error pressing {button}: {e}")
            return False
    
    def get_emulator_window_title(self) -> str:
        """Return the mGBA emulator window title keyword"""
        return "mGBA"
    
    def mash(self, button: str, count: int = 3, delay: float = 0.1) -> bool:
        """Press a button repeatedly (useful for dialogue skipping)"""
        if not self.validate_button(button):
            return False
            
        try:
            for _ in range(count):
                self.press(button, duration=0.05)
                time.sleep(delay)
            return True
        except Exception as e:
            print(f"[GBA Controller] Error mashing {button}: {e}")
            return False