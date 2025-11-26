"""
Base controller class for game input systems.
All platform-specific controllers should inherit from this.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class BaseController(ABC):
    """Abstract base class for all game controllers"""
    
    def __init__(self, name: str):
        self.name = name
        self.keymap: Dict[str, str] = {}
        self.available_buttons: List[str] = []
        
    @abstractmethod
    def get_keymap(self) -> Dict[str, str]:
        """Return the button to key mapping for this controller"""
        pass
    
    @abstractmethod
    def get_available_buttons(self) -> List[str]:
        """Return list of available buttons for this controller"""
        pass
    
    @abstractmethod
    def press(self, button: str, duration: float = 0.2) -> bool:
        """Press a button. Returns True if successful."""
        pass
    
    @abstractmethod
    def get_emulator_window_title(self) -> str:
        """Return the expected emulator window title keyword"""
        pass
    
    def validate_button(self, button: str) -> bool:
        """Check if button is valid for this controller"""
        return button in self.available_buttons
    
    def get_controller_info(self) -> Dict[str, any]:
        """Return controller information for display"""
        return {
            "name": self.name,
            "buttons": self.available_buttons,
            "emulator": self.get_emulator_window_title()
        }