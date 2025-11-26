"""
Controller Manager for handling multiple controller types and selection.
"""

import time
from typing import Dict, List, Optional, Type
from .base_controller import BaseController
from .gba_controller import GBAController

class ControllerManager:
    """Manages multiple controller types and handles controller selection"""
    
    def __init__(self):
        self.controllers: Dict[str, BaseController] = {}
        self.current_controller: Optional[BaseController] = None
        self._register_default_controllers()
    
    def _register_default_controllers(self):
        """Register all available controller types"""
        # Add GBA controller
        gba = GBAController()
        self.controllers[gba.name] = gba
        
        # Set GBA as default for now
        self.current_controller = gba
        
        # TODO: Add more controllers here as they're implemented
        # nes = NESController()
        # self.controllers[nes.name] = nes
        # 
        # snes = SNESController() 
        # self.controllers[snes.name] = snes
    
    def get_available_controllers(self) -> List[str]:
        """Get list of available controller names"""
        return list(self.controllers.keys())
    
    def select_controller(self, controller_name: str) -> bool:
        """Select a controller by name"""
        if controller_name in self.controllers:
            self.current_controller = self.controllers[controller_name]
            print(f"[Controller Manager] Selected: {controller_name}")
            return True
        else:
            print(f"[Controller Manager] Error: Controller '{controller_name}' not found")
            return False
    
    def get_current_controller(self) -> Optional[BaseController]:
        """Get the currently selected controller"""
        return self.current_controller
    
    def get_current_controller_name(self) -> str:
        """Get the name of the currently selected controller"""
        return self.current_controller.name if self.current_controller else "None"
    
    def press(self, button: str, duration: float = 0.2) -> bool:
        """Press a button using the current controller"""
        if not self.current_controller:
            print("[Controller Manager] Error: No controller selected")
            return False
        
        return self.current_controller.press(button, duration)
    
    def get_available_buttons(self) -> List[str]:
        """Get available buttons for current controller"""
        if not self.current_controller:
            return []
        return self.current_controller.get_available_buttons()
    
    def get_emulator_window_title(self) -> str:
        """Get emulator window title for current controller"""
        if not self.current_controller:
            return "Unknown"
        return self.current_controller.get_emulator_window_title()
    
    def get_controller_info(self) -> Dict[str, any]:
        """Get information about current controller"""
        if not self.current_controller:
            return {"name": "None", "buttons": [], "emulator": "Unknown"}
        return self.current_controller.get_controller_info()
    
    def register_controller(self, controller: BaseController):
        """Register a new controller type"""
        self.controllers[controller.name] = controller
        print(f"[Controller Manager] Registered: {controller.name}")
    
    def mash(self, button: str, count: int = 3, delay: float = 0.1) -> bool:
        """Mash a button using current controller (if supported)"""
        if not self.current_controller:
            return False
            
        # Check if controller has mash method
        if hasattr(self.current_controller, 'mash'):
            return self.current_controller.mash(button, count, delay)
        else:
            # Fallback: simulate mashing with regular presses
            for _ in range(count):
                self.press(button, 0.05)
                time.sleep(delay)
            return True