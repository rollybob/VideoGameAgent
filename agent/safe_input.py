"""
Safe Input System - Only sends input to the target emulator window
Prevents accidental input to other applications when emulator loses focus
"""
import time
import ctypes
from ctypes import wintypes, windll
import pygetwindow as gw
from typing import Optional

# Windows API constants
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_CHAR = 0x0102
WM_SYSCOMMAND = 0x0112
SC_RESTORE = 0xF120

# Virtual key codes for gamepad/controller inputs
VK_CODES = {
    'up': 0x26,        # VK_UP
    'down': 0x28,      # VK_DOWN  
    'left': 0x25,      # VK_LEFT
    'right': 0x27,     # VK_RIGHT
    'a': 0x5A,         # Z key (commonly mapped to A)
    'b': 0x58,         # X key (commonly mapped to B)
    'start': 0x0D,     # Enter key
    'select': 0x08,    # Backspace key
    'l': 0x41,         # A key (L shoulder)
    'r': 0x53,         # S key (R shoulder)
    'enter': 0x0D,
    'backspace': 0x08,
    'space': 0x20,
    'z': 0x5A,
    'x': 0x58
}

class SafeEmulatorInput:
    """
    Handles input specifically for emulator windows with safety checks
    """
    
    def __init__(self, emulator_title_keyword="mGBA"):
        self.emulator_title_keyword = emulator_title_keyword
        self.target_window = None
        self.last_focus_check = 0
        self.focus_check_interval = 1.0  # Check focus every second
        
        # Windows API functions
        self.user32 = windll.user32
        self.kernel32 = windll.kernel32
        
        print(f"Safe input system initialized for '{emulator_title_keyword}' emulator")
    
    def find_target_window(self) -> Optional[object]:
        """Find the emulator window"""
        try:
            windows = gw.getWindowsWithTitle(self.emulator_title_keyword)
            if windows:
                return windows[0]
            return None
        except Exception as e:
            print(f"Error finding emulator window: {e}")
            return None
    
    def is_emulator_window_valid(self) -> bool:
        """Check if emulator window exists and is accessible"""
        current_time = time.time()
        
        # Only check periodically to avoid performance impact
        if current_time - self.last_focus_check > self.focus_check_interval:
            self.target_window = self.find_target_window()
            self.last_focus_check = current_time
        
        return self.target_window is not None
    
    def is_emulator_minimized(self) -> bool:
        """Check if emulator window is minimized"""
        if not self.target_window:
            return True
        
        try:
            # Check if window is minimized
            return self.target_window.isMinimized
        except:
            return True
    
    def restore_emulator_if_minimized(self) -> bool:
        """Restore emulator window if it's minimized, but don't steal focus"""
        if not self.target_window:
            return False
        
        try:
            if self.target_window.isMinimized:
                # Restore window without activating it (doesn't steal focus)
                hwnd = self.target_window._hWnd
                self.user32.ShowWindow(hwnd, 9)  # SW_RESTORE = 9
                time.sleep(0.1)  # Brief pause for window to restore
                return True
        except Exception as e:
            print(f"Error restoring window: {e}")
        
        return False
    
    def send_key_to_emulator(self, key: str, duration: float = 0.1) -> bool:
        """
        Send a key press directly to the emulator window without stealing focus
        Returns True if successful, False if failed or unsafe
        """
        # Safety checks
        if not self.is_emulator_window_valid():
            print(f"Emulator window not found or invalid")
            return False
        
        if self.is_emulator_minimized():
            print("Emulator is minimized - not sending input for safety")
            return False
        
        key_lower = key.lower()
        if key_lower not in VK_CODES:
            print(f"Unknown key: {key}")
            return False
        
        try:
            # Get window handle
            hwnd = self.target_window._hWnd
            vk_code = VK_CODES[key_lower]
            
            # Send keydown message directly to window
            self.user32.PostMessageW(hwnd, WM_KEYDOWN, vk_code, 0)
            
            # Hold the key for specified duration
            time.sleep(duration)
            
            # Send keyup message
            self.user32.PostMessageW(hwnd, WM_KEYUP, vk_code, 0)
            
            return True
            
        except Exception as e:
            print(f"Error sending key '{key}' to emulator: {e}")
            return False
    
    def send_key_sequence(self, keys: list, key_duration: float = 0.1, sequence_delay: float = 0.05) -> bool:
        """
        Send a sequence of key presses to the emulator
        """
        if not self.is_emulator_window_valid():
            return False
        
        success_count = 0
        for key in keys:
            if self.send_key_to_emulator(key, key_duration):
                success_count += 1
                if sequence_delay > 0:
                    time.sleep(sequence_delay)
            else:
                break
        
        return success_count == len(keys)
    
    def is_safe_to_send_input(self) -> tuple:
        """
        Comprehensive safety check before sending input
        Returns (is_safe: bool, reason: str)
        """
        # Check if emulator window exists
        if not self.is_emulator_window_valid():
            return False, "Emulator window not found"
        
        # Check if emulator is minimized
        if self.is_emulator_minimized():
            return False, "Emulator is minimized"
        
        # Check if emulator window is too small (might be covered)
        try:
            if self.target_window.width < 100 or self.target_window.height < 100:
                return False, "Emulator window too small (possibly covered)"
        except:
            return False, "Cannot access emulator window properties"
        
        # Additional safety: check if there are other applications in focus
        # that might intercept the input
        try:
            current_foreground = self.user32.GetForegroundWindow()
            emulator_handle = self.target_window._hWnd
            
            # If something else has focus and emulator is not visible, it's not safe
            if current_foreground != emulator_handle:
                # Check if emulator window is still visible on screen
                rect = wintypes.RECT()
                if self.user32.GetWindowRect(emulator_handle, ctypes.byref(rect)):
                    # If emulator is off-screen or hidden, don't send input
                    if rect.right <= 0 or rect.bottom <= 0:
                        return False, "Emulator window not visible on screen"
                
        except Exception as e:
            return False, f"Safety check failed: {e}"
        
        return True, "Safe to send input"
    
    def safe_key_press(self, key: str, duration: float = 0.1) -> bool:
        """
        Ultra-safe key press that includes all safety checks
        """
        is_safe, reason = self.is_safe_to_send_input()
        if not is_safe:
            print(f"Input blocked for safety: {reason}")
            return False
        
        return self.send_key_to_emulator(key, duration)
    
    def get_emulator_status(self) -> dict:
        """Get detailed status of the emulator window for debugging"""
        status = {
            'window_found': False,
            'window_minimized': True,
            'window_size': (0, 0),
            'window_position': (0, 0),
            'is_safe': False,
            'safety_reason': 'Unknown'
        }
        
        if self.is_emulator_window_valid():
            status['window_found'] = True
            status['window_minimized'] = self.is_emulator_minimized()
            
            try:
                status['window_size'] = (self.target_window.width, self.target_window.height)
                status['window_position'] = (self.target_window.left, self.target_window.top)
            except:
                pass
        
        is_safe, reason = self.is_safe_to_send_input()
        status['is_safe'] = is_safe
        status['safety_reason'] = reason
        
        return status

# Global instance for easy access
safe_input = SafeEmulatorInput()

def safe_emulator_key_press(key: str, duration: float = 0.1) -> bool:
    """Convenience function for safe key presses"""
    return safe_input.safe_key_press(key, duration)

def get_emulator_input_status() -> dict:
    """Convenience function to get emulator status"""
    return safe_input.get_emulator_status()