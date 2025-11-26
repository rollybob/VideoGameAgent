#!/usr/bin/env python3
"""
🎯 DIRECT WINDOW CAPTURE
========================

Direct capture from mGBA window using Windows API.
Gets pure window content without desktop/overlay interference.
"""

import cv2
import numpy as np
import win32gui
import win32ui
import win32con
from ctypes import windll
from PIL import Image

class DirectWindowCapture:
    """Direct window content capture without desktop interference"""
    
    def __init__(self, window_handle):
        self.hwnd = window_handle
        self.cropped_x = 0
        self.cropped_y = 0
        self.offset_x = 0
        self.offset_y = 0
        
    def get_window_size(self):
        """Get the actual window content size"""
        rect = win32gui.GetWindowRect(self.hwnd)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        return width, height
        
    def get_client_size(self):
        """Get the client area size (excludes window borders/title bar)"""
        client_rect = win32gui.GetClientRect(self.hwnd)
        return client_rect[2], client_rect[3]
        
    def capture_window_direct(self):
        """Capture window content directly using Windows API"""
        try:
            # Get window and client dimensions
            window_rect = win32gui.GetWindowRect(self.hwnd)
            client_rect = win32gui.GetClientRect(self.hwnd)
            
            # Calculate the actual game area (excluding title bar and borders)
            # This gets just the game content, not the window chrome
            width = client_rect[2]
            height = client_rect[3]
            
            if width <= 0 or height <= 0:
                return None
                
            # Get window device context
            hwndDC = win32gui.GetWindowDC(self.hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # Create bitmap
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # Copy the client area (game content) to our bitmap
            # This captures ONLY the game area, not the title bar or borders
            result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 1)
            
            if result:
                # Convert to PIL Image
                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)
                
                img = Image.frombuffer(
                    'RGB',
                    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                    bmpstr, 'raw', 'BGRX', 0, 1
                )
                
                # Convert to numpy array (OpenCV format)
                img_array = np.array(img)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Cleanup
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(self.hwnd, hwndDC)
                
                return img_array
            else:
                # Cleanup on failure
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(self.hwnd, hwndDC)
                return None
                
        except Exception as e:
            print(f"Direct capture error: {e}")
            return None
            
    def capture_with_fallback(self):
        """Capture with fallback to screen grab if direct capture fails"""
        # Try direct capture first
        direct_result = self.capture_window_direct()
        
        if direct_result is not None:
            return direct_result
            
        # Fallback to screen grab (original method)
        print("Direct capture failed, using screen grab fallback")
        try:
            from PIL import ImageGrab
            rect = win32gui.GetWindowRect(self.hwnd)
            screenshot = ImageGrab.grab(bbox=rect)
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Fallback capture also failed: {e}")
            return None

def test_direct_capture():
    """Test the direct capture functionality"""
    print("Testing direct window capture...")
    
    # Find mGBA window
    def find_mgba_window():
        windows = []
        def enum_callback(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if window_text and ('mgba' in window_text.lower() or 'pokemon' in window_text.lower()):
                    windows.append((hwnd, window_text))
        
        win32gui.EnumWindows(enum_callback, windows)
        return windows
    
    emulator_windows = find_mgba_window()
    
    if not emulator_windows:
        print("No mGBA window found!")
        return
        
    # Use first found window
    hwnd, title = emulator_windows[0]
    print(f"Testing capture on: {title}")
    
    # Create capturer
    capturer = DirectWindowCapture(hwnd)
    
    # Test capture
    result = capturer.capture_with_fallback()
    
    if result is not None:
        print(f"Capture successful! Image shape: {result.shape}")
        
        # Save test image
        cv2.imwrite("test_direct_capture.png", result)
        print("Test image saved as: test_direct_capture.png")
        
        # Show basic info
        height, width = result.shape[:2]
        print(f"Captured image: {width}x{height}")
        
        # Check if it looks like a game screen
        if width >= 240 and height >= 160:
            print("✅ Looks like a valid game capture!")
        else:
            print("⚠️ Unusual dimensions for a game screen")
            
    else:
        print("❌ Capture failed!")

if __name__ == "__main__":
    test_direct_capture()