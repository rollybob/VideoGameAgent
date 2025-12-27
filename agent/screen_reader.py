import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import time
import pytesseract
from PIL import Image

# Use centralized configuration for Tesseract path
try:
    from config import get_tesseract_cmd
    tesseract_path = get_tesseract_cmd()
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
except ImportError:
    # Fallback if config not available
    import os
    pytesseract.pytesseract.tesseract_cmd = os.environ.get(
        'TESSERACT_CMD',
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

def find_emulator_window(title_keyword="mGBA", debug=False):
    windows = gw.getWindowsWithTitle(title_keyword)
    if not windows:
        raise RuntimeError(f"No emulator window found with title containing '{title_keyword}'")
    
    if debug:
        print(f"Found {len(windows)} windows with '{title_keyword}':")
        for i, win in enumerate(windows):
            print(f"  {i}: {win.title} - {win.left},{win.top} {win.width}x{win.height}")
    
    return windows[0]

def find_emulator_window_only(title_keyword="mGBA", exclude_keywords=None):
    """
    Find emulator window while excluding windows that might contain overlays
    """
    if exclude_keywords is None:
        exclude_keywords = ["VGA", "Agent", "AI", "Control"]
    
    all_windows = gw.getAllWindows()
    emulator_windows = []
    
    for window in all_windows:
        title = window.title.lower()
        if title_keyword.lower() in title:
            # Check if this window contains overlay keywords
            has_overlay_keywords = any(keyword.lower() in title for keyword in exclude_keywords)
            if not has_overlay_keywords and window.width > 100 and window.height > 100:
                emulator_windows.append(window)
    
    if not emulator_windows:
        # Fallback to original method
        return find_emulator_window(title_keyword)
    
    # Return the largest window (likely the main emulator)
    return max(emulator_windows, key=lambda w: w.width * w.height)

def capture_window(window):
    """
    Grab a screenshot of the emulator window without forcing it to the front.
    """
    left, top = window.left, window.top
    width, height = window.width, window.height

    if width <= 0 or height <= 0:
        raise ValueError("Window size is invalid. Cannot capture screen.")

    bbox = (left, top, width, height)
    screenshot = pyautogui.screenshot(region=bbox)
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return frame

def capture_emulator_game_area(window, debug=False):
    """
    Capture only the game area, excluding window decorations and overlays.
    Uses multiple strategies to get clean game screen capture.
    """
    left, top = window.left, window.top
    width, height = window.width, window.height

    if width <= 0 or height <= 0:
        raise ValueError("Window size is invalid. Cannot capture screen.")

    if debug:
        print(f"Window bounds: {left}, {top}, {width}x{height}")

    # Strategy 1: Try to capture just the emulator's client area (no titlebar)
    try:
        # Get the actual client area of the window (excludes titlebar/borders)
        import ctypes
        from ctypes import wintypes
        
        # Get window handle
        hwnd = window._hWnd
        
        # Get client rectangle (interior area only)
        client_rect = wintypes.RECT()
        ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(client_rect))
        
        # Convert client coordinates to screen coordinates
        client_point = wintypes.POINT()
        client_point.x = 0
        client_point.y = 0
        ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(client_point))
        
        # Calculate actual client area in screen coordinates
        client_left = client_point.x
        client_top = client_point.y
        client_width = client_rect.right - client_rect.left
        client_height = client_rect.bottom - client_rect.top
        
        if debug:
            print(f"Client area: {client_left}, {client_top}, {client_width}x{client_height}")
        
        # Capture just the client area
        bbox = (client_left, client_top, client_width, client_height)
        screenshot = pyautogui.screenshot(region=bbox)
        client_frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Now crop out emulator UI from the client area
        game_frame = crop_emulator_ui(client_frame, debug=debug)
        
        if debug:
            print(f"Final game frame size: {game_frame.shape}")
            
        return game_frame
        
    except Exception as e:
        if debug:
            print(f"Client area capture failed: {e}")
        
        # Fallback: use full window capture with aggressive cropping
        bbox = (left, top, width, height)
        screenshot = pyautogui.screenshot(region=bbox)
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        return crop_emulator_ui(frame, debug=debug)

def crop_emulator_ui(frame, debug=False):
    """
    Crop emulator UI elements to get just the game screen.
    Uses multiple detection methods.
    """
    h, w, _ = frame.shape
    
    if debug:
        print(f"Cropping frame of size: {w}x{h}")
    
    # Method 1: Try to detect the game area automatically
    game_area = detect_game_screen_region(frame)
    
    if game_area:
        x1, y1, x2, y2 = game_area
        if debug:
            print(f"Detected game area: {x1},{y1} to {x2},{y2}")
        game_frame = frame[y1:y2, x1:x2]
        
        # Validate the detected area isn't too small
        if game_frame.shape[0] > 100 and game_frame.shape[1] > 100:
            return game_frame
        elif debug:
            print("Detected area too small, using fallback")
    
    # Method 2: Conservative cropping based on typical emulator layouts
    # Most emulators have menu bars at top, status at bottom
    
    # For mGBA, typical layout:
    # - Menu bar at top (~30-40px)  
    # - Status bar at bottom (~20-25px)
    # - Some padding on sides
    
    crop_top = max(0, int(h * 0.08))      # Remove top 8% (menu bar)
    crop_bottom = min(h, int(h * 0.95))    # Remove bottom 5% (status bar)  
    crop_left = max(0, int(w * 0.03))      # Remove left 3%
    crop_right = min(w, int(w * 0.97))     # Remove right 3%
    
    if debug:
        print(f"Conservative crop: top={crop_top}, bottom={crop_bottom}, left={crop_left}, right={crop_right}")
    
    cropped_frame = frame[crop_top:crop_bottom, crop_left:crop_right]
    
    # Method 3: If still looks like it has UI, be more aggressive
    if has_likely_ui_elements(cropped_frame):
        if debug:
            print("Still detecting UI, applying aggressive crop")
        
        # More aggressive cropping
        crop_top = max(0, int(h * 0.15))     # Remove top 15%
        crop_bottom = min(h, int(h * 0.90))   # Remove bottom 10%
        crop_left = max(0, int(w * 0.08))     # Remove left 8%
        crop_right = min(w, int(w * 0.92))    # Remove right 8%
        
        cropped_frame = frame[crop_top:crop_bottom, crop_left:crop_right]
    
    return cropped_frame

def has_likely_ui_elements(frame):
    """
    Check if frame likely contains UI elements that should be cropped out.
    """
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Check edges for UI elements (menus, toolbars typically at edges)
    edge_thickness = min(20, int(h * 0.05))
    
    # Check top edge for menu bars (typically dark or light bars)
    top_edge = gray[:edge_thickness, :]
    top_uniformity = np.std(top_edge)
    
    # Check bottom edge for status bars
    bottom_edge = gray[-edge_thickness:, :]
    bottom_uniformity = np.std(bottom_edge)
    
    # UI elements typically have low variance (uniform colors)
    # Game content typically has high variance (detailed pixels)
    return top_uniformity < 20 or bottom_uniformity < 20

def capture_emulator_direct(window, debug=False):
    """
    Alternative capture method that tries to avoid overlays by targeting the emulator process directly
    """
    try:
        import ctypes
        from ctypes import wintypes
        import win32gui
        import win32ui
        import win32con
        
        # Get window handle
        hwnd = window._hWnd
        
        if debug:
            print(f"Attempting direct capture of window handle: {hwnd}")
        
        # Get window device context
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        
        # Get window dimensions
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        width = right - left
        height = bottom - top
        
        if debug:
            print(f"Client rect: {width}x{height}")
        
        # Create bitmap
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)
        
        # Copy the window content (this should exclude overlays from other processes)
        result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)  # PW_RENDERFULLCONTENT
        
        if not result:
            # Fallback to BitBlt
            saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
        
        # Convert to numpy array
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        
        img = np.frombuffer(bmpstr, dtype='uint8')
        img.shape = (height, width, 4)  # BGRA format
        
        # Convert BGRA to BGR
        img = img[:, :, :3]  # Remove alpha channel
        img = img[:, :, ::-1]  # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Back to BGR for OpenCV
        
        # Clean up
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        if debug:
            print(f"Direct capture successful: {img.shape}")
            
        return img
        
    except Exception as e:
        if debug:
            print(f"Direct capture failed: {e}")
        return None

def capture_emulator_without_overlay(window, debug=False):
    """
    Master function that tries multiple approaches to get clean emulator capture
    """
    if debug:
        print("=== Attempting clean emulator capture ===")
    
    # Method 1: Try direct window capture (should exclude overlays from other processes)
    try:
        direct_capture = capture_emulator_direct(window, debug=debug)
        if direct_capture is not None and direct_capture.size > 0:
            if debug:
                print("+ Direct capture successful")
            # Still apply UI cropping to remove emulator's own UI
            return crop_emulator_ui(direct_capture, debug=debug)
    except Exception as e:
        if debug:
            print(f"- Direct capture failed: {e}")
    
    # Method 2: Try to find a cleaner window
    try:
        clean_window = find_emulator_window_only("mGBA")
        if clean_window._hWnd != window._hWnd:
            if debug:
                print(f"+ Found different emulator window: {clean_window.title}")
            return capture_emulator_game_area(clean_window, debug=debug)
    except Exception as e:
        if debug:
            print(f"- Clean window search failed: {e}")
    
    # Method 3: Fallback to original method
    if debug:
        print("-> Falling back to original capture method")
    return capture_emulator_game_area(window, debug=debug)

def detect_game_screen_region(frame):
    """
    Detect the actual game screen region within the emulator window.
    Returns (x1, y1, x2, y2) coordinates of the game area, or None if not found.
    """
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find edges to detect screen boundaries
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for the largest rectangular contour (likely the game screen)
    best_area = 0
    best_rect = None
    
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If it's roughly rectangular (4 corners)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            
            # Must be reasonably large (at least 25% of total area)
            if area > (w * h * 0.25) and area > best_area:
                x, y, rect_w, rect_h = cv2.boundingRect(contour)
                
                # Aspect ratio should be reasonable for a game screen
                aspect_ratio = rect_w / rect_h
                if 1.2 < aspect_ratio < 2.0:  # GBA is roughly 1.5:1
                    best_area = area
                    best_rect = (x, y, x + rect_w, y + rect_h)
    
    return best_rect

def extract_text_from_screen(debug=False):
    window = find_emulator_window()
    frame = capture_window(window)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    pil_img = Image.fromarray(thresh)
    text = pytesseract.image_to_string(pil_img, config='--oem 3 --psm 6')

    if debug:
        print("\n[OCR Result]:\n", text)
    return text

def detect_game_state(frame):
    if frame is None or frame.size == 0:
        return "Unknown"
    
    h, w, _ = frame.shape
    roi = frame[int(h * 0.3):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
    
    # Convert to different color spaces for better detection
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculate various pixel counts
    black_pixels = np.sum(gray < 40)
    white_pixels = np.sum(gray > 200)
    total_pixels = roi.shape[0] * roi.shape[1]
    
    # Color detection for various environments
    green_mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([80, 255, 255]))
    green_pixels = np.sum(green_mask > 0)
    
    blue_mask = cv2.inRange(hsv, np.array([100, 40, 40]), np.array([130, 255, 255]))
    blue_pixels = np.sum(blue_mask > 0)
    
    red_mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    red_pixels = np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)
    
    # Pink detection (Pokemon Center healing machines)
    pink_mask = cv2.inRange(hsv, np.array([140, 50, 50]), np.array([170, 255, 255]))
    pink_pixels = np.sum(pink_mask > 0)
    
    # Yellow detection (shops, electric attacks)
    yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
    yellow_pixels = np.sum(yellow_mask > 0)
    
    # Calculate percentages
    black_pct = black_pixels / total_pixels
    white_pct = white_pixels / total_pixels
    green_pct = green_pixels / total_pixels
    blue_pct = blue_pixels / total_pixels
    red_pct = red_pixels / total_pixels
    pink_pct = pink_pixels / total_pixels
    yellow_pct = yellow_pixels / total_pixels
    
    # Advanced state detection logic
    # PRIORITIZE BATTLE DETECTION - check for battle UI first before environmental detection
    
    # Battle detection - ONLY use UI elements, ignore colors
    if detect_actual_battle_state(frame):
        return "Battle"
    
    # Check for specific locations (after battle detection)
    elif detect_pokemon_center(frame, pink_pct, white_pct):
        return "Pokemon Center"
    elif detect_shop(frame, yellow_pct, white_pct):
        return "Shop"
    elif detect_gym(frame, blue_pct, red_pct):
        return "Gym"
    
    # Check for dialogue/menu states - be more lenient
    elif black_pct > 0.3 or white_pct > 0.5:
        return "Dialogue/Menu"
    
    # Environment detection (after all other checks)
    elif green_pct > 0.25:
        return "Overworld"
    elif blue_pct > 0.3:
        return "Water/Flying"
    elif black_pct > 0.2:
        # Distinguish between Indoor and Cave environments
        if detect_cave_environment(frame, black_pct, gray):
            return "Cave"
        else:
            return "Indoor"
    else:
        return "Overworld"

def analyze_screen_regions(frame):
    """Analyzes different regions of the screen for specific UI elements"""
    if frame is None or frame.size == 0:
        return {}
    
    h, w, _ = frame.shape
    
    # Define regions of interest
    top_region = frame[0:int(h*0.2), :]  # UI elements, menus
    bottom_region = frame[int(h*0.8):h, :]  # Text boxes, dialogue
    center_region = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]  # Main game area
    
    analysis = {
        'has_text_box': detect_text_box(bottom_region),
        'has_menu': detect_menu(top_region),
        'player_visible': detect_player_sprite(center_region),
        'health_bar_visible': detect_health_bar(top_region)
    }
    
    return analysis

def detect_text_box(region):
    """Detects presence of dialogue/text boxes"""
    if region.size == 0:
        return False
    
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for rectangular contours that could be text boxes
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Minimum size for text box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 2 < aspect_ratio < 8:  # Text boxes are typically wide
                return True
    return False

def detect_menu(region):
    """Detects presence of menus or UI elements"""
    if region.size == 0:
        return False
    
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # Look for high contrast areas typical of UI elements
    contrast = cv2.Laplacian(gray, cv2.CV_64F).var()
    return contrast > 500

def detect_player_sprite(region):
    """Basic detection of player character sprite"""
    if region.size == 0:
        return False
    
    # This is a simple heuristic - in practice you'd want template matching
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Look for common player sprite colors (this would need game-specific tuning)
    player_colors = [
        ([0, 100, 100], [10, 255, 255]),    # Red clothes
        ([20, 100, 100], [30, 255, 255]),   # Yellow/brown hair
        ([100, 100, 100], [120, 255, 255]), # Blue clothes
    ]
    
    for lower, upper in player_colors:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if np.sum(mask > 0) > 100:  # Minimum pixels to consider sprite present
            return True
    return False

def detect_health_bar(region):
    """Detects health/status bars"""
    if region.size == 0:
        return False
    
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Look for green health bars
    green_mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))
    # Look for red health bars (low health)
    red_mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    
    total_health_pixels = np.sum(green_mask > 0) + np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)
    return total_health_pixels > 50

def detect_wild_battle(frame, red_pct, green_pct):
    """Detects wild Pokemon battles using UI elements instead of just colors"""
    # First check if we have actual battle UI
    has_battle_ui = detect_actual_battle_state(frame)
    
    if not has_battle_ui:
        return False
    
    h, w, _ = frame.shape
    background_region = frame[int(h*0.1):int(h*0.6), :]  # Upper portion where background is visible
    
    hsv_bg = cv2.cvtColor(background_region, cv2.COLOR_BGR2HSV)
    # Wild battles often have green/brown nature backgrounds
    nature_mask = cv2.inRange(hsv_bg, np.array([25, 30, 30]), np.array([85, 255, 255]))
    nature_pct = np.sum(nature_mask > 0) / (background_region.shape[0] * background_region.shape[1])
    
    # Only return true if we have both UI elements AND nature background
    return has_battle_ui and nature_pct > 0.15

def detect_trainer_battle(frame, red_pct, blue_pct):
    """Detects trainer battles using UI elements instead of just colors"""
    # First check if we have actual battle UI
    has_battle_ui = detect_actual_battle_state(frame)
    
    if not has_battle_ui:
        return False
    
    h, w, _ = frame.shape
    background_region = frame[int(h*0.1):int(h*0.6), :]
    
    hsv_bg = cv2.cvtColor(background_region, cv2.COLOR_BGR2HSV)
    # Trainer battles often have more blue/structured backgrounds
    structure_mask = cv2.inRange(hsv_bg, np.array([90, 30, 30]), np.array([130, 255, 255]))
    structure_pct = np.sum(structure_mask > 0) / (background_region.shape[0] * background_region.shape[1])
    
    # Only return true if we have both UI elements AND structured background
    return has_battle_ui and structure_pct > 0.1

def detect_pokemon_center(frame, pink_pct, white_pct):
    """Detects Pokemon Centers - distinctive pink healing machines and white/clean interior"""
    # Pokemon Centers have characteristic pink healing machines and clean white interiors
    return pink_pct > 0.08 and white_pct > 0.3

def detect_shop(frame, yellow_pct, white_pct):
    """Detects shops/marts - often have yellow signage and clean interiors"""
    h, w, _ = frame.shape
    top_region = frame[0:int(h*0.4), :]  # Check upper area for shop signs
    
    hsv_top = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
    # Look for yellow/orange shop signs
    shop_sign_mask = cv2.inRange(hsv_top, np.array([15, 100, 100]), np.array([35, 255, 255]))
    sign_pct = np.sum(shop_sign_mask > 0) / (top_region.shape[0] * top_region.shape[1])
    
    # More strict criteria to avoid false positives with menus
    # Require both significant yellow signage AND specific shop indicators
    return sign_pct > 0.1 and white_pct > 0.4 and yellow_pct > 0.08

def detect_gym(frame, blue_pct, red_pct):
    """Detects Pokemon Gyms - often have distinctive colored themes"""
    h, w, _ = frame.shape
    
    # First check if this might be a battle - if so, don't detect gym
    # Battles have moving elements and UI, gyms are more static
    battle_ui_present = detect_actual_battle_state(frame)
    if battle_ui_present:
        return False  # Don't detect gym if battle UI is present
    
    # Gyms often have strong thematic colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Check for gym-specific color schemes
    # Electric gym: yellow, Water gym: blue, Fire gym: red, etc.
    electric_mask = cv2.inRange(hsv, np.array([20, 150, 150]), np.array([30, 255, 255]))
    electric_pct = np.sum(electric_mask > 0) / (frame.shape[0] * frame.shape[1])
    
    # Much more conservative gym detection - require very specific conditions
    # Strong thematic coloring AND no battle indicators
    return ((blue_pct > 0.5 or red_pct > 0.2 or electric_pct > 0.2) and 
            (blue_pct + red_pct + electric_pct) > 0.4 and
            not battle_ui_present)

def detect_cave_environment(frame, black_pct, gray) -> bool:
    """Distinguish cave from indoor environments"""
    if frame is None:
        return False
    
    h, w, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Caves typically have:
    # 1. More irregular/rough textures (higher edge density)
    # 2. Brown/gray rocky colors
    # 3. Very dark areas (shadows)
    # 4. Less uniform lighting than indoor areas
    
    # Check for rocky/earthy colors (brown, gray, dark colors)
    brown_mask = cv2.inRange(hsv, np.array([10, 50, 20]), np.array([20, 255, 150]))
    brown_pct = np.sum(brown_mask > 0) / (h * w)
    
    # Check for very dark areas (deep shadows in caves)
    very_dark_mask = gray < 20
    very_dark_pct = np.sum(very_dark_mask) / (h * w)
    
    # Check edge density (caves have more irregular textures)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / (h * w)
    
    # Check color variance (caves have more varied colors than clean indoor spaces)
    color_std = np.std(hsv[:,:,1])  # Saturation variance
    
    # Cave indicators
    has_rocky_colors = brown_pct > 0.1
    has_deep_shadows = very_dark_pct > 0.3
    has_rough_texture = edge_density > 0.15
    has_color_variation = color_std > 30
    
    # Cave if multiple indicators present
    cave_score = sum([has_rocky_colors, has_deep_shadows, has_rough_texture, has_color_variation])
    
    return cave_score >= 2  # At least 2 cave indicators

def detect_battle_transition(frame) -> bool:
    """DISABLED - was too aggressive. Detect actual battle transitions by looking for black particle effects and screen flashes"""
    # TEMPORARILY DISABLED - was causing too many false positives
    return False
    
    # Original code commented out:
    # if frame is None:
    #     return False
    # 
    # h, w, _ = frame.shape
    # 
    # # Convert to grayscale for better particle detection
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 
    # # Detect black particles (battle transition effect)
    # black_threshold = 30  # Very dark pixels
    # black_mask = gray < black_threshold
    # black_pct = np.sum(black_mask) / (h * w)
    # 
    # # Detect sudden brightness changes (screen flash effect)
    # bright_mask = gray > 200
    # bright_pct = np.sum(bright_mask) / (h * w)
    # 
    # # Battle transitions have lots of moving black particles
    # if black_pct > 0.3:  # More than 30% very dark pixels
    #     return True
    # 
    # # Screen flash effect during transition
    # if bright_pct > 0.7:  # More than 70% bright pixels (flash)
    #     return True
    # 
    # # Check for the characteristic "swirl" pattern of battle transitions
    # # Look for edge density which increases during particle effects
    # edges = cv2.Canny(gray, 50, 150)
    # edge_density = np.sum(edges > 0) / (h * w)
    # 
    # if edge_density > 0.15:  # High edge density suggests particle movement
    #     return True
    # 
    # return False

def detect_actual_battle_state(frame) -> bool:
    """Enhanced battle detection - look for health bars and battle UI elements"""
    if frame is None:
        return False
    
    h, w, _ = frame.shape
    
    # Check multiple battle indicators
    battle_indicators = 0
    
    # 1. Look for health bars in the top region
    top_region = frame[0:int(h*0.25), :]  # Top 25% where health bars appear
    if detect_health_bars_detailed(top_region):
        battle_indicators += 2  # Health bars are strong indicator
    
    # 2. Look for battle menu UI in bottom region
    bottom_region = frame[int(h*0.7):h, :]  # Bottom 30% where battle menus appear
    if detect_battle_menu_ui(bottom_region):
        battle_indicators += 2  # Battle menu is strong indicator
    
    # 3. Look for HP/PP text indicators
    if detect_battle_text_elements(frame):
        battle_indicators += 1
    
    # 4. Look for move selection boxes (4-option grid)
    if detect_move_selection_grid(bottom_region):
        battle_indicators += 2
    
    # Need at least 2 indicators for confident battle detection
    return battle_indicators >= 2

def detect_health_bars_detailed(region) -> bool:
    """Detect health bars more precisely"""
    if region.size == 0:
        return False
    
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, w, _ = region.shape
    
    # Look for green health bars (common in Pokemon)
    green_mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))
    
    # Look for yellow health bars (low health)
    yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    
    # Look for red health bars (critical health)
    red_mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Health bars are typically horizontal rectangles
    health_pixels = np.sum(green_mask > 0) + np.sum(yellow_mask > 0) + np.sum(red_mask > 0)
    
    # Also look for the characteristic horizontal pattern of health bars
    if health_pixels > 50:  # Minimum pixels for health bar
        # Check if the colored pixels form horizontal patterns
        for mask in [green_mask, yellow_mask, red_mask]:
            if np.sum(mask > 0) > 20:
                # Find contours to check shape
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w_c, h_c = cv2.boundingRect(contour)
                    aspect_ratio = w_c / h_c if h_c > 0 else 0
                    # Health bars are wide and short
                    if aspect_ratio > 3 and w_c > 30:
                        return True
    
    return False

def detect_battle_menu_ui(region) -> bool:
    """Detect battle menu UI elements"""
    if region.size == 0:
        return False
    
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Look for menu boxes with clear borders
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    menu_boxes = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Reasonable size for menu elements
            x, y, w_c, h_c = cv2.boundingRect(contour)
            aspect_ratio = w_c / h_c if h_c > 0 else 0
            # Battle menus are typically rectangular
            if 1.5 < aspect_ratio < 6 and w_c > 50 and h_c > 20:
                menu_boxes += 1
    
    return menu_boxes >= 1

def detect_battle_text_elements(frame) -> bool:
    """Look for battle-specific text elements like HP, PP"""
    # This is a simplified check - in practice you'd use OCR
    # For now, just check for text-like patterns in likely locations
    h, w, _ = frame.shape
    
    # Check areas where battle text typically appears
    text_regions = [
        frame[0:int(h*0.2), int(w*0.6):w],  # Top-right (opponent info)
        frame[int(h*0.5):int(h*0.7), 0:int(w*0.4)],  # Mid-left (player info)
    ]
    
    for region in text_regions:
        if region.size == 0:
            continue
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # Look for text-like patterns (high contrast areas)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Text has many small contours in organized patterns
        small_contours = [c for c in contours if 10 < cv2.contourArea(c) < 200]
        if len(small_contours) > 5:  # Likely text
            return True
    
    return False

def detect_move_selection_grid(region) -> bool:
    """Detect the 2x2 move selection grid in battle"""
    if region.size == 0:
        return False
    
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Look for a grid pattern (multiple rectangular selections)
    edges = cv2.Canny(gray, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for multiple similar-sized rectangles (move options)
    rectangles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area < 2000:  # Move buttons are medium-sized
            x, y, w_c, h_c = cv2.boundingRect(contour)
            aspect_ratio = w_c / h_c if h_c > 0 else 0
            if 1.2 < aspect_ratio < 4:  # Move buttons are rectangular
                rectangles.append((x, y, w_c, h_c))
    
    # Battle move selection typically has 2-4 options
    return len(rectangles) >= 2