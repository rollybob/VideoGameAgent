#!/usr/bin/env python3
"""
🤖 AI-ASSISTED DATA COLLECTOR
=============================

Live assisted labeling system:
- AI attempts object detection first
- User corrects/confirms detections
- Much faster than manual labeling
- Auto-minimizes window during capture
"""

import cv2
import numpy as np
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from PIL import Image, ImageTk, ImageGrab
import win32gui
import win32con
import win32ui
from ctypes import windll

class AssistedDataCollector:
    """AI-assisted data collection tool"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI-Assisted Pokemon Data Collector")
        self.root.geometry("1800x1000")  # Wider for better game view
        
        # Window management
        self.always_on_top = False
        
        # Data storage
        self.current_screenshot = None
        self.ai_detections = []
        self.confirmed_labels = []
        self.emulator_window = None
        
        # Live capture
        self.live_mode = False
        self.capture_thread = None
        self.live_paused = False  # Pause live mode during editing
        
        # Manual annotation mode
        self.annotation_mode = False
        self.drawing_box = False
        self.box_start_pos = None
        self.manual_annotations = []
        
        # Attention system
        self.attention_mode = False
        self.attention_points = []
        
        # Status variable (not displayed to maximize view space)
        self.main_status_var = tk.StringVar(value="Ready")
        
        # Directories
        self.data_dir = Path("assisted_training_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Object types for Pokemon games
        self.object_types = [
            "hp_bar", "player_character", "pokemon_sprite", "npc_character", 
            "menu_box", "dialogue_box", "item_icon", "building", "tree",
            "battle_menu", "text_area", "status_icon", "map_element", "button"
        ]
        
        # AI detection confidence threshold
        self.confidence_threshold = 0.5
        
        # Display settings
        self.zoom_level = 2.0  # Default 2x zoom
        self.selected_detection_idx = -1
        
        # Current annotation type for manual drawing
        self.current_annotation_type = "player_character"
        
        self.setup_gui()
        self.setup_directories()
        
    def setup_directories(self):
        """Setup directory structure"""
        subdirs = ["screenshots", "confirmed", "metadata"]
        for subdir in subdirs:
            (self.data_dir / subdir).mkdir(exist_ok=True)
            
    def setup_gui(self):
        """Setup the main GUI interface"""
        # Create main panels
        self.create_control_panel()
        self.create_live_display_panel()
        self.create_detection_panel()
        
    def create_control_panel(self):
        """Create control panel with scrollable content"""
        # Main control frame container
        control_container = ttk.LabelFrame(self.root, text="Live AI-Assisted Collection", padding=5)
        control_container.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Create canvas and scrollbar for scrollable content
        canvas = tk.Canvas(control_container, width=280)
        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=canvas.yview)
        
        # Scrollable frame inside canvas
        control_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        control_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=control_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", on_mousewheel)
        
        # Bind mouse wheel to canvas when mouse enters
        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', bind_mousewheel)
        canvas.bind('<Leave>', unbind_mousewheel)
        
        # Emulator setup
        ttk.Label(control_frame, text="Emulator Setup", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        self.emulator_var = tk.StringVar(value="No emulator found")
        ttk.Label(control_frame, textvariable=self.emulator_var, wraplength=250).pack(anchor=tk.W)
        
        ttk.Button(control_frame, text="Find Emulator Window", 
                  command=self.find_emulator_window).pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Live mode controls
        ttk.Label(control_frame, text="Live Detection Mode", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        self.live_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Enable Live Mode", 
                       variable=self.live_var, command=self.toggle_live_mode).pack(anchor=tk.W, pady=2)
        
        # Confidence threshold
        ttk.Label(control_frame, text="AI Confidence Threshold:").pack(anchor=tk.W, pady=(10, 2))
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(control_frame, from_=0.1, to=0.9, orient=tk.HORIZONTAL,
                                   variable=self.confidence_var, length=200)
        confidence_scale.pack(fill=tk.X, pady=2)
        
        self.confidence_label_var = tk.StringVar(value="0.5")
        ttk.Label(control_frame, textvariable=self.confidence_label_var).pack(anchor=tk.W)
        confidence_scale.config(command=self.update_confidence_display)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Window controls
        ttk.Label(control_frame, text="Window Options:", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        self.on_top_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Always on top", 
                       variable=self.on_top_var, command=self.toggle_always_on_top).pack(anchor=tk.W, pady=2)
        
        ttk.Button(control_frame, text="Pause Live Mode", 
                  command=self.toggle_live_pause).pack(fill=tk.X, pady=2)
        
        self.pause_status_var = tk.StringVar(value="Live: Active")
        ttk.Label(control_frame, textvariable=self.pause_status_var, font=("Arial", 9)).pack(anchor=tk.W)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Zoom controls
        ttk.Label(control_frame, text="Display Zoom:", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Zoom slider (1.0x to 5.0x with 0.1 increments)
        self.zoom_var = tk.DoubleVar(value=2.0)
        zoom_slider = ttk.Scale(control_frame, from_=1.0, to=5.0, orient=tk.HORIZONTAL,
                               variable=self.zoom_var, length=220)
        zoom_slider.pack(fill=tk.X, pady=2)
        zoom_slider.config(command=self.on_zoom_changed)
        
        self.zoom_label_var = tk.StringVar(value="Current: 2.0x")
        ttk.Label(control_frame, textvariable=self.zoom_label_var, font=("Arial", 9)).pack(anchor=tk.W)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Attention system controls
        ttk.Label(control_frame, text="Attention System", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        self.attention_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Enable Click Attention", 
                       variable=self.attention_var, command=self.toggle_attention_mode).pack(anchor=tk.W, pady=2)
        
        ttk.Button(control_frame, text="Clear Attention Points", 
                  command=self.clear_attention_points).pack(fill=tk.X, pady=2)
        
        attention_help = tk.Text(control_frame, height=3, width=25, wrap=tk.WORD, font=("Arial", 8))
        attention_help.pack(fill=tk.X, pady=2)
        attention_help.insert(tk.END, 
            "Click areas in live view to highlight them. "
            "Attention circles fade over time. "
            "Use to focus AI on specific regions.")
        attention_help.config(state=tk.DISABLED)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Manual annotation controls
        ttk.Label(control_frame, text="Manual Annotation", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        self.annotation_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Enable Manual Drawing", 
                       variable=self.annotation_var, command=self.toggle_annotation_mode).pack(anchor=tk.W, pady=2)
        
        ttk.Label(control_frame, text="Draw Object Type:").pack(anchor=tk.W, pady=(5, 2))
        self.annotation_type_var = tk.StringVar(value="player_character")
        annotation_combo = ttk.Combobox(control_frame, textvariable=self.annotation_type_var, 
                                      values=self.object_types, state="readonly", width=18)
        annotation_combo.pack(fill=tk.X, pady=2)
        
        instructions_manual = tk.Text(control_frame, height=4, width=25, wrap=tk.WORD, font=("Arial", 8))
        instructions_manual.pack(fill=tk.X, pady=2)
        instructions_manual.insert(tk.END, 
            "Manual Mode:\n"
            "• Enable drawing mode above\n"
            "• Click and drag to draw boxes\n"
            "• Right-click to delete last box")
        instructions_manual.config(state=tk.DISABLED)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Manual controls
        ttk.Label(control_frame, text="Quick Actions", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Button(control_frame, text="Capture & Analyze", 
                  command=self.manual_capture_analyze).pack(fill=tk.X, pady=2)
        
        ttk.Button(control_frame, text="Clear All Detections", 
                  command=self.clear_all_detections).pack(fill=tk.X, pady=2)
        
        ttk.Button(control_frame, text="Focus AI on Attention Areas", 
                  command=self.focus_ai_on_attention).pack(fill=tk.X, pady=2)
        
        ttk.Button(control_frame, text="Confirm All Detections", 
                  command=self.confirm_all_detections).pack(fill=tk.X, pady=2)
        
        ttk.Button(control_frame, text="Save Confirmed Labels", 
                  command=self.save_confirmed_labels).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Statistics
        ttk.Label(control_frame, text="Collection Stats", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        self.stats_var = tk.StringVar(value="Confirmed: 0\nSaved: 0")
        ttk.Label(control_frame, textvariable=self.stats_var, justify=tk.LEFT).pack(anchor=tk.W)
        
        ttk.Button(control_frame, text="View Detailed Stats", 
                  command=self.show_detailed_stats).pack(fill=tk.X, pady=5)
        
    def create_live_display_panel(self):
        """Create live display panel"""
        display_frame = ttk.LabelFrame(self.root, text="Live Game View", padding=10)
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)
        
        # Canvas for live display (much larger - no status below it)
        self.live_canvas = tk.Canvas(display_frame, bg='black', width=800, height=600)
        self.live_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind click events for corrections and manual annotation
        self.live_canvas.bind("<Button-1>", self.on_canvas_left_click)
        self.live_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.live_canvas.bind("<ButtonRelease-1>", self.on_canvas_left_release)
        self.live_canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.live_canvas.focus_set()
        
        # Live status will be moved to status panel
        
    def create_detection_panel(self):
        """Create AI detection results panel"""
        detection_frame = ttk.LabelFrame(self.root, text="AI Detections & Corrections", padding=10)
        detection_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        
        # Detection list
        ttk.Label(detection_frame, text="AI Detected Objects:", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        
        # Scrollable list of detections
        list_frame = ttk.Frame(detection_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Enable multiple selection for batch operations
        self.detection_listbox = tk.Listbox(list_frame, height=15, font=("Courier", 9), 
                                          selectmode=tk.EXTENDED)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        
        self.detection_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.detection_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.detection_listbox.yview)
        
        # Bind selection events
        self.detection_listbox.bind('<<ListboxSelect>>', self.on_detection_select)
        
        # Add help text for multi-selection
        help_label = tk.Label(list_frame, text="Ctrl+Click for multi-select", 
                             font=("Arial", 8), fg="gray")
        help_label.pack(anchor=tk.W, pady=(2, 0))
        
        # Correction controls
        ttk.Label(detection_frame, text="Correction Controls:", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(15, 5))
        
        # Object type correction
        ttk.Label(detection_frame, text="Correct Object Type:").pack(anchor=tk.W)
        self.correction_var = tk.StringVar(value=self.object_types[0])
        self.correction_combo = ttk.Combobox(detection_frame, textvariable=self.correction_var, 
                                           values=self.object_types, state="readonly")
        self.correction_combo.pack(fill=tk.X, pady=2)
        
        # Bind combo selection to maintain detection selection
        self.correction_combo.bind('<<ComboboxSelected>>', self.on_correction_selected)
        
        # Single selection buttons
        ttk.Label(detection_frame, text="Single Selection:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        
        ttk.Button(detection_frame, text="✓ Confirm Detection", 
                  command=self.confirm_selected_detection).pack(fill=tk.X, pady=1)
        
        ttk.Button(detection_frame, text="✏ Correct & Confirm", 
                  command=self.correct_and_confirm).pack(fill=tk.X, pady=1)
        
        ttk.Button(detection_frame, text="✗ Delete Detection", 
                  command=self.delete_selected_detection).pack(fill=tk.X, pady=1)
        
        # Batch operation buttons
        ttk.Label(detection_frame, text="Batch Operations:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        
        ttk.Button(detection_frame, text="🔄 Batch Correct Selected", 
                  command=self.batch_correct_selected).pack(fill=tk.X, pady=1)
        
        ttk.Button(detection_frame, text="✓ Batch Confirm Selected", 
                  command=self.batch_confirm_selected).pack(fill=tk.X, pady=1)
        
        ttk.Button(detection_frame, text="✗ Batch Delete Selected", 
                  command=self.batch_delete_selected).pack(fill=tk.X, pady=1)
        
        ttk.Separator(detection_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Confirmed labels
        ttk.Label(detection_frame, text="Confirmed Labels:", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        
        self.confirmed_listbox = tk.Listbox(detection_frame, height=8, font=("Courier", 9))
        self.confirmed_listbox.pack(fill=tk.X, pady=5)
        
        
        
    def find_emulator_window(self):
        """Find emulator window with dropdown selection"""
        windows = []
        
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)
                
                # Skip our own window
                if "AI-Assisted Pokemon Data Collector" in window_text:
                    return
                    
                # Look for potential emulator windows
                if window_text and (
                    'mgba' in window_text.lower() or 
                    'pokemon' in window_text.lower() or 
                    'gba' in window_text.lower() or
                    'emulator' in window_text.lower() or
                    'mgba' in class_name.lower() or
                    'visualboy' in window_text.lower() or
                    'advance' in window_text.lower() or
                    # Also include any window with substantial content
                    (len(window_text) > 3 and 
                     class_name not in ['Shell_TrayWnd', 'DV2ControlHost', 'Windows.UI.Core.CoreWindow'])
                ):
                    windows.append((hwnd, window_text, class_name))
        
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        if not windows:
            messagebox.showwarning("No Windows Found", 
                                 "No potential emulator windows found.\n"
                                 "Please ensure your emulator is running and visible.")
            return
            
        if len(windows) == 1:
            # Only one window found, use it
            self.emulator_window = windows[0][0]
            self.emulator_var.set(f"Selected: {windows[0][1]}")
            self.main_status_var.set("Emulator window selected! Enable live mode or use manual capture.")
        else:
            # Multiple windows found, let user choose
            self.choose_emulator_window(windows)
            
    def choose_emulator_window(self, windows):
        """Let user choose from multiple detected windows"""
        choice_window = tk.Toplevel(self.root)
        choice_window.title("Choose Emulator Window")
        choice_window.geometry("600x400")
        choice_window.transient(self.root)
        choice_window.grab_set()
        
        # Center the window
        choice_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 100,
            self.root.winfo_rooty() + 100
        ))
        
        ttk.Label(choice_window, text="Multiple windows found. Choose your emulator:", 
                 font=("Arial", 12)).pack(pady=10)
        
        # Create frame for listbox and scrollbar
        list_frame = ttk.Frame(choice_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Listbox with scrollbar
        listbox = tk.Listbox(list_frame, height=15, font=("Courier", 10))
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        
        # Populate listbox with window information
        for i, (hwnd, title, class_name) in enumerate(windows):
            # Get window size for additional info
            try:
                rect = win32gui.GetWindowRect(hwnd)
                width = rect[2] - rect[0]
                height = rect[3] - rect[1]
                size_info = f"({width}x{height})"
            except:
                size_info = "(size unknown)"
                
            display_text = f"{i+1:2d}. {title[:50]:50s} {size_info:15s} [{class_name}]"
            listbox.insert(tk.END, display_text)
            
        # Pre-select first item
        listbox.selection_set(0)
        
        # Button frame
        button_frame = ttk.Frame(choice_window)
        button_frame.pack(pady=10)
        
        def select_window():
            selection = listbox.curselection()
            if selection:
                selected_idx = selection[0]
                self.emulator_window = windows[selected_idx][0]
                self.emulator_var.set(f"Selected: {windows[selected_idx][1]}")
                self.main_status_var.set("Emulator window selected! Enable live mode or use manual capture.")
                choice_window.destroy()
            else:
                messagebox.showwarning("No Selection", "Please select a window first!")
                
        def cancel_selection():
            choice_window.destroy()
            
        def refresh_windows():
            choice_window.destroy()
            self.find_emulator_window()  # Recursively call to refresh
            
        ttk.Button(button_frame, text="Select This Window", 
                  command=select_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Refresh List", 
                  command=refresh_windows).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=cancel_selection).pack(side=tk.LEFT, padx=5)
                  
        # Help text
        help_text = tk.Text(choice_window, height=4, wrap=tk.WORD, font=("Arial", 9))
        help_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        help_text.insert(tk.END, 
            "Tips:\n"
            "• Look for your emulator name (mGBA, VisualBoyAdvance, etc.)\n"
            "• Game window size is usually 240x160 or multiples (480x320, 720x480)\n"
            "• If unsure, try 'Refresh List' after focusing your emulator window")
        help_text.config(state=tk.DISABLED)
        
    def toggle_live_mode(self):
        """Toggle live detection mode"""
        if not self.emulator_window:
            self.live_var.set(False)
            messagebox.showwarning("No Emulator", "Please find the emulator window first!")
            return
            
        self.live_mode = self.live_var.get()
        
        if self.live_mode:
            self.start_live_capture()
        else:
            self.stop_live_capture()
            
    def start_live_capture(self):
        """Start live capture and detection"""
        # Live mode status removed to maximize view window space
        self.main_status_var.set("Live mode active")
        
        def live_capture_loop():
            fps_counter = 0
            fps_start_time = time.time()
            
            while self.live_mode:
                try:
                    # Check if live mode is paused
                    if self.live_paused:
                        time.sleep(0.5)
                        continue
                        
                    # Capture screenshot without minimizing any windows
                    screenshot = self.capture_emulator_screenshot()
                    
                    if screenshot is not None:
                        # Run AI detection only if not paused
                        detections = self.run_ai_detection(screenshot)
                        
                        # Update display only if still not paused
                        if not self.live_paused:
                            self.root.after(0, lambda: self.update_live_display(screenshot, detections))
                        
                        # Update FPS
                        fps_counter += 1
                        if time.time() - fps_start_time >= 1.0:
                            fps = fps_counter
                            # FPS display removed to minimize status bar clutter
                            pass
                            fps_counter = 0
                            fps_start_time = time.time()
                    
                    time.sleep(0.2)  # ~5 FPS to be less aggressive
                    
                except Exception as e:
                    print(f"Live capture error: {e}")
                    time.sleep(0.5)
                    
        self.capture_thread = threading.Thread(target=live_capture_loop, daemon=True)
        self.capture_thread.start()
        
    def stop_live_capture(self):
        """Stop live capture"""
        self.live_mode = False
        # Live mode status removed to maximize view window space
        self.main_status_var.set("Live mode stopped")
        
    def capture_emulator_screenshot(self):
        """Capture screenshot directly from emulator window (no overlay interference)"""
        try:
            # Check if emulator window still exists
            if not win32gui.IsWindow(self.emulator_window):
                print("Emulator window no longer exists")
                return None
                
            # Try direct window capture first (pure window content)
            direct_result = self.capture_window_direct()
            if direct_result is not None:
                return direct_result
                
            # Fallback to screen grab if direct capture fails
            print("Direct capture failed, using screen grab fallback")
            rect = win32gui.GetWindowRect(self.emulator_window)
            
            # Check if window is minimized or has zero size
            if rect[2] - rect[0] <= 0 or rect[3] - rect[1] <= 0:
                print("Emulator window is minimized or has zero size")
                return None
            
            # Fallback: Capture window content with potential overlay
            screenshot = ImageGrab.grab(bbox=rect)
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            return screenshot_cv
            
        except Exception as e:
            print(f"Screenshot capture error: {e}")
            return None
            
    def capture_window_direct(self):
        """Capture pure window content using Windows API (no overlays)"""
        try:
            # Get client area dimensions (excludes title bar and borders)
            client_rect = win32gui.GetClientRect(self.emulator_window)
            width = client_rect[2]
            height = client_rect[3]
            
            if width <= 0 or height <= 0:
                return None
                
            # Get window device context
            hwndDC = win32gui.GetWindowDC(self.emulator_window)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # Create bitmap
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # Copy the window content to our bitmap (pure window, no overlays)
            result = windll.user32.PrintWindow(self.emulator_window, saveDC.GetSafeHdc(), 1)
            
            if result:
                # Convert to PIL Image
                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)
                
                img = Image.frombuffer(
                    'RGB',
                    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                    bmpstr, 'raw', 'BGRX', 0, 1
                )
                
                # Convert to OpenCV format
                img_array = np.array(img)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Cleanup Windows API resources
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(self.emulator_window, hwndDC)
                
                return img_array
            else:
                # Cleanup on failure
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(self.emulator_window, hwndDC)
                return None
                
        except Exception as e:
            print(f"Direct window capture error: {e}")
            return None
            
    def run_ai_detection(self, screenshot):
        """Run AI object detection on screenshot"""
        # This is a simplified detection system
        # In a real implementation, you'd use a trained model
        
        detections = []
        
        # Convert to different color spaces for detection
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        
        height, width = screenshot.shape[:2]
        
        # Detect potential HP bars (red/green horizontal rectangles)
        hp_detections = self.detect_hp_bars(screenshot, hsv)
        detections.extend(hp_detections)
        
        # Detect text areas (white text on dark background)
        text_detections = self.detect_text_areas(screenshot, gray)
        detections.extend(text_detections)
        
        # Detect menu boxes (rectangular areas with borders)
        menu_detections = self.detect_menu_boxes(screenshot, gray)
        detections.extend(menu_detections)
        
        # Detect potential character sprites (colored areas)
        sprite_detections = self.detect_sprites(screenshot, hsv)
        detections.extend(sprite_detections)
        
        # Filter by confidence and remove duplicates
        filtered_detections = self.filter_detections(detections)
        
        return filtered_detections
        
    def detect_hp_bars(self, image, hsv):
        """Detect HP bars with ultra-strict filtering to eliminate false positives"""
        detections = []
        height, width = image.shape[:2]
        
        # Only look for HP bars in UI regions to avoid trees/objects
        ui_regions = [
            (0, 0, width, int(height * 0.25)),  # Top UI area
            (0, int(height * 0.75), int(width * 0.6), height)  # Bottom-left UI area
        ]
        
        for region_x, region_y, region_w, region_h in ui_regions:
            # Extract UI region
            roi = hsv[region_y:region_h, region_x:region_w]
            if roi.size == 0:
                continue
                
            # Very specific HP bar colors (avoid any green that could be trees)
            # Red HP bars (critical health)
            red_lower = np.array([0, 180, 180])  # Very saturated red only
            red_upper = np.array([8, 255, 255])
            red_mask = cv2.inRange(roi, red_lower, red_upper)
            
            # Yellow HP bars (low health) 
            yellow_lower = np.array([22, 180, 180])  # Very saturated yellow only
            yellow_upper = np.array([32, 255, 255])
            yellow_mask = cv2.inRange(roi, yellow_lower, yellow_upper)
            
            # Green HP bars (full health) - VERY specific to avoid trees
            green_lower = np.array([50, 200, 200])  # Only very bright, saturated greens
            green_upper = np.array([70, 255, 255])
            green_mask = cv2.inRange(roi, green_lower, green_upper)
            
            # Combine HP bar colors
            hp_mask = cv2.bitwise_or(red_mask, yellow_mask)
            hp_mask = cv2.bitwise_or(hp_mask, green_mask)
            
            # Very aggressive noise removal
            kernel_h = np.ones((1, 5), np.uint8)  # Horizontal kernel
            hp_mask = cv2.morphologyEx(hp_mask, cv2.MORPH_CLOSE, kernel_h)
            hp_mask = cv2.morphologyEx(hp_mask, cv2.MORPH_OPEN, kernel_h)
            
            contours, _ = cv2.findContours(hp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Adjust coordinates back to full image
                x += region_x
                y += region_y
                
                # EXTREMELY strict HP bar requirements
                aspect_ratio = w / h if h > 0 else 0
                fill_ratio = area / (w * h) if (w * h) > 0 else 0
                
                if (aspect_ratio > 4.0 and  # Must be very wide relative to height
                    25 < w < 80 and  # Specific HP bar width range
                    3 < h < 8 and   # Very specific height range
                    area > 50 and   # Minimum substantial area
                    fill_ratio > 0.8):  # Must be almost completely filled
                    
                    # Additional validation: check surrounding area for UI context
                    surrounding_area = hsv[max(0, y-10):min(height, y+h+10), 
                                         max(0, x-10):min(width, x+w+10)]
                    
                    # HP bars are usually surrounded by darker UI elements
                    avg_surrounding_value = np.mean(surrounding_area[:, :, 2])
                    
                    if avg_surrounding_value < 150:  # Dark UI background
                        confidence = 0.9  # High confidence for strict matches
                        
                        detections.append({
                            'type': 'hp_bar',
                            'bbox': [x, y, x + w, y + h],
                            'confidence': confidence,
                            'method': 'ultra_strict_hp_detection'
                        })
                
        return detections
        
    def detect_text_areas(self, image, gray):
        """Detect text areas using edge detection with strict filtering"""
        detections = []
        image_height, image_width = image.shape[:2]
        
        # Use edge detection to find text regions
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect text characters
        kernel = np.ones((3, 10), np.uint8)  # Horizontal kernel for text
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # STRICT filtering to prevent false positives:
            # 1. Text areas are wider than tall
            # 2. Reasonable text dimensions (not tiny, not huge)
            # 3. Maximum 50% of screen width (was 80%)
            # 4. Maximum 30% of screen height (new)
            # 5. Minimum aspect ratio for text
            # 6. Not at screen edges (likely UI borders)
            
            aspect_ratio = w / h if h > 0 else 0
            area_ratio = (w * h) / (image_width * image_height)
            
            if (w > h and                                    # Wider than tall
                w > 30 and h > 8 and                       # Reasonable minimum size
                w < image_width * 0.5 and                  # Max 50% width (stricter)
                h < image_height * 0.3 and                 # Max 30% height (new)
                aspect_ratio > 1.5 and aspect_ratio < 15 and  # Text-like aspect ratio
                area_ratio < 0.15 and                      # Max 15% of total area (new)
                x > 5 and y > 5 and                       # Not at very edge
                x + w < image_width - 5 and               # Not at right edge
                y + h < image_height - 5):                # Not at bottom edge
                
                confidence = min(0.7, (w * h) / 10000)
                
                detections.append({
                    'type': 'text_area',
                    'bbox': [x, y, x + w, y + h],
                    'confidence': confidence,
                    'method': 'edge_detection_filtered'
                })
                
        return detections
        
    def detect_menu_boxes(self, image, gray):
        """Detect menu boxes using contour detection"""
        detections = []
        
        # Use threshold to find dark/light contrasts
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Look for rectangular shapes that could be menus
            if len(approx) >= 4 and w > 50 and h > 30:
                area_ratio = cv2.contourArea(contour) / (w * h)
                
                if area_ratio > 0.7:  # Fairly rectangular
                    confidence = min(0.6, area_ratio)
                    
                    detections.append({
                        'type': 'menu_box',
                        'bbox': [x, y, x + w, y + h],
                        'confidence': confidence,
                        'method': 'contour_detection'
                    })
                    
        return detections
        
    def detect_sprites(self, image, hsv):
        """Detect potential character/Pokemon sprites with improved accuracy"""
        detections = []
        height, width = image.shape[:2]
        
        # Multi-stage sprite detection
        
        # 1. Character-like sprites (varied colors, medium size)
        character_detections = self.detect_character_sprites(image, hsv)
        detections.extend(character_detections)
        
        # 2. Pokemon sprites (bright colors, specific sizes)
        pokemon_detections = self.detect_pokemon_sprites(image, hsv)
        detections.extend(pokemon_detections)
        
        return detections
    
    def detect_character_sprites(self, image, hsv):
        """Detect player character and NPC sprites with very strict filtering"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        
        # Much more restrictive character detection to avoid false positives
        
        # Skin tone detection (very specific range)
        skin_lower = np.array([5, 50, 80])
        skin_upper = np.array([20, 180, 255])
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
        
        # Human clothing colors (avoid green/brown that trees have)
        blue_lower = np.array([100, 80, 80])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        red_lower = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        
        # Combine human-specific colors only
        character_mask = cv2.bitwise_or(skin_mask, blue_mask)
        character_mask = cv2.bitwise_or(character_mask, red_mask)
        
        # Very strict cleanup
        kernel = np.ones((2, 2), np.uint8)
        character_mask = cv2.morphologyEx(character_mask, cv2.MORPH_CLOSE, kernel)
        character_mask = cv2.morphologyEx(character_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(character_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # VERY strict character sprite requirements:
            aspect_ratio = h / w if w > 0 else 0
            fill_ratio = area / (w * h) if (w * h) > 0 else 0
            
            # Must be in walkable areas (not in tree/building regions)
            in_walkable_area = (
                y > height * 0.3 and  # Not in sky/UI area
                y < height * 0.9 and  # Not at very bottom
                x > width * 0.1 and   # Not at edges
                x < width * 0.9
            )
            
            if (1.2 < aspect_ratio < 2.2 and  # Character proportions
                14 < w < 32 and 18 < h < 48 and  # Exact character size range
                area > 200 and area < 1000 and  # Reasonable area
                fill_ratio > 0.4 and  # Must be substantially filled
                in_walkable_area):  # Must be in logical location
                
                # Additional validation: check if colors make sense for characters
                roi = hsv[y:y+h, x:x+w]
                avg_saturation = np.mean(roi[:, :, 1])
                avg_value = np.mean(roi[:, :, 2])
                
                # Characters have moderate saturation, avoid very green areas
                if 40 < avg_saturation < 200 and avg_value > 60:
                    confidence = 0.7
                    
                    # Boost for typical character locations
                    if height * 0.5 < y < height * 0.8:
                        confidence += 0.2
                    
                    char_type = 'player_character' if y > height * 0.65 else 'npc_character'
                    
                    detections.append({
                        'type': char_type,
                        'bbox': [x, y, x + w, y + h],
                        'confidence': min(0.9, confidence),
                        'method': 'strict_character_detection'
                    })
                
        return detections
    
    def detect_pokemon_sprites(self, image, hsv):
        """Detect Pokemon sprites with ultra-strict filtering to avoid trees/bushes"""
        detections = []
        height, width = image.shape[:2]
        
        # Only detect in battle scenes - check if this looks like a battle
        is_battle_scene = self.is_battle_scene(image)
        if not is_battle_scene:
            return detections  # Skip Pokemon detection if not in battle
        
        # Pokemon-specific colors (avoid green/brown of trees)
        # Focus on colors that Pokemon have but trees don't
        
        # Bright reds (fire Pokemon)
        red_lower = np.array([0, 150, 150])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        
        # Blues (water Pokemon) 
        blue_lower = np.array([100, 150, 150])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Purples (psychic Pokemon)
        purple_lower = np.array([130, 100, 100])
        purple_upper = np.array([160, 255, 255])
        purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
        
        # Yellows (electric Pokemon)
        yellow_lower = np.array([20, 150, 150])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Combine Pokemon-specific colors (excluding greens/browns)
        pokemon_mask = cv2.bitwise_or(red_mask, blue_mask)
        pokemon_mask = cv2.bitwise_or(pokemon_mask, purple_mask)
        pokemon_mask = cv2.bitwise_or(pokemon_mask, yellow_mask)
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        pokemon_mask = cv2.morphologyEx(pokemon_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(pokemon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Very strict Pokemon sprite requirements
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            fill_ratio = area / (w * h) if (w * h) > 0 else 0
            
            # Must be in battle positions (not scattered around map)
            in_battle_position = (
                (y < height * 0.4 and x > width * 0.3 and x < width * 0.8) or  # Enemy position
                (y > height * 0.6 and x > width * 0.1 and x < width * 0.6)     # Player position
            )
            
            if (aspect_ratio < 1.8 and  # Not too elongated
                32 < w < 80 and 32 < h < 80 and  # Battle sprite size
                area > 600 and area < 4000 and  # Substantial but not huge
                fill_ratio > 0.5 and  # Well-filled shape
                in_battle_position):  # Must be in battle positions
                
                confidence = 0.8
                
                detections.append({
                    'type': 'pokemon_sprite',
                    'bbox': [x, y, x + w, y + h],
                    'confidence': confidence,
                    'method': 'strict_pokemon_detection'
                })
                
        return detections
    
    def is_battle_scene(self, image):
        """Check if the current image looks like a battle scene"""
        height, width = image.shape[:2]
        
        # Battle scenes typically have:
        # - Dark backgrounds in certain areas
        # - UI elements at specific positions
        # - Different color distribution than overworld
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check for dark background areas typical in battles
        upper_area = gray[0:int(height*0.4), :]
        lower_area = gray[int(height*0.6):, :]
        
        upper_dark_ratio = np.sum(upper_area < 50) / upper_area.size
        lower_ui_bright_ratio = np.sum(lower_area > 150) / lower_area.size
        
        # Battle scenes have dark upper areas and bright UI lower areas
        return upper_dark_ratio > 0.3 and lower_ui_bright_ratio > 0.2
        
    def filter_detections(self, detections):
        """Filter and deduplicate detections"""
        # Filter by confidence threshold
        filtered = [d for d in detections if d['confidence'] >= self.confidence_var.get()]
        
        # Remove overlapping detections (simple NMS)
        final_detections = []
        
        for detection in sorted(filtered, key=lambda x: x['confidence'], reverse=True):
            bbox1 = detection['bbox']
            
            # Check overlap with existing detections
            overlap = False
            for existing in final_detections:
                bbox2 = existing['bbox']
                
                # Calculate IoU
                x1 = max(bbox1[0], bbox2[0])
                y1 = max(bbox1[1], bbox2[1])
                x2 = min(bbox1[2], bbox2[2])
                y2 = min(bbox1[3], bbox2[3])
                
                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                    union = area1 + area2 - intersection
                    
                    if union > 0 and intersection / union > 0.3:  # 30% overlap threshold
                        overlap = True
                        break
                        
            if not overlap:
                final_detections.append(detection)
                
        return final_detections
    
    def toggle_always_on_top(self):
        """Toggle always on top window mode"""
        self.always_on_top = self.on_top_var.get()
        self.root.attributes('-topmost', self.always_on_top)
        
        if self.always_on_top:
            self.main_status_var.set("Window set to always on top")
        else:
            self.main_status_var.set("Window normal mode")
    
    def toggle_live_pause(self):
        """Toggle pause for live mode to allow editing"""
        if self.live_mode:
            self.live_paused = not self.live_paused
            
            if self.live_paused:
                self.pause_status_var.set("Live: PAUSED")
                self.main_status_var.set("Live mode paused - safe to edit detections")
            else:
                self.pause_status_var.set("Live: Active")
                self.main_status_var.set("Live mode resumed")
        else:
            self.pause_status_var.set("Live: Inactive")
            messagebox.showinfo("Not Active", "Live mode is not currently active")
    
    def toggle_annotation_mode(self):
        """Toggle manual annotation drawing mode"""
        self.annotation_mode = self.annotation_var.get()
        
        if self.annotation_mode:
            # Disable attention mode if it's active
            if self.attention_mode:
                self.attention_mode = False
                self.attention_var.set(False)
            self.main_status_var.set("Manual annotation mode - click and drag to draw boxes")
            # Pause live mode while annotating
            if self.live_mode:
                self.live_paused = True
                self.pause_status_var.set("Live: PAUSED (annotating)")
        else:
            self.main_status_var.set("Manual annotation disabled")
            # Resume live mode if it was active
            if self.live_mode:
                self.live_paused = False
                self.pause_status_var.set("Live: Active")
    
    def clear_all_detections(self):
        """Clear all AI detections and manual annotations"""
        self.ai_detections = []
        self.manual_annotations = []
        self.selected_detection_idx = -1
        
        # Update displays
        self.update_detection_list()
        
        # Refresh display if we have a screenshot
        if self.current_screenshot is not None:
            self.set_zoom(self.zoom_level)
            
        self.main_status_var.set("All detections cleared")
    
    def toggle_attention_mode(self):
        """Toggle click-to-highlight attention mode"""
        self.attention_mode = self.attention_var.get()
        
        if self.attention_mode:
            # Disable annotation mode if it's active
            if self.annotation_mode:
                self.annotation_mode = False
                self.annotation_var.set(False)
            self.main_status_var.set("Attention mode - click areas to highlight for AI focus")
        else:
            self.main_status_var.set("Attention mode disabled")
    
    def clear_attention_points(self):
        """Clear all attention points and attention-generated detections"""
        self.attention_points = []
        
        # Also remove attention-generated detections
        self.ai_detections = [d for d in self.ai_detections if d.get('method') != 'attention_focused']
        self.update_detection_list()
        
        if self.current_screenshot is not None:
            self.set_zoom(self.zoom_level)  # Refresh display
        self.main_status_var.set("Attention points and detections cleared")
    
    def focus_ai_on_attention(self):
        """Run AI detection focused on attention areas"""
        if not self.attention_points:
            messagebox.showinfo("No Attention Points", "Click areas first to set attention points!")
            return
            
        if self.current_screenshot is None:
            messagebox.showwarning("No Screenshot", "No current screenshot to analyze!")
            return
            
        print(f"DEBUG: Running focused detection on {len(self.attention_points)} attention points")
        
        # Run focused detection on attention areas
        focused_detections = self.run_focused_detection(self.current_screenshot, self.attention_points)
        
        print(f"DEBUG: Found {len(focused_detections)} focused detections")
        
        # Add to current detections
        self.ai_detections.extend(focused_detections)
        self.update_detection_list()
        self.set_zoom(self.zoom_level)  # Refresh display
        
        self.main_status_var.set(f"Found {len(focused_detections)} objects in attention areas")
    
    def run_focused_detection(self, image, attention_points):
        """Run AGGRESSIVE AI detection focused on specific attention areas"""
        detections = []
        
        print(f"DEBUG: Processing {len(attention_points)} attention points")
        
        for i, (x, y, radius, timestamp) in enumerate(attention_points):
            print(f"DEBUG: Processing attention point {i+1}: ({x:.1f}, {y:.1f}) radius={radius}")
            
            # Create larger region of interest around attention point
            roi_x1 = max(0, int(x - radius))
            roi_y1 = max(0, int(y - radius))
            roi_x2 = min(image.shape[1], int(x + radius))
            roi_y2 = min(image.shape[0], int(y + radius))
            
            print(f"DEBUG: ROI bounds: ({roi_x1}, {roi_y1}) to ({roi_x2}, {roi_y2})")
            
            # Extract ROI
            roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size == 0:
                print(f"DEBUG: ROI is empty, skipping attention point {i+1}")
                continue
                
            print(f"DEBUG: ROI shape: {roi.shape}")
                
            # Run VERY aggressive detection on ROI (lower thresholds)
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Much more aggressive detection for attention areas
            roi_detections = []
            roi_detections.extend(self.aggressive_attention_detection(roi, roi_hsv, roi_gray))
            
            print(f"DEBUG: Found {len(roi_detections)} detections in attention point {i+1}")
            
            # Adjust detection coordinates back to full image
            for detection in roi_detections:
                bbox = detection['bbox']
                detection['bbox'] = [
                    bbox[0] + roi_x1,
                    bbox[1] + roi_y1,
                    bbox[2] + roi_x1,
                    bbox[3] + roi_y1
                ]
                detection['method'] = f"attention_focused"
                detection['confidence'] = min(0.9, detection['confidence'] + 0.2)  # Big boost
                detection['attention_timestamp'] = time.time()  # Mark when this was created
                print(f"DEBUG: Adjusted detection bbox: {detection['bbox']} type: {detection['type']}")
                
            detections.extend(roi_detections)
            
        print(f"DEBUG: Total focused detections: {len(detections)}")
        return detections
    
    def aggressive_attention_detection(self, roi, roi_hsv, roi_gray):
        """ULTRA-AGGRESSIVE detection for attention areas - ALWAYS find something"""
        detections = []
        height, width = roi.shape[:2]
        
        if height < 5 or width < 5:
            return detections
        
        print(f"DEBUG: Aggressive detection on ROI {width}x{height}")
        
        # GUARANTEED DETECTION METHOD 1: Create bounding box for entire attention area
        # This ensures we ALWAYS have at least one detection
        fallback_detection = {
            'type': 'attention_area',
            'bbox': [0, 0, width, height],
            'confidence': 0.6,
            'method': 'attention_fallback'
        }
        
        found_specific = False
        
        # Method 1: ANY edge or contrast
        edges = cv2.Canny(roi_gray, 10, 50)  # Ultra-low thresholds
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Method 2: ANY color variation at all
        saturation = roi_hsv[:, :, 1]
        value = roi_hsv[:, :, 2]
        
        # Find any areas with ANY saturation or brightness variation
        color_masks = [
            saturation > 20,  # Any slight color
            value > 80,       # Any bright area
            value < 150,      # Any darker area
        ]
        
        # Method 3: Gradient-based detection (any directional change)
        grad_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mask = gradient_magnitude > 10  # Very low threshold
        
        # Combine all detection methods
        all_masks = [edges_dilated > 0, gradient_mask] + color_masks
        
        for i, mask in enumerate(all_masks):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Accept ANY contour with area > 10 pixels
                area = cv2.contourArea(contour)
                if area > 10:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Expand bounding box slightly for better visibility
                    x = max(0, x - 2)
                    y = max(0, y - 2)
                    w = min(width - x, w + 4)
                    h = min(height - y, h + 4)
                    
                    detection = {
                        'type': self.guess_object_type_from_context(roi, x, y, w, h),
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 0.7 + (area / (width * height)) * 0.2,  # Higher confidence for larger areas
                        'method': f'aggressive_method_{i+1}'
                    }
                    detections.append(detection)
                    found_specific = True
                    print(f"DEBUG: Found detection via method {i+1}: {detection['type']} at {detection['bbox']}")
        
        # If no specific detections found, use the fallback
        if not found_specific:
            detections.append(fallback_detection)
            print(f"DEBUG: Using fallback detection for entire attention area")
        
        # Limit to top 3 detections to avoid overwhelming the user
        detections = detections[:3]
        
        print(f"DEBUG: Returning {len(detections)} aggressive detections")
        return detections
    
    def guess_object_type_from_context(self, roi, x, y, w, h):
        """Guess object type based on shape, size, and context"""
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
        area = w * h
        roi_area = roi.shape[0] * roi.shape[1]
        
        # Analyze colors in the bounding box
        roi_region = roi[y:y+h, x:x+w]
        roi_region_hsv = cv2.cvtColor(roi_region, cv2.COLOR_BGR2HSV)
        
        # Get dominant colors
        mean_hue = np.mean(roi_region_hsv[:, :, 0])
        mean_sat = np.mean(roi_region_hsv[:, :, 1])
        mean_val = np.mean(roi_region_hsv[:, :, 2])
        
        # Object type guessing based on characteristics
        if aspect_ratio > 3:  # Very wide/tall
            if mean_sat > 100:  # Colorful
                return "hp_bar"
            else:
                return "menu_element"
        elif aspect_ratio < 1.5 and w > 20 and h > 20:  # Square-ish and decent size
            if mean_sat > 80:  # Colorful
                return "pokemon_sprite"
            else:
                return "player_character"
        elif area > roi_area * 0.3:  # Large area
            return "text_area"
        elif aspect_ratio > 1.2 and aspect_ratio < 2.5:  # Rectangular
            if mean_val > 150:  # Bright
                return "ui_element"
            else:
                return "npc_character"
        else:
            return "unknown_object"
        
    def update_live_display(self, screenshot, detections):
        """Update live display with screenshot and detections"""
        # Don't update if paused (preserves current detections for editing)
        if self.live_paused:
            return
            
        self.current_screenshot = screenshot.copy()
        
        # Preserve attention-generated detections by merging with new live detections
        current_time = time.time()
        attention_max_age = 60.0  # Keep attention detections for 60 seconds
        
        # Filter out expired attention detections
        attention_detections = [
            d for d in self.ai_detections 
            if (d.get('method') == 'attention_focused' and 
                current_time - d.get('attention_timestamp', 0) < attention_max_age)
        ]
        
        live_detections = detections.copy()
        
        # Combine: keep fresh attention detections + add new live detections
        self.ai_detections = attention_detections + live_detections
        
        # Update the detection list to show merged detections
        self.update_detection_list()
        
        # Draw detections on screenshot
        display_image = screenshot.copy()
        
        # Draw attention points first (so they appear behind detections)
        current_time = time.time()
        for x, y, radius, timestamp in self.attention_points:
            # Fade attention points over time
            age = current_time - timestamp
            max_age = 30.0  # 30 seconds
            
            if age < max_age:
                # Calculate fade alpha (1.0 = fresh, 0.0 = expired)
                alpha = 1.0 - (age / max_age)
                
                # Draw pulsing attention circle
                pulse_factor = 0.8 + 0.2 * abs(np.sin(current_time * 3))  # Pulse effect
                current_radius = int(radius * pulse_factor)
                
                # Draw outer attention ring
                color_intensity = int(255 * alpha)
                cv2.circle(display_image, (int(x), int(y)), current_radius, 
                          (0, color_intensity, 255), 3)  # Orange attention ring
                
                # Draw inner dot
                cv2.circle(display_image, (int(x), int(y)), 3, 
                          (0, color_intensity, 255), -1)
        
        # Clean up expired attention points
        self.attention_points = [(x, y, r, t) for x, y, r, t in self.attention_points 
                                if current_time - t < max_age]
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            obj_type = detection['type']
            confidence = detection['confidence']
            
            # Color coding
            colors = {
                'hp_bar': (0, 255, 0),      # Green
                'text_area': (255, 255, 0),  # Cyan
                'menu_box': (0, 255, 255),   # Yellow
                'pokemon_sprite': (255, 0, 0), # Blue
                'player_character': (255, 0, 255), # Magenta
                'npc_character': (255, 100, 0)  # Orange
            }
            color = colors.get(obj_type, (255, 255, 255))
            
            # Highlight selected detection
            thickness = 3 if i == self.selected_detection_idx else 2
            
            # Draw bounding box
            cv2.rectangle(display_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            
            # Draw label
            label = f"{i+1}: {obj_type} ({confidence:.2f})"
            cv2.putText(display_image, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Update canvas
        self.display_image_on_canvas(display_image, self.live_canvas)
        
        # Update detection list
        self.update_detection_list()
        
    def display_image_on_canvas(self, image, canvas):
        """Display image on canvas with proper scaling and zoom"""
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(50, lambda: self.display_image_on_canvas(image, canvas))
            return
            
        # Apply base scaling to fit canvas, then apply zoom
        img_height, img_width = image.shape[:2]
        base_scale_x = canvas_width / img_width
        base_scale_y = canvas_height / img_height
        base_scale = min(base_scale_x, base_scale_y) * 0.8  # Leave margin
        
        # Apply zoom level
        final_scale = base_scale * self.zoom_level
        
        new_width = int(img_width * final_scale)
        new_height = int(img_height * final_scale)
        
        resized = cv2.resize(image, (new_width, new_height))
        
        # Convert for Tkinter
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        canvas.delete("all")
        
        # Center the image (allow scrolling if image is larger than canvas)
        x_offset = max(0, (canvas_width - new_width) // 2)
        y_offset = max(0, (canvas_height - new_height) // 2)
        
        canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
        
        # Store scaling info for coordinate transformation
        self.image_scale = final_scale
        self.image_offset = (x_offset, y_offset)
        self.image_display_size = (new_width, new_height)
        
        # Keep reference to prevent garbage collection
        canvas.image = photo
        
    def update_detection_list(self):
        """Update the detection list display"""
        self.detection_listbox.delete(0, tk.END)
        
        for i, detection in enumerate(self.ai_detections):
            obj_type = detection['type']
            confidence = detection['confidence']
            bbox = detection['bbox']
            
            list_item = f"{i+1:2d}: {obj_type:15s} ({confidence:.2f}) [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"
            self.detection_listbox.insert(tk.END, list_item)
            
    def manual_capture_analyze(self):
        """Manual capture and analyze"""
        if not self.emulator_window:
            messagebox.showwarning("No Emulator", "Please find the emulator window first!")
            return
            
        # Brief minimize only for manual capture to avoid GUI interference
        self.root.iconify()
        time.sleep(0.3)  # Longer pause for manual capture
        
        screenshot = self.capture_emulator_screenshot()
        
        # Restore window
        self.root.deiconify()
        
        if screenshot is not None:
            detections = self.run_ai_detection(screenshot)
            self.update_live_display(screenshot, detections)
            self.main_status_var.set(f"Captured and analyzed - found {len(detections)} objects")
        else:
            messagebox.showerror("Capture Failed", "Failed to capture screenshot")
            
    def update_confidence_display(self, value):
        """Update confidence threshold display"""
        self.confidence_label_var.set(f"{float(value):.1f}")
        self.confidence_threshold = float(value)
        
    def on_detection_select(self, event):
        """Handle detection selection - supports multi-selection WITHOUT overriding dropdown"""
        selection = self.detection_listbox.curselection()
        if selection:
            if len(selection) == 1:
                # Single selection - update dropdown ONLY if user hasn't manually changed it
                detection_idx = selection[0]
                if detection_idx < len(self.ai_detections):
                    self.selected_detection_idx = detection_idx
                    detection = self.ai_detections[detection_idx]
                    
                    # Only auto-update dropdown if it's currently empty or matches the detection
                    current_dropdown = self.correction_var.get()
                    if not current_dropdown or current_dropdown == detection['type']:
                        self.correction_var.set(detection['type'])
                    
                    self.main_status_var.set(f"Selected: {detection['type']} (confidence: {detection['confidence']:.2f})")
                    self.refresh_display_highlight()
            else:
                # Multiple selection - NEVER change dropdown, user controls it for batch operations
                self.selected_detection_idx = selection[0]  # Keep first for highlighting
                types = [self.ai_detections[i]['type'] for i in selection if i < len(self.ai_detections)]
                type_counts = {}
                for t in types:
                    type_counts[t] = type_counts.get(t, 0) + 1
                type_summary = ", ".join([f"{t}({c})" for t, c in type_counts.items()])
                self.main_status_var.set(f"Selected {len(selection)} detections: {type_summary} | Dropdown: {self.correction_var.get()}")
                self.refresh_display_highlight()
        else:
            self.selected_detection_idx = -1
            self.main_status_var.set("No detection selected")
    
    def refresh_display_highlight(self):
        """Refresh display with highlighting (separate method to avoid focus issues)"""
        if hasattr(self, 'current_screenshot') and self.current_screenshot is not None:
            self.set_zoom(self.zoom_level)
    
    def on_correction_selected(self, event):
        """Handle correction dropdown selection - NO interference with listbox selection"""
        corrected_type = self.correction_var.get()
        selection = self.detection_listbox.curselection()
        
        if selection:
            if len(selection) == 1:
                # Single selection preview
                detection = self.ai_detections[selection[0]]
                self.main_status_var.set(f"Will correct '{detection['type']}' → '{corrected_type}' (click 'Correct & Confirm')")
            else:
                # Multi-selection preview
                self.main_status_var.set(f"Will batch correct {len(selection)} detections → '{corrected_type}' (click 'Batch Correct')")
            
            # DO NOT touch the listbox selection - let user control it
            
        else:
            self.main_status_var.set(f"Select detection(s) first, then use '{corrected_type}' correction")
                
    def confirm_selected_detection(self):
        """Confirm selected detection as correct"""
        if self.selected_detection_idx < 0 or self.selected_detection_idx >= len(self.ai_detections):
            messagebox.showwarning("No Selection", "Please select a detection to confirm")
            return
            
        detection = self.ai_detections[self.selected_detection_idx].copy()
        detection['confirmed'] = True
        detection['user_verified'] = True
        
        self.confirmed_labels.append(detection)
        self.update_confirmed_list()
        self.update_stats()
        
        # Keep selection for easier workflow
        self.main_status_var.set(f"Confirmed {detection['type']} detection")
            
    def correct_and_confirm(self):
        """Correct object type and confirm - BULLETPROOF VERSION"""
        # Validate selection
        if self.selected_detection_idx < 0 or self.selected_detection_idx >= len(self.ai_detections):
            messagebox.showwarning("No Selection", "Please select a detection from the list first!")
            return
            
        # Get the corrected type from dropdown
        corrected_type = self.correction_var.get()
        if not corrected_type or corrected_type not in self.object_types:
            messagebox.showwarning("Invalid Type", "Please select a valid object type from the dropdown!")
            return
            
        # Get the detection to correct
        detection = self.ai_detections[self.selected_detection_idx].copy()
        old_type = detection['type']
        
        # Apply correction
        detection['type'] = corrected_type
        detection['confirmed'] = True
        detection['user_corrected'] = True
        detection['correction_timestamp'] = time.time()
        
        # Add to confirmed labels
        self.confirmed_labels.append(detection)
        
        # Update the original detection for immediate visual feedback
        self.ai_detections[self.selected_detection_idx]['type'] = corrected_type
        
        # Update all displays
        self.update_detection_list()
        self.update_confirmed_list()
        self.update_stats()
        
        # Refresh visual display
        self.refresh_display_highlight()
        
        # Success feedback
        self.main_status_var.set(f"SUCCESS: Corrected '{old_type}' to '{corrected_type}' and confirmed!")
        
        # Keep the selection active for easier workflow
        self.detection_listbox.selection_clear(0, tk.END)
        self.detection_listbox.selection_set(self.selected_detection_idx)
        self.detection_listbox.see(self.selected_detection_idx)
            
    def delete_selected_detection(self):
        """Delete selected detection"""
        if self.selected_detection_idx < 0 or self.selected_detection_idx >= len(self.ai_detections):
            return
            
        deleted_detection = self.ai_detections[self.selected_detection_idx]
        del self.ai_detections[self.selected_detection_idx]
        
        # Reset selection
        self.selected_detection_idx = -1
        
        # Update displays
        self.update_detection_list()
        self.set_zoom(self.zoom_level)  # Refresh display
        
        self.main_status_var.set(f"Deleted {deleted_detection['type']} detection")
    
    def batch_correct_selected(self):
        """Correct multiple selected detections to the same type"""
        selection = self.detection_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select detections to correct (Ctrl+Click for multiple)")
            return
        
        corrected_type = self.correction_var.get()
        if not corrected_type or corrected_type not in self.object_types:
            messagebox.showwarning("Invalid Type", "Please select a valid object type from the dropdown!")
            return
        
        # Confirm batch operation
        selected_types = [self.ai_detections[i]['type'] for i in selection if i < len(self.ai_detections)]
        unique_types = list(set(selected_types))
        
        confirm_msg = f"Correct {len(selection)} detections to '{corrected_type}'?\n\nCurrent types: {', '.join(unique_types)}"
        if not messagebox.askyesno("Batch Correction", confirm_msg):
            return
        
        corrected_count = 0
        for idx in reversed(selection):  # Reverse to maintain indices during modifications
            if idx < len(self.ai_detections):
                detection = self.ai_detections[idx].copy()
                old_type = detection['type']
                
                # Apply correction
                detection['type'] = corrected_type
                detection['confirmed'] = True
                detection['user_corrected'] = True
                detection['correction_timestamp'] = time.time()
                
                # Add to confirmed labels
                self.confirmed_labels.append(detection)
                
                # Update the original detection for visual feedback
                self.ai_detections[idx]['type'] = corrected_type
                
                corrected_count += 1
        
        # Update displays
        self.update_detection_list()
        self.update_confirmed_list()
        self.update_stats()
        self.refresh_display_highlight()
        
        self.main_status_var.set(f"Batch corrected {corrected_count} detections to '{corrected_type}'")
    
    def batch_confirm_selected(self):
        """Confirm multiple selected detections as correct"""
        selection = self.detection_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select detections to confirm (Ctrl+Click for multiple)")
            return
        
        # Confirm batch operation
        confirm_msg = f"Confirm {len(selection)} detections as correct?"
        if not messagebox.askyesno("Batch Confirmation", confirm_msg):
            return
        
        confirmed_count = 0
        for idx in selection:
            if idx < len(self.ai_detections):
                detection = self.ai_detections[idx].copy()
                detection['confirmed'] = True
                detection['user_verified'] = True
                self.confirmed_labels.append(detection)
                confirmed_count += 1
        
        # Update displays
        self.update_confirmed_list()
        self.update_stats()
        
        self.main_status_var.set(f"Batch confirmed {confirmed_count} detections")
    
    def batch_delete_selected(self):
        """Delete multiple selected detections"""
        selection = self.detection_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select detections to delete (Ctrl+Click for multiple)")
            return
        
        # Confirm batch operation
        confirm_msg = f"Delete {len(selection)} selected detections?"
        if not messagebox.askyesno("Batch Deletion", confirm_msg):
            return
        
        # Delete in reverse order to maintain indices
        deleted_count = 0
        for idx in reversed(selection):
            if idx < len(self.ai_detections):
                self.ai_detections.pop(idx)
                deleted_count += 1
        
        # Update displays
        self.update_detection_list()
        self.set_zoom(self.zoom_level)  # Refresh display
        
        self.main_status_var.set(f"Deleted {deleted_count} detections")
            
    def confirm_all_detections(self):
        """Confirm all current detections"""
        for detection in self.ai_detections:
            confirmed_detection = detection.copy()
            confirmed_detection['confirmed'] = True
            confirmed_detection['user_verified'] = True
            self.confirmed_labels.append(confirmed_detection)
            
        self.update_confirmed_list()
        self.update_stats()
        self.main_status_var.set(f"Confirmed {len(self.ai_detections)} detections")
        
    def update_confirmed_list(self):
        """Update confirmed labels list"""
        self.confirmed_listbox.delete(0, tk.END)
        
        for i, label in enumerate(self.confirmed_labels):
            obj_type = label['type']
            confidence = label['confidence']
            status = "✓" if label.get('user_verified') else "✏" if label.get('user_corrected') else "?"
            
            list_item = f"{status} {obj_type} ({confidence:.2f})"
            self.confirmed_listbox.insert(tk.END, list_item)
            
    def update_stats(self):
        """Update statistics display"""
        confirmed_count = len(self.confirmed_labels)
        
        # Count by type
        type_counts = {}
        for label in self.confirmed_labels:
            obj_type = label['type']
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
            
        stats_text = f"Confirmed: {confirmed_count}\nTypes: {len(type_counts)}"
        self.stats_var.set(stats_text)
        
    def save_confirmed_labels(self):
        """Save confirmed labels to files"""
        if not self.confirmed_labels:
            messagebox.showwarning("No Labels", "No confirmed labels to save!")
            return
            
        if self.current_screenshot is None:
            messagebox.showwarning("No Screenshot", "No current screenshot to save!")
            return
            
        # Generate timestamp for filenames
        timestamp = int(time.time() * 1000)
        
        # Save screenshot
        screenshot_path = self.data_dir / "screenshots" / f"game_{timestamp}.png"
        cv2.imwrite(str(screenshot_path), self.current_screenshot)
        
        # Save confirmed screenshot with labels drawn
        labeled_image = self.current_screenshot.copy()
        for i, label in enumerate(self.confirmed_labels):
            bbox = label['bbox']
            obj_type = label['type']
            
            cv2.rectangle(labeled_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(labeled_image, f"{i+1}: {obj_type}", (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                       
        confirmed_path = self.data_dir / "confirmed" / f"game_{timestamp}_labeled.png"
        cv2.imwrite(str(confirmed_path), labeled_image)
        
        # Save metadata
        metadata = {
            'screenshot_path': str(screenshot_path),
            'timestamp': timestamp,
            'labels': self.confirmed_labels,
            'image_size': self.current_screenshot.shape[:2],
            'collection_method': 'ai_assisted'
        }
        
        metadata_path = self.data_dir / "metadata" / f"game_{timestamp}_labels.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Clear confirmed labels for next capture
        self.confirmed_labels = []
        self.update_confirmed_list()
        self.update_stats()
        
        messagebox.showinfo("Saved", f"Saved labels for screenshot: game_{timestamp}.png")
        self.main_status_var.set(f"Labels saved successfully!")
        
    def show_detailed_stats(self):
        """Show detailed collection statistics"""
        messagebox.showinfo("Stats", "Detailed stats feature coming soon!")
        
    def on_zoom_changed(self, value):
        """Handle zoom slider changes"""
        zoom_level = float(value)
        self.set_zoom(zoom_level)
    
    def set_zoom(self, zoom_level):
        """Set zoom level and update display"""
        self.zoom_level = zoom_level
        self.zoom_label_var.set(f"Current: {zoom_level:.1f}x")
        
        # Update slider if called from elsewhere (like button clicks)
        if hasattr(self, 'zoom_var'):
            self.zoom_var.set(zoom_level)
        
        # Update display if we have a current screenshot
        if self.current_screenshot is not None:
            # Redraw with new zoom level
            display_image = self.current_screenshot.copy()
            
            # Draw detections on screenshot with updated scaling
            for i, detection in enumerate(self.ai_detections):
                bbox = detection['bbox']
                obj_type = detection['type']
                confidence = detection['confidence']
                
                # Color coding
                colors = {
                    'hp_bar': (0, 255, 0),      # Green
                    'text_area': (255, 255, 0),  # Cyan
                    'menu_box': (0, 255, 255),   # Yellow
                    'pokemon_sprite': (255, 0, 0), # Blue
                    'player_character': (255, 0, 255) # Magenta
                }
                color = colors.get(obj_type, (255, 255, 255))
                
                # Highlight selected detection
                thickness = 3 if i == self.selected_detection_idx else 2
                
                # Draw bounding box
                cv2.rectangle(display_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
                
                # Draw label
                label = f"{i+1}: {obj_type} ({confidence:.2f})"
                cv2.putText(display_image, label, (bbox[0], bbox[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Update display
            self.display_image_on_canvas(display_image, self.live_canvas)
    
    def on_canvas_left_click(self, event):
        """Handle left clicks on canvas for detection selection, manual annotation, or attention"""
        if self.attention_mode:
            # Add attention point
            if hasattr(self, 'image_scale'):
                # Convert canvas coordinates to image coordinates
                img_x = (event.x - self.image_offset[0]) / self.image_scale
                img_y = (event.y - self.image_offset[1]) / self.image_scale
                
                # Add attention point with timestamp
                attention_point = (img_x, img_y, 40, time.time())  # 40px radius
                self.attention_points.append(attention_point)
                
                # Refresh display to show new attention point
                self.set_zoom(self.zoom_level)
                
                self.main_status_var.set(f"Added attention point - {len(self.attention_points)} total")
        elif self.annotation_mode:
            # Start manual annotation
            self.box_start_pos = (event.x, event.y)
            self.drawing_box = True
        else:
            # Normal detection selection
            self.on_detection_click(event)
    
    def on_canvas_drag(self, event):
        """Handle mouse drag for manual annotation"""
        if self.annotation_mode and self.drawing_box and self.box_start_pos:
            # Clear previous temporary box
            self.live_canvas.delete("temp_annotation_box")
            
            # Draw current box
            self.live_canvas.create_rectangle(
                self.box_start_pos[0], self.box_start_pos[1], event.x, event.y,
                outline="red", width=2, tags="temp_annotation_box"
            )
    
    def on_canvas_left_release(self, event):
        """Handle mouse release for manual annotation"""
        if self.annotation_mode and self.drawing_box and self.box_start_pos:
            # Finish drawing the box
            self.drawing_box = False
            
            # Convert canvas coordinates to image coordinates
            if hasattr(self, 'image_scale'):
                start_img_x = (self.box_start_pos[0] - self.image_offset[0]) / self.image_scale
                start_img_y = (self.box_start_pos[1] - self.image_offset[1]) / self.image_scale
                end_img_x = (event.x - self.image_offset[0]) / self.image_scale
                end_img_y = (event.y - self.image_offset[1]) / self.image_scale
                
                # Ensure coordinates are in correct order
                x1, x2 = sorted([start_img_x, end_img_x])
                y1, y2 = sorted([start_img_y, end_img_y])
                
                # Only add if box is big enough
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    annotation = {
                        'type': self.annotation_type_var.get(),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': 1.0,
                        'method': 'manual_annotation',
                        'manual': True
                    }
                    
                    self.manual_annotations.append(annotation)
                    self.main_status_var.set(f"Added manual {annotation['type']} annotation")
                    
                    # Add to AI detections for display
                    self.ai_detections.append(annotation)
                    self.update_detection_list()
                    
                    # Refresh display
                    self.set_zoom(self.zoom_level)
                    
            # Clean up
            self.live_canvas.delete("temp_annotation_box")
            self.box_start_pos = None
    
    def on_canvas_right_click(self, event):
        """Handle right-click for context menu or deleting annotations"""
        if self.annotation_mode:
            # Delete last manual annotation
            if self.manual_annotations:
                removed = self.manual_annotations.pop()
                # Also remove from ai_detections if it was added there
                self.ai_detections = [d for d in self.ai_detections if not d.get('manual') or d != removed]
                self.update_detection_list()
                self.set_zoom(self.zoom_level)
                self.main_status_var.set(f"Deleted last manual annotation ({removed['type']})")
        else:
            # Normal right-click detection menu
            self.on_detection_right_click(event)
    
    def on_detection_click(self, event):
        """Handle clicks on detections in live view"""
        if not hasattr(self, 'image_scale') or not self.ai_detections:
            return
            
        # Convert canvas coordinates to image coordinates
        canvas_x = event.x
        canvas_y = event.y
        
        # Account for image offset and scaling
        img_x = (canvas_x - self.image_offset[0]) / self.image_scale
        img_y = (canvas_y - self.image_offset[1]) / self.image_scale
        
        # Find which detection was clicked
        for i, detection in enumerate(self.ai_detections):
            bbox = detection['bbox']
            if (bbox[0] <= img_x <= bbox[2] and bbox[1] <= img_y <= bbox[3]):
                # Select this detection
                self.selected_detection_idx = i
                self.detection_listbox.selection_clear(0, tk.END)
                self.detection_listbox.selection_set(i)
                self.detection_listbox.see(i)
                self.correction_var.set(detection['type'])
                
                # Update display to highlight selected detection
                self.set_zoom(self.zoom_level)  # Refresh display
                break
                
        # Handle start of manual annotation if no detection was clicked
        if self.annotation_mode:
            self.box_start_pos = (event.x, event.y)
            self.drawing_box = True
        
    def on_detection_right_click(self, event):
        """Handle right-clicks for quick actions"""
        if not hasattr(self, 'image_scale') or not self.ai_detections:
            return
            
        # Convert canvas coordinates to image coordinates
        canvas_x = event.x
        canvas_y = event.y
        
        # Account for image offset and scaling
        img_x = (canvas_x - self.image_offset[0]) / self.image_scale
        img_y = (canvas_y - self.image_offset[1]) / self.image_scale
        
        # Find which detection was right-clicked
        for i, detection in enumerate(self.ai_detections):
            bbox = detection['bbox']
            if (bbox[0] <= img_x <= bbox[2] and bbox[1] <= img_y <= bbox[3]):
                # Show context menu for quick actions
                self.show_detection_context_menu(event, i)
                break
    
    def show_detection_context_menu(self, event, detection_idx):
        """Show context menu for detection actions"""
        if detection_idx >= len(self.ai_detections):
            return
            
        detection = self.ai_detections[detection_idx]
        
        # Create context menu
        context_menu = tk.Menu(self.root, tearoff=0)
        context_menu.add_command(label=f"Confirm '{detection['type']}'", 
                                command=lambda: self.quick_confirm_detection(detection_idx))
        context_menu.add_separator()
        
        # Add quick correction options for common types
        common_types = ['hp_bar', 'text_area', 'menu_box', 'pokemon_sprite', 'player_character']
        for obj_type in common_types:
            if obj_type != detection['type']:  # Don't show current type
                context_menu.add_command(label=f"Correct to '{obj_type}'", 
                                        command=lambda t=obj_type: self.quick_correct_detection(detection_idx, t))
        
        context_menu.add_separator()
        context_menu.add_command(label="Delete Detection", 
                                command=lambda: self.quick_delete_detection(detection_idx))
        
        # Show menu at cursor position
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()
    
    def quick_confirm_detection(self, detection_idx):
        """Quickly confirm a detection"""
        if detection_idx < len(self.ai_detections):
            detection = self.ai_detections[detection_idx].copy()
            detection['confirmed'] = True
            detection['user_verified'] = True
            
            self.confirmed_labels.append(detection)
            self.update_confirmed_list()
            self.update_stats()
            
    def quick_correct_detection(self, detection_idx, new_type):
        """Quickly correct and confirm a detection"""
        if detection_idx < len(self.ai_detections):
            detection = self.ai_detections[detection_idx].copy()
            detection['type'] = new_type
            detection['confirmed'] = True
            detection['user_corrected'] = True
            
            self.confirmed_labels.append(detection)
            self.update_confirmed_list()
            self.update_stats()
            
    def quick_delete_detection(self, detection_idx):
        """Quickly delete a detection"""
        if detection_idx < len(self.ai_detections):
            del self.ai_detections[detection_idx]
            self.selected_detection_idx = -1
            self.update_detection_list()
            self.set_zoom(self.zoom_level)  # Refresh display
        
    def run(self):
        """Run the assisted data collector"""
        print("AI-Assisted Pokemon Data Collector")
        print("=" * 40)
        print("Features:")
        print("- AI detects objects automatically")
        print("- You just confirm or correct detections")
        print("- Much faster than manual labeling")
        print("- Auto-minimizes window during capture")
        print("- Live mode for continuous collection")
        print("\nStarting GUI...")
        
        self.root.mainloop()

def main():
    """Main function"""
    collector = AssistedDataCollector()
    collector.run()

if __name__ == "__main__":
    main()