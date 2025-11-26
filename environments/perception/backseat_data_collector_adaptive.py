#!/usr/bin/env python3
"""
🚗 ADAPTIVE BACKSEAT DRIVING DATA COLLECTOR
==========================================

Enhanced with flexible class management system:
- Dynamic class discovery from all models
- Backward/forward compatibility between models
- Automatic class mapping and adaptation
- Extensible to new object types without breaking existing models

Key improvements:
- Uses ClassRegistry for dynamic class management
- AdaptiveModelWrapper for model compatibility
- Unified interface regardless of model class architecture
- Real-time class system adaptation
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
from PIL import Image, ImageTk, ImageGrab, ImageDraw
import win32gui
import win32con
import win32ui
from ctypes import windll
from collections import defaultdict
from datetime import datetime

# Import our adaptive system
from class_registry import ClassRegistry
from adaptive_model_wrapper import AdaptiveModelWrapper, ModelManager
from model_manager import ModelManager as BasicModelManager

class AdaptiveBackseatDataCollector:
    """Enhanced data collector with adaptive class management"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Adaptive Backseat AI Trainer")
        self.root.geometry("1800x1000")
        
        # Adaptive system components
        self.class_registry = ClassRegistry()
        self.model_manager = ModelManager()
        self.current_model_wrapper = None
        
        # Available classes (dynamically loaded)
        self.available_classes = self.class_registry.get_all_classes()
        self.active_classes = self.available_classes.copy()
        
        # Core systems
        self.current_screenshot = None
        self.ai_detections = []
        self.confirmed_labels = []
        
        # Emulator window
        self.emulator_window = None
        
        # Live mode
        self.live_mode = False
        self.capture_thread = None
        self.live_paused = False
        
        # Detection state
        self.selected_detection = None
        self.detection_method = "adaptive"
        
        # Detection confidence thresholds
        self.auto_accept_threshold = 0.9
        self.ask_confirmation_threshold = 0.6
        
        # Manual draw mode state
        self.draw_mode = False
        self.drawing = False
        self.draw_start_x = None
        self.draw_start_y = None
        self.draw_current_rect = None
        
        # Data management
        self.data_dir = Path("adaptive_training_data")
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "screenshots").mkdir(exist_ok=True)
        (self.data_dir / "metadata").mkdir(exist_ok=True)
        
        self.setup_ui()
        self.setup_adaptive_ai()
        
    def setup_ui(self):
        """Setup enhanced UI with adaptive class controls"""
        
        # Main layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - controls
        left_panel = ttk.Frame(main_frame, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0,5))
        left_panel.pack_propagate(False)
        
        # Right panel - image display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # === ADAPTIVE SYSTEM CONTROLS ===
        adaptive_frame = ttk.LabelFrame(left_panel, text="Adaptive AI System", padding=5)
        adaptive_frame.pack(fill=tk.X, pady=(0,5))
        
        # Class registry status
        registry_info_frame = ttk.Frame(adaptive_frame)
        registry_info_frame.pack(fill=tk.X, pady=(0,5))
        
        ttk.Label(registry_info_frame, text="Class Registry:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        self.registry_status_label = ttk.Label(registry_info_frame, text="Loading...", foreground="blue")
        self.registry_status_label.pack(anchor=tk.W)
        
        ttk.Button(registry_info_frame, text="Refresh Registry", 
                  command=self.refresh_class_registry).pack(anchor=tk.W, pady=(2,0))
        
        # Active class selection
        class_frame = ttk.LabelFrame(left_panel, text="Active Classes", padding=5)
        class_frame.pack(fill=tk.X, pady=(0,5))
        
        ttk.Label(class_frame, text="Select classes to work with:", font=("Arial", 9)).pack(anchor=tk.W)
        
        # Scrollable class list
        class_list_frame = ttk.Frame(class_frame)
        class_list_frame.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        
        self.class_listbox = tk.Listbox(class_list_frame, selectmode=tk.MULTIPLE, height=8)
        class_scrollbar = ttk.Scrollbar(class_list_frame, orient=tk.VERTICAL, command=self.class_listbox.yview)
        self.class_listbox.configure(yscrollcommand=class_scrollbar.set)
        
        self.class_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        class_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Class management buttons
        class_btn_frame = ttk.Frame(class_frame)
        class_btn_frame.pack(fill=tk.X, pady=(5,0))
        
        ttk.Button(class_btn_frame, text="Select All", 
                  command=self.select_all_classes).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(class_btn_frame, text="Clear All", 
                  command=self.clear_all_classes).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(class_btn_frame, text="Apply Selection", 
                  command=self.apply_class_selection).pack(side=tk.LEFT)
        
        # Add new class
        new_class_frame = ttk.Frame(class_frame)
        new_class_frame.pack(fill=tk.X, pady=(5,0))
        
        ttk.Label(new_class_frame, text="Add new class:").pack(anchor=tk.W)
        self.new_class_var = tk.StringVar()
        new_class_entry = ttk.Entry(new_class_frame, textvariable=self.new_class_var)
        new_class_entry.pack(fill=tk.X, pady=(2,0))
        
        ttk.Button(new_class_frame, text="Add Class", 
                  command=self.add_new_class).pack(anchor=tk.W, pady=(2,0))
        
        # === MODEL SELECTION ===
        model_frame = ttk.LabelFrame(left_panel, text="Model Selection", padding=5)
        model_frame.pack(fill=tk.X, pady=(0,5))\n        \n        ttk.Label(model_frame, text="Active Model:", font=("Arial", 9, "bold")).pack(anchor=tk.W)\n        \n        # Model dropdown\n        self.model_var = tk.StringVar()\n        self.model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, \n                                          state="readonly", width=35)\n        self.model_dropdown.pack(fill=tk.X, pady=(2,0))\n        self.model_dropdown.bind('<<ComboboxSelected>>', self.on_model_changed)\n        \n        # Model info\n        self.model_info_label = ttk.Label(model_frame, text="No model loaded", foreground="red")\n        self.model_info_label.pack(anchor=tk.W, pady=(5,0))\n        \n        # Model compatibility info\n        self.compatibility_label = ttk.Label(model_frame, text="", foreground="blue")\n        self.compatibility_label.pack(anchor=tk.W, pady=(2,0))\n        \n        ttk.Button(model_frame, text="Refresh Models", \n                  command=self.refresh_available_models).pack(anchor=tk.W, pady=(5,0))\n        \n        # === LIVE FEEDBACK CONTROLS ===\n        feedback_frame = ttk.LabelFrame(left_panel, text="Live Feedback Controls", padding=5)\n        feedback_frame.pack(fill=tk.X, pady=(0,5))\n        \n        ttk.Label(feedback_frame, text="Interaction:", font=("Arial", 9, "bold")).pack(anchor=tk.W)\n        ttk.Label(feedback_frame, text="• Right-click detection = WRONG", foreground="red").pack(anchor=tk.W)\n        ttk.Label(feedback_frame, text="• Ctrl+click detection = CORRECT", foreground="green").pack(anchor=tk.W)\n        ttk.Label(feedback_frame, text="• Drag to adjust bbox", foreground="blue").pack(anchor=tk.W)\n        \n        # Confidence thresholds\n        thresh_frame = ttk.LabelFrame(left_panel, text="Detection Thresholds", padding=5)\n        thresh_frame.pack(fill=tk.X, pady=(0,5))\n        \n        ttk.Label(thresh_frame, text="Auto-accept above:").pack(anchor=tk.W)\n        self.auto_accept_var = tk.DoubleVar(value=0.9)\n        ttk.Scale(thresh_frame, from_=0.5, to=1.0, variable=self.auto_accept_var, \n                 orient=tk.HORIZONTAL, length=200).pack(fill=tk.X)\n        self.auto_accept_label = ttk.Label(thresh_frame, text="90%")\n        self.auto_accept_label.pack(anchor=tk.W)\n        \n        # Bind threshold updates\n        self.auto_accept_var.trace('w', self.update_threshold_labels)\n        \n        # === CAPTURE CONTROLS ===\n        capture_frame = ttk.LabelFrame(left_panel, text="Capture Controls", padding=5)\n        capture_frame.pack(fill=tk.X, pady=(0,5))\n        \n        # Window selection\n        ttk.Label(capture_frame, text="Target Window:").pack(anchor=tk.W)\n        self.window_var = tk.StringVar()\n        self.window_dropdown = ttk.Combobox(capture_frame, textvariable=self.window_var, state="readonly")\n        self.window_dropdown.pack(fill=tk.X, pady=(2,5))\n        \n        ttk.Button(capture_frame, text="Refresh Windows", command=self.refresh_windows).pack(fill=tk.X)\n        \n        # Capture buttons\n        btn_frame = ttk.Frame(capture_frame)\n        btn_frame.pack(fill=tk.X, pady=(5,0))\n        \n        self.capture_btn = ttk.Button(btn_frame, text="Capture Screenshot", command=self.capture_screenshot)\n        self.capture_btn.pack(side=tk.LEFT, padx=(0,5))\n        \n        self.live_btn = ttk.Button(btn_frame, text="Start Live Mode", command=self.toggle_live_mode)\n        self.live_btn.pack(side=tk.LEFT)\n        \n        # === IMAGE DISPLAY ===\n        # Create canvas for image display\n        self.canvas = tk.Canvas(right_panel, bg='gray90')\n        self.canvas.pack(fill=tk.BOTH, expand=True)\n        \n        # Bind mouse events\n        self.canvas.bind("<Button-1>", self.on_canvas_click)\n        self.canvas.bind("<Button-3>", self.on_canvas_right_click)\n        self.canvas.bind("<Control-Button-1>", self.on_canvas_ctrl_click)\n        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)\n        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)\n        \n        # Status bar\n        self.status_label = ttk.Label(self.root, text="Ready - Select window and capture screenshot", \n                                     relief=tk.SUNKEN, anchor=tk.W)\n        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)\n        \n        # Initialize UI state\n        self.refresh_windows()\n        self.refresh_class_registry()\n        self.populate_class_list()\n        self.refresh_available_models()\n    \n    def setup_adaptive_ai(self):\n        """Initialize adaptive AI detection system"""\n        self.current_model_wrapper = None\n        print("[ADAPTIVE AI] Initialized adaptive detection system")\n        print(f"[ADAPTIVE AI] Available classes: {len(self.available_classes)}")\n        for class_name in self.available_classes:\n            print(f"  • {class_name}")\n    \n    def refresh_class_registry(self):\n        """Refresh the class registry and update UI"""\n        try:\n            # Force registry to scan models again\n            self.class_registry.scan_and_update_models()\n            self.available_classes = self.class_registry.get_all_classes()\n            \n            # Update status\n            class_count = len(self.available_classes)\n            self.registry_status_label.config(\n                text=f"{class_count} classes from registry", \n                foreground="green"\n            )\n            \n            print(f"[ADAPTIVE AI] Registry refreshed: {class_count} classes")\n            \n            # Update class list\n            self.populate_class_list()\n            \n        except Exception as e:\n            self.registry_status_label.config(text="Registry error", foreground="red")\n            print(f"[ADAPTIVE AI] Registry refresh error: {e}")\n    \n    def populate_class_list(self):\n        """Populate the class selection listbox"""\n        self.class_listbox.delete(0, tk.END)\n        \n        for class_name in sorted(self.available_classes):\n            self.class_listbox.insert(tk.END, class_name)\n            \n            # Select if in active classes\n            if class_name in self.active_classes:\n                self.class_listbox.selection_set(tk.END)\n    \n    def select_all_classes(self):\n        """Select all classes in the listbox"""\n        self.class_listbox.selection_set(0, tk.END)\n    \n    def clear_all_classes(self):\n        """Clear all class selections"""\n        self.class_listbox.selection_clear(0, tk.END)\n    \n    def apply_class_selection(self):\n        """Apply the selected classes as active classes"""\n        selected_indices = self.class_listbox.curselection()\n        self.active_classes = [self.available_classes[i] for i in selected_indices]\n        \n        if self.active_classes:\n            # Update model manager with new class system\n            self.model_manager.set_active_classes(self.active_classes)\n            \n            # Reload current model if needed\n            if self.current_model_wrapper:\n                self.current_model_wrapper.set_target_classes(self.active_classes)\n            \n            self.status_label.config(text=f"Active classes: {', '.join(self.active_classes[:3])}{'...' if len(self.active_classes) > 3 else ''}")\n            print(f"[ADAPTIVE AI] Active classes updated: {len(self.active_classes)} classes")\n        else:\n            messagebox.showwarning("Selection Required", "Please select at least one class!")\n    \n    def add_new_class(self):\n        """Add a new class to the registry"""\n        new_class = self.new_class_var.get().strip()\n        \n        if not new_class:\n            messagebox.showwarning("Input Required", "Please enter a class name!")\n            return\n        \n        if new_class in self.available_classes:\n            messagebox.showwarning("Duplicate Class", f"Class '{new_class}' already exists!")\n            return\n        \n        # Add to registry (will be saved when a model is trained with this class)\n        self.available_classes.append(new_class)\n        self.class_registry.register_class(new_class, "user_defined", "manually_added")\n        \n        # Update UI\n        self.populate_class_list()\n        self.new_class_var.set("")\n        \n        messagebox.showinfo("Class Added", f"New class '{new_class}' added successfully!\\n\\nYou can now use it for labeling. It will be registered with models when they are trained.")\n        print(f"[ADAPTIVE AI] Added new class: {new_class}")\n    \n    def refresh_available_models(self):\n        """Refresh the list of available models"""\n        try:\n            basic_manager = BasicModelManager()\n            available_models = basic_manager.get_all_model_versions()\n            \n            model_options = []\n            self.model_paths = {}\n            \n            for name, path, creation_time in available_models:\n                display_name = f"{name} ({creation_time.strftime('%Y-%m-%d %H:%M')})"  \n                model_options.append(display_name)\n                self.model_paths[display_name] = path\n            \n            if model_options:\n                self.model_dropdown['values'] = model_options\n                # Auto-select latest model\n                self.model_var.set(model_options[0])\n                self.load_selected_model()\n                print(f"[ADAPTIVE AI] Found {len(model_options)} available models")\n            else:\n                self.model_dropdown['values'] = ["No models available"]\n                self.model_var.set("No models available")\n                self.model_info_label.config(text="No trained models found", foreground="red")\n                \n        except Exception as e:\n            print(f"[ADAPTIVE AI] Error refreshing models: {e}")\n            self.model_dropdown['values'] = ["Error loading models"]\n            self.model_var.set("Error loading models")\n    \n    def on_model_changed(self, event=None):\n        """Handle model selection change"""\n        self.load_selected_model()\n    \n    def load_selected_model(self):\n        """Load the selected model with adaptive wrapper"""\n        selected_name = self.model_var.get()\n        \n        if selected_name in ["No models available", "Error loading models"]:\n            return\n        \n        if selected_name not in self.model_paths:\n            return\n        \n        model_path = self.model_paths[selected_name]\n        \n        try:\n            print(f"[ADAPTIVE AI] Loading model: {selected_name}")\n            \n            # Create adaptive wrapper\n            self.current_model_wrapper = AdaptiveModelWrapper(\n                str(model_path), \n                target_classes=self.active_classes\n            )\n            \n            if self.current_model_wrapper.model is not None:\n                # Update UI\n                native_classes = self.current_model_wrapper.get_native_classes()\n                target_classes = self.current_model_wrapper.get_target_classes()\n                \n                self.model_info_label.config(\n                    text=f"✓ Loaded: {model_path.parent.parent.name}\\nNative: {len(native_classes)} classes\\nTarget: {len(target_classes)} classes", \n                    foreground="green"\n                )\n                \n                # Show compatibility info\n                mapping = self.current_model_wrapper.get_class_mapping()\n                mapped_count = sum(1 for k, v in mapping.items() if k != v)\n                \n                if mapped_count > 0:\n                    self.compatibility_label.config(\n                        text=f"Mapping {mapped_count} classes for compatibility", \n                        foreground="orange"\n                    )\n                else:\n                    self.compatibility_label.config(\n                        text="Direct compatibility - no mapping needed", \n                        foreground="green"\n                    )\n                \n                print(f"[ADAPTIVE AI] Successfully loaded adaptive model")\n                print(f"[ADAPTIVE AI] Native classes: {native_classes}")\n                print(f"[ADAPTIVE AI] Target classes: {target_classes}")\n                \n            else:\n                raise Exception("Failed to load model")\n                \n        except Exception as e:\n            self.current_model_wrapper = None\n            self.model_info_label.config(text=f"Error: {e}", foreground="red")\n            self.compatibility_label.config(text="", foreground="black")\n            print(f"[ADAPTIVE AI] Error loading model: {e}")\n    \n    def detect_objects(self, image):\n        """Run adaptive AI object detection"""\n        detections = []\n        \n        if self.current_model_wrapper and self.current_model_wrapper.model:\n            try:\n                results = self.current_model_wrapper.predict(image, conf=0.25)\n                \n                if results:\n                    for result in results:\n                        if hasattr(result, 'boxes') and result.boxes is not None:\n                            boxes = result.boxes.xyxy.cpu().numpy()\n                            confidences = result.boxes.conf.cpu().numpy()\n                            classes = result.boxes.cls.cpu().numpy().astype(int)\n                            \n                            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):\n                                x1, y1, x2, y2 = box.astype(int)\n                                \n                                # Get class name from target classes\n                                if cls < len(self.active_classes):\n                                    obj_type = self.active_classes[cls]\n                                else:\n                                    obj_type = f"unknown_class_{cls}"\n                                \n                                detection = {\n                                    'type': obj_type,\n                                    'bbox': [x1, y1, x2, y2],\n                                    'confidence': float(conf),\n                                    'source': 'adaptive_ai'\n                                }\n                                detections.append(detection)\n                \n                print(f"[ADAPTIVE AI] Found {len(detections)} detections")\n                \n            except Exception as e:\n                print(f"[ADAPTIVE AI] Detection error: {e}")\n        \n        return detections\n    \n    # === WINDOW AND CAPTURE METHODS ===\n    def refresh_windows(self):\n        """Refresh available windows list"""\n        def enum_windows_callback(hwnd, windows):\n            if win32gui.IsWindowVisible(hwnd):\n                window_text = win32gui.GetWindowText(hwnd)\n                if window_text and len(window_text.strip()) > 0:\n                    windows.append((hwnd, window_text))\n            return True\n        \n        windows = []\n        win32gui.EnumWindows(enum_windows_callback, windows)\n        \n        # Sort windows by title\n        windows.sort(key=lambda x: x[1].lower())\n        \n        # Update dropdown\n        window_titles = [title for _, title in windows]\n        self.window_dropdown['values'] = window_titles\n        \n        # Try to find and select mGBA or emulator window\n        for title in window_titles:\n            if any(keyword in title.lower() for keyword in ['mgba', 'emulator', 'pokemon', 'gameboy']):\n                self.window_var.set(title)\n                break\n        \n        # Store window handles\n        self.windows_dict = {title: hwnd for hwnd, title in windows}\n    \n    def capture_screenshot(self):\n        """Capture screenshot of selected window"""\n        window_title = self.window_var.get()\n        if not window_title or window_title not in self.windows_dict:\n            messagebox.showwarning("No Window", "Please select a window to capture!")\n            return\n        \n        try:\n            hwnd = self.windows_dict[window_title]\n            self.emulator_window = hwnd\n            \n            # Get window rectangle\n            rect = win32gui.GetWindowRect(hwnd)\n            x, y, x2, y2 = rect\n            width = x2 - x\n            height = y2 - y\n            \n            # Capture window\n            wDC = win32gui.GetWindowDC(hwnd)\n            dcObj = win32ui.CreateDCFromHandle(wDC)\n            cDC = dcObj.CreateCompatibleDC()\n            dataBitMap = win32ui.CreateBitmap()\n            dataBitMap.CreateCompatibleBitmap(dcObj, width, height)\n            cDC.SelectObject(dataBitMap)\n            cDC.BitBlt((0, 0), (width, height), dcObj, (0, 0), win32con.SRCCOPY)\n            \n            # Convert to numpy array\n            signedIntsArray = dataBitMap.GetBitmapBits(True)\n            img = np.frombuffer(signedIntsArray, dtype=np.uint8)\n            img.shape = (height, width, 4)\n            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)\n            \n            # Cleanup\n            dcObj.DeleteDC()\n            cDC.DeleteDC()\n            win32gui.ReleaseDC(hwnd, wDC)\n            win32gui.DeleteObject(dataBitMap.GetHandle())\n            \n            self.current_screenshot = img\n            self.process_screenshot()\n            \n        except Exception as e:\n            messagebox.showerror("Capture Failed", f"Failed to capture screenshot: {e}")\n    \n    def process_screenshot(self):\n        """Process captured screenshot with AI detection"""\n        if self.current_screenshot is None:\n            return\n        \n        # Run AI detection\n        self.ai_detections = self.detect_objects(self.current_screenshot)\n        \n        # Display results\n        self.display_image_with_detections()\n        \n        # Update status\n        detection_count = len(self.ai_detections)\n        self.status_label.config(text=f"Captured screenshot - Found {detection_count} detections")\n        \n        print(f"[CAPTURE] Processed screenshot with {detection_count} detections")\n    \n    def display_image_with_detections(self):\n        """Display image with AI detections overlaid"""\n        if self.current_screenshot is None:\n            return\n        \n        # Create display image\n        display_img = self.current_screenshot.copy()\n        \n        # Draw detections\n        for i, detection in enumerate(self.ai_detections):\n            bbox = detection['bbox']\n            confidence = detection['confidence']\n            obj_type = detection['type']\n            \n            x1, y1, x2, y2 = bbox\n            \n            # Color based on confidence\n            if confidence >= self.auto_accept_threshold:\n                color = (0, 255, 0)  # Green for high confidence\n            elif confidence >= self.ask_confirmation_threshold:\n                color = (255, 165, 0)  # Orange for medium confidence\n            else:\n                color = (255, 0, 0)  # Red for low confidence\n            \n            # Draw bounding box\n            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)\n            \n            # Draw label\n            label = f"{obj_type} ({confidence:.2f})"\n            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]\n            cv2.rectangle(display_img, (x1, y1-20), (x1+label_size[0], y1), color, -1)\n            cv2.putText(display_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)\n        \n        # Convert to PhotoImage and display\n        img_pil = Image.fromarray(display_img)\n        \n        # Resize to fit canvas\n        canvas_width = self.canvas.winfo_width()\n        canvas_height = self.canvas.winfo_height()\n        \n        if canvas_width > 1 and canvas_height > 1:\n            img_pil.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)\n        \n        self.photo = ImageTk.PhotoImage(img_pil)\n        \n        # Clear canvas and display image\n        self.canvas.delete("all")\n        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)\n    \n    def toggle_live_mode(self):\n        """Toggle live capture mode"""\n        if not self.live_mode:\n            self.start_live_mode()\n        else:\n            self.stop_live_mode()\n    \n    def start_live_mode(self):\n        """Start live capture mode"""\n        if not self.window_var.get() or self.window_var.get() not in self.windows_dict:\n            messagebox.showwarning("No Window", "Please select a window first!")\n            return\n        \n        self.live_mode = True\n        self.live_paused = False\n        self.live_btn.config(text="Stop Live Mode")\n        \n        # Start capture thread\n        self.capture_thread = threading.Thread(target=self.live_capture_loop)\n        self.capture_thread.daemon = True\n        self.capture_thread.start()\n        \n        self.status_label.config(text="Live mode started - capturing continuously...")\n    \n    def stop_live_mode(self):\n        """Stop live capture mode"""\n        self.live_mode = False\n        self.live_btn.config(text="Start Live Mode")\n        self.status_label.config(text="Live mode stopped")\n    \n    def live_capture_loop(self):\n        """Continuous capture loop for live mode"""\n        while self.live_mode:\n            if not self.live_paused:\n                try:\n                    self.capture_screenshot()\n                    time.sleep(0.5)  # Capture every 500ms\n                except Exception as e:\n                    print(f"[LIVE] Capture error: {e}")\n                    break\n            else:\n                time.sleep(0.1)\n    \n    # === MOUSE EVENT HANDLERS ===\n    def on_canvas_click(self, event):\n        """Handle canvas click"""\n        print(f"[CANVAS] Click at ({event.x}, {event.y})")\n    \n    def on_canvas_right_click(self, event):\n        """Handle right-click on detection (mark as wrong)"""\n        print(f"[CANVAS] Right-click at ({event.x}, {event.y}) - marking as wrong")\n    \n    def on_canvas_ctrl_click(self, event):\n        """Handle ctrl+click on detection (mark as correct)"""\n        print(f"[CANVAS] Ctrl+click at ({event.x}, {event.y}) - marking as correct")\n    \n    def on_canvas_drag(self, event):\n        """Handle canvas drag"""\n        pass\n    \n    def on_canvas_release(self, event):\n        """Handle mouse release"""\n        pass\n    \n    def update_threshold_labels(self, *args):\n        """Update threshold display labels"""\n        auto_val = self.auto_accept_var.get()\n        self.auto_accept_label.config(text=f"{auto_val:.0%}")\n        self.auto_accept_threshold = auto_val\n    \n    def run(self):\n        """Run the application"""\n        print("[ADAPTIVE AI] Starting Adaptive Backseat Data Collector")\n        self.class_registry.print_registry_status()\n        \n        try:\n            self.root.mainloop()\n        except KeyboardInterrupt:\n            print("\\n[ADAPTIVE AI] Shutting down...")\n        finally:\n            self.stop_live_mode()\n\n\nif __name__ == "__main__":\n    app = AdaptiveBackseatDataCollector()\n    app.run()