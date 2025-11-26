#!/usr/bin/env python3
"""
🚗 BACKSEAT DRIVING DATA COLLECTOR (FIXED)
==========================================

Enhanced AI-assisted data collection with live feedback:
- Right-click detections to mark as wrong
- Ctrl+click to confirm correct detections  
- Live confidence voting system
- Active learning (AI asks for help on uncertain detections)
- Feedback database for continuous learning

This creates a human-AI collaboration loop for rapid model improvement!
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
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil  # For memory monitoring
from PIL import Image, ImageTk, ImageGrab, ImageDraw
import win32gui
import win32con
import win32ui
from ctypes import windll
from collections import defaultdict
from datetime import datetime
from online_learning_system import OnlineLearningIntegration
from model_manager import get_latest_model_path
from yolo_exporter import save_yolo_frame, CANONICAL_CLASSES

# ADAPTIVE SYSTEM IMPORTS
from class_registry import ClassRegistry
from adaptive_model_wrapper import AdaptiveModelWrapper
from shared_class_manager import get_shared_class_manager, cleanup_shared_manager

class LiveFeedbackSystem:
    """Manages live feedback and learning"""
    
    def __init__(self):
        self.feedback_db = []
        self.confidence_votes = defaultdict(list)
        self.false_positives = []
        self.true_positives = []
        self.corrections = []
        self.uncertainty_queue = []
        
    def add_negative_feedback(self, detection, reason="user_marked_wrong"):
        """User right-clicked = this detection is wrong"""
        feedback = {
            'type': 'negative',
            'detection': detection,
            'reason': reason,
            'timestamp': time.time(),
            'user_action': 'right_click'
        }
        self.feedback_db.append(feedback)
        self.false_positives.append(detection)
        print(f"[FEEDBACK] Marked as wrong: {detection['type']} (confidence: {detection.get('confidence', 0):.2f})")
        
    def add_positive_feedback(self, detection, reason="user_confirmed"):
        """User ctrl+clicked = this detection is correct"""
        feedback = {
            'type': 'positive',
            'detection': detection,
            'reason': reason,
            'timestamp': time.time(),
            'user_action': 'ctrl_click'
        }
        self.feedback_db.append(feedback)
        self.true_positives.append(detection)
        print(f"[FEEDBACK] Confirmed correct: {detection['type']} (confidence: {detection.get('confidence', 0):.2f})")
        
    def should_ask_for_confirmation(self, detection):
        """Active learning: should we ask user about this detection?"""
        confidence = detection.get('confidence', 0)
        obj_type = detection.get('type', '')
        
        if confidence < 0.6:
            return True, f"I'm only {confidence:.0%} sure this is a {obj_type}. Can you confirm?"
            
        if confidence < 0.8 and np.random.random() < 0.1:
            return True, f"Quick check: Is this {obj_type} detection correct?"
            
        return False, ""
        
    def get_learning_priorities(self):
        """What should the model focus on learning next?"""
        priorities = []
        
        # Analyze false positives
        fp_types = defaultdict(int)
        for fp in self.false_positives:
            fp_types[fp['type']] += 1
            
        for obj_type, count in fp_types.items():
            if count >= 3:
                priorities.append(f"Reduce false positives for {obj_type}")
                
        return priorities
        
    def export_feedback_for_training(self, output_path="feedback_training_data.json"):
        """Export feedback data for model retraining"""
        training_data = {
            'feedback_sessions': self.feedback_db,
            'false_positives': self.false_positives,
            'true_positives': self.true_positives,
            'corrections': self.corrections,
            'learning_priorities': self.get_learning_priorities(),
            'export_timestamp': time.time(),
            'total_feedback_events': len(self.feedback_db)
        }
        
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
            
        print(f"[EXPORT] Feedback data saved to {output_path}")
        return training_data

class BackseatDataCollector:
    """Enhanced data collector with live feedback capabilities"""
    
    def __init__(self):
        print("[INIT] Creating Tkinter window...")
        self.root = tk.Tk()
        self.root.title("Adaptive Backseat AI Trainer (Enhanced)")
        self.root.geometry("1800x1000")
        print("[INIT] Tkinter window created")
        
        # Core systems
        print("[INIT] Initializing feedback system...")
        self.feedback_system = LiveFeedbackSystem()
        self.current_screenshot = None
        self.ai_detections = []
        self.confirmed_labels = []
        print("[INIT] Feedback system initialized")
        
        # ADAPTIVE SYSTEM: Initialize flexible class management
        print("[INIT] Initializing class registry...")
        self.class_registry = ClassRegistry()
        print("[INIT] Class registry initialized")
        
        # Initialize shared class management (deferred to avoid blocking)
        print("[INIT] Deferring shared class management initialization...")
        self.class_manager = None
        self.available_classes = []
        self.active_classes = []
        print("[INIT] Shared class management deferred")
        
        self.current_model_wrapper = None
        
        # Emulator window
        self.emulator_window = None
        
        # Live mode
        self.live_mode = False
        self.capture_thread = None
        self.live_paused = False
        
        # Enhanced threading infrastructure (Phase 1)
        self._thread_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.background_tasks = ThreadPoolExecutor(
            max_workers=3, 
            thread_name_prefix="BackseatBG"
        )
        
        # Phase 2: Advanced Pipeline Infrastructure
        self.pipeline_mode = False  # Enable high-performance pipeline
        self.screenshot_thread = None
        self.ai_processing_thread = None
        self.memory_manager_thread = None
        
        # Phase 3: Multi-Window Monitoring
        self.multi_window_mode = False
        self.monitored_windows = {}  # {window_title: {'hwnd': hwnd, 'thread': thread, 'queue': queue}}
        self.window_threads = {}
        self.window_results_queue = queue.Queue(maxsize=50)  # Combined results from all windows
        
        # Phase 3: Model Hot-Swapping
        self.model_swap_queue = queue.Queue(maxsize=2)  # Queue for model swap requests
        self.model_swap_lock = threading.Lock()  # Protect model swapping operations
        
        # Phase 3: Distributed Processing Framework
        self.distributed_mode = False
        self.processing_nodes = {}  # {node_id: {'url': str, 'status': str, 'last_ping': time}}
        self.distributed_queue = queue.Queue(maxsize=100)  # Tasks for distributed processing
        self.distributed_results_queue = queue.Queue(maxsize=50)  # Results from nodes
        
        # Thread-safe queues for pipeline
        self.screenshot_queue = queue.Queue(maxsize=30)  # 3-second buffer at 10 FPS
        self.ai_results_queue = queue.Queue(maxsize=10)  # AI results buffer
        
        # Performance tracking
        self.performance_stats = {
            'screenshots_captured': 0,
            'ai_detections_processed': 0,
            'last_screenshot_time': 0,
            'last_ai_time': 0,
            'screenshot_fps': 0.0,
            'ai_fps': 0.0,
            'memory_usage_mb': 0.0,
            'last_cleanup_time': 0,
            'screenshot_errors': 0,
            'ai_errors': 0,
            'total_errors': 0
        }
        
        # Error recovery settings
        self.error_recovery = {
            'max_consecutive_errors': 5,
            'error_backoff_time': 1.0,
            'recovery_attempts': 0,
            'last_error_time': 0
        }
        
        # Start background memory manager
        self.start_memory_manager()
        
        # Live feedback state
        self.selected_detection = None
        self.right_click_enabled = True
        self.active_learning_enabled = True
        
        # Detection confidence thresholds
        self.auto_accept_threshold = 0.9
        self.ask_confirmation_threshold = 0.6
        
        # Manual draw mode state
        self.draw_mode = False
        self.drawing = False
        self.draw_start_x = None
        self.draw_start_y = None
        self.draw_current_rect = None
        
        # Focus mode state
        self.focus_mode = False
        self.focus_areas = []  # List of temporary focus areas
        self.focus_current_rect = None
        
        # Data management
        self.data_dir = Path("backseat_training_data")
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "screenshots").mkdir(exist_ok=True)
        (self.data_dir / "metadata").mkdir(exist_ok=True)
        (self.data_dir / "feedback").mkdir(exist_ok=True)
        
        print("[INIT] Setting up UI...")
        self.setup_ui()
        print("[INIT] UI setup complete")
        
        # Initialize loading state
        self.loading_complete = False
        
        # Start loading animation now that everything is set up
        self.start_loading_animation()
        
        # Start background loading thread (non-blocking)
        self.background_thread = threading.Thread(target=self.initialize_heavy_systems_threaded, daemon=True)
        self.background_thread.start()
        print("[INIT] Heavy system initialization started in background thread")
    
    def initialize_heavy_systems_threaded(self):
        """Initialize heavy systems in background thread"""
        print("[THREAD] Starting heavy system initialization in background...")
        
        try:
            print("[THREAD] Initializing shared class management...")
            self.class_manager = get_shared_class_manager()
            print("[THREAD] Class manager created")
            
            self.available_classes = self.class_manager.get_classes()
            print(f"[THREAD] Got {len(self.available_classes)} classes")
            
            self.active_classes = self.available_classes.copy()
            print("[THREAD] Active classes set")
            
            self.class_manager.subscribe(self._on_classes_changed, "BackseatCollector")
            print("[THREAD] Subscribed to class changes")
            
            print("[THREAD] Deferring AI detector setup to main thread...")
            # Defer AI detector setup to main thread now that classes are loaded
            self.root.after(0, self.setup_ai_detector_deferred)
            
            print("[THREAD] Initializing class registry...")
            try:
                # ADAPTIVE SYSTEM: Initialize class registry
                self.refresh_class_registry()
                print("[THREAD] Class registry initialized successfully")
            except Exception as e:
                print(f"[THREAD] Class registry initialization failed: {e}")
                import traceback
                traceback.print_exc()
            
            print("[THREAD] Integrating online learning system...")
            try:
                # Integrate online learning system
                self.online_learning = OnlineLearningIntegration.integrate_with_backseat_collector(self)
                print("[THREAD] Online learning system integrated successfully")
            except Exception as e:
                print(f"[THREAD] Online learning integration failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Update status on main thread
            self.root.after(0, self.on_loading_complete)
            
            print("[THREAD] Heavy system initialization complete")
            
        except Exception as e:
            print(f"[THREAD] Error during heavy initialization: {e}")
            import traceback
            traceback.print_exc()
            
            # Update status on main thread
            error_msg = f"Error during initialization: {e}"
            self.root.after(0, lambda: self.on_loading_error(error_msg))
    
    def on_loading_complete(self):
        """Called on main thread when background loading is complete"""
        self.loading_complete = True
        if hasattr(self, 'status_bar'):
            self.status_bar.config(text="Ready - All systems loaded")
        print("[MAIN] Background loading completed successfully")
    
    def on_loading_error(self, error_msg):
        """Called on main thread when background loading fails"""
        if hasattr(self, 'status_bar'):
            self.status_bar.config(text=error_msg)
        print(f"[MAIN] Background loading failed: {error_msg}")
    
    def start_loading_animation(self):
        """Start loading animation in status bar"""
        self.loading_dots = 0
        self.animate_loading()
    
    def animate_loading(self):
        """Animate loading dots"""
        if not self.loading_complete:
            dots = '.' * (self.loading_dots % 4)
            self.status_bar.config(text=f"🔄 Loading system components{dots}")
            self.loading_dots += 1
            self.root.after(500, self.animate_loading)
    
    def setup_ai_detector_deferred(self):
        """Setup AI detector on main thread with populated classes"""
        print("[MAIN] Setting up AI detector with classes loaded...")
        try:
            self.setup_ai_detector()
            print(f"[MAIN] AI detector setup completed with {len(self.active_classes)} classes")
        except Exception as e:
            print(f"[MAIN] AI detector setup failed: {e}")
            import traceback
            traceback.print_exc()
        
    def setup_ui(self):
        """Setup enhanced UI with feedback controls"""
        
        # Main layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - scrollable controls container
        left_container = ttk.Frame(main_frame, width=300)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0,5))
        left_container.pack_propagate(False)
        
        # Create scrollable left panel
        left_canvas = tk.Canvas(left_container, width=280)
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        left_panel = ttk.Frame(left_canvas)
        
        # Configure scrolling
        left_panel.bind('<Configure>', lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=left_panel, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        # Pack scrollable components
        left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Right panel - image display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # === BACKSEAT DRIVING CONTROLS ===
        feedback_frame = ttk.LabelFrame(left_panel, text="Backseat Driving Controls", padding=5)
        feedback_frame.pack(fill=tk.X, pady=(0,5))
        
        ttk.Label(feedback_frame, text="Live Feedback:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        ttk.Label(feedback_frame, text="• Right-click detection = WRONG", foreground="red").pack(anchor=tk.W)
        ttk.Label(feedback_frame, text="• Ctrl+click detection = CORRECT", foreground="green").pack(anchor=tk.W)
        ttk.Label(feedback_frame, text="• Drag to adjust bbox", foreground="blue").pack(anchor=tk.W)
        
        # Active learning controls
        active_frame = ttk.Frame(feedback_frame)
        active_frame.pack(fill=tk.X, pady=(5,0))
        
        self.active_learning_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(active_frame, text="Ask for uncertain detections", 
                       variable=self.active_learning_var).pack(anchor=tk.W)
        
        # ADAPTIVE SYSTEM: Class Management UI
        adaptive_frame = ttk.LabelFrame(left_panel, text="Adaptive Class System", padding=5)
        adaptive_frame.pack(fill=tk.X, pady=(0,5))
        
        # Class registry status
        ttk.Label(adaptive_frame, text="Class Registry:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        self.registry_status_label = ttk.Label(adaptive_frame, text="Loading...", foreground="blue")
        self.registry_status_label.pack(anchor=tk.W)
        
        # Quick add new class
        new_class_frame = ttk.Frame(adaptive_frame)
        new_class_frame.pack(fill=tk.X, pady=(5,0))
        
        ttk.Label(new_class_frame, text="Add new class:").pack(anchor=tk.W)
        self.new_class_var = tk.StringVar()
        new_class_entry = ttk.Entry(new_class_frame, textvariable=self.new_class_var, width=20)
        new_class_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        ttk.Button(new_class_frame, text="Add", command=self.add_new_class, width=6).pack(side=tk.RIGHT)
        
        ttk.Button(adaptive_frame, text="Manage Classes", 
                  command=self.show_class_manager).pack(anchor=tk.W, pady=(5,0))

        # Confidence thresholds
        thresh_frame = ttk.LabelFrame(left_panel, text="Confidence Thresholds", padding=5)
        thresh_frame.pack(fill=tk.X, pady=(0,5))
        
        ttk.Label(thresh_frame, text="Auto-accept above:").pack(anchor=tk.W)
        self.auto_accept_var = tk.DoubleVar(value=0.9)
        ttk.Scale(thresh_frame, from_=0.5, to=1.0, variable=self.auto_accept_var, 
                 orient=tk.HORIZONTAL, length=200).pack(fill=tk.X)
        self.auto_accept_label = ttk.Label(thresh_frame, text="90%")
        self.auto_accept_label.pack(anchor=tk.W)
        
        ttk.Label(thresh_frame, text="Ask confirmation below:").pack(anchor=tk.W)
        self.ask_confirm_var = tk.DoubleVar(value=0.6)
        ttk.Scale(thresh_frame, from_=0.2, to=0.9, variable=self.ask_confirm_var,
                 orient=tk.HORIZONTAL, length=200).pack(fill=tk.X)
        self.ask_confirm_label = ttk.Label(thresh_frame, text="60%")
        self.ask_confirm_label.pack(anchor=tk.W)
        
        # Update threshold labels
        self.auto_accept_var.trace('w', self.update_threshold_labels)
        self.ask_confirm_var.trace('w', self.update_threshold_labels)
        
        # === FEEDBACK STATISTICS ===
        stats_frame = ttk.LabelFrame(left_panel, text="Learning Statistics", padding=5)
        stats_frame.pack(fill=tk.X, pady=(0,5))
        
        # Stats text with scrollbar
        stats_container = ttk.Frame(stats_frame)
        stats_container.pack(fill=tk.BOTH, expand=True)
        
        stats_scrollbar = ttk.Scrollbar(stats_container)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.stats_text = tk.Text(stats_container, height=8, width=35, font=("Courier", 8),
                                 yscrollcommand=stats_scrollbar.set, wrap=tk.WORD)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.config(command=self.stats_text.yview)
        
        # === EMULATOR SETUP ===
        emulator_frame = ttk.LabelFrame(left_panel, text="Emulator Setup", padding=5)
        emulator_frame.pack(fill=tk.X, pady=(0,5))
        
        self.emulator_var = tk.StringVar(value="No emulator found")
        ttk.Label(emulator_frame, textvariable=self.emulator_var, wraplength=250).pack(anchor=tk.W)
        
        ttk.Button(emulator_frame, text="Find Emulator Window", 
                  command=self.find_emulator_window).pack(fill=tk.X, pady=2)
        
        # === MODEL SELECTION ===
        model_frame = ttk.LabelFrame(left_panel, text="AI Model Selection", padding=5)
        model_frame.pack(fill=tk.X, pady=(0,5))
        
        ttk.Label(model_frame, text="Select Model:").pack(anchor=tk.W)
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                         state="readonly", width=35)
        self.model_dropdown.pack(fill=tk.X, pady=(0,5))
        self.model_dropdown.bind('<<ComboboxSelected>>', self.on_model_changed)
        
        # Phase 3: Hot-swap controls
        hotswap_frame = ttk.Frame(model_frame)
        hotswap_frame.pack(fill=tk.X, pady=(5,0))
        
        ttk.Button(hotswap_frame, text="🔄 Hot-Swap", 
                  command=self.hot_swap_model).pack(side=tk.LEFT, padx=(0,5))
        self.hot_swap_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(hotswap_frame, text="Live Swap", 
                       variable=self.hot_swap_enabled).pack(side=tk.LEFT)
        
        # Model info display
        self.model_info_label = ttk.Label(model_frame, text="No model loaded", 
                                        wraplength=250, foreground="gray")
        self.model_info_label.pack(anchor=tk.W)
        
        # Refresh models button
        ttk.Button(model_frame, text="Refresh Available Models", 
                  command=self.refresh_available_models).pack(fill=tk.X, pady=(5,0))
        
        # === WINDOW CONTROLS ===
        window_frame = ttk.LabelFrame(left_panel, text="Window Controls", padding=5)
        window_frame.pack(fill=tk.X, pady=(0,5))
        
        self.always_on_top_var = tk.BooleanVar()
        ttk.Checkbutton(window_frame, text="Always on top", 
                       variable=self.always_on_top_var, command=self.toggle_always_on_top).pack(anchor=tk.W, pady=2)
        
        self.transparent_var = tk.BooleanVar()
        ttk.Checkbutton(window_frame, text="Semi-transparent (see emulator below)", 
                       variable=self.transparent_var, command=self.toggle_transparency).pack(anchor=tk.W, pady=2)
        
        # === CAPTURE CONTROLS ===
        capture_frame = ttk.LabelFrame(left_panel, text="Capture Controls", padding=5)
        capture_frame.pack(fill=tk.X, pady=(0,5))
        
        self.live_var = tk.BooleanVar()
        ttk.Checkbutton(capture_frame, text="Enable Live Mode", 
                       variable=self.live_var, command=self.toggle_live_mode).pack(anchor=tk.W, pady=2)
        
        # Phase 2: High-performance pipeline toggle
        self.pipeline_var = tk.BooleanVar()
        pipeline_cb = ttk.Checkbutton(capture_frame, text="🚀 High-Performance Pipeline (10+ FPS)", 
                                     variable=self.pipeline_var, command=self.toggle_pipeline_mode)
        pipeline_cb.pack(anchor=tk.W, pady=2)
        
        # Phase 3: Multi-window monitoring toggle
        self.multi_window_var = tk.BooleanVar()
        multi_cb = ttk.Checkbutton(capture_frame, text="🖥️ Multi-Window Monitoring", 
                                  variable=self.multi_window_var, command=self.toggle_multi_window_mode)
        multi_cb.pack(anchor=tk.W, pady=2)
        
        # Multi-window controls
        multi_controls = ttk.Frame(capture_frame)
        multi_controls.pack(fill=tk.X, pady=2)
        
        ttk.Button(multi_controls, text="Add Window", 
                  command=self.add_window_dialog).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(multi_controls, text="Remove Window", 
                  command=self.remove_window_dialog).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(multi_controls, text="List Windows", 
                  command=self.show_monitored_windows).pack(side=tk.LEFT)
        
        # Performance statistics display
        perf_frame = ttk.LabelFrame(capture_frame, text="Performance Stats", padding=5)
        perf_frame.pack(fill=tk.X, pady=(5,0))
        
        self.perf_stats_label = ttk.Label(perf_frame, text="Not running", font=("Arial", 8))
        self.perf_stats_label.pack(anchor=tk.W)
        
        # Start performance monitoring
        self.update_performance_display()
        
        # Phase 3: Advanced Controls
        advanced_frame = ttk.LabelFrame(capture_frame, text="Advanced Controls", padding=5)
        advanced_frame.pack(fill=tk.X, pady=(5,0))
        
        adv_row1 = ttk.Frame(advanced_frame)
        adv_row1.pack(fill=tk.X, pady=2)
        
        ttk.Button(adv_row1, text="⚙️ Settings", 
                  command=self.show_advanced_settings).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(adv_row1, text="📊 Analytics", 
                  command=self.show_analytics_dashboard).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(adv_row1, text="🔧 Debug", 
                  command=self.show_debug_panel).pack(side=tk.LEFT)
        
        ttk.Button(capture_frame, text="Take Screenshot", 
                  command=self.capture_screenshot).pack(fill=tk.X, pady=(0,2))
        
        ttk.Button(capture_frame, text="Export Feedback Data", 
                  command=self.export_feedback_data_async).pack(fill=tk.X, pady=(0,2))
        
        ttk.Button(capture_frame, text="Online Learning Status", 
                  command=self.show_online_learning_status).pack(fill=tk.X, pady=(0,2))
        
        # GPU acceleration toggle
        gpu_frame = ttk.Frame(capture_frame)
        gpu_frame.pack(fill=tk.X, pady=(5,0))
        
        self.use_gpu_var = tk.BooleanVar(value=self.detect_gpu_available())
        ttk.Checkbutton(gpu_frame, text="Use GPU acceleration", 
                       variable=self.use_gpu_var, command=self.update_gpu_status).pack(side=tk.LEFT)
        
        self.gpu_status_label = ttk.Label(gpu_frame, text="", font=("Arial", 8))
        self.gpu_status_label.pack(side=tk.LEFT, padx=(10,0))
        self.update_gpu_status()
        
        # === YOLO EXPORT ===
        yolo_frame = ttk.LabelFrame(left_panel, text="YOLO Export", padding=5)
        yolo_frame.pack(fill=tk.X, pady=(0,5))

        self.yolo_images_out = tk.StringVar(value=r"F:\GameAgentUSB\datasets\gba\images\train")
        self.yolo_labels_out = tk.StringVar(value=r"F:\GameAgentUSB\datasets\gba\labels\train")

        row = ttk.Frame(yolo_frame); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Images out:").pack(side=tk.LEFT)
        img_entry = ttk.Entry(row, textvariable=self.yolo_images_out)
        img_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(row, text="Browse", command=lambda: self._choose_dir(self.yolo_images_out)).pack(side=tk.RIGHT)

        row2 = ttk.Frame(yolo_frame); row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Labels out:").pack(side=tk.LEFT)
        lbl_entry = ttk.Entry(row2, textvariable=self.yolo_labels_out)
        lbl_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(row2, text="Browse", command=lambda: self._choose_dir(self.yolo_labels_out)).pack(side=tk.RIGHT)

        ttk.Button(yolo_frame, text="Save current frame → YOLO (Ctrl+S)",
                   command=self.save_current_frame_to_yolo_async).pack(fill=tk.X, pady=(6,0))
        
        # === MANUAL ANNOTATION ===
        manual_frame = ttk.LabelFrame(left_panel, text="Manual Annotation", padding=5)
        manual_frame.pack(fill=tk.X, pady=(0,5))
        
        self.draw_mode_var = tk.BooleanVar()
        ttk.Checkbutton(manual_frame, text="Draw Box Mode (Drag to create boxes)", 
                       variable=self.draw_mode_var, command=self.toggle_draw_mode).pack(anchor=tk.W, pady=2)
        
        # Label selection for manual boxes
        tk.Label(manual_frame, text="New box label:").pack(anchor=tk.W, pady=(5,2))
        self.manual_label_var = tk.StringVar(value="missed_object")
        # Updated to match competitive trainer object classes
        manual_labels = [
            "building", "dialogue_box", "hp_bar", "menu_box", "npc_character", 
            "player_character", "pokemon_sprite", "text_area", "tree", 
            "pokeball/item", "grass", "water", "exp._bar", "level_indicator",
            "missed_object", "other"
        ]
        self.manual_label_dropdown = ttk.Combobox(manual_frame, textvariable=self.manual_label_var, 
                                                values=manual_labels, width=20)
        self.manual_label_dropdown.pack(fill=tk.X, pady=(0,5))
        
        # === FOCUS MODE ===
        focus_frame = ttk.LabelFrame(left_panel, text="Focus Mode (Temporary)", padding=5)
        focus_frame.pack(fill=tk.X, pady=(0,5))
        
        self.focus_mode_var = tk.BooleanVar()
        ttk.Checkbutton(focus_frame, text="Focus Mode (Drag to highlight areas temporarily)", 
                       variable=self.focus_mode_var, command=self.toggle_focus_mode).pack(anchor=tk.W, pady=2)
        
        # Focus duration
        tk.Label(focus_frame, text="Focus duration:").pack(anchor=tk.W, pady=(5,2))
        self.focus_duration_var = tk.StringVar(value="10")
        focus_duration_frame = ttk.Frame(focus_frame)
        focus_duration_frame.pack(fill=tk.X, pady=(0,2))
        
        ttk.Entry(focus_duration_frame, textvariable=self.focus_duration_var, width=5).pack(side=tk.LEFT)
        tk.Label(focus_duration_frame, text=" seconds").pack(side=tk.LEFT)
        
        ttk.Button(focus_frame, text="Clear All Focus Areas", 
                  command=self.clear_focus_areas).pack(fill=tk.X, pady=(5,0))
        
        # === DETECTION LIST ===
        list_frame = ttk.LabelFrame(left_panel, text="AI Detections", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Detection listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.detection_listbox = tk.Listbox(list_container, height=15, font=("Courier", 9),
                                           yscrollcommand=scrollbar.set, selectmode=tk.EXTENDED)
        self.detection_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.detection_listbox.yview)
        
        # Quick action buttons
        action_frame = ttk.Frame(list_frame)
        action_frame.pack(fill=tk.X, pady=(5,0))
        
        ttk.Button(action_frame, text="Confirm Selected", 
                  command=self.confirm_selected_detections).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(action_frame, text="Mark Wrong", 
                  command=self.mark_selected_wrong).pack(side=tk.LEFT)
        
        # === IMAGE DISPLAY ===
        self.setup_image_display(right_panel)
        
        # Bind Ctrl+S hotkey for YOLO export (async)
        self.root.bind("<Control-s>", lambda e: self.save_current_frame_to_yolo_async())
        
    def setup_image_display(self, parent):
        """Setup image display with click handlers"""
        
        # Image frame
        image_frame = ttk.Frame(parent)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        self.image_canvas = tk.Canvas(image_frame, bg='black')
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind events for interactive feedback
        self.image_canvas.bind('<Button-1>', self.on_canvas_click)
        self.image_canvas.bind('<Button-3>', self.on_canvas_right_click)  # Right click
        self.image_canvas.bind('<Control-Button-1>', self.on_canvas_ctrl_click)  # Ctrl+click
        
        # Draw mode bindings
        self.image_canvas.bind('<ButtonPress-1>', self.on_canvas_press)
        self.image_canvas.bind('<B1-Motion>', self.on_canvas_drag)
        self.image_canvas.bind('<ButtonRelease-1>', self.on_canvas_release)
        
        # Status bar
        self.status_bar = ttk.Label(parent, text="🔄 Loading system components...", 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Loading animation will be started after init is complete
        
    def setup_ai_detector(self):
        """Initialize AI detection system (main thread only)"""
        self.ai_model = None
        self.detection_method = "simple"
        self.current_model_path = None
        
        # Initialize available models and load the first one
        self.refresh_available_models()
        
    def refresh_available_models(self):
        """Refresh the list of available models and populate dropdown"""
        try:
            from model_manager import ModelManager
            manager = ModelManager()
            available_models = manager.get_all_model_versions()
            
            model_names = []
            self.model_paths = {}  # Map model names to paths
            
            for name, path, creation_time in available_models:
                display_name = f"{name} ({creation_time.strftime('%Y-%m-%d %H:%M')})"
                model_names.append(display_name)
                self.model_paths[display_name] = path
            
            # Thread-safe UI updates
            import threading
            if threading.current_thread() is threading.main_thread():
                self._update_model_dropdown_ui(model_names)
            else:
                self.root.after(0, lambda: self._update_model_dropdown_ui(model_names))
                
        except Exception as e:
            print(f"[AI] Error refreshing models: {e}")
            # Thread-safe error handling
            import threading
            if threading.current_thread() is threading.main_thread():
                self._update_model_dropdown_error()
            else:
                self.root.after(0, self._update_model_dropdown_error)
    
    def _update_model_dropdown_ui(self, model_names):
        """Update model dropdown UI (main thread only)"""
        if model_names:
            if hasattr(self, 'model_dropdown'):
                self.model_dropdown['values'] = model_names
                # Auto-select the latest model (first in list)
                self.model_var.set(model_names[0])
                self.load_selected_model_async()
                print(f"[AI] Found {len(model_names)} available models")
        else:
            if hasattr(self, 'model_dropdown'):
                self.model_dropdown['values'] = ["No models available"]
                self.model_var.set("No models available")
            if hasattr(self, 'model_info_label'):
                self.model_info_label.config(text="No trained models found", foreground="red")
            print("[AI] No trained models found")
    
    def _update_model_dropdown_error(self):
        """Update model dropdown with error (main thread only)"""
        if hasattr(self, 'model_dropdown'):
            self.model_dropdown['values'] = ["Error loading models"]
            self.model_var.set("Error loading models")
    
    def on_model_changed(self, event=None):
        """Handle model selection change with background loading"""
        self.load_selected_model_async()
        
    def unload_current_model(self):
        """Safely unload the current model to prevent conflicts"""
        if self.ai_model is not None:
            try:
                # Clear the model from memory
                del self.ai_model
                self.ai_model = None
                self.detection_method = "simple"
                self.current_model_path = None
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                
                print("[AI] Previous model unloaded successfully")
                return True
            except Exception as e:
                print(f"[AI] Error unloading model: {e}")
                return False
        return True
    
    def load_selected_model_async(self):
        """Load the selected model in background to prevent UI freezing"""
        selected_name = self.model_var.get()
        
        if selected_name == "No models available" or selected_name == "Error loading models":
            return
            
        if selected_name in self.model_paths:
            model_path = self.model_paths[selected_name]
            
            # Don't reload if it's the same model
            if self.current_model_path == model_path:
                return
            
            # Show loading state immediately
            self.model_info_label.config(text="🔄 Loading model...", foreground="blue")
            self.status_bar.config(text=f"Loading model: {selected_name}...")
            
            # Submit background task
            future = self.background_tasks.submit(self._load_model_worker, selected_name, model_path)
            
            # Monitor completion
            def check_loading():
                if future.done():
                    try:
                        result = future.result()
                        self._on_model_loaded_success(result)
                    except Exception as e:
                        self._on_model_loaded_error(e, selected_name)
                else:
                    # Update loading animation
                    current_text = self.model_info_label.cget("text")
                    dots = len([c for c in current_text if c == '.'])
                    new_dots = '.' * ((dots % 3) + 1)
                    self.model_info_label.config(text=f"🔄 Loading model{new_dots}")
                    self.root.after(300, check_loading)  # Check again in 300ms
            
            self.root.after(100, check_loading)
    
    def _load_model_worker(self, selected_name, model_path):
        """Background worker for model loading"""
        print(f"[BACKGROUND] Loading model: {selected_name}")
        
        # Unload current model first (this can be slow)
        if not self.unload_current_model():
            raise Exception("Failed to unload previous model")
        
        # Load new model (this is the slow part)
        print(f"[ADAPTIVE AI] Loading model with adaptive wrapper...")
        current_model_wrapper = AdaptiveModelWrapper(
            str(model_path), 
            target_classes=self.active_classes
        )
        
        if current_model_wrapper.model is None:
            raise Exception("Failed to load model with adaptive wrapper")
        
        # Return all the data needed for UI update
        model_name = model_path.parent.parent.name
        native_classes = current_model_wrapper.get_native_classes()
        target_classes = current_model_wrapper.get_target_classes()
        mapping = current_model_wrapper.get_class_mapping()
        mapped_count = sum(1 for k, v in mapping.items() if k != v)
        
        return {
            'wrapper': current_model_wrapper,
            'model_path': model_path,
            'model_name': model_name,
            'native_classes': native_classes,
            'target_classes': target_classes,
            'mapped_count': mapped_count
        }
    
    def _on_model_loaded_success(self, result):
        """Handle successful model loading in UI thread"""
        # Update instance variables
        self.current_model_wrapper = result['wrapper']
        self.ai_model = result['wrapper'].model
        self.detection_method = "adaptive_yolo"
        self.current_model_path = result['model_path']
        
        # Update UI
        info_text = f"✓ Loaded: {result['model_name']}\n"
        info_text += f"Native: {len(result['native_classes'])} classes\n"
        info_text += f"Active: {len(result['target_classes'])} classes"
        
        if result['mapped_count'] > 0:
            info_text += f"\nMapped: {result['mapped_count']} classes"
            color = "orange"
        else:
            info_text += f"\nDirect compatibility"
            color = "green"
        
        self.model_info_label.config(text=info_text, foreground=color)
        self.status_bar.config(text=f"✓ Model loaded successfully: {result['model_name']}")
        
        print(f"[BACKGROUND] Successfully loaded: {result['model_name']}")
        print(f"[ADAPTIVE AI] Class compatibility: {len(result['native_classes'])} -> {len(result['target_classes'])}")
        if result['mapped_count'] > 0:
            print(f"[ADAPTIVE AI] Applied {result['mapped_count']} class mappings for compatibility")
    
    def _on_model_loaded_error(self, error, selected_name):
        """Handle model loading errors in UI thread"""
        error_msg = f"Error loading {selected_name}: {str(error)}"
        self.model_info_label.config(text=error_msg, foreground="red")
        self.status_bar.config(text=f"❌ Failed to load model: {selected_name}")
        print(f"[BACKGROUND] {error_msg}")
        
        # Clear model state on error
        self.ai_model = None
        self.current_model_wrapper = None
        self.detection_method = "simple"

    def load_selected_model(self):
        """Load the selected model from dropdown (DEPRECATED - use load_selected_model_async)"""
        selected_name = self.model_var.get()
        
        if selected_name == "No models available" or selected_name == "Error loading models":
            return
            
        if selected_name in self.model_paths:
            model_path = self.model_paths[selected_name]
            
            # Don't reload if it's the same model
            if self.current_model_path == model_path:
                return
                
            print(f"[AI] Loading model: {selected_name}")
            
            # Unload current model first
            if not self.unload_current_model():
                self.model_info_label.config(text="Error unloading previous model", foreground="red")
                return
            
            try:
                # ADAPTIVE SYSTEM: Use adaptive wrapper for universal compatibility
                print(f"[ADAPTIVE AI] Loading model with adaptive wrapper...")
                
                self.current_model_wrapper = AdaptiveModelWrapper(
                    str(model_path), 
                    target_classes=self.active_classes
                )
                
                if self.current_model_wrapper.model is not None:
                    # Keep compatibility with existing code
                    self.ai_model = self.current_model_wrapper.model
                    self.detection_method = "adaptive_yolo"
                    self.current_model_path = model_path
                    
                    # Enhanced info display
                    model_name = model_path.parent.parent.name
                    native_classes = self.current_model_wrapper.get_native_classes()
                    target_classes = self.current_model_wrapper.get_target_classes()
                    mapping = self.current_model_wrapper.get_class_mapping()
                    mapped_count = sum(1 for k, v in mapping.items() if k != v)
                    
                    info_text = f"✓ Loaded: {model_name}\n"
                    info_text += f"Native: {len(native_classes)} classes\n"
                    info_text += f"Active: {len(target_classes)} classes"
                    
                    if mapped_count > 0:
                        info_text += f"\nMapped: {mapped_count} classes"
                        color = "orange"
                    else:
                        info_text += f"\nDirect compatibility"
                        color = "green"
                    
                    self.model_info_label.config(text=info_text, foreground=color)
                    
                    print(f"[ADAPTIVE AI] Successfully loaded: {model_name}")
                    print(f"[ADAPTIVE AI] Class compatibility: {len(native_classes)} -> {len(target_classes)}")
                    if mapped_count > 0:
                        print(f"[ADAPTIVE AI] Applied {mapped_count} class mappings for compatibility")
                else:
                    raise Exception("Failed to load model with adaptive wrapper")
                
            except ImportError:
                self.ai_model = None
                self.current_model_wrapper = None
                self.detection_method = "simple"
                self.model_info_label.config(text="YOLO not available", foreground="red")
                print("[AI] YOLO not available")
            except Exception as e:
                self.ai_model = None
                self.current_model_wrapper = None
                self.detection_method = "simple"
                self.model_info_label.config(text=f"Error loading model: {e}", foreground="red")
                print(f"[ADAPTIVE AI] Error loading model: {e}")
        else:
            print(f"[AI] Model path not found for: {selected_name}")
            
    def update_threshold_labels(self, *args):
        """Update threshold display labels"""
        auto_val = self.auto_accept_var.get()
        ask_val = self.ask_confirm_var.get()
        
        self.auto_accept_label.config(text=f"{auto_val:.0%}")
        self.ask_confirm_label.config(text=f"{ask_val:.0%}")
        
        # Update thresholds
        self.auto_accept_threshold = auto_val
        self.ask_confirmation_threshold = ask_val
        
    # Note: detect_objects method is now implemented later in the file for hot-swapping support
        
    def simple_object_detection(self, image):
        """Fallback simple detection method"""
        h, w = image.shape[:2]
        
        mock_detections = [
            {
                'type': 'demo_object',
                'bbox': [w//4, h//4, w//2, h//2],
                'confidence': 0.5,
                'method': 'simple_cv',
                'model_suggestion': True,
                'needs_confirmation': True
            }
        ]
        
        return mock_detections
        
    def on_canvas_right_click(self, event):
        """Handle right-click on detection (re-label or mark as wrong)"""
        # Find detection directly without overlapping dialog for right-click
        detection = self.get_detection_at_point_direct(event.x, event.y)
        if detection:
            self.show_relabel_dialog(detection)
            
    def get_detection_at_point_direct(self, x, y):
        """Find detection at canvas coordinates without overlapping dialog"""
        if not hasattr(self, 'current_screenshot_display') or not self.ai_detections:
            return None
            
        if hasattr(self, 'display_scale'):
            img_x = int(x / self.display_scale)
            img_y = int(y / self.display_scale)
            
            # Find all overlapping detections
            overlapping_detections = []
            for detection in self.ai_detections:
                bbox = detection['bbox']
                if (bbox[0] <= img_x <= bbox[2] and 
                    bbox[1] <= img_y <= bbox[3]):
                    overlapping_detections.append(detection)
            
            if overlapping_detections:
                # For right-click, just return the first (smallest area) detection
                overlapping_detections.sort(key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]))
                return overlapping_detections[0]
                    
        return None
            
    def on_canvas_ctrl_click(self, event):
        """Handle Ctrl+click on detection (confirm as correct)"""
        detection = self.get_detection_at_point(event.x, event.y)
        if detection:
            self.feedback_system.add_positive_feedback(detection)
            detection['user_confirmed'] = True
            self.update_display()
            self.update_stats_display()
            
    def show_relabel_dialog(self, detection):
        """Show dialog to re-label or remove detection"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Re-label Detection")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Current detection info
        current_label = detection.get('type', 'unknown')
        confidence = detection.get('confidence', 0.0)
        
        tk.Label(dialog, text=f"Current Detection:", font=("Arial", 12, "bold")).pack(pady=5)
        tk.Label(dialog, text=f"Label: {current_label}").pack()
        tk.Label(dialog, text=f"Confidence: {confidence:.1%}").pack()
        
        tk.Label(dialog, text="\nWhat should this actually be?", font=("Arial", 10, "bold")).pack(pady=(20,5))
        
        # Re-label options
        relabel_frame = ttk.Frame(dialog)
        relabel_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Available labels
        # Updated to match competitive trainer object classes
        available_labels = [
            "building", "dialogue_box", "hp_bar", "menu_box", "npc_character", 
            "player_character", "pokemon_sprite", "text_area", "tree", 
            "pokeball/item", "grass", "water", "exp._bar", "level_indicator",
            "other"
        ]
        
        label_var = tk.StringVar(value="menu_box" if current_label == "text_area" else current_label)
        
        # Create dropdown for new label
        tk.Label(relabel_frame, text="Correct label:").pack(anchor=tk.W)
        label_dropdown = ttk.Combobox(relabel_frame, textvariable=label_var, values=available_labels, width=25)
        label_dropdown.pack(fill=tk.X, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        
        def relabel_detection():
            new_label = label_var.get().strip()
            if new_label and new_label != current_label:
                # Add negative feedback for old label
                self.feedback_system.add_negative_feedback(detection, f"mislabeled_as_{current_label}")
                
                # Update detection with new label
                detection['type'] = new_label
                detection['relabeled'] = True
                detection['original_type'] = current_label
                
                # Add positive feedback for new label
                self.feedback_system.add_positive_feedback(detection, "user_relabeled")
                
                print(f"[RELABEL] {current_label} -> {new_label}")
                
                # Update display
                self.update_display()
                self.update_stats_display()
                
            dialog.destroy()
            
        def remove_detection():
            # Just mark as wrong and remove
            self.feedback_system.add_negative_feedback(detection, "false_positive")
            self.remove_detection(detection)
            self.update_display()
            self.update_stats_display()
            dialog.destroy()
            
        def cancel_action():
            dialog.destroy()
        
        ttk.Button(button_frame, text="Re-label", command=relabel_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Remove (Wrong)", command=remove_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel_action).pack(side=tk.LEFT, padx=5)
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
            
    def get_detection_at_point(self, x, y):
        """Find detection at canvas coordinates with smart overlapping handling"""
        if not hasattr(self, 'current_screenshot_display') or not self.ai_detections:
            return None
            
        if hasattr(self, 'display_scale'):
            img_x = int(x / self.display_scale)
            img_y = int(y / self.display_scale)
            
            # Find all overlapping detections
            overlapping_detections = []
            for detection in self.ai_detections:
                bbox = detection['bbox']
                if (bbox[0] <= img_x <= bbox[2] and 
                    bbox[1] <= img_y <= bbox[3]):
                    overlapping_detections.append(detection)
            
            if not overlapping_detections:
                return None
                
            # If multiple detections, handle intelligently
            if len(overlapping_detections) == 1:
                return overlapping_detections[0]
            else:
                # Show selection dialog for overlapping detections
                return self.handle_overlapping_detections(overlapping_detections, x, y)
                    
        return None
        
    def handle_overlapping_detections(self, detections, click_x, click_y):
        """Handle multiple overlapping detections at click point"""
        if len(detections) == 1:
            return detections[0]
            
        # Create a quick selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Multiple Detections Found")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg='lightgray')
        
        selected_detection = None
        
        tk.Label(dialog, text="Multiple detections found:", font=("Arial", 10, "bold")).pack(pady=5)
        tk.Label(dialog, text="Which one do you want to select?").pack(pady=5)
        
        # List detections with details
        listbox_frame = ttk.Frame(dialog)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        listbox = tk.Listbox(listbox_frame, height=6)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        for i, detection in enumerate(detections):
            label = detection.get('type', 'unknown')
            confidence = detection.get('confidence', 0.0)
            bbox = detection['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            listbox.insert(tk.END, f"{i+1}. {label} ({confidence:.1%}) - Area: {area}px")
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        def select_detection():
            nonlocal selected_detection
            selection = listbox.curselection()
            if selection:
                selected_detection = detections[selection[0]]
            dialog.destroy()
            
        def cancel_selection():
            dialog.destroy()
        
        select_btn = ttk.Button(button_frame, text="SELECT THIS DETECTION", command=select_detection)
        select_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=cancel_selection)
        cancel_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Auto-select first item
        listbox.selection_set(0)
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Wait for dialog to close
        self.root.wait_window(dialog)
        
        return selected_detection
        
    def remove_detection(self, detection_to_remove):
        """Remove a detection from the current list"""
        self.ai_detections = [d for d in self.ai_detections if d != detection_to_remove]
        self.update_detection_list()
        
    def confirm_selected_detections(self):
        """Confirm all selected detections as correct"""
        selection = self.detection_listbox.curselection()
        for idx in selection:
            if idx < len(self.ai_detections):
                detection = self.ai_detections[idx]
                self.feedback_system.add_positive_feedback(detection)
                detection['user_confirmed'] = True
                
        self.update_display()
        self.update_stats_display()
        
    def mark_selected_wrong(self):
        """Mark selected detections as wrong"""
        selection = self.detection_listbox.curselection()
        to_remove = []
        
        for idx in reversed(list(selection)):
            if idx < len(self.ai_detections):
                detection = self.ai_detections[idx]
                self.feedback_system.add_negative_feedback(detection)
                to_remove.append(detection)
                
        for detection in to_remove:
            self.remove_detection(detection)
            
        self.update_display()
        self.update_stats_display()
        
    def update_stats_display(self):
        """Update the feedback statistics display"""
        stats = f"""LEARNING STATS:
Positive feedback: {len(self.feedback_system.true_positives)}
Negative feedback: {len(self.feedback_system.false_positives)}
Total feedback: {len(self.feedback_system.feedback_db)}

ONLINE LEARNING:
"""
        
        # Add online learning stats if available
        if hasattr(self, 'online_learning') and self.online_learning:
            online_stats = self.online_learning.get_learning_stats()
            stats += f"Real-time updates: {online_stats.get('model_updates', 0)}\n"
            stats += f"Confidence boosts: {online_stats.get('confidence_improvements', 0)}\n"
            stats += f"Queue size: {online_stats.get('positive_queue_size', 0)} + {online_stats.get('negative_queue_size', 0)}\n"
            stats += f"Next update in: {max(0, 30 - int(online_stats.get('time_since_last_update', 0)))}s\n"
        
        stats += "\nLEARNING PRIORITIES:\n"
        
        priorities = self.feedback_system.get_learning_priorities()
        for i, priority in enumerate(priorities[:3], 1):
            stats += f"{i}. {priority}\n"
            
        if not priorities:
            stats += "(No specific priorities yet)"
            
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
        
    def export_feedback_data_async(self):
        """Export feedback data in background to prevent UI blocking"""
        # Show immediate feedback
        self.status_bar.config(text="🔄 Exporting feedback data in background...")
        
        # Submit background task
        future = self.background_tasks.submit(self._export_feedback_worker)
        
        # Monitor completion
        def check_export():
            if future.done():
                try:
                    result = future.result()
                    self._on_export_success(result)
                except Exception as e:
                    self._on_export_error(e)
            else:
                # Update progress animation
                current_text = self.status_bar.cget("text")
                if "Exporting" in current_text:
                    dots = len([c for c in current_text if c == '.'])
                    new_dots = '.' * ((dots % 3) + 1)
                    self.status_bar.config(text=f"🔄 Exporting feedback data{new_dots}")
                self.root.after(500, check_export)  # Check again in 500ms
        
        self.root.after(100, check_export)
    
    def _export_feedback_worker(self):
        """Background worker for feedback export"""
        print("[BACKGROUND] Starting feedback export...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feedback_file = self.data_dir / "feedback" / f"feedback_session_{timestamp}.json"
        
        # This is the potentially slow operation
        feedback_data = self.feedback_system.export_feedback_for_training(str(feedback_file))
        
        return {
            'file_path': feedback_file,
            'feedback_data': feedback_data,
            'timestamp': timestamp
        }
    
    def _on_export_success(self, result):
        """Handle successful export in UI thread"""
        feedback_data = result['feedback_data']
        feedback_file = result['file_path']
        
        self.status_bar.config(text=f"✅ Export complete: {feedback_file.name}")
        
        messagebox.showinfo("Feedback Exported", 
                           f"Feedback data exported to:\n{feedback_file}\n\n"
                           f"Total feedback events: {len(feedback_data['feedback_sessions'])}\n"
                           f"True positives: {len(feedback_data['true_positives'])}\n"
                           f"False positives: {len(feedback_data['false_positives'])}")
        
        print(f"[BACKGROUND] Export completed: {feedback_file}")
    
    def _on_export_error(self, error):
        """Handle export errors in UI thread"""
        error_msg = f"Export failed: {str(error)}"
        self.status_bar.config(text=f"❌ {error_msg}")
        messagebox.showerror("Export Error", error_msg)
        print(f"[BACKGROUND] {error_msg}")

    def export_feedback_data(self):
        """Export feedback data for model improvement (DEPRECATED - use export_feedback_data_async)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feedback_file = self.data_dir / "feedback" / f"feedback_session_{timestamp}.json"
        
        feedback_data = self.feedback_system.export_feedback_for_training(str(feedback_file))
        
        messagebox.showinfo("Feedback Exported", 
                           f"Feedback data exported to:\n{feedback_file}\n\n"
                           f"Total feedback events: {len(feedback_data['feedback_sessions'])}\n"
                           f"True positives: {len(feedback_data['true_positives'])}\n"
                           f"False positives: {len(feedback_data['false_positives'])}")
        
    def show_online_learning_status(self):
        """Show detailed online learning status"""
        if not hasattr(self, 'online_learning') or not self.online_learning:
            messagebox.showinfo("Online Learning", "Online learning system not available")
            return
            
        stats = self.online_learning.get_learning_stats()
        
        status_text = f"""ONLINE LEARNING STATUS:

Real-time Updates Applied: {stats.get('model_updates', 0)}
Confidence Improvements: {stats.get('confidence_improvements', 0)}

FEEDBACK QUEUES:
Positive feedback pending: {stats.get('positive_queue_size', 0)}
Negative feedback pending: {stats.get('negative_queue_size', 0)}

LEARNING BUFFERS:
Positive samples: {stats.get('positive_buffer_size', 0)}/100
Negative samples: {stats.get('negative_buffer_size', 0)}/100

TIMING:
Time since last update: {int(stats.get('time_since_last_update', 0))}s
Next automatic update: {max(0, 30 - int(stats.get('time_since_last_update', 0)))}s

TOTALS:
Total positive feedback: {stats.get('positive_feedback_count', 0)}
Total negative feedback: {stats.get('negative_feedback_count', 0)}

STATUS: {'ACTIVE' if stats.get('model_updates', 0) >= 0 else 'INACTIVE'}"""

        messagebox.showinfo("Online Learning Status", status_text)
        
    def capture_screenshot(self):
        """Capture screenshot with AI assistance and feedback"""
        self.status_bar.config(text="Capturing screenshot...")
        
        # Capture from emulator if available, otherwise full screen
        if self.emulator_window:
            screenshot_np = self.capture_from_emulator()
            if screenshot_np is None:
                return
        else:
            screenshot = ImageGrab.grab()
            screenshot_np = np.array(screenshot)
        
        # Run AI detection
        detections = self.detect_objects(screenshot_np)
        
        self.current_screenshot = screenshot_np
        self.ai_detections = detections
        
        self.update_display()
        self.update_detection_list()
        self.update_stats_display()
        
        self.status_bar.config(text=f"Screenshot captured - {len(detections)} objects detected")
        
    def update_display(self):
        """Update image display with detections and feedback indicators"""
        if self.current_screenshot is None:
            return
            
        # Convert to PIL Image
        display_image = Image.fromarray(self.current_screenshot)
        
        # Calculate display size
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Scale image to fit canvas
            img_width, img_height = display_image.size
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            self.display_scale = min(scale_x, scale_y)
            
            new_width = int(img_width * self.display_scale)
            new_height = int(img_height * self.display_scale)
            
            display_image = display_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Draw detections on image
            display_image = self.draw_detections_with_feedback(display_image)
            
            # Update canvas
            self.current_screenshot_display = ImageTk.PhotoImage(display_image)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.current_screenshot_display)
            
    def draw_detections_with_feedback(self, image):
        """Draw detections with color coding for feedback status"""
        draw = ImageDraw.Draw(image)
        
        for detection in self.ai_detections:
            bbox = detection['bbox']
            scaled_bbox = [
                int(bbox[0] * self.display_scale),
                int(bbox[1] * self.display_scale),
                int(bbox[2] * self.display_scale),
                int(bbox[3] * self.display_scale)
            ]
            
            # Color coding based on status
            if detection.get('manual'):
                color = 'yellow'  # Manual user-drawn detection
                width = 4
            elif detection.get('user_confirmed'):
                color = 'green'  # Confirmed correct
                width = 3
            elif detection.get('relabeled'):
                color = 'purple' # Re-labeled detection
                width = 3
            elif detection.get('auto_accepted'):
                color = 'blue'   # Auto-accepted high confidence
                width = 2
            elif detection.get('needs_confirmation'):
                color = 'orange' # Needs user confirmation
                width = 2
            else:
                color = 'red'    # Default/uncertain
                width = 2
                
            # Draw bounding box
            draw.rectangle(scaled_bbox, outline=color, width=width)
            
            # Draw label with confidence
            label = f"{detection['type']} ({detection.get('confidence', 0):.2f})"
            draw.text((scaled_bbox[0], scaled_bbox[1]-15), label, fill=color)
            
        # Draw focus areas (temporary highlights)
        current_time = time.time()
        for focus_area in self.focus_areas:
            if current_time < focus_area['expires_at']:
                bbox = focus_area['bbox']
                scaled_bbox = [
                    int(bbox[0] * self.display_scale),
                    int(bbox[1] * self.display_scale),
                    int(bbox[2] * self.display_scale),
                    int(bbox[3] * self.display_scale)
                ]
                
                # Draw cyan dashed border for focus areas
                # PIL doesn't support dashed lines, so draw multiple small rectangles for dashed effect
                for i in range(0, scaled_bbox[2] - scaled_bbox[0], 10):
                    if i % 20 < 10:  # Create dash pattern
                        x1 = scaled_bbox[0] + i
                        x2 = min(scaled_bbox[0] + i + 8, scaled_bbox[2])
                        # Top line
                        draw.rectangle([x1, scaled_bbox[1], x2, scaled_bbox[1]+2], fill='cyan')
                        # Bottom line  
                        draw.rectangle([x1, scaled_bbox[3]-2, x2, scaled_bbox[3]], fill='cyan')
                
                for i in range(0, scaled_bbox[3] - scaled_bbox[1], 10):
                    if i % 20 < 10:  # Create dash pattern
                        y1 = scaled_bbox[1] + i
                        y2 = min(scaled_bbox[1] + i + 8, scaled_bbox[3])
                        # Left line
                        draw.rectangle([scaled_bbox[0], y1, scaled_bbox[0]+2, y2], fill='cyan')
                        # Right line
                        draw.rectangle([scaled_bbox[2]-2, y1, scaled_bbox[2], y2], fill='cyan')
                
                # Add focus area label
                remaining = int(focus_area['expires_at'] - current_time)
                focus_label = f"FOCUS ({remaining}s)"
                draw.text((scaled_bbox[0], scaled_bbox[3]+5), focus_label, fill='cyan')
            
        return image
        
    def update_detection_list(self):
        """Update the detection listbox"""
        self.detection_listbox.delete(0, tk.END)
        
        for i, detection in enumerate(self.ai_detections):
            confidence = detection.get('confidence', 0)
            obj_type = detection['type']
            status = ""
            
            if detection.get('manual'):
                status = " [MANUAL]"
            elif detection.get('relabeled'):
                status = " [RELABELED]"
            elif detection.get('user_confirmed'):
                status = " [CONFIRMED]"
            elif detection.get('auto_accepted'):
                status = " [AUTO]"
            elif detection.get('needs_confirmation'):
                status = " [?]"
                
            display_text = f"{obj_type} {confidence:.2f}{status}"
            self.detection_listbox.insert(tk.END, display_text)
            
    def find_emulator_window(self):
        """Find emulator window with dropdown selection"""
        windows = []
        
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)
                
                # Skip our own window
                if "Backseat Driving AI Trainer" in window_text:
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
            self.status_bar.config(text="Emulator window selected! Enable live mode or take screenshot.")
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
            choice_window.winfo_screenwidth() // 2 - 300,
            choice_window.winfo_screenheight() // 2 - 200
        ))
        
        ttk.Label(choice_window, text="Multiple windows found. Choose your emulator:", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # Listbox for window selection
        list_frame = ttk.Frame(choice_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        window_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=("Courier", 10))
        window_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=window_listbox.yview)
        
        # Populate listbox
        for i, (hwnd, title, class_name) in enumerate(windows):
            try:
                rect = win32gui.GetWindowRect(hwnd)
                width = rect[2] - rect[0]
                height = rect[3] - rect[1]
                display_text = f"{title} [{width}x{height}] ({class_name})"
            except:
                display_text = f"{title} ({class_name})"
            window_listbox.insert(tk.END, display_text)
        
        # Buttons
        button_frame = ttk.Frame(choice_window)
        button_frame.pack(pady=10)
        
        def select_window():
            selection = window_listbox.curselection()
            if selection:
                self.emulator_window = windows[selection[0]][0]
                self.emulator_var.set(f"Selected: {windows[selection[0]][1]}")
                self.status_bar.config(text="Emulator window selected! Enable live mode or take screenshot.")
                choice_window.destroy()
            else:
                messagebox.showwarning("No Selection", "Please select a window first.")
                
        def cancel_selection():
            choice_window.destroy()
            
        ttk.Button(button_frame, text="Select This Window", 
                  command=select_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=cancel_selection).pack(side=tk.LEFT, padx=5)
                  
        # Help text
        help_text = tk.Text(choice_window, height=3, wrap=tk.WORD, font=("Arial", 9))
        help_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        help_text.insert(tk.END, 
            "Tips: Look for your emulator name (mGBA, VisualBoyAdvance, etc.). "
            "Game window size is usually 240x160 or multiples (480x320, 720x480).")
        help_text.config(state=tk.DISABLED)
        
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
            print(f"Direct capture error: {e}")
            return None
            
    def capture_from_emulator(self):
        """Legacy method name for compatibility"""
        result = self.capture_emulator_screenshot()
        if result is not None:
            # Convert from BGR to RGB for PIL compatibility
            return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return None
            
    def toggle_pipeline_mode(self):
        """Toggle high-performance pipeline mode"""
        self.pipeline_mode = self.pipeline_var.get()
        
        if self.pipeline_mode:
            # Disable regular live mode when using pipeline
            if self.live_var.get():
                self.live_var.set(False)
                self.toggle_live_mode()
            
            self.start_pipeline_mode()
        else:
            self.stop_pipeline_mode()
    
    def start_pipeline_mode(self):
        """Start high-performance screenshot/AI pipeline"""
        print("[PIPELINE] Starting high-performance pipeline...")
        
        with self._thread_lock:
            if ((self.screenshot_thread and self.screenshot_thread.is_alive()) or
                (self.ai_processing_thread and self.ai_processing_thread.is_alive())):
                return
                
            # Clear stop signal
            self._stop_event.clear()
            
            # Clear queues
            while not self.screenshot_queue.empty():
                try:
                    self.screenshot_queue.get_nowait()
                except queue.Empty:
                    break
            while not self.ai_results_queue.empty():
                try:
                    self.ai_results_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Start pipeline threads
            self.screenshot_thread = threading.Thread(
                target=self.screenshot_capture_loop,
                daemon=True,
                name="ScreenshotCapture"
            )
            self.ai_processing_thread = threading.Thread(
                target=self.ai_processing_loop,
                daemon=True, 
                name="AIProcessing"
            )
            
            self.screenshot_thread.start()
            self.ai_processing_thread.start()
        
        self.status_bar.config(text="🚀 High-performance pipeline active (10+ FPS)")
        
        # Start UI update loop
        self.root.after(100, self.update_pipeline_display)
    
    def stop_pipeline_mode(self):
        """Stop high-performance pipeline"""
        print("[PIPELINE] Stopping pipeline...")
        
        with self._thread_lock:
            self.pipeline_mode = False
            self._stop_event.set()
        
        self.status_bar.config(text="Pipeline stopped")
    
    def screenshot_capture_loop(self):
        """High-frequency screenshot capture thread (10-15 FPS)"""
        print("[PIPELINE] Screenshot capture thread started")
        
        while not self._stop_event.is_set() and self.pipeline_mode:
            try:
                # Capture screenshot
                screenshot = self.capture_emulator_screenshot()
                
                if screenshot is not None:
                    timestamp = time.time()
                    
                    # Add to queue (non-blocking)
                    try:
                        # Convert BGR to RGB for consistency
                        screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
                        self.screenshot_queue.put_nowait({
                            'image': screenshot_rgb,
                            'timestamp': timestamp
                        })
                        
                        # Update performance stats
                        self.performance_stats['screenshots_captured'] += 1
                        self.performance_stats['last_screenshot_time'] = timestamp
                        
                        # Reset error recovery on successful operations
                        if self.error_recovery['recovery_attempts'] > 0 and self.performance_stats['screenshots_captured'] % 10 == 0:
                            print("[RECOVERY] Screenshot capture successful, resetting error counter")
                            self.error_recovery['recovery_attempts'] = 0
                        
                    except queue.Full:
                        # Queue full, skip this frame (prevents memory buildup)
                        pass
                
                # High frequency: 100ms = 10 FPS
                time.sleep(0.1)
                
            except Exception as e:
                # Use enhanced error handling
                backoff_time = self.handle_error_with_recovery(e, "Screenshot Capture", "screenshot")
                if backoff_time == float('inf'):
                    break  # Stop thread if recovery says to stop
                time.sleep(backoff_time)
        
        print("[PIPELINE] Screenshot capture thread ended")
    
    def ai_processing_loop(self):
        """AI processing thread (3-5 FPS, processes best frames)"""
        print("[PIPELINE] AI processing thread started")
        
        while not self._stop_event.is_set() and self.pipeline_mode:
            try:
                # Get screenshot from queue (blocking with timeout)
                try:
                    screenshot_data = self.screenshot_queue.get(timeout=0.5)
                    screenshot = screenshot_data['image']
                    timestamp = screenshot_data['timestamp']
                    
                    # Process with AI
                    detections = self.detect_objects(screenshot)
                    
                    # Add to results queue
                    try:
                        self.ai_results_queue.put_nowait({
                            'image': screenshot,
                            'detections': detections,
                            'timestamp': timestamp
                        })
                        
                        # Update performance stats
                        self.performance_stats['ai_detections_processed'] += 1
                        self.performance_stats['last_ai_time'] = time.time()
                        
                        # Reset error recovery on successful operations
                        if self.error_recovery['recovery_attempts'] > 0:
                            print("[RECOVERY] AI processing successful, resetting error counter")
                            self.error_recovery['recovery_attempts'] = 0
                        
                    except queue.Full:
                        # Results queue full, drop oldest result
                        try:
                            self.ai_results_queue.get_nowait()
                            self.ai_results_queue.put_nowait({
                                'image': screenshot,
                                'detections': detections,
                                'timestamp': timestamp
                            })
                        except queue.Empty:
                            pass
                    
                except queue.Empty:
                    # No screenshots available, continue waiting
                    continue
                
            except Exception as e:
                # Use enhanced error handling
                backoff_time = self.handle_error_with_recovery(e, "AI Processing", "ai") 
                if backoff_time == float('inf'):
                    break  # Stop thread if recovery says to stop
                time.sleep(backoff_time)
        
        print("[PIPELINE] AI processing thread ended")
    
    def update_pipeline_display(self):
        """Update UI with pipeline results"""
        if not self.pipeline_mode:
            return
            
        try:
            # Get latest AI results (non-blocking)
            result = self.ai_results_queue.get_nowait()
            
            # Update display
            self.current_screenshot = result['image']
            self.ai_detections = result['detections']
            
            self.update_display()
            self.update_detection_list()
            
            # Calculate and display FPS
            current_time = time.time()
            if self.performance_stats['last_screenshot_time'] > 0:
                screenshot_fps = 1.0 / max(0.1, current_time - self.performance_stats['last_screenshot_time'])
                self.performance_stats['screenshot_fps'] = screenshot_fps
            
            if self.performance_stats['last_ai_time'] > 0:
                ai_fps = 1.0 / max(0.3, current_time - self.performance_stats['last_ai_time'])
                self.performance_stats['ai_fps'] = ai_fps
            
            # Update status with FPS info
            status_text = (f"🚀 Pipeline: {self.performance_stats['screenshot_fps']:.1f} cap FPS, "
                         f"{self.performance_stats['ai_fps']:.1f} AI FPS, {len(result['detections'])} objects")
            self.status_bar.config(text=status_text)
            
        except queue.Empty:
            # No new results, just update status
            pass
        
        # Schedule next update
        if self.pipeline_mode:
            self.root.after(50, self.update_pipeline_display)  # 20 FPS UI updates
    
    def start_memory_manager(self):
        """Start background memory management thread"""
        if self.memory_manager_thread and self.memory_manager_thread.is_alive():
            return
            
        self.memory_manager_thread = threading.Thread(
            target=self.memory_management_loop,
            daemon=True,
            name="MemoryManager"
        )
        self.memory_manager_thread.start()
        print("[MEMORY] Background memory manager started")
    
    def memory_management_loop(self):
        """Background memory management and cleanup"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        while not self._stop_event.is_set():
            try:
                # Get current memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                self.performance_stats['memory_usage_mb'] = memory_mb
                
                current_time = time.time()
                
                # Cleanup every 30 seconds or if memory exceeds 500MB
                should_cleanup = (
                    (current_time - self.performance_stats['last_cleanup_time'] > 30) or
                    (memory_mb > 500)
                )
                
                if should_cleanup:
                    print(f"[MEMORY] Running cleanup - Current usage: {memory_mb:.1f} MB")
                    
                    # Clear old queue items if queues are getting full
                    if hasattr(self, 'screenshot_queue') and self.screenshot_queue.qsize() > 20:
                        cleared = 0
                        while self.screenshot_queue.qsize() > 15:
                            try:
                                self.screenshot_queue.get_nowait()
                                cleared += 1
                            except queue.Empty:
                                break
                        if cleared > 0:
                            print(f"[MEMORY] Cleared {cleared} old screenshots from queue")
                    
                    if hasattr(self, 'ai_results_queue') and self.ai_results_queue.qsize() > 7:
                        cleared = 0
                        while self.ai_results_queue.qsize() > 5:
                            try:
                                self.ai_results_queue.get_nowait()
                                cleared += 1
                            except queue.Empty:
                                break
                        if cleared > 0:
                            print(f"[MEMORY] Cleared {cleared} old AI results from queue")
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Check memory after cleanup
                    new_memory_info = process.memory_info()
                    new_memory_mb = new_memory_info.rss / (1024 * 1024)
                    freed_mb = memory_mb - new_memory_mb
                    
                    if freed_mb > 1:  # Only log if significant memory was freed
                        print(f"[MEMORY] Cleanup freed {freed_mb:.1f} MB, now using {new_memory_mb:.1f} MB")
                    
                    self.performance_stats['last_cleanup_time'] = current_time
                    self.performance_stats['memory_usage_mb'] = new_memory_mb
                
                # Check every 10 seconds
                time.sleep(10)
                
            except Exception as e:
                print(f"[MEMORY] Memory management error: {e}")
                time.sleep(30)  # Wait longer if there's an error
        
        print("[MEMORY] Memory management thread ended")
    
    def handle_error_with_recovery(self, error, operation_name, error_type='general'):
        """Enhanced error handling with smart recovery"""
        current_time = time.time()
        
        # Track error statistics
        self.performance_stats['total_errors'] += 1
        if error_type == 'screenshot':
            self.performance_stats['screenshot_errors'] += 1
        elif error_type == 'ai':
            self.performance_stats['ai_errors'] += 1
        
        # Check if this is part of a consecutive error sequence
        time_since_last_error = current_time - self.error_recovery['last_error_time']
        
        if time_since_last_error < 10:  # Errors within 10 seconds are consecutive
            self.error_recovery['recovery_attempts'] += 1
        else:
            # Reset recovery attempts if errors are spaced out
            self.error_recovery['recovery_attempts'] = 1
        
        self.error_recovery['last_error_time'] = current_time
        
        # Determine recovery strategy
        consecutive_errors = self.error_recovery['recovery_attempts']
        error_msg = str(error)
        
        print(f"[ERROR] {operation_name}: {error_msg} (attempt {consecutive_errors})")
        
        # Progressive backoff based on error count
        if consecutive_errors <= 2:
            backoff_time = 0.5
            recovery_action = "quick retry"
        elif consecutive_errors <= 4:
            backoff_time = 2.0
            recovery_action = "medium backoff"
            
            # Try model reload for AI errors
            if error_type == 'ai' and consecutive_errors == 3:
                self.root.after(0, self._attempt_model_recovery)
                recovery_action = "model recovery + backoff"
                
        elif consecutive_errors <= self.error_recovery['max_consecutive_errors']:
            backoff_time = 5.0
            recovery_action = "long backoff"
            
            # Clear queues to prevent cascade failures
            if hasattr(self, 'screenshot_queue'):
                while not self.screenshot_queue.empty():
                    try:
                        self.screenshot_queue.get_nowait()
                    except queue.Empty:
                        break
            if hasattr(self, 'ai_results_queue'):
                while not self.ai_results_queue.empty():
                    try:
                        self.ai_results_queue.get_nowait()
                    except queue.Empty:
                        break
                        
        else:
            # Too many consecutive errors - escalate
            print(f"[ERROR] {operation_name}: Too many consecutive errors ({consecutive_errors}), escalating...")
            
            if self.pipeline_mode:
                # Disable pipeline mode and fallback to regular mode
                print("[ERROR] Disabling high-performance pipeline due to errors")
                self.root.after(0, self._disable_pipeline_mode)
                return 10.0  # Wait longer before retry
            else:
                # Disable live mode entirely
                print("[ERROR] Disabling live mode due to persistent errors")
                self.root.after(0, self._disable_live_mode)
                return float('inf')  # Don't retry
        
        print(f"[RECOVERY] Strategy: {recovery_action}, backoff: {backoff_time}s")
        
        # Update status bar with error info
        error_summary = f"⚠️ {operation_name} error #{consecutive_errors} - recovering..."
        self.root.after(0, lambda: self.status_bar.config(text=error_summary, foreground="orange"))
        
        return backoff_time
    
    def _attempt_model_recovery(self):
        """Attempt to recover from AI model errors"""
        print("[RECOVERY] Attempting AI model recovery...")
        try:
            if hasattr(self, 'current_model_wrapper') and self.current_model_wrapper:
                # Try to reload the current model
                current_path = self.current_model_path
                if current_path:
                    print("[RECOVERY] Reloading current model...")
                    self.load_selected_model_async()
                    self.status_bar.config(text="🔄 Attempting model recovery...", foreground="blue")
        except Exception as e:
            print(f"[RECOVERY] Model recovery failed: {e}")
    
    def _disable_pipeline_mode(self):
        """Fallback from pipeline mode to regular live mode"""
        if self.pipeline_var.get():
            self.pipeline_var.set(False)
            self.toggle_pipeline_mode()
            
        # Enable regular live mode as fallback
        if not self.live_var.get():
            self.live_var.set(True)
            self.toggle_live_mode()
    
    def _disable_live_mode(self):
        """Disable all live modes due to persistent errors"""
        if self.pipeline_var.get():
            self.pipeline_var.set(False)
        if self.live_var.get():
            self.live_var.set(False)
            
        self.toggle_live_mode()
        self.status_bar.config(text="❌ Live mode disabled due to persistent errors", foreground="red")
    
    def update_performance_display(self):
        """Update performance statistics display"""
        try:
            stats = self.performance_stats
            
            # Build performance summary
            if self.pipeline_mode:
                # Pipeline mode stats
                perf_text = (
                    f"📊 Cap: {stats['screenshot_fps']:.1f} FPS | "
                    f"AI: {stats['ai_fps']:.1f} FPS | "
                    f"Mem: {stats['memory_usage_mb']:.0f} MB\n"
                    f"Screenshots: {stats['screenshots_captured']} | "
                    f"AI Processed: {stats['ai_detections_processed']} | "
                    f"Errors: {stats['total_errors']}"
                )
            elif self.live_mode:
                # Regular live mode stats
                combined_fps = 1.0 / max(0.3, time.time() - stats['last_ai_time']) if stats['last_ai_time'] > 0 else 0.0
                perf_text = (
                    f"📊 Live: {combined_fps:.1f} FPS | "
                    f"Mem: {stats['memory_usage_mb']:.0f} MB\n"
                    f"AI Processed: {stats['ai_detections_processed']} | "
                    f"Errors: {stats['total_errors']}"
                )
            else:
                # Idle stats
                perf_text = (
                    f"📊 Idle | "
                    f"Mem: {stats['memory_usage_mb']:.0f} MB | "
                    f"Total Errors: {stats['total_errors']}"
                )
            
            self.perf_stats_label.config(text=perf_text)
            
            # Color coding based on performance
            if stats['total_errors'] > 10:
                color = "red"
            elif stats['total_errors'] > 5:
                color = "orange"
            elif (self.pipeline_mode and stats['screenshot_fps'] < 5) or (self.live_mode and combined_fps < 1):
                color = "orange"
            else:
                color = "black"
                
            self.perf_stats_label.config(foreground=color)
            
        except Exception as e:
            print(f"[PERF] Error updating performance display: {e}")
            self.perf_stats_label.config(text="Performance stats unavailable")
        
        # Update every 2 seconds
        self.root.after(2000, self.update_performance_display)
    
    def toggle_multi_window_mode(self):
        """Toggle multi-window monitoring mode"""
        self.multi_window_mode = self.multi_window_var.get()
        
        if self.multi_window_mode:
            # Disable other modes when using multi-window
            if self.pipeline_var.get():
                self.pipeline_var.set(False)
                self.toggle_pipeline_mode()
            if self.live_var.get():
                self.live_var.set(False)
                self.toggle_live_mode()
                
            self.start_multi_window_monitoring()
        else:
            self.stop_multi_window_monitoring()
    
    def add_window_dialog(self):
        """Dialog to add a window for monitoring"""
        # Get list of available windows
        windows = []
        def enum_window_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if window_title and len(window_title) > 3:  # Filter out empty/short titles
                    windows.append((hwnd, window_title))
            return True
        
        win32gui.EnumWindows(enum_window_callback, windows)
        
        if not windows:
            messagebox.showwarning("No Windows", "No suitable windows found for monitoring")
            return
        
        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Window for Monitoring")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select window to monitor:", font=("Arial", 10, "bold")).pack(pady=10)
        
        # Window list
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        window_list = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        window_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=window_list.yview)
        
        # Populate window list
        for hwnd, title in windows:
            display_text = f"{title} (ID: {hwnd})"
            window_list.insert(tk.END, display_text)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def add_selected():
            selection = window_list.curselection()
            if selection:
                idx = selection[0]
                hwnd, title = windows[idx]
                self.add_window_to_monitoring(hwnd, title)
                dialog.destroy()
        
        ttk.Button(button_frame, text="Add Window", command=add_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def add_window_to_monitoring(self, hwnd, title):
        """Add a window to the monitoring list"""
        if title in self.monitored_windows:
            messagebox.showinfo("Already Monitored", f"Window '{title}' is already being monitored")
            return
        
        # Create monitoring infrastructure for this window
        window_queue = queue.Queue(maxsize=20)
        
        window_info = {
            'hwnd': hwnd,
            'title': title,
            'queue': window_queue,
            'thread': None,
            'active': False
        }
        
        self.monitored_windows[title] = window_info
        
        print(f"[MULTI-WINDOW] Added window for monitoring: {title}")
        self.status_bar.config(text=f"Added window: {title}")
        
        # Start monitoring if multi-window mode is active
        if self.multi_window_mode:
            self.start_window_monitoring(title)
    
    def start_window_monitoring(self, window_title):
        """Start monitoring a specific window"""
        if window_title not in self.monitored_windows:
            return
            
        window_info = self.monitored_windows[window_title]
        
        if window_info['thread'] and window_info['thread'].is_alive():
            return  # Already running
        
        # Start monitoring thread for this window
        window_info['thread'] = threading.Thread(
            target=self.window_monitoring_loop,
            args=(window_title,),
            daemon=True,
            name=f"WindowMonitor-{window_title[:10]}"
        )
        window_info['active'] = True
        window_info['thread'].start()
        
        print(f"[MULTI-WINDOW] Started monitoring: {window_title}")
    
    def window_monitoring_loop(self, window_title):
        """Monitoring loop for a specific window"""
        window_info = self.monitored_windows.get(window_title)
        if not window_info:
            return
            
        hwnd = window_info['hwnd']
        window_queue = window_info['queue']
        
        print(f"[MULTI-WINDOW] Monitoring thread started for: {window_title}")
        
        while not self._stop_event.is_set() and self.multi_window_mode and window_info['active']:
            try:
                # Capture screenshot from specific window
                screenshot = self.capture_window_screenshot(hwnd)
                
                if screenshot is not None:
                    timestamp = time.time()
                    
                    # Process with AI (reuse existing detect_objects method)
                    detections = self.detect_objects(screenshot)
                    
                    # Add to combined results queue
                    try:
                        self.window_results_queue.put_nowait({
                            'window_title': window_title,
                            'image': screenshot,
                            'detections': detections,
                            'timestamp': timestamp
                        })
                    except queue.Full:
                        # Drop oldest result if queue is full
                        try:
                            self.window_results_queue.get_nowait()
                            self.window_results_queue.put_nowait({
                                'window_title': window_title,
                                'image': screenshot,
                                'detections': detections,
                                'timestamp': timestamp
                            })
                        except queue.Empty:
                            pass
                
                # Monitor at 2 FPS per window to avoid overwhelming the system
                time.sleep(0.5)
                
            except Exception as e:
                backoff_time = self.handle_error_with_recovery(e, f"Window Monitoring ({window_title})", "screenshot")
                if backoff_time == float('inf'):
                    break
                time.sleep(backoff_time)
        
        print(f"[MULTI-WINDOW] Monitoring thread ended for: {window_title}")
    
    def capture_window_screenshot(self, hwnd):
        """Capture screenshot from a specific window"""
        try:
            # Get window rectangle
            rect = win32gui.GetWindowRect(hwnd)
            x, y, x2, y2 = rect
            width = x2 - x
            height = y2 - y
            
            if width <= 0 or height <= 0:
                return None
            
            # Capture window
            wDC = win32gui.GetWindowDC(hwnd)
            dcObj = win32ui.CreateDCFromHandle(wDC)
            cDC = dcObj.CreateCompatibleDC()
            dataBitMap = win32ui.CreateBitmap()
            dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
            cDC.SelectObject(dataBitMap)
            
            # Copy window content
            cDC.BitBlt((0, 0), (width, height), dcObj, (0, 0), win32con.SRCCOPY)
            
            # Convert to numpy array
            signedIntsArray = dataBitMap.GetBitmapBits(True)
            img = np.frombuffer(signedIntsArray, dtype=np.uint8)
            img.shape = (height, width, 4)
            
            # Cleanup
            dcObj.DeleteDC()
            cDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, wDC)
            win32gui.DeleteObject(dataBitMap.GetHandle())
            
            # Convert BGRA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            return img
            
        except Exception as e:
            print(f"[MULTI-WINDOW] Screenshot capture error for window {hwnd}: {e}")
            return None
    
    def start_multi_window_monitoring(self):
        """Start monitoring all configured windows"""
        if not self.monitored_windows:
            messagebox.showinfo("No Windows", "Please add windows to monitor first using 'Add Window'")
            self.multi_window_var.set(False)
            return
        
        print("[MULTI-WINDOW] Starting multi-window monitoring...")
        
        # Clear stop event
        self._stop_event.clear()
        
        # Start monitoring each window
        for window_title in self.monitored_windows:
            self.start_window_monitoring(window_title)
        
        # Start combined results processor
        self.root.after(100, self.update_multi_window_display)
        
        window_count = len(self.monitored_windows)
        self.status_bar.config(text=f"🖥️ Multi-window monitoring: {window_count} windows active")
    
    def stop_multi_window_monitoring(self):
        """Stop monitoring all windows"""
        print("[MULTI-WINDOW] Stopping multi-window monitoring...")
        
        # Mark all windows as inactive
        for window_info in self.monitored_windows.values():
            window_info['active'] = False
        
        self.status_bar.config(text="Multi-window monitoring stopped")
    
    def update_multi_window_display(self):
        """Update UI with multi-window results"""
        if not self.multi_window_mode:
            return
        
        try:
            # Get latest results from any window
            result = self.window_results_queue.get_nowait()
            
            # Update display with the latest result
            self.current_screenshot = result['image']
            self.ai_detections = result['detections']
            
            self.update_display()
            self.update_detection_list()
            
            # Update status with multi-window info
            window_title = result['window_title']
            detection_count = len(result['detections'])
            active_windows = sum(1 for w in self.monitored_windows.values() if w['active'])
            
            status_text = f"🖥️ Multi-window: {active_windows} active, latest: {window_title} ({detection_count} objects)"
            self.status_bar.config(text=status_text)
            
        except queue.Empty:
            # No new results
            pass
        
        # Schedule next update
        if self.multi_window_mode:
            self.root.after(100, self.update_multi_window_display)
    
    def remove_window_dialog(self):
        """Dialog to remove a monitored window"""
        if not self.monitored_windows:
            messagebox.showinfo("No Windows", "No windows are currently being monitored")
            return
        
        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Remove Monitored Window")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select window to remove:", font=("Arial", 10, "bold")).pack(pady=10)
        
        # Window list
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        
        window_list = tk.Listbox(list_frame)
        window_list.pack(fill=tk.BOTH, expand=True)
        
        # Populate with monitored windows
        for title in self.monitored_windows:
            window_list.insert(tk.END, title)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def remove_selected():
            selection = window_list.curselection()
            if selection:
                idx = selection[0]
                title = list(self.monitored_windows.keys())[idx]
                self.remove_window_from_monitoring(title)
                dialog.destroy()
        
        ttk.Button(button_frame, text="Remove Window", command=remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def remove_window_from_monitoring(self, window_title):
        """Remove a window from monitoring"""
        if window_title not in self.monitored_windows:
            return
        
        # Stop the monitoring thread
        window_info = self.monitored_windows[window_title]
        window_info['active'] = False
        
        # Remove from monitored windows
        del self.monitored_windows[window_title]
        
        print(f"[MULTI-WINDOW] Removed window from monitoring: {window_title}")
        self.status_bar.config(text=f"Removed window: {window_title}")
    
    def show_monitored_windows(self):
        """Show dialog with currently monitored windows"""
        if not self.monitored_windows:
            messagebox.showinfo("No Windows", "No windows are currently being monitored")
            return
        
        # Create info dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Monitored Windows")
        dialog.geometry("500x300")
        dialog.transient(self.root)
        
        ttk.Label(dialog, text="Currently Monitored Windows:", font=("Arial", 10, "bold")).pack(pady=10)
        
        # Create text area with scrollbar
        frame = ttk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_area = tk.Text(frame, yscrollcommand=scrollbar.set, wrap=tk.WORD)
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_area.yview)
        
        # Add window info
        for title, info in self.monitored_windows.items():
            status = "🟢 Active" if info['active'] else "🔴 Inactive"
            text_area.insert(tk.END, f"{status} {title}\n")
            text_area.insert(tk.END, f"  └─ ID: {info['hwnd']}\n\n")
        
        text_area.config(state=tk.DISABLED)
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def hot_swap_model(self):
        """Hot-swap the AI model without stopping capture"""
        if not self.hot_swap_enabled.get():
            # Fall back to regular model loading
            self.load_selected_model_async()
            return
        
        selected_name = self.model_var.get()
        
        if selected_name == "No models available" or selected_name == "Error loading models":
            messagebox.showwarning("Invalid Model", "Please select a valid model for hot-swapping")
            return
            
        if selected_name not in self.model_paths:
            messagebox.showwarning("Model Not Found", f"Model '{selected_name}' not found in available models")
            return
        
        model_path = self.model_paths[selected_name]
        
        # Don't hot-swap if it's the same model
        if self.current_model_path == model_path:
            messagebox.showinfo("Same Model", "Selected model is already loaded")
            return
        
        # Check if any capture mode is running
        is_capturing = (self.live_mode or self.pipeline_mode or self.multi_window_mode)
        
        if not is_capturing:
            # No capture running, use regular loading
            self.load_selected_model_async()
            return
        
        print(f"[HOT-SWAP] Initiating hot-swap to model: {selected_name}")
        
        # Add to hot-swap queue
        try:
            swap_request = {
                'model_name': selected_name,
                'model_path': model_path,
                'timestamp': time.time()
            }
            
            self.model_swap_queue.put_nowait(swap_request)
            
            # Update status
            self.status_bar.config(text=f"🔄 Hot-swapping to: {selected_name}...")
            
            # Start processing hot-swap requests if not already running
            if not hasattr(self, '_hotswap_processor_running') or not self._hotswap_processor_running:
                self._hotswap_processor_running = True
                self.root.after(100, self.process_model_swap_requests)
                
        except queue.Full:
            messagebox.showwarning("Swap Queue Full", "Too many pending model swap requests. Please wait.")
    
    def process_model_swap_requests(self):
        """Process model hot-swap requests in a thread-safe manner"""
        try:
            # Check if there's a swap request
            swap_request = self.model_swap_queue.get_nowait()
            
            print(f"[HOT-SWAP] Processing swap request: {swap_request['model_name']}")
            
            # Update UI to show swapping
            self.model_info_label.config(text="🔄 Hot-swapping model...", foreground="blue")
            
            # Submit hot-swap as background task
            future = self.background_tasks.submit(self._execute_hot_swap, swap_request)
            
            # Monitor completion
            def check_hotswap():
                if future.done():
                    try:
                        result = future.result()
                        self._on_hot_swap_success(result)
                    except Exception as e:
                        self._on_hot_swap_error(e, swap_request['model_name'])
                else:
                    # Update animation
                    current_text = self.model_info_label.cget("text")
                    if "Hot-swapping" in current_text:
                        dots = len([c for c in current_text if c == '.'])
                        new_dots = '.' * ((dots % 3) + 1)
                        self.model_info_label.config(text=f"🔄 Hot-swapping model{new_dots}")
                    self.root.after(300, check_hotswap)
            
            self.root.after(100, check_hotswap)
            
        except queue.Empty:
            # No swap requests, stop processing
            self._hotswap_processor_running = False
            return
        
        # Continue processing if there might be more requests
        self.root.after(500, self.process_model_swap_requests)
    
    def _execute_hot_swap(self, swap_request):
        """Execute the hot-swap in a background thread"""
        model_name = swap_request['model_name']
        model_path = swap_request['model_path']
        
        print(f"[HOT-SWAP] Executing background swap for: {model_name}")
        
        # Use the model swap lock to ensure thread safety
        with self.model_swap_lock:
            # Load the new model
            print(f"[HOT-SWAP] Loading new model: {model_name}")
            
            new_model_wrapper = AdaptiveModelWrapper(
                str(model_path), 
                target_classes=self.active_classes
            )
            
            if new_model_wrapper.model is None:
                raise Exception(f"Failed to load model: {model_name}")
            
            # Prepare swap data
            model_info = {
                'wrapper': new_model_wrapper,
                'model_path': model_path,
                'model_name': model_name,
                'native_classes': new_model_wrapper.get_native_classes(),
                'target_classes': new_model_wrapper.get_target_classes(),
                'mapped_count': sum(1 for k, v in new_model_wrapper.get_class_mapping().items() if k != v)
            }
            
            return model_info
    
    def _on_hot_swap_success(self, result):
        """Handle successful hot-swap completion"""
        print(f"[HOT-SWAP] Successfully loaded new model: {result['model_name']}")
        
        # Atomically swap the model (this is the critical section)
        with self.model_swap_lock:
            old_model = self.current_model_wrapper
            
            # Update to new model
            self.current_model_wrapper = result['wrapper']
            self.ai_model = result['wrapper'].model
            self.detection_method = "adaptive_yolo"
            self.current_model_path = result['model_path']
            
            # Clear old model from memory
            if old_model:
                try:
                    del old_model
                    import gc
                    gc.collect()
                except:
                    pass
        
        # Update UI
        info_text = f"✅ Hot-swapped: {result['model_name']}\n"
        info_text += f"Native: {len(result['native_classes'])} classes\n"
        info_text += f"Active: {len(result['target_classes'])} classes"
        
        if result['mapped_count'] > 0:
            info_text += f"\nMapped: {result['mapped_count']} classes"
            color = "orange"
        else:
            info_text += f"\nDirect compatibility"
            color = "green"
        
        self.model_info_label.config(text=info_text, foreground=color)
        self.status_bar.config(text=f"✅ Hot-swap complete: {result['model_name']} (capture continues)")
        
        print(f"[HOT-SWAP] Hot-swap complete. Capture operations continued without interruption.")
    
    def _on_hot_swap_error(self, error, model_name):
        """Handle hot-swap errors"""
        error_msg = f"Hot-swap failed for {model_name}: {str(error)}"
        print(f"[HOT-SWAP] {error_msg}")
        
        self.model_info_label.config(text=f"❌ Hot-swap failed: {model_name}", foreground="red")
        self.status_bar.config(text=f"❌ Hot-swap failed: {model_name} (using previous model)")
        
        # Show user-friendly error message
        messagebox.showerror("Hot-Swap Failed", 
                           f"Failed to hot-swap to model '{model_name}'.\n\n"
                           f"Error: {str(error)}\n\n"
                           f"The previous model is still active and capture continues.")
    
    def detect_objects(self, screenshot):
        """Thread-safe object detection that supports hot-swapping"""
        # Use model swap lock for thread safety during hot-swaps
        with self.model_swap_lock:
            # Call the original detect_objects method
            return self._detect_objects_internal(screenshot)
    
    def _detect_objects_internal(self, screenshot):
        """Internal object detection (original detect_objects logic)"""
        if self.current_model_wrapper and self.current_model_wrapper.model:
            try:
                # Detect with adaptive wrapper
                results = self.current_model_wrapper.predict(screenshot)
                detections = []
                
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes.cpu().numpy()
                        
                        for i in range(len(boxes.xyxy)):
                            bbox = boxes.xyxy[i]
                            conf = boxes.conf[i]
                            cls_id = int(boxes.cls[i])
                            
                            if conf >= 0.1:  # Confidence threshold (lowered to see more detections)
                                # Get class name through adaptive wrapper
                                class_name = self.current_model_wrapper.get_class_name(cls_id)
                                
                                detections.append({
                                    'type': class_name,
                                    'confidence': float(conf),
                                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                                })
                
                return detections
                
            except Exception as e:
                print(f"[DETECTION] Error in adaptive detection: {e}")
                return []
        
        return []
    
    # Phase 3: Distributed Processing Framework
    def setup_distributed_processing(self):
        """Setup distributed processing framework (placeholder for future)"""
        print("[DISTRIBUTED] Distributed processing framework initialized")
        print("[DISTRIBUTED] Note: This is a framework placeholder for future cloud processing")
        
        # Framework components that would be implemented:
        # 1. Node discovery and registration
        # 2. Task serialization and distribution  
        # 3. Result aggregation
        # 4. Load balancing
        # 5. Fault tolerance and retry logic
        
        distributed_info = {
            'framework_ready': True,
            'supported_operations': [
                'ai_inference',
                'image_preprocessing', 
                'batch_processing',
                'model_training'
            ],
            'node_types': [
                'local_gpu',
                'cloud_instance',
                'edge_device'
            ],
            'protocols': [
                'http_rest',
                'grpc',
                'websocket'
            ]
        }
        
        return distributed_info
    
    def add_processing_node(self, node_id, node_url, node_type='local'):
        """Add a processing node to the distributed system"""
        print(f"[DISTRIBUTED] Adding processing node: {node_id} ({node_type})")
        
        self.processing_nodes[node_id] = {
            'url': node_url,
            'type': node_type,
            'status': 'pending',
            'last_ping': time.time(),
            'processed_tasks': 0,
            'avg_response_time': 0.0
        }
        
        # In a real implementation, this would:
        # 1. Ping the node to verify it's online
        # 2. Exchange capability information
        # 3. Setup secure communication
        # 4. Add to load balancer
        
        print(f"[DISTRIBUTED] Framework ready for node {node_id} integration")
    
    def submit_distributed_task(self, task_type, task_data):
        """Submit a task for distributed processing"""
        task = {
            'id': f"task_{int(time.time()*1000)}",
            'type': task_type,
            'data': task_data,
            'timestamp': time.time(),
            'priority': 'normal'
        }
        
        try:
            self.distributed_queue.put_nowait(task)
            print(f"[DISTRIBUTED] Task {task['id']} queued for processing")
            return task['id']
        except queue.Full:
            print("[DISTRIBUTED] Task queue full - implementing backpressure")
            return None
    
    def get_distributed_processing_status(self):
        """Get status of distributed processing system"""
        total_nodes = len(self.processing_nodes)
        active_nodes = sum(1 for node in self.processing_nodes.values() 
                          if node['status'] == 'active')
        pending_tasks = self.distributed_queue.qsize()
        
        return {
            'total_nodes': total_nodes,
            'active_nodes': active_nodes,
            'pending_tasks': pending_tasks,
            'framework_status': 'ready' if total_nodes > 0 else 'no_nodes',
            'distributed_mode': self.distributed_mode
        }
    
    # Phase 3: Advanced UI Controls  
    def show_advanced_settings(self):
        """Show advanced configuration panel"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Advanced Settings")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Threading Settings Tab
        threading_frame = ttk.Frame(notebook)
        notebook.add(threading_frame, text="Threading")
        
        ttk.Label(threading_frame, text="Threading Configuration", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Pipeline settings
        pipeline_group = ttk.LabelFrame(threading_frame, text="Pipeline Settings", padding=10)
        pipeline_group.pack(fill=tk.X, pady=5)
        
        ttk.Label(pipeline_group, text="Screenshot FPS Target:").pack(anchor=tk.W)
        screenshot_fps = tk.Scale(pipeline_group, from_=1, to=30, orient=tk.HORIZONTAL)
        screenshot_fps.set(10)
        screenshot_fps.pack(fill=tk.X, pady=2)
        
        ttk.Label(pipeline_group, text="AI Processing FPS Target:").pack(anchor=tk.W)
        ai_fps = tk.Scale(pipeline_group, from_=1, to=10, orient=tk.HORIZONTAL)  
        ai_fps.set(3)
        ai_fps.pack(fill=tk.X, pady=2)
        
        # Memory Settings Tab
        memory_frame = ttk.Frame(notebook)
        notebook.add(memory_frame, text="Memory")
        
        ttk.Label(memory_frame, text="Memory Management", font=("Arial", 12, "bold")).pack(pady=10)
        
        mem_group = ttk.LabelFrame(memory_frame, text="Memory Limits", padding=10)
        mem_group.pack(fill=tk.X, pady=5)
        
        ttk.Label(mem_group, text="Memory Cleanup Threshold (MB):").pack(anchor=tk.W)
        mem_threshold = tk.Scale(mem_group, from_=100, to=2000, orient=tk.HORIZONTAL)
        mem_threshold.set(500)
        mem_threshold.pack(fill=tk.X, pady=2)
        
        ttk.Label(mem_group, text="Queue Size Limits:").pack(anchor=tk.W)
        ttk.Label(mem_group, text=f"Screenshot Queue: {self.screenshot_queue.maxsize}").pack(anchor=tk.W)
        ttk.Label(mem_group, text=f"AI Results Queue: {self.ai_results_queue.maxsize}").pack(anchor=tk.W)
        
        # Error Recovery Tab
        error_frame = ttk.Frame(notebook)
        notebook.add(error_frame, text="Error Recovery")
        
        ttk.Label(error_frame, text="Error Recovery Settings", font=("Arial", 12, "bold")).pack(pady=10)
        
        error_group = ttk.LabelFrame(error_frame, text="Recovery Thresholds", padding=10)
        error_group.pack(fill=tk.X, pady=5)
        
        ttk.Label(error_group, text="Max Consecutive Errors:").pack(anchor=tk.W)
        max_errors = tk.Scale(error_group, from_=1, to=20, orient=tk.HORIZONTAL)
        max_errors.set(self.error_recovery['max_consecutive_errors'])
        max_errors.pack(fill=tk.X, pady=2)
        
        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def show_analytics_dashboard(self):
        """Show analytics and performance dashboard"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Analytics Dashboard")
        dialog.geometry("700x600")
        dialog.transient(self.root)
        
        # Create notebook for different analytics views
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Performance Analytics Tab
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="Performance")
        
        ttk.Label(perf_frame, text="Performance Analytics", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Performance metrics
        stats = self.performance_stats
        
        metrics_text = tk.Text(perf_frame, height=15, width=80)
        metrics_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        analytics_content = f"""
PERFORMANCE METRICS
{'=' * 50}

📊 Capture Performance:
   Screenshots Captured: {stats['screenshots_captured']}
   Current Screenshot FPS: {stats['screenshot_fps']:.2f}
   AI Detections Processed: {stats['ai_detections_processed']}
   Current AI FPS: {stats['ai_fps']:.2f}

💾 Memory Usage:
   Current Memory: {stats['memory_usage_mb']:.1f} MB
   Last Cleanup: {time.strftime('%H:%M:%S', time.localtime(stats['last_cleanup_time'])) if stats['last_cleanup_time'] > 0 else 'Never'}

⚠️ Error Statistics:
   Total Errors: {stats['total_errors']}
   Screenshot Errors: {stats['screenshot_errors']}
   AI Processing Errors: {stats['ai_errors']}
   Recovery Attempts: {self.error_recovery['recovery_attempts']}

🔄 Threading Status:
   Pipeline Mode: {'Active' if self.pipeline_mode else 'Inactive'}
   Live Mode: {'Active' if self.live_mode else 'Inactive'}
   Multi-Window Mode: {'Active' if self.multi_window_mode else 'Inactive'}
   
📈 Queue Status:
   Screenshot Queue: {self.screenshot_queue.qsize()}/{self.screenshot_queue.maxsize}
   AI Results Queue: {self.ai_results_queue.qsize()}/{self.ai_results_queue.maxsize}
   Background Tasks: {len([f for f in self.background_tasks._threads if f.running()]) if hasattr(self.background_tasks, '_threads') else 'N/A'}

🖥️ Multi-Window Status:
   Monitored Windows: {len(self.monitored_windows)}
   Active Windows: {sum(1 for w in self.monitored_windows.values() if w.get('active', False))}
"""
        
        metrics_text.insert(tk.END, analytics_content)
        metrics_text.config(state=tk.DISABLED)
        
        # System Info Tab
        system_frame = ttk.Frame(notebook)
        notebook.add(system_frame, text="System")
        
        ttk.Label(system_frame, text="System Information", font=("Arial", 12, "bold")).pack(pady=10)
        
        system_text = tk.Text(system_frame, height=15, width=80)
        system_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        try:
            import platform
            process = psutil.Process()
            
            system_content = f"""
SYSTEM INFORMATION
{'=' * 50}

🖥️ Platform:
   OS: {platform.system()} {platform.release()}
   Architecture: {platform.architecture()[0]}
   Processor: {platform.processor()}

🧠 Memory:
   Process Memory: {process.memory_info().rss / (1024*1024):.1f} MB
   Memory Percent: {process.memory_percent():.1f}%
   
⚡ CPU:
   CPU Percent: {process.cpu_percent():.1f}%
   Thread Count: {process.num_threads()}
   
📁 Files:
   Open Files: {len(process.open_files())}
   
🔧 Python:
   Version: {platform.python_version()}
   Implementation: {platform.python_implementation()}
"""
        except Exception as e:
            system_content = f"System information unavailable: {e}"
        
        system_text.insert(tk.END, system_content)
        system_text.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def show_debug_panel(self):
        """Show debug and diagnostic panel"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Debug Panel")
        dialog.geometry("600x400")
        dialog.transient(self.root)
        
        ttk.Label(dialog, text="Debug & Diagnostics", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Debug controls
        debug_controls = ttk.Frame(dialog)
        debug_controls.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(debug_controls, text="🔍 Test Model", 
                  command=self.debug_test_model).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(debug_controls, text="🧹 Force Cleanup", 
                  command=self.debug_force_cleanup).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(debug_controls, text="📊 Queue Stats", 
                  command=self.debug_show_queue_stats).pack(side=tk.LEFT)
        
        # Debug log
        ttk.Label(dialog, text="Debug Log:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10,0))
        
        self.debug_log = tk.Text(dialog, height=15, width=70)
        self.debug_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add some initial debug info
        debug_info = f"""[{time.strftime('%H:%M:%S')}] Debug panel opened
[{time.strftime('%H:%M:%S')}] Current model: {getattr(self, 'current_model_path', 'None')}
[{time.strftime('%H:%M:%S')}] Threading modes - Pipeline: {self.pipeline_mode}, Live: {self.live_mode}, Multi-window: {self.multi_window_mode}
[{time.strftime('%H:%M:%S')}] Error recovery attempts: {self.error_recovery['recovery_attempts']}
"""
        self.debug_log.insert(tk.END, debug_info)
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def debug_test_model(self):
        """Debug: Test current model with sample data"""
        if hasattr(self, 'debug_log'):
            self.debug_log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] Testing current model...\n")
            
            if self.current_model_wrapper:
                # Create a test image
                test_image = np.zeros((240, 320, 3), dtype=np.uint8)  # GBA resolution
                test_image.fill(128)  # Gray image
                
                try:
                    detections = self._detect_objects_internal(test_image)
                    self.debug_log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] Model test successful - {len(detections)} detections\n")
                except Exception as e:
                    self.debug_log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] Model test failed: {e}\n")
            else:
                self.debug_log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] No model loaded\n")
            
            self.debug_log.see(tk.END)
    
    def debug_force_cleanup(self):
        """Debug: Force memory cleanup"""
        if hasattr(self, 'debug_log'):
            import gc
            before_mem = self.performance_stats['memory_usage_mb']
            
            self.debug_log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] Forcing memory cleanup...\n")
            
            # Clear queues
            cleared_screenshots = 0
            while not self.screenshot_queue.empty():
                try:
                    self.screenshot_queue.get_nowait()
                    cleared_screenshots += 1
                except queue.Empty:
                    break
            
            cleared_results = 0
            while not self.ai_results_queue.empty():
                try:
                    self.ai_results_queue.get_nowait()
                    cleared_results += 1
                except queue.Empty:
                    break
            
            # Force garbage collection
            gc.collect()
            
            after_mem = self.performance_stats['memory_usage_mb']
            freed = before_mem - after_mem
            
            self.debug_log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] Cleanup complete:\n")
            self.debug_log.insert(tk.END, f"  - Cleared {cleared_screenshots} screenshots, {cleared_results} results\n")
            self.debug_log.insert(tk.END, f"  - Memory freed: {freed:.1f} MB\n")
            self.debug_log.see(tk.END)
    
    def debug_show_queue_stats(self):
        """Debug: Show detailed queue statistics"""
        if hasattr(self, 'debug_log'):
            self.debug_log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] Queue Statistics:\n")
            self.debug_log.insert(tk.END, f"  Screenshot Queue: {self.screenshot_queue.qsize()}/{self.screenshot_queue.maxsize}\n")
            self.debug_log.insert(tk.END, f"  AI Results Queue: {self.ai_results_queue.qsize()}/{self.ai_results_queue.maxsize}\n")
            self.debug_log.insert(tk.END, f"  Model Swap Queue: {self.model_swap_queue.qsize()}/{self.model_swap_queue.maxsize}\n")
            self.debug_log.insert(tk.END, f"  Distributed Queue: {self.distributed_queue.qsize()}/{self.distributed_queue.maxsize}\n")
            
            if self.multi_window_mode:
                self.debug_log.insert(tk.END, f"  Window Results Queue: {self.window_results_queue.qsize()}/{self.window_results_queue.maxsize}\n")
            
            self.debug_log.see(tk.END)

    def toggle_live_mode(self):
        """Toggle live capture mode"""
        if not self.emulator_window:
            messagebox.showwarning("No Emulator", "Please find and select an emulator window first!")
            self.live_var.set(False)
            return
        
        # Disable pipeline mode when using regular live mode
        if self.live_var.get() and self.pipeline_var.get():
            self.pipeline_var.set(False)
            self.toggle_pipeline_mode()
            
        self.live_mode = self.live_var.get()
        
        if self.live_mode:
            self.start_live_capture()
        else:
            self.stop_live_capture()
            
    def start_live_capture(self):
        """Start live capture with enhanced thread safety"""
        with self._thread_lock:
            if self.capture_thread and self.capture_thread.is_alive():
                return
                
            # Clear any previous stop signal
            self._stop_event.clear()
            self.live_paused = False
            
            self.capture_thread = threading.Thread(
                target=self.live_capture_loop, 
                daemon=True,
                name="LiveCapture"
            )
            self.capture_thread.start()
            
        self.status_bar.config(text="Live mode active - providing feedback...")
        
    def stop_live_capture(self):
        """Stop live capture with proper synchronization"""
        with self._thread_lock:
            self.live_mode = False
            self._stop_event.set()
            
        self.status_bar.config(text="Live mode stopped")
        
    def live_capture_loop(self):
        """Live capture loop with enhanced synchronization"""
        print("Starting enhanced live capture loop...")
        while not self._stop_event.is_set() and self.live_mode and not self.live_paused:
            try:
                # Capture from emulator using proper method
                screenshot = self.capture_emulator_screenshot()
                if screenshot is not None:
                    print(f"Captured screenshot: {screenshot.shape}")
                    # Run AI detection
                    detections = self.detect_objects(screenshot)
                    print(f"Detected {len(detections)} objects")
                    
                    # Update UI in main thread
                    self.root.after(0, lambda s=screenshot, d=detections: self.update_live_display(s, d))
                else:
                    print("Screenshot capture returned None")
                    
                time.sleep(0.2)  # Update every 200ms for better responsiveness
                
            except Exception as e:
                print(f"Live capture error: {e}")
                time.sleep(0.5)
        print("Live capture loop ended")
                
    def update_live_display(self, screenshot, detections):
        """Update display with live capture and detections"""
        # Convert from BGR (OpenCV) to RGB (PIL) format
        if screenshot is not None:
            screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            self.current_screenshot = screenshot_rgb
            
            # Clean up expired focus areas
            current_time = time.time()
            expired_count = len(self.focus_areas)
            self.focus_areas = [area for area in self.focus_areas if current_time < area['expires_at']]
            expired_count -= len(self.focus_areas)
            if expired_count > 0:
                print(f"[FOCUS] Expired {expired_count} focus areas")
            
            # Filter detections: remove overlaps and suppressed regions
            filtered_detections = []
            for detection in detections:
                if not self.is_detection_suppressed(detection):
                    # Apply auto-accept logic for high confidence detections
                    confidence = detection.get('confidence', 0.0)
                    if confidence >= self.auto_accept_threshold and not detection.get('manual', False):
                        detection['auto_accepted'] = True
                        print(f"[AUTO-ACCEPT] {detection['type']} ({confidence:.1%}) auto-accepted")
                    elif confidence < self.ask_confirmation_threshold and not detection.get('manual', False):
                        detection['needs_confirmation'] = True
                        
                    filtered_detections.append(detection)
                    
            # Apply Non-Maximum Suppression to remove overlapping boxes
            if len(filtered_detections) > 1:
                filtered_detections = self.fix_overlapping_detections(filtered_detections)
            
            # Preserve manual detections with context awareness
            manual_detections = [d for d in self.ai_detections if d.get('manual', False)]
            context_valid_manual = self.filter_manual_detections_by_context(manual_detections, filtered_detections)
            self.ai_detections = filtered_detections + context_valid_manual
            
            # Force display update
            self.update_display()
            self.update_detection_list()
            print(f"[LIVE] Updated display with {len(filtered_detections)} detections")
        
    def toggle_always_on_top(self):
        """Toggle always on top mode"""
        self.root.attributes('-topmost', self.always_on_top_var.get())
        
    def toggle_transparency(self):
        """Toggle window transparency"""
        if self.transparent_var.get():
            self.root.attributes('-alpha', 0.85)  # 85% opacity
        else:
            self.root.attributes('-alpha', 1.0)   # 100% opacity
            
    def fix_overlapping_detections(self, detections):
        """Remove overlapping detections using Non-Maximum Suppression"""
        if len(detections) <= 1:
            return detections
            
        # Convert to format for NMS
        boxes = []
        scores = []
        for detection in detections:
            bbox = detection['bbox']
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            scores.append(detection.get('confidence', 0.5))
            
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        try:
            # Use OpenCV's NMS if available
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.3, 0.4)
            
            if len(indices) > 0:
                # Keep only non-overlapping detections
                keep_indices = indices.flatten()
                filtered_detections = [detections[i] for i in keep_indices]
                return filtered_detections
        except:
            # Fallback: simple overlap removal
            pass
            
        return detections
        
    def remove_detection_permanently(self, detection_to_remove):
        """Permanently remove a detection and prevent it from reappearing"""
        # Remove from current detections
        self.ai_detections = [d for d in self.ai_detections if d != detection_to_remove]
        
        # Add to suppression list to prevent redetection
        if not hasattr(self, 'suppressed_regions'):
            self.suppressed_regions = []
            
        # Store the region to suppress future detections
        bbox = detection_to_remove['bbox']
        suppression_region = {
            'bbox': bbox,
            'type': detection_to_remove['type'],
            'timestamp': time.time(),
            'reason': 'user_marked_wrong'
        }
        self.suppressed_regions.append(suppression_region)
        
        # Clean old suppressions (older than 10 seconds)
        current_time = time.time()
        self.suppressed_regions = [
            r for r in self.suppressed_regions 
            if current_time - r['timestamp'] < 10
        ]
        
        print(f"[SUPPRESSED] Removed {detection_to_remove['type']} detection permanently")
        
    def is_detection_suppressed(self, detection):
        """Check if this detection should be suppressed based on user feedback"""
        if not hasattr(self, 'suppressed_regions'):
            return False
            
        bbox = detection['bbox']
        
        for suppressed in self.suppressed_regions:
            # Check if bounding boxes overlap significantly
            s_bbox = suppressed['bbox']
            
            # Calculate intersection over union (IoU)
            x1 = max(bbox[0], s_bbox[0])
            y1 = max(bbox[1], s_bbox[1])
            x2 = min(bbox[2], s_bbox[2])
            y2 = min(bbox[3], s_bbox[3])
            
            if x1 < x2 and y1 < y2:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                area2 = (s_bbox[2] - s_bbox[0]) * (s_bbox[3] - s_bbox[1])
                union = area1 + area2 - intersection
                
                if union > 0:
                    iou = intersection / union
                    if iou > 0.5:  # 50% overlap threshold
                        return True
                        
        return False
        
    def on_canvas_right_click(self, event):
        """Handle right-click on detection (mark as wrong) - ENHANCED"""
        detection = self.get_detection_at_point(event.x, event.y)
        if detection:
            self.feedback_system.add_negative_feedback(detection)
            # Use permanent removal instead of simple removal
            self.remove_detection_permanently(detection)
            self.update_display()
            self.update_stats_display()
            self.update_detection_list()
            
    def toggle_draw_mode(self):
        """Toggle manual draw box mode"""
        self.draw_mode = self.draw_mode_var.get()
        if self.draw_mode:
            self.status_bar.config(text="Draw Mode: Drag to create bounding boxes for missed objects")
        else:
            self.status_bar.config(text="Ready for backseat driving feedback...")
            
    def toggle_focus_mode(self):
        """Toggle temporary focus mode"""
        self.focus_mode = self.focus_mode_var.get()
        if self.focus_mode:
            self.status_bar.config(text="Focus Mode: Drag to highlight areas temporarily (they auto-expire)")
        else:
            self.status_bar.config(text="Ready for backseat driving feedback...")
            
    def clear_focus_areas(self):
        """Clear all focus areas"""
        self.focus_areas.clear()
        print(f"[FOCUS] Cleared all focus areas")
        self.update_display()
            
    def on_canvas_press(self, event):
        """Handle mouse press for draw mode and focus mode"""
        if self.draw_mode:
            self.drawing = True
            self.draw_start_x = event.x
            self.draw_start_y = event.y
        elif self.focus_mode:
            self.drawing = True  # Reuse drawing state for focus mode
            self.draw_start_x = event.x
            self.draw_start_y = event.y
            
    def on_canvas_drag(self, event):
        """Handle mouse drag for draw mode and focus mode"""
        if (self.draw_mode or self.focus_mode) and self.drawing:
            # Remove previous rectangle if exists
            if self.draw_current_rect:
                self.image_canvas.delete(self.draw_current_rect)
            elif self.focus_current_rect:
                self.image_canvas.delete(self.focus_current_rect)
                
            # Draw new rectangle with different colors for different modes
            if self.draw_mode:
                self.draw_current_rect = self.image_canvas.create_rectangle(
                    self.draw_start_x, self.draw_start_y, event.x, event.y,
                    outline='yellow', width=3, tags='manual_box'
                )
            elif self.focus_mode:
                self.focus_current_rect = self.image_canvas.create_rectangle(
                    self.draw_start_x, self.draw_start_y, event.x, event.y,
                    outline='cyan', width=2, tags='focus_box', dash=(5, 5)
                )
            
    def on_canvas_release(self, event):
        """Handle mouse release for draw mode and focus mode"""
        if self.draw_mode and self.drawing:
            self.drawing = False
            
            # Convert canvas coordinates to image coordinates
            if hasattr(self, 'display_scale') and self.draw_current_rect:
                # Calculate bounding box in image coordinates
                x1 = int(min(self.draw_start_x, event.x) / self.display_scale)
                y1 = int(min(self.draw_start_y, event.y) / self.display_scale)
                x2 = int(max(self.draw_start_x, event.x) / self.display_scale)
                y2 = int(max(self.draw_start_y, event.y) / self.display_scale)
                
                # Minimum box size check
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    # Create manual detection
                    manual_detection = {
                        'type': self.manual_label_var.get(),
                        'bbox': [x1, y1, x2, y2],
                        'confidence': 1.0,  # Manual detections have 100% confidence
                        'manual': True,
                        'source': 'user_drawn',
                        'created_at': time.time()  # Timestamp for context filtering
                    }
                    
                    # Add to detections list
                    self.ai_detections.append(manual_detection)
                    
                    # Add positive feedback for the manual detection
                    self.feedback_system.add_positive_feedback(manual_detection, "user_manual_annotation")
                    
                    print(f"[MANUAL] Created {manual_detection['type']} detection at {[x1, y1, x2, y2]}")
                    
                    # Update display
                    self.update_display()
                    self.update_stats_display()
                    
                else:
                    print("[MANUAL] Box too small, ignored")
                
                # Clean up temporary rectangle
                if self.draw_current_rect:
                    self.image_canvas.delete(self.draw_current_rect)
                    self.draw_current_rect = None
                    
        elif self.focus_mode and self.drawing:
            self.drawing = False
            
            # Convert canvas coordinates to image coordinates for focus area
            if hasattr(self, 'display_scale') and self.focus_current_rect:
                x1 = int(min(self.draw_start_x, event.x) / self.display_scale)
                y1 = int(min(self.draw_start_y, event.y) / self.display_scale)
                x2 = int(max(self.draw_start_x, event.x) / self.display_scale)
                y2 = int(max(self.draw_start_y, event.y) / self.display_scale)
                
                # Minimum box size check
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    # Get focus duration from UI
                    try:
                        duration = float(self.focus_duration_var.get())
                    except ValueError:
                        duration = 10.0  # Default to 10 seconds
                    
                    # Create temporary focus area
                    focus_area = {
                        'bbox': [x1, y1, x2, y2],
                        'created_at': time.time(),
                        'expires_at': time.time() + duration,
                        'duration': duration
                    }
                    
                    self.focus_areas.append(focus_area)
                    
                    print(f"[FOCUS] Created focus area at {[x1, y1, x2, y2]} for {duration}s")
                    
                    # Update display
                    self.update_display()
                    
                else:
                    print("[FOCUS] Box too small, ignored")
                
                # Clean up temporary rectangle
                if self.focus_current_rect:
                    self.image_canvas.delete(self.focus_current_rect)
                    self.focus_current_rect = None
            
    def on_canvas_click(self, event):
        """Handle canvas click"""
        # Don't process clicks if in draw mode or focus mode
        if self.draw_mode or self.focus_mode:
            return
            
    def filter_manual_detections_by_context(self, manual_detections, current_ai_detections):
        """Filter manual detections based on current game context"""
        if not manual_detections:
            return []
            
        # Detect current game state based on AI detections
        current_context = self.detect_game_context(current_ai_detections)
        
        valid_manual = []
        for manual_det in manual_detections:
            # Check if manual detection is still contextually valid
            if self.is_manual_detection_valid_in_context(manual_det, current_context):
                valid_manual.append(manual_det)
            else:
                # Log removal for user awareness
                print(f"[CONTEXT] Removed manual {manual_det['type']} detection due to game state change")
                
                # Add negative feedback for context-invalid manual detection
                self.feedback_system.add_negative_feedback(manual_det, "context_invalid")
                
        return valid_manual
        
    def detect_game_context(self, ai_detections):
        """Detect current game context from AI detections"""
        detection_types = [d['type'] for d in ai_detections]
        
        # Battle context indicators
        if any(t in detection_types for t in ['hp_bar', 'pokemon_sprite']):
            return 'battle'
            
        # Menu context indicators  
        if any(t in detection_types for t in ['menu_box', 'dialogue_box']):
            return 'menu'
            
        # Pokemon Center context
        if any(t in detection_types for t in ['pokeball/item']) and 'npc_character' in detection_types:
            return 'pokemon_center'
            
        # Overworld context (default)
        if any(t in detection_types for t in ['player_character', 'npc_character', 'building', 'tree', 'water', 'grass']):
            return 'overworld'
            
        return 'unknown'
        
    def is_manual_detection_valid_in_context(self, manual_detection, current_context):
        """Check if a manual detection is valid in the current context"""
        manual_type = manual_detection['type']
        creation_time = manual_detection.get('created_at', 0)
        current_time = time.time()
        
        # Auto-expire manual detections after 30 seconds regardless of context
        if current_time - creation_time > 30:
            print(f"[CONTEXT] Auto-expiring manual {manual_type} after 30 seconds")
            return False
        
        # Context-specific validation rules (updated to match competitive trainer)
        context_rules = {
            'battle': ['hp_bar', 'pokemon_sprite', 'exp._bar', 'level_indicator'],
            'menu': ['menu_box', 'dialogue_box', 'text_area'],
            'pokemon_center': ['pokeball/item', 'npc_character', 'menu_box'],
            'overworld': ['player_character', 'npc_character', 'building', 'tree', 'water', 'grass', 'pokeball/item'],
            'unknown': []  # Keep manual detections in unknown contexts
        }
        
        # If manual detection type is not in current context rules, it's probably invalid
        valid_types = context_rules.get(current_context, [])
        
        # Special cases - some types are valid across multiple contexts
        universal_types = ['player_character', 'text_area', 'item']
        
        if manual_type in universal_types or manual_type in valid_types or current_context == 'unknown':
            return True
        else:
            return False
        
    # ADAPTIVE SYSTEM METHODS
    def refresh_class_registry(self):
        """Refresh the class registry and update UI"""
        try:
            # Scan models for backward compatibility
            self.class_registry.scan_and_update_models()
            
            # Use shared class manager as primary source
            self.available_classes = self.class_manager.get_classes()
            class_count = len(self.available_classes)
            
            # Also sync registry classes with shared manager
            registry_classes = self.class_registry.get_all_classes()
            for cls in registry_classes:
                if cls not in self.available_classes:
                    self.class_manager.add_class(cls, "ClassRegistry")
            
            # Update UI (thread-safe)
            import threading
            if threading.current_thread() is threading.main_thread():
                if hasattr(self, 'registry_status_label'):
                    self.registry_status_label.config(
                        text=f"{class_count} classes available", 
                        foreground="green"
                    )
            else:
                # Defer to main thread
                def update_ui():
                    if hasattr(self, 'registry_status_label'):
                        self.registry_status_label.config(
                            text=f"{class_count} classes available", 
                            foreground="green"
                        )
                self.root.after(0, update_ui)
            
            print(f"[ADAPTIVE AI] Registry refreshed: {class_count} classes from shared manager")
            
        except Exception as e:
            self.registry_status_label.config(text="Registry error", foreground="red")
            print(f"[ADAPTIVE AI] Registry refresh error: {e}")

    def add_new_class(self):
        """Add a new class to the registry"""
        new_class = self.new_class_var.get().strip()
        
        if not new_class:
            messagebox.showwarning("Input Required", "Please enter a class name!")
            return
        
        if new_class in self.available_classes:
            messagebox.showwarning("Duplicate Class", f"Class '{new_class}' already exists!")
            return
        
        # Add through shared manager (will notify all subscribers)
        if self.class_manager.add_class(new_class, "BackseatCollector"):
            self.new_class_var.set("")
            messagebox.showinfo("Class Added", 
                               f"New class '{new_class}' added successfully!\n\n"
                               f"This class is now available in all perception tools.")
            print(f"[ADAPTIVE AI] Added new class: {new_class}")
    
    def _on_classes_changed(self, new_classes):
        """Callback when classes are updated externally"""
        print(f"[BackseatCollector] Class list updated: {len(new_classes)} classes")
        self.available_classes = new_classes
        
        # Update active classes to include any new ones
        for cls in new_classes:
            if cls not in self.active_classes:
                self.active_classes.append(cls)
        
        # Update model wrapper if loaded
        if self.current_model_wrapper:
            self.current_model_wrapper.set_target_classes(self.active_classes)
        
        # Update UI
        self.root.after(0, self.refresh_class_registry)
        self.root.after(0, self._update_manual_labels_dropdown)
    
    def _update_manual_labels_dropdown(self):
        """Update manual annotation dropdown with current classes"""
        if hasattr(self, 'manual_label_dropdown'):
            current_classes = self.available_classes.copy()
            current_classes.extend(['missed_object', 'other'])  # Always include these
            self.manual_label_dropdown.config(values=current_classes)

    def detect_gpu_available(self):
        """Detect if GPU is available for training"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def update_gpu_status(self):
        """Update GPU status display"""
        if self.use_gpu_var.get():
            if self.detect_gpu_available():
                self.gpu_status_label.config(text="✅ GPU Ready", foreground="green")
            else:
                self.gpu_status_label.config(text="❌ GPU Not Available", foreground="red")
                self.use_gpu_var.set(False)  # Auto-disable if not available
        else:
            self.gpu_status_label.config(text="💻 CPU Mode", foreground="blue")

    def show_class_manager(self):
        """Show detailed class management window"""
        class_window = tk.Toplevel(self.root)
        class_window.title("Adaptive Class Manager")
        class_window.geometry("600x400")
        class_window.transient(self.root)
        
        # Available classes
        ttk.Label(class_window, text="Available Classes:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(10,5))
        
        list_frame = ttk.Frame(class_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        class_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=class_listbox.yview)
        class_listbox.configure(yscrollcommand=scrollbar.set)
        
        class_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate listbox
        for class_name in sorted(self.available_classes):
            class_listbox.insert(tk.END, class_name)
            if class_name in self.active_classes:
                class_listbox.selection_set(tk.END)
        
        # Buttons
        btn_frame = ttk.Frame(class_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def apply_selection():
            selected_indices = class_listbox.curselection()
            self.active_classes = [self.available_classes[i] for i in selected_indices]
            
            if self.current_model_wrapper:
                self.current_model_wrapper.set_target_classes(self.active_classes)
            
            messagebox.showinfo("Updated", f"Active classes updated: {len(self.active_classes)} selected")
            class_window.destroy()
        
        ttk.Button(btn_frame, text="Select All", 
                  command=lambda: class_listbox.selection_set(0, tk.END)).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(btn_frame, text="Clear All", 
                  command=lambda: class_listbox.selection_clear(0, tk.END)).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(btn_frame, text="Apply Selection", command=apply_selection).pack(side=tk.RIGHT)

    def _choose_dir(self, var):
        from tkinter import filedialog
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    def _collect_current_detections_for_export(self):
        """Return list of dicts: {'type': str, 'bbox': [x1,y1,x2,y2]}"""
        out = []
        for det in (self.ai_detections or []):
            bbox = det.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
            out.append({'type': det.get('type','other'), 'bbox': [int(b) for b in bbox]})
        return out

    def _current_image_rgb(self):
        # Your display pipeline keeps self.current_screenshot as RGB in many paths
        # If it's BGR in your build, convert before export.
        return self.current_screenshot

    def save_current_frame_to_yolo_async(self, stem=None):
        """Save current frame to YOLO format in background"""
        if self.current_screenshot is None:
            self.status_bar.config(text="No frame to export. Take a screenshot or enable Live Mode.")
            return
            
        # Show immediate feedback
        self.status_bar.config(text="🔄 Saving YOLO frame in background...")
        
        # Submit background task
        future = self.background_tasks.submit(self._save_yolo_worker, stem)
        
        # Monitor completion
        def check_save():
            if future.done():
                try:
                    result = future.result()
                    self._on_yolo_save_success(result)
                except Exception as e:
                    self._on_yolo_save_error(e)
            else:
                # Update progress animation  
                current_text = self.status_bar.cget("text")
                if "Saving YOLO" in current_text:
                    dots = len([c for c in current_text if c == '.'])
                    new_dots = '.' * ((dots % 3) + 1)
                    self.status_bar.config(text=f"🔄 Saving YOLO frame{new_dots}")
                self.root.after(200, check_save)
        
        self.root.after(50, check_save)
    
    def _save_yolo_worker(self, stem):
        """Background worker for YOLO export"""
        print("[BACKGROUND] Starting YOLO export...")
        
        # Generate stem if not provided
        if not stem:
            import time
            stem = f"cap_{int(time.time()*1000)}"

        # Get paths
        images_out = Path(self.yolo_images_out.get()).expanduser()
        labels_out = Path(self.yolo_labels_out.get()).expanduser()
        
        # Collect detections
        dets = self._collect_current_detections_for_export()
        
        # This is the potentially slow operation (large image processing)
        img_path, lbl_path = save_yolo_frame(
            image_rgb=self._current_image_rgb(),
            detections=dets,
            labels_out=labels_out,
            images_out=images_out,
            classes=CANONICAL_CLASSES,
            stem=stem,
        )
        
        return {
            'img_path': img_path,
            'lbl_path': lbl_path,
            'detections_count': len(dets)
        }
    
    def _on_yolo_save_success(self, result):
        """Handle successful YOLO save in UI thread"""
        img_path = result['img_path'] 
        lbl_path = result['lbl_path']
        det_count = result['detections_count']
        
        self.status_bar.config(text=f"✅ Saved: {img_path.name} / {lbl_path.name} ({det_count} objects)")
        print(f"[BACKGROUND] YOLO export completed: {img_path} and {lbl_path}")
    
    def _on_yolo_save_error(self, error):
        """Handle YOLO save errors in UI thread"""
        error_msg = f"YOLO export failed: {str(error)}"
        self.status_bar.config(text=f"❌ {error_msg}")
        print(f"[BACKGROUND] {error_msg}")

    def save_current_frame_to_yolo(self, stem=None):
        """Save current frame to YOLO format (DEPRECATED - use save_current_frame_to_yolo_async)"""
        if self.current_screenshot is None:
            self.status_bar.config(text="No frame to export. Take a screenshot or enable Live Mode.")
            return
        # choose a stem from timestamp if not provided
        import time
        if not stem:
            stem = f"cap_{int(time.time()*1000)}"

        # make sure we have sane folders
        images_out = Path(self.yolo_images_out.get()).expanduser()
        labels_out = Path(self.yolo_labels_out.get()).expanduser()

        dets = self._collect_current_detections_for_export()
        try:
            img_path, lbl_path = save_yolo_frame(
                image_rgb=self._current_image_rgb(),
                detections=dets,
                labels_out=labels_out,
                images_out=images_out,
                classes=CANONICAL_CLASSES,
                stem=stem,
            )
            self.status_bar.config(text=f"Saved: {img_path.name} / {lbl_path.name}")
            print(f"[YOLO] Wrote {img_path} and {lbl_path}")
        except Exception as e:
            self.status_bar.config(text=f"YOLO export failed: {e}")
            print(f"[YOLO] Export error: {e}")

    def shutdown(self):
        """Enhanced shutdown with proper thread cleanup"""
        print("[SHUTDOWN] Starting enhanced cleanup...")
        
        # Phase 1: Stop live mode gracefully
        print("[SHUTDOWN] Stopping live capture...")
        if self.live_mode:
            with self._thread_lock:
                self.live_mode = False
                self._stop_event.set()
        
        # Stop pipeline mode if running
        if hasattr(self, 'pipeline_mode') and self.pipeline_mode:
            print("[SHUTDOWN] Stopping pipeline mode...")
            self.pipeline_mode = False
            
        # Stop multi-window mode if running  
        if hasattr(self, 'multi_window_mode') and self.multi_window_mode:
            print("[SHUTDOWN] Stopping multi-window mode...")
            self.multi_window_mode = False
            if hasattr(self, 'monitored_windows'):
                for window_info in self.monitored_windows.values():
                    window_info['active'] = False
            
        # Wait for all capture threads to finish
        threads_to_wait = []
        if hasattr(self, 'capture_thread') and self.capture_thread and self.capture_thread.is_alive():
            threads_to_wait.append(('Live Capture', self.capture_thread))
        if hasattr(self, 'screenshot_thread') and self.screenshot_thread and self.screenshot_thread.is_alive():
            threads_to_wait.append(('Screenshot Capture', self.screenshot_thread))
        if hasattr(self, 'ai_processing_thread') and self.ai_processing_thread and self.ai_processing_thread.is_alive():
            threads_to_wait.append(('AI Processing', self.ai_processing_thread))
        if hasattr(self, 'memory_manager_thread') and self.memory_manager_thread and self.memory_manager_thread.is_alive():
            threads_to_wait.append(('Memory Manager', self.memory_manager_thread))
        
        # Add multi-window monitoring threads
        if hasattr(self, 'monitored_windows'):
            for title, window_info in self.monitored_windows.items():
                if window_info.get('thread') and window_info['thread'].is_alive():
                    threads_to_wait.append((f'Window Monitor ({title[:20]})', window_info['thread']))
            
        for thread_name, thread in threads_to_wait:
            print(f"[SHUTDOWN] Waiting for {thread_name} thread...")
            try:
                thread.join(timeout=3.0)  # Wait max 3 seconds per thread
                if thread.is_alive():
                    print(f"[SHUTDOWN] Warning: {thread_name} thread didn't stop gracefully")
                else:
                    print(f"[SHUTDOWN] {thread_name} thread stopped cleanly")
            except Exception as e:
                print(f"[SHUTDOWN] Error waiting for {thread_name} thread: {e}")
        
        # Phase 2: Shutdown background task pool
        print("[SHUTDOWN] Shutting down background tasks...")
        if hasattr(self, 'background_tasks') and self.background_tasks:
            try:
                # Cancel any pending tasks
                for future in self.background_tasks._threads:
                    if hasattr(future, 'cancel'):
                        future.cancel()
                
                # Shutdown thread pool gracefully
                self.background_tasks.shutdown(wait=True, timeout=5.0)
                print("[SHUTDOWN] Background tasks stopped cleanly")
            except Exception as e:
                print(f"[SHUTDOWN] Error shutting down background tasks: {e}")
                # Force shutdown if graceful fails
                try:
                    self.background_tasks.shutdown(wait=False)
                except:
                    pass
        
        # Phase 3: Shutdown other systems
        print("[SHUTDOWN] Shutting down other systems...")
        
        # Shutdown online learning system
        if hasattr(self, 'online_learning') and self.online_learning:
            try:
                self.online_learning.shutdown()
                print("[SHUTDOWN] Online learning system stopped")
            except Exception as e:
                print(f"[SHUTDOWN] Error shutting down online learning: {e}")
        
        # Cleanup shared class manager
        try:
            cleanup_shared_manager()
            print("[SHUTDOWN] Shared class manager cleanup complete")
        except Exception as e:
            print(f"[SHUTDOWN] Error cleaning up class manager: {e}")
        
        # Phase 4: Final cleanup
        print("[SHUTDOWN] Final cleanup...")
        
        # Clear model references to help with garbage collection
        if hasattr(self, 'ai_model'):
            self.ai_model = None
        if hasattr(self, 'current_model_wrapper'):
            self.current_model_wrapper = None
            
        # Destroy GUI
        try:
            self.root.quit()
            self.root.destroy()
            print("[SHUTDOWN] GUI destroyed successfully")
        except Exception as e:
            print(f"[SHUTDOWN] Error destroying GUI: {e}")
        
        print("[SHUTDOWN] Enhanced cleanup complete!")
        
    def run(self):
        """Start the application"""
        self.update_stats_display()
        
        # Handle window close properly
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.shutdown()

def main():
    """Run the backseat driving data collector"""
    print("[BACKSEAT] Starting Backseat Driving AI Trainer...")
    print("Right-click detections to mark wrong, Ctrl+click to confirm correct!")
    
    try:
        app = BackseatDataCollector()
        print("[BACKSEAT] Application initialized successfully")
        app.run()
    except Exception as e:
        print(f"[ERROR] Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()