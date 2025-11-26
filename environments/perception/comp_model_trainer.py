#!/usr/bin/env python3
"""
🏆 COMPETITIVE MODEL TRAINER
============================

Real-time competitive learning system where multiple models compete 
to correctly identify objects you draw bounding boxes around.

Workflow:
1. Load 4 best models
2. User draws bounding box on game screen
3. All 4 models predict what object it is
4. User clicks correct answer from buttons
5. Wrong models get penalty, correct ones get reward
6. Models adapt and improve through competition

This is essentially "model battle royale" with human supervision!
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import win32gui
import win32ui
import win32con
from ctypes import windll
from PIL import Image, ImageTk, ImageDraw
import pickle
import os

from model_manager import ModelManager
from lightweight_training_viz import show_training_popup
from shared_class_manager import get_shared_class_manager, cleanup_shared_manager

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] YOLO not available - install with: pip install ultralytics")

@dataclass
class ModelCompetitor:
    """Represents a model in the competition"""
    name: str
    model: Optional[object]
    path: Path
    score: int = 0
    correct_predictions: int = 0
    total_predictions: int = 0
    confidence_sum: float = 0.0
    last_prediction: Optional[str] = None
    last_confidence: float = 0.0
    
    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
    
    @property
    def avg_confidence(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.confidence_sum / self.total_predictions

class CompetitiveModelTrainer:
    """Main competitive training application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🏆 Competitive Model Trainer")
        self.root.geometry("1400x900")
        
        # Competition state
        self.competitors: List[ModelCompetitor] = []
        self.selected_models = [None, None, None, None]  # Track individually selected models
        self.is_running = False
        self.current_bbox = None
        self.current_image = None
        self.drawing = False
        self.start_pos = None
        self.emulator_window = None
        
        # Load persistent data
        self.data_file = Path("comp_model_trainer_data.pkl")
        
        # Initialize shared class management
        self.class_manager = get_shared_class_manager()
        self.object_classes = self.class_manager.get_classes()
        self.class_manager.subscribe(self._on_classes_changed, "CompetitiveTrainer")
        
        self.load_persistent_data()
        
        # Competition results
        self.session_data = {
            'start_time': datetime.now(),
            'competitions': [],
            'model_performances': {},
            'training_data': []  # Store screenshots and labels for training
        }
        
        # Create training data directory
        self.training_data_dir = Path("competitive_session_training")
        self.training_data_dir.mkdir(exist_ok=True)
        (self.training_data_dir / "images").mkdir(exist_ok=True)
        (self.training_data_dir / "labels").mkdir(exist_ok=True)
        
        # Setup GUI
        self.setup_gui()
        
        # Available models list
        self.available_models = []
        
        # Load available models for dropdowns
        self.refresh_available_models()
        
    def load_persistent_data(self):
        """Load persistent training data (classes managed by SharedClassManager)"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    # Classes are now managed centrally - don't override
                    # self.object_classes managed by SharedClassManager
                    # Load any other persistent data here
                print(f"Loaded session data (classes managed centrally: {len(self.object_classes)})")
            except Exception as e:
                print(f"Error loading persistent data: {e}")
        else:
            print(f"Using shared class management: {len(self.object_classes)} classes")
    
    def save_persistent_data(self):
        """Save persistent object classes and training data"""
        try:
            data = {
                'object_classes': self.object_classes,
                'save_time': datetime.now().isoformat()
            }
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved persistent data with {len(self.object_classes)} object classes")
        except Exception as e:
            print(f"Error saving persistent data: {e}")
    
    def find_emulator_window(self):
        """Find and capture emulator window"""
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if any(emu in window_title.lower() for emu in ['mgba', 'visualboy', 'no$gba', 'bgb', 'sameboy', 'emulator']):
                    windows.append((hwnd, window_title))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        if windows:
            # Use the first emulator window found
            self.emulator_window = windows[0][0]
            window_title = windows[0][1]
            self.window_status_label.config(text=f"✓ Found: {window_title}")
            return True
        else:
            self.window_status_label.config(text="❌ No emulator window found")
            return False
    
    def capture_emulator_screen(self):
        """Capture the emulator window screen"""
        if not self.find_emulator_window():
            messagebox.showwarning("Warning", "No emulator window found! Please open an emulator.")
            return
        
        try:
            # Get window dimensions
            left, top, right, bottom = win32gui.GetWindowRect(self.emulator_window)
            width = right - left
            height = bottom - top
            
            # Capture window
            hwndDC = win32gui.GetWindowDC(self.emulator_window)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # Copy the window content
            saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
            
            # Convert to OpenCV format
            signedIntsArray = saveBitMap.GetBitmapBits(True)
            img = np.frombuffer(signedIntsArray, dtype='uint8')
            img.shape = (height, width, 4)
            
            # Convert BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Clean up
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.emulator_window, hwndDC)
            
            self.current_image = img
            self.display_image(img)
            
            self.window_status_label.config(text="✓ Screen captured successfully")
            
        except Exception as e:
            messagebox.showerror("Capture Error", f"Failed to capture screen: {e}")
            print(f"Screen capture error: {e}")
    
    def setup_gui(self):
        """Setup the competitive training interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🏆 Competitive Model Trainer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Left panel - Model selection and scoreboard
        self.setup_model_selection_panel(main_frame)
        
        # Center panel - Image display and drawing
        self.setup_image_panel(main_frame)
        
        # Right panel - Controls and feedback
        self.setup_control_panel(main_frame)
        
        # Bottom panel - Object class buttons (dynamic)
        self.setup_class_buttons(main_frame)
        
    def setup_model_selection_panel(self, parent):
        """Setup the model selection and scoreboard"""
        left_frame = ttk.Frame(parent)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Model Selection Section
        selection_frame = ttk.LabelFrame(left_frame, text="Model Selection", padding="10")
        selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Individual model selection dropdowns
        self.model_vars = []
        self.model_dropdowns = []
        
        for i in range(4):
            model_frame = ttk.Frame(selection_frame)
            model_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(model_frame, text=f"Competitor {i+1}:", width=12).pack(side=tk.LEFT)
            
            model_var = tk.StringVar()
            dropdown = ttk.Combobox(model_frame, textvariable=model_var, state="readonly", width=25)
            dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
            
            self.model_vars.append(model_var)
            self.model_dropdowns.append(dropdown)
        
        # Load models button
        load_button = ttk.Button(selection_frame, text="Load Selected Models", 
                               command=self.load_selected_models)
        load_button.pack(pady=(10, 0))
        
        # Scoreboard Section
        self.setup_scoreboard(left_frame)
        
    def setup_scoreboard(self, parent):
        """Setup the model performance scoreboard"""
        scoreboard_frame = ttk.LabelFrame(parent, text="Model Competition Scoreboard", padding="10")
        scoreboard_frame.pack(fill=tk.BOTH, expand=True)
        
        # Headers
        headers = ["Model", "Score", "Accuracy", "Avg Conf", "Last Prediction"]
        for i, header in enumerate(headers):
            label = ttk.Label(scoreboard_frame, text=header, font=('Arial', 10, 'bold'))
            label.grid(row=0, column=i, padx=5, pady=5, sticky=tk.W)
        
        # Model rows (will be populated by update_scoreboard)
        self.scoreboard_labels = []
        for i in range(4):  # 4 models max
            row_labels = []
            for j in range(len(headers)):
                label = ttk.Label(scoreboard_frame, text="", width=12)
                label.grid(row=i+1, column=j, padx=5, pady=2, sticky=tk.W)
                row_labels.append(label)
            self.scoreboard_labels.append(row_labels)
        
        # Competition stats
        stats_frame = ttk.Frame(scoreboard_frame)
        stats_frame.grid(row=6, column=0, columnspan=len(headers), pady=(20, 0), sticky=(tk.W, tk.E))
        
        self.total_competitions_label = ttk.Label(stats_frame, text="Total Competitions: 0")
        self.total_competitions_label.pack(anchor=tk.W)
        
        self.training_data_label = ttk.Label(stats_frame, text="Training Examples: 0")
        self.training_data_label.pack(anchor=tk.W)
        
        self.session_time_label = ttk.Label(stats_frame, text="Session Time: 0:00")
        self.session_time_label.pack(anchor=tk.W)
        
    def setup_image_panel(self, parent):
        """Setup the main image display and drawing area"""
        image_frame = ttk.LabelFrame(parent, text="Game Screen - Draw Bounding Box", padding="10")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Canvas for image display and drawing
        self.canvas = tk.Canvas(image_frame, width=600, height=400, bg='black')
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.update_drawing)
        self.canvas.bind("<ButtonRelease-1>", self.finish_drawing)
        
        # Capture controls
        capture_frame = ttk.Frame(image_frame)
        capture_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.capture_button = ttk.Button(capture_frame, text="📷 Capture Game Screen", 
                                       command=self.capture_emulator_screen)
        self.capture_button.pack(side=tk.LEFT)
        
        self.window_status_label = ttk.Label(capture_frame, text="No game window found")
        self.window_status_label.pack(side=tk.RIGHT)
        
    def setup_control_panel(self, parent):
        """Setup the control panel with competition controls"""
        control_frame = ttk.LabelFrame(parent, text="Competition Controls", padding="10")
        control_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Competition status
        self.status_label = ttk.Label(control_frame, text="Ready for competition", 
                                     font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=(0, 20))
        
        # Current predictions display
        predictions_frame = ttk.LabelFrame(control_frame, text="Model Predictions", padding="10")
        predictions_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.prediction_labels = []
        for i in range(4):
            label = ttk.Label(predictions_frame, text=f"Model {i+1}: No prediction", 
                            font=('Arial', 10))
            label.pack(anchor=tk.W, pady=2)
            self.prediction_labels.append(label)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(control_frame, text="How to Use", padding="10")
        instructions_frame.pack(fill=tk.X, pady=(0, 20))
        
        instructions = [
            "1. Capture game screen",
            "2. Draw box around object", 
            "3. Click correct object type",
            "4. Screenshot saved & models learn!"
        ]
        
        for instruction in instructions:
            ttk.Label(instructions_frame, text=instruction).pack(anchor=tk.W)
        
        # Session controls
        session_frame = ttk.Frame(control_frame)
        session_frame.pack(fill=tk.X)
        
        self.save_session_button = ttk.Button(session_frame, text="💾 Save Session", 
                                            command=self.save_session)
        self.save_session_button.pack(fill=tk.X, pady=2)
        
        self.reset_scores_button = ttk.Button(session_frame, text="🔄 Reset Scores", 
                                            command=self.reset_scores)
        self.reset_scores_button.pack(fill=tk.X, pady=2)
        
        # GPU acceleration toggle
        gpu_frame = ttk.Frame(session_frame)
        gpu_frame.pack(fill=tk.X, pady=2)
        
        self.use_gpu_var = tk.BooleanVar(value=self.detect_gpu_available())
        self.gpu_checkbox = ttk.Checkbutton(gpu_frame, text="Use GPU acceleration", 
                                          variable=self.use_gpu_var,
                                          command=self.update_gpu_status)
        self.gpu_checkbox.pack(side=tk.LEFT)
        
        self.gpu_status_label = ttk.Label(gpu_frame, text="", font=('Arial', 8))
        self.gpu_status_label.pack(side=tk.LEFT, padx=(10, 0))
        self.update_gpu_status()
        
        # Training visualization toggle
        viz_frame = ttk.Frame(session_frame)
        viz_frame.pack(fill=tk.X, pady=2)
        
        self.show_viz_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_frame, text="Show training visualization", 
                       variable=self.show_viz_var).pack(side=tk.LEFT)
        
        self.train_button = ttk.Button(session_frame, text="🎯 Train Top Models", 
                                     command=self.train_top_models)
        self.train_button.pack(fill=tk.X, pady=2)
        
    def setup_class_buttons(self, parent):
        """Setup object class selection buttons"""
        self.buttons_frame = ttk.LabelFrame(parent, text="Select Correct Object Type", padding="10")
        self.buttons_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Create buttons in a grid
        self.class_buttons = []
        self.create_class_buttons()
        
        # Initially disable buttons
        self.enable_class_buttons(False)
    
    def create_class_buttons(self):
        """Create the actual class buttons"""
        # Clear existing buttons
        for widget in self.buttons_frame.winfo_children():
            widget.destroy()
        self.class_buttons = []
        
        cols = 3
        for i, obj_class in enumerate(self.object_classes):
            row = i // cols
            col = i % cols
            
            button = ttk.Button(self.buttons_frame, text=obj_class.replace('_', ' ').title(), 
                              command=lambda cls=obj_class: self.submit_correct_answer(cls))
            button.grid(row=row, column=col, padx=5, pady=5, sticky=(tk.W, tk.E))
            self.buttons_frame.columnconfigure(col, weight=1)
            self.class_buttons.append(button)
        
        # Add new object type button
        add_button_frame = ttk.Frame(self.buttons_frame)
        add_button_frame.grid(row=(len(self.object_classes) // cols) + 1, column=0, columnspan=cols, pady=10, sticky=(tk.W, tk.E))
        
        self.add_type_button = ttk.Button(add_button_frame, text="+ Add New Object Type", 
                                        command=self.add_new_object_type)
        self.add_type_button.pack()
        
    def load_competing_models(self):
        """Load the 4 best models for competition"""
        print("Loading competing models...")
        
        if not YOLO_AVAILABLE:
            messagebox.showerror("Error", "YOLO not available. Install with: pip install ultralytics")
            return
        
        # Get all available models
        manager = ModelManager()
        all_models = manager.get_all_model_versions()
        
        if len(all_models) == 0:
            messagebox.showwarning("Warning", "No trained models found!")
            return
        
        # Select top 4 models (you could add logic to pick diverse models)
        selected_models = all_models[:4]
        
        for name, path, creation_time in selected_models:
            try:
                print(f"Loading {name}...")
                model = YOLO(str(path))
                
                competitor = ModelCompetitor(
                    name=name,
                    model=model,
                    path=path
                )
                
                self.competitors.append(competitor)
                print(f"✓ Loaded {name}")
                
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
        
        print(f"Competition ready with {len(self.competitors)} models!")
        self.update_scoreboard()
        
    def update_scoreboard(self):
        """Update the scoreboard display"""
        for i, competitor in enumerate(self.competitors):
            if i < len(self.scoreboard_labels):
                labels = self.scoreboard_labels[i]
                labels[0].config(text=competitor.name.replace('pokemon_object_detector', 'Model'))
                labels[1].config(text=str(competitor.score))
                labels[2].config(text=f"{competitor.accuracy:.1%}")
                labels[3].config(text=f"{competitor.avg_confidence:.2f}")
                labels[4].config(text=competitor.last_prediction or "None")
        
        # Update session stats
        total_competitions = len(self.session_data['competitions'])
        self.total_competitions_label.config(text=f"Total Competitions: {total_competitions}")
        
        # Update training data count
        training_examples = len(self.session_data['training_data'])
        self.training_data_label.config(text=f"Training Examples: {training_examples}")
        
        # Update session time
        session_time = datetime.now() - self.session_data['start_time']
        minutes = int(session_time.total_seconds() // 60)
        seconds = int(session_time.total_seconds() % 60)
        self.session_time_label.config(text=f"Session Time: {minutes}:{seconds:02d}")
        
        
    def display_image(self, image):
        """Display image on canvas"""
        # Convert BGR to RGB for Tkinter
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Canvas is initialized
            image_pil = image_pil.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and display
        self.photo = ImageTk.PhotoImage(image_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
    def start_drawing(self, event):
        """Start drawing a bounding box"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please capture a screen first!")
            return
            
        self.drawing = True
        self.start_pos = (event.x, event.y)
        self.current_bbox = None
        
    def update_drawing(self, event):
        """Update the bounding box while drawing"""
        if not self.drawing or self.start_pos is None:
            return
            
        # Clear previous rectangle
        self.canvas.delete("bbox")
        
        # Draw new rectangle
        self.canvas.create_rectangle(
            self.start_pos[0], self.start_pos[1], 
            event.x, event.y,
            outline="red", width=2, tags="bbox"
        )
        
    def finish_drawing(self, event):
        """Finish drawing and start model competition"""
        if not self.drawing or self.start_pos is None:
            return
            
        self.drawing = False
        
        # Store bounding box coordinates
        x1, y1 = self.start_pos
        x2, y2 = event.x, event.y
        
        # Normalize coordinates to image space
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if self.current_image is not None:
            img_height, img_width = self.current_image.shape[:2]
            
            # Convert canvas coordinates to image coordinates
            x1 = int(x1 * img_width / canvas_width)
            y1 = int(y1 * img_height / canvas_height)
            x2 = int(x2 * img_width / canvas_width)
            y2 = int(y2 * img_height / canvas_height)
            
            # Ensure proper bounding box format
            self.current_bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            
            # Run model competition (but don't save screenshot yet)
            self.run_model_competition()
            
    def run_model_competition(self):
        """Run all models on the drawn bounding box"""
        if self.current_bbox is None or self.current_image is None:
            return
            
        self.status_label.config(text="🏁 Models competing...")
        
        # Extract the cropped region
        x1, y1, x2, y2 = self.current_bbox
        cropped_image = self.current_image[y1:y2, x1:x2]
        
        if cropped_image.size == 0:
            messagebox.showwarning("Warning", "Bounding box too small!")
            return
        
        print(f"Running competition on bbox: {self.current_bbox}")
        print(f"Original image shape: {self.current_image.shape}")
        print(f"Cropped region shape: {cropped_image.shape}")
        
        # Try both approaches: full image analysis and cropped analysis
        use_full_image = True  # Set to True to run on full image, False for cropped
        
        # Run each model 
        predictions = []
        
        for i, competitor in enumerate(self.competitors):
            try:
                if use_full_image:
                    print(f"Running inference for {competitor.name} on full image {self.current_image.shape}")
                    # Run YOLO inference on full image with very low confidence threshold
                    results = competitor.model(self.current_image, verbose=False, conf=0.01)
                    inference_image = self.current_image
                else:
                    print(f"Running inference for {competitor.name} on cropped region {cropped_image.shape}")
                    # Run YOLO inference on cropped region with very low confidence threshold
                    results = competitor.model(cropped_image, verbose=False, conf=0.01)
                    inference_image = cropped_image
                
                print(f"Results for {competitor.name}: {len(results)} result objects")
                
                # Get best prediction
                best_prediction = None
                best_confidence = 0.0
                
                for result in results:
                    print(f"Result boxes: {result.boxes is not None and len(result.boxes) if result.boxes is not None else 'None'}")
                    if result.boxes is not None and len(result.boxes) > 0:
                        # Get detections
                        confidences = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                        
                        print(f"Found {len(confidences)} detections with confidences: {confidences}")
                        print(f"Classes: {classes}")
                        print(f"Model class names: {list(competitor.model.names.values())}")
                        
                        if use_full_image:
                            # Find detection that overlaps most with our drawn bounding box
                            best_overlap = 0
                            best_idx = -1
                            
                            for idx, box in enumerate(boxes):
                                # Calculate intersection over union with our bounding box
                                overlap = self.calculate_bbox_overlap(self.current_bbox, box)
                                print(f"Detection {idx}: bbox={box}, overlap={overlap:.3f}, conf={confidences[idx]:.3f}")
                                
                                if overlap > best_overlap and overlap > 0.05:  # Minimum 5% overlap
                                    best_overlap = overlap
                                    best_idx = idx
                            
                            if best_idx >= 0:
                                best_confidence = float(confidences[best_idx])
                                class_id = int(classes[best_idx])
                                best_prediction = competitor.model.names[class_id]
                                print(f"Best overlapping detection: {best_prediction} with confidence {best_confidence} and overlap {best_overlap:.3f}")
                            else:
                                print(f"No detections overlap with drawn bounding box")
                        else:
                            # Use most confident detection from cropped image
                            max_idx = np.argmax(confidences)
                            best_confidence = float(confidences[max_idx])
                            class_id = int(classes[max_idx])
                            best_prediction = competitor.model.names[class_id]
                            print(f"Best detection: {best_prediction} with confidence {best_confidence}")
                    else:
                        print(f"No detections found for {competitor.name}")
                
                # Store prediction
                competitor.last_prediction = best_prediction or "no_detection"
                competitor.last_confidence = best_confidence
                
                # Update prediction display
                pred_text = f"{competitor.name.replace('pokemon_object_detector', 'Model')}: "
                if best_prediction:
                    pred_text += f"{best_prediction} ({best_confidence:.2f})"
                else:
                    pred_text += "No detection"
                    
                self.prediction_labels[i].config(text=pred_text)
                
                predictions.append((competitor, best_prediction, best_confidence))
                
            except Exception as e:
                print(f"Error running {competitor.name}: {e}")
                competitor.last_prediction = "error"
                competitor.last_confidence = 0.0
                self.prediction_labels[i].config(text=f"{competitor.name}: Error")
        
        # Enable class buttons for user feedback
        self.enable_class_buttons(True)
        self.status_label.config(text="👆 Click correct object type to save & train")
    
    def calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate intersection over union between two bounding boxes"""
        # bbox1 is (x1, y1, x2, y2) from our drawing
        # bbox2 is [x1, y1, x2, y2] from YOLO detection
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def save_training_data(self, correct_class):
        """Save screenshot and create training data for later model training.
        
        This is called ONLY when the user clicks an object type button to confirm
        the correct answer, ensuring high-quality training data without accidents.
        """
        if self.current_image is None or self.current_bbox is None:
            return None
        
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_filename = f"training_{timestamp}.jpg"
            label_filename = f"training_{timestamp}.txt"
            
            image_path = self.training_data_dir / "images" / image_filename
            label_path = self.training_data_dir / "labels" / label_filename
            
            # Save the screenshot
            cv2.imwrite(str(image_path), self.current_image)
            
            # Convert bounding box to YOLO format
            img_height, img_width = self.current_image.shape[:2]
            x1, y1, x2, y2 = self.current_bbox
            
            # YOLO format: class_id center_x center_y width height (all normalized)
            center_x = (x1 + x2) / 2 / img_width
            center_y = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # ADAPTIVE SYSTEM: Map new classes to base model classes to maintain compatibility
            # This ensures competitive models use the same architecture as base models
            # The adaptive system will handle the reverse mapping during inference
            class_mapping = {
                'pokeball/item': 'pokemon_sprite',  # Map pokeball to pokemon_sprite
                'exp._bar': 'hp_bar',               # Map exp bar to hp_bar
                'level_indicator': 'text_area',     # Map level indicator to text_area
                'grass': 'tree',                    # Map grass to tree (natural objects)
                'water': 'tree',                    # Map water to tree (natural objects)
            }
            
            # Apply mapping
            mapped_class = class_mapping.get(correct_class, correct_class)
            
            # Get class ID for base model classes
            base_classes = ['building', 'dialogue_box', 'hp_bar', 'menu_box', 'npc_character', 
                          'player_character', 'pokemon_sprite', 'text_area', 'tree']
            
            try:
                class_id = base_classes.index(mapped_class)
            except ValueError:
                print(f"Warning: Class '{mapped_class}' not in base classes, using 0")
                class_id = 0
            
            # Save label file
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            # Store training data info
            training_entry = {
                'image_file': image_filename,
                'label_file': label_filename,
                'class': correct_class,
                'bbox': self.current_bbox,
                'timestamp': timestamp
            }
            
            self.session_data['training_data'].append(training_entry)
            
            print(f"Saved training data: {image_filename} -> {correct_class}")
            return training_entry
            
        except Exception as e:
            print(f"Error saving training data: {e}")
            return None
        
    def submit_correct_answer(self, correct_class):
        """Process user's correct answer and update model scores"""
        if self.current_bbox is None:
            return
            
        self.enable_class_buttons(False)
        
        # Save screenshot and create training data ONLY when user confirms the correct answer
        # This prevents bad training data from accidental draws or mis-clicks
        training_data_entry = self.save_training_data(correct_class)
        
        # Score each model
        competition_result = {
            'timestamp': datetime.now(),
            'bounding_box': self.current_bbox,
            'correct_answer': correct_class,
            'predictions': [],
            'screenshot_file': training_data_entry['image_file'] if training_data_entry else None
        }
        
        for competitor in self.competitors:
            # Update statistics
            competitor.total_predictions += 1
            competitor.confidence_sum += competitor.last_confidence
            
            # Check if prediction was correct
            was_correct = competitor.last_prediction == correct_class
            
            if was_correct:
                competitor.correct_predictions += 1
                competitor.score += 10  # Reward for correct prediction
                result = "✓ CORRECT"
            else:
                competitor.score -= 5   # Penalty for wrong prediction
                result = "✗ WRONG"
            
            # Store competition result
            competition_result['predictions'].append({
                'model': competitor.name,
                'prediction': competitor.last_prediction,
                'confidence': competitor.last_confidence,
                'correct': was_correct,
                'score_change': 10 if was_correct else -5
            })
            
            print(f"{competitor.name}: {result} - {competitor.last_prediction} (conf: {competitor.last_confidence:.2f})")
        
        # Save competition result
        self.session_data['competitions'].append(competition_result)
        
        # Update display
        self.update_scoreboard()
        self.status_label.config(text="✅ Training data saved! Draw another box.")
        
        # Clear predictions
        for label in self.prediction_labels:
            label.config(text="Ready for next competition")
        
        # Clear bounding box
        self.canvas.delete("bbox")
        self.current_bbox = None
        
    def enable_class_buttons(self, enabled):
        """Enable or disable class selection buttons"""
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in self.class_buttons:
            button.config(state=state)
            
    def reset_scores(self):
        """Reset all model scores"""
        if messagebox.askyesno("Reset Scores", "Reset all model scores to zero?"):
            for competitor in self.competitors:
                competitor.score = 0
                competitor.correct_predictions = 0
                competitor.total_predictions = 0
                competitor.confidence_sum = 0.0
                competitor.last_prediction = None
                competitor.last_confidence = 0.0
            
            self.session_data['competitions'] = []
            self.session_data['start_time'] = datetime.now()
            
            self.update_scoreboard()
            
    def save_session(self):
        """Save the competition session data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"competition_session_{timestamp}.json"
        
        # Prepare session data for JSON serialization
        session_export = {
            'start_time': self.session_data['start_time'].isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_competitions': len(self.session_data['competitions']),
            'model_performances': {},
            'competitions': []
        }
        
        # Add model performance summaries
        for competitor in self.competitors:
            session_export['model_performances'][competitor.name] = {
                'final_score': competitor.score,
                'accuracy': competitor.accuracy,
                'total_predictions': competitor.total_predictions,
                'correct_predictions': competitor.correct_predictions,
                'average_confidence': competitor.avg_confidence
            }
        
        # Add competition details (convert datetime objects)
        for comp in self.session_data['competitions']:
            comp_export = comp.copy()
            comp_export['timestamp'] = comp['timestamp'].isoformat()
            session_export['competitions'].append(comp_export)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(session_export, f, indent=2)
        
        messagebox.showinfo("Session Saved", f"Competition session saved to {filename}")
    
    def refresh_available_models(self):
        """Refresh the list of available models for dropdowns"""
        manager = ModelManager()
        self.available_models = manager.get_all_model_versions()
        
        # Update dropdown options
        model_names = ["None"] + [name for name, path, time in self.available_models]
        
        for i, dropdown in enumerate(self.model_dropdowns):
            dropdown['values'] = model_names
            if i < len(model_names) - 1:  # Auto-select first few models
                dropdown.set(model_names[i + 1] if i + 1 < len(model_names) else "None")
            else:
                dropdown.set("None")
    
    def load_selected_models(self):
        """Load the individually selected models for competition"""
        print("Loading selected models...")
        
        if not YOLO_AVAILABLE:
            messagebox.showerror("Error", "YOLO not available. Install with: pip install ultralytics")
            return
        
        # Clear existing competitors
        self.competitors = []
        
        # Load each selected model
        for i, model_var in enumerate(self.model_vars):
            selected_name = model_var.get()
            if selected_name and selected_name != "None":
                # Find the model path
                model_path = None
                for name, path, time in self.available_models:
                    if name == selected_name:
                        model_path = path
                        break
                
                if model_path:
                    try:
                        print(f"Loading {selected_name}...")
                        model = YOLO(str(model_path))
                        
                        competitor = ModelCompetitor(
                            name=selected_name,
                            model=model,
                            path=model_path
                        )
                        
                        self.competitors.append(competitor)
                        print(f"✓ Loaded {selected_name}")
                        
                    except Exception as e:
                        print(f"✗ Failed to load {selected_name}: {e}")
                        messagebox.showerror("Load Error", f"Failed to load {selected_name}: {e}")
        
        print(f"Competition ready with {len(self.competitors)} models!")
        self.update_scoreboard()
        
        if len(self.competitors) == 0:
            messagebox.showwarning("Warning", "No models loaded! Please select at least one model.")
    
    def add_new_object_type(self):
        """Add a new object type dynamically"""
        new_type = simpledialog.askstring("Add Object Type", "Enter new object type name:")
        
        if new_type and new_type.strip():
            # Clean the name
            new_type = new_type.strip().lower().replace(' ', '_')
            
            if new_type not in self.object_classes:
                # Add to object classes
                self.object_classes.append(new_type)
                
                # Save persistent data
                self.save_persistent_data()
                
                # Rebuild the class buttons
                self.create_class_buttons()
                
                # Keep buttons disabled until a bounding box is drawn
                self.enable_class_buttons(False)
                
                messagebox.showinfo("Success", f"Added new object type: {new_type.replace('_', ' ').title()}")
                print(f"Added new object type: {new_type}")
            else:
                messagebox.showwarning("Duplicate", f"Object type '{new_type}' already exists!")
        elif new_type is not None:  # User clicked OK but entered empty string
            messagebox.showwarning("Invalid", "Please enter a valid object type name!")
    
    def create_training_dataset(self):
        """Create YOLO dataset from collected training data"""
        if len(self.session_data['training_data']) == 0:
            print("No training data collected")
            return None
        
        # Get the top performing model to match its class structure
        top_models = self.get_top_models(1)
        if len(top_models) > 0:
            try:
                # Load the best model to get its class names
                from ultralytics import YOLO
                best_model = YOLO(str(top_models[0].path))
                base_classes = list(best_model.names.values())
                print(f"Using base model class structure: {base_classes}")
            except Exception as e:
                print(f"Warning: Could not load base model classes: {e}")
                print("Using shared class structure")
                base_classes = self.object_classes  # Use shared classes (16 classes)
        else:
            # Fallback to shared classes
            base_classes = self.object_classes  # Use shared classes (16 classes)
        
        # Create data.yaml file with base model classes (ADAPTIVE APPROACH)
        # Use the base model's classes to ensure architecture compatibility
        yaml_content = f"""# Competitive training dataset from session {self.session_data['start_time']}
# ADAPTIVE SYSTEM: Using base model class structure to maintain compatibility
# This ensures that new competitive models can be used with the adaptive system
path: {self.training_data_dir.absolute()}
train: images
val: images  # Using same data for validation

nc: {len(base_classes)}
names: {base_classes}
"""
        
        yaml_path = self.training_data_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content.strip())
        
        print(f"Created training dataset with {len(self.session_data['training_data'])} images")
        return yaml_path
    
    def get_top_models(self, n=2):
        """Get the top N performing models from this session"""
        if len(self.competitors) == 0:
            return []
        
        # Sort competitors by accuracy, then by score
        sorted_competitors = sorted(
            self.competitors, 
            key=lambda x: (x.accuracy, x.score), 
            reverse=True
        )
        
        return sorted_competitors[:n]
    
    def train_top_models(self):
        """Train the top 2 performing models on the session data"""
        if not YOLO_AVAILABLE:
            messagebox.showerror("Error", "YOLO not available for training")
            return False
            
        # Check if already training
        if hasattr(self, 'training_thread') and self.training_thread.is_alive():
            messagebox.showwarning("Warning", "Training already in progress!")
            return False
        
        # Create dataset
        yaml_path = self.create_training_dataset()
        if yaml_path is None:
            messagebox.showwarning("Warning", "No training data to train on!")
            return False
        
        # Get top models
        top_models = self.get_top_models(2)
        if len(top_models) == 0:
            messagebox.showwarning("Warning", "No models to train!")
            return False
        
        # Start training in separate thread
        self.training_thread = threading.Thread(
            target=self._train_models_thread, 
            args=(top_models, yaml_path),
            daemon=True
        )
        self.training_thread.start()
        
        # Update UI to show training status
        self.train_button.config(text="🔄 Training... (Click to stop)", command=self.stop_training)
        return True
    
    def _train_models_thread(self, top_models, yaml_path):
        """Background thread for training models with early stopping"""
        self.training_active = True
        
        print(f"Training top {len(top_models)} models on {len(self.session_data['training_data'])} examples...")
        
        success_count = 0
        for i, competitor in enumerate(top_models):
            if not self.training_active:  # Check if user stopped training
                print("Training stopped by user")
                break
                
            try:
                print(f"Training {competitor.name}...")
                
                # Update status in main thread
                self.root.after(0, lambda: self.status_label.config(
                    text=f"🔄 Training model {i+1}/{len(top_models)}: {competitor.name}"
                ))
                
                # Create new model name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_model_name = f"{competitor.name}_competitive_{timestamp}"
                
                # Launch visualization popup if enabled
                training_dir = Path("models") / new_model_name
                if self.show_viz_var.get() and i == 0:  # Only show for first model
                    self.root.after(0, lambda: self._launch_viz_popup(training_dir))
                
                # Start training from the existing model
                model = YOLO(str(competitor.path))
                
                # Determine device and batch size based on GPU setting
                device = '0' if self.use_gpu_var.get() and self.detect_gpu_available() else 'cpu'
                batch_size = 8 if device != 'cpu' else 4  # Larger batch for GPU
                
                # IMPROVED TRAINING: Conservative parameters to prevent degradation
                data_size = len(self.session_data['training_data'])
                
                # Adaptive training configuration based on data size
                if data_size < 20:
                    # Small dataset - use proven successful parameters
                    epochs, lr0, patience = 50, 0.01, 10
                    strategy = "standard_training_small"
                elif data_size < 50:
                    # Medium dataset - full training like original trainer
                    epochs, lr0, patience = 75, 0.01, 10
                    strategy = "standard_training_medium"  
                else:
                    # Larger dataset - extended training
                    epochs, lr0, patience = 100, 0.01, 10
                    strategy = "standard_training_large"
                
                print(f"Training strategy: {strategy} (data_size={data_size}, lr={lr0}, epochs={epochs})")
                
                results = model.train(
                    data=str(yaml_path),
                    epochs=epochs,  # Adaptive epochs based on data size
                    imgsz=640,
                    batch=min(batch_size, 4),  # Smaller batch for stability
                    save=True,
                    project="models",
                    name=new_model_name,
                    exist_ok=True,
                    pretrained=True,
                    optimizer='auto',  # Use default optimizer like successful models
                    lr0=lr0,  # Standard learning rate
                    lrf=0.1,  # Higher final learning rate
                    plots=False,
                    verbose=True,  # Enable verbose for debugging
                    device=device,
                    # Conservative regularization
                    patience=patience,  # Adaptive patience
                    save_period=max(epochs//5, 3),  # Save less frequently
                    close_mosaic=max(epochs//3, 5),  # Close mosaic later
                    # Reduced augmentation for small datasets
                    hsv_h=0.01,  # Minimal HSV augmentation
                    hsv_s=0.3,
                    hsv_v=0.2,
                    degrees=0.0,  # No rotation
                    translate=0.05,  # Minimal translation
                    scale=0.2,  # Minimal scaling
                    shear=0.0,  # No shear
                    perspective=0.0,  # No perspective
                    flipud=0.0,  # No vertical flip
                    fliplr=0.3,  # Minimal horizontal flip
                    mosaic=0.3,  # Reduced mosaic
                    mixup=0.0,  # No mixup for small datasets
                    copy_paste=0.0,  # No copy-paste
                    # Regularization
                    dropout=0.0,
                    weight_decay=0.0005,
                    warmup_epochs=2,
                    workers=1,  # Single worker for stability
                    single_cls=False,
                )
                
                print(f"✓ Successfully trained {new_model_name}")
                success_count += 1
                
            except Exception as e:
                print(f"✗ Failed to train {competitor.name}: {e}")
                if not self.training_active:
                    print("Training was interrupted")
                    break
        
        # Update UI in main thread when training completes
        self.root.after(0, self._training_complete, success_count, len(top_models))
    
    def _training_complete(self, success_count, total_models):
        """Handle training completion in main thread"""
        self.training_active = False
        self.train_button.config(text="🎯 Train Top Models", command=self.train_top_models)
        
        if success_count > 0:
            self.status_label.config(text=f"✅ Training complete: {success_count}/{total_models} models")
            messagebox.showinfo("Training Complete", f"Successfully trained {success_count} models!")
        else:
            self.status_label.config(text="❌ Training failed")
            messagebox.showerror("Training Failed", "Failed to train any models")
    
    def stop_training(self):
        """Stop training process"""
        if hasattr(self, 'training_active'):
            self.training_active = False
            self.status_label.config(text="🛑 Stopping training...")
            self.train_button.config(text="🎯 Train Top Models", command=self.train_top_models)
            print("Training stop requested by user")
    
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
    
    def _launch_viz_popup(self, training_dir):
        """Launch training visualization popup"""
        try:
            self.training_popup = show_training_popup(training_dir, self.root)
        except Exception as e:
            print(f"Failed to launch training visualization: {e}")
    
    def _on_classes_changed(self, new_classes):
        """Callback when classes are updated externally"""
        print(f"[CompTrainer] Class list updated: {len(new_classes)} classes")
        self.object_classes = new_classes
        
        # Update UI if class buttons exist
        if hasattr(self, 'class_buttons'):
            self.root.after(0, self.create_class_buttons)
    
    def on_closing(self):
        """Handle window closing with training prompt"""
        if len(self.session_data['training_data']) > 0:
            # Ask user about training
            result = messagebox.askyesnocancel(
                "Training Available", 
                f"You have {len(self.session_data['training_data'])} training examples collected.\n\n"
                "Do you want to train the top models before quitting?\n\n"
                "• Yes: Train models and quit\n"
                "• No: Quit without training\n"
                "• Cancel: Return to trainer"
            )
            
            if result is None:  # Cancel
                return
            elif result:  # Yes - train first
                print("Training models before exit...")
                training_success = self.train_top_models()
                if training_success:
                    print("Training completed successfully!")
                else:
                    print("Training had issues, but continuing with exit...")
        
        # Save session data before closing
        self.save_session()
        
        # Cleanup shared class manager
        cleanup_shared_manager()
        
        # Close the application
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """Start the competitive training application"""
        print("🏆 Starting competitive model trainer")
        print("Select models using the dropdowns and click 'Load Selected Models'")
        
        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.root.mainloop()

if __name__ == "__main__":
    if not YOLO_AVAILABLE:
        print("Please install YOLO: pip install ultralytics")
        exit(1)
    
    app = CompetitiveModelTrainer()
    app.run()