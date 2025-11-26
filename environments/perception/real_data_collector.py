#!/usr/bin/env python3
"""
🎮 REAL GAME DATA COLLECTOR
===========================

Interactive tool for collecting and labeling real game screenshots.
Features:
- Screenshot capture from emulator window
- Interactive bounding box labeling
- Automatic data augmentation
- Progress tracking and validation
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

class RealDataCollector:
    """Interactive tool for collecting real game data"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pokemon Game Data Collector")
        self.root.geometry("1200x800")
        
        # Data storage
        self.collected_screenshots = []
        self.current_screenshot = None
        self.current_labels = []
        self.emulator_window = None
        
        # Labeling state
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_label_type = "hp_bar"
        
        # Directories
        self.data_dir = Path("real_training_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Object types for Pokemon games
        self.object_types = [
            "hp_bar", "player_character", "pokemon_sprite", "npc_character", 
            "menu_box", "dialogue_box", "item_icon", "building", "tree",
            "battle_menu", "text_area", "status_icon", "map_element"
        ]
        
        self.setup_gui()
        self.setup_directories()
        
    def setup_directories(self):
        """Setup directory structure for collected data"""
        subdirs = ["screenshots", "labeled", "augmented", "metadata"]
        for subdir in subdirs:
            (self.data_dir / subdir).mkdir(exist_ok=True)
            
    def setup_gui(self):
        """Setup the main GUI interface"""
        # Main frame layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        self.setup_control_panel(main_frame)
        
        # Right panel - Image display and labeling
        self.setup_image_panel(main_frame)
        
        # Bottom panel - Progress and status
        self.setup_status_panel(main_frame)
        
    def setup_control_panel(self, parent):
        """Setup control panel with capture and labeling options"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Emulator detection
        ttk.Label(control_frame, text="Step 1: Find Emulator", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self.emulator_var = tk.StringVar(value="No emulator found")
        ttk.Label(control_frame, textvariable=self.emulator_var, wraplength=200).pack(anchor=tk.W)
        
        ttk.Button(control_frame, text="Find mGBA Window", 
                  command=self.find_emulator_window).pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Screenshot capture
        ttk.Label(control_frame, text="Step 2: Capture Screenshots", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(control_frame, text="Capture Screenshot", 
                  command=self.capture_screenshot).pack(fill=tk.X, pady=2)
        
        ttk.Button(control_frame, text="Auto Capture (5s)", 
                  command=self.start_auto_capture).pack(fill=tk.X, pady=2)
        
        self.auto_capture_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Continuous capture", 
                       variable=self.auto_capture_var).pack(anchor=tk.W, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Labeling controls
        ttk.Label(control_frame, text="Step 3: Label Objects", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(control_frame, text="Object Type:").pack(anchor=tk.W)
        self.label_type_var = tk.StringVar(value=self.object_types[0])
        label_combo = ttk.Combobox(control_frame, textvariable=self.label_type_var, 
                                  values=self.object_types, state="readonly", width=18)
        label_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(control_frame, text="Instructions:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        instructions = tk.Text(control_frame, height=6, width=25, wrap=tk.WORD, font=("Arial", 9))
        instructions.pack(fill=tk.X, pady=2)
        instructions.insert(tk.END, 
            "1. Select object type above\n"
            "2. Click and drag on image to draw box\n"
            "3. Right-click to delete last box\n"
            "4. Press 'S' to save labels\n"
            "5. Press 'N' for next screenshot")
        instructions.config(state=tk.DISABLED)
        
        ttk.Button(control_frame, text="Save Current Labels", 
                  command=self.save_current_labels).pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Clear All Labels", 
                  command=self.clear_labels).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Data management
        ttk.Label(control_frame, text="Data Management", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(control_frame, text="Load Screenshots", 
                  command=self.load_screenshots).pack(fill=tk.X, pady=2)
        
        ttk.Button(control_frame, text="Generate Augmented Data", 
                  command=self.generate_augmented_data).pack(fill=tk.X, pady=2)
        
        ttk.Button(control_frame, text="View Collection Stats", 
                  command=self.show_collection_stats).pack(fill=tk.X, pady=2)
        
    def setup_image_panel(self, parent):
        """Setup image display and labeling panel"""
        image_frame = ttk.LabelFrame(parent, text="Screenshot Labeling", padding=10)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        self.canvas = tk.Canvas(image_frame, bg='black', width=600, height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for labeling
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing)
        self.canvas.bind("<Button-3>", self.delete_last_label)  # Right click
        
        # Bind keyboard events
        self.canvas.bind("<Key>", self.handle_keypress)
        self.canvas.focus_set()
        
    def setup_status_panel(self, parent):
        """Setup status and progress panel"""
        status_frame = ttk.LabelFrame(self.root, text="Status & Progress", padding=10)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
        
        # Progress information
        progress_info = ttk.Frame(status_frame)
        progress_info.pack(fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready to capture screenshots")
        ttk.Label(progress_info, textvariable=self.status_var).pack(side=tk.LEFT)
        
        self.progress_var = tk.StringVar(value="Screenshots: 0 | Labeled: 0")
        ttk.Label(progress_info, textvariable=self.progress_var).pack(side=tk.RIGHT)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(status_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
    def find_emulator_window(self):
        """Find and select the mGBA emulator window"""
        windows = []
        
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if window_text and ('mgba' in window_text.lower() or 'pokemon' in window_text.lower() or 'gba' in window_text.lower()):
                    windows.append((hwnd, window_text))
        
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        if not windows:
            messagebox.showwarning("No Emulator Found", 
                                 "No mGBA or GBA emulator window found.\n"
                                 "Make sure your emulator is running and visible.")
            return
            
        if len(windows) == 1:
            self.emulator_window = windows[0][0]
            self.emulator_var.set(f"Found: {windows[0][1]}")
            self.status_var.set("Emulator window detected! Ready to capture.")
        else:
            # Multiple windows found, let user choose
            self.choose_emulator_window(windows)
            
    def choose_emulator_window(self, windows):
        """Let user choose from multiple detected windows"""
        choice_window = tk.Toplevel(self.root)
        choice_window.title("Choose Emulator Window")
        choice_window.geometry("400x300")
        
        ttk.Label(choice_window, text="Multiple windows found. Choose one:", 
                 font=("Arial", 12)).pack(pady=10)
        
        listbox = tk.Listbox(choice_window, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for hwnd, title in windows:
            listbox.insert(tk.END, title)
            
        def select_window():
            selection = listbox.curselection()
            if selection:
                self.emulator_window = windows[selection[0]][0]
                self.emulator_var.set(f"Selected: {windows[selection[0]][1]}")
                self.status_var.set("Emulator window selected! Ready to capture.")
                choice_window.destroy()
                
        ttk.Button(choice_window, text="Select", command=select_window).pack(pady=10)
        
    def capture_screenshot(self):
        """Capture screenshot from the emulator window"""
        if not self.emulator_window:
            messagebox.showwarning("No Emulator", "Please find the emulator window first!")
            return
            
        try:
            # Get window position and size
            rect = win32gui.GetWindowRect(self.emulator_window)
            
            # Capture the window
            screenshot = ImageGrab.grab(bbox=rect)
            
            # Convert to OpenCV format
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Store the screenshot
            timestamp = int(time.time() * 1000)
            filename = f"screenshot_{timestamp}.png"
            filepath = self.data_dir / "screenshots" / filename
            
            cv2.imwrite(str(filepath), screenshot_cv)
            
            # Update current screenshot for labeling
            self.current_screenshot = screenshot_cv.copy()
            self.current_screenshot_path = filepath
            self.current_labels = []
            
            # Display in canvas
            self.display_screenshot(screenshot_cv)
            
            # Update progress
            self.collected_screenshots.append({
                'path': str(filepath),
                'timestamp': timestamp,
                'labeled': False
            })
            
            self.update_progress()
            self.status_var.set(f"Screenshot captured: {filename}")
            
        except Exception as e:
            messagebox.showerror("Capture Error", f"Failed to capture screenshot: {str(e)}")
            
    def display_screenshot(self, screenshot):
        """Display screenshot in the canvas for labeling"""
        # Resize image to fit canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:  # Canvas not initialized yet
            self.root.after(100, lambda: self.display_screenshot(screenshot))
            return
            
        img_height, img_width = screenshot.shape[:2]
        
        # Calculate scaling factor
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y) * 0.9  # Leave some margin
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized = cv2.resize(screenshot, (new_width, new_height))
        
        # Convert to PIL format for Tkinter
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        
        # Center the image
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo)
        
        # Store scaling info for coordinate transformation
        self.image_scale = scale
        self.image_offset = (x_offset, y_offset)
        self.image_size = (new_width, new_height)
        
    def start_drawing(self, event):
        """Start drawing bounding box"""
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
        
    def draw_rectangle(self, event):
        """Draw rectangle while dragging"""
        if self.drawing:
            # Remove previous temporary rectangle
            self.canvas.delete("temp_rect")
            
            # Draw current rectangle
            self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline="red", width=2, tags="temp_rect"
            )
            
    def end_drawing(self, event):
        """Finish drawing bounding box and add label"""
        if not self.drawing:
            return
            
        self.drawing = False
        
        # Calculate bounding box in original image coordinates
        x1, y1 = self.canvas_to_image_coords(self.start_x, self.start_y)
        x2, y2 = self.canvas_to_image_coords(event.x, event.y)
        
        # Ensure x1 < x2 and y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
            
        # Only add if box is big enough
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            label = {
                'type': self.label_type_var.get(),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': 1.0
            }
            
            self.current_labels.append(label)
            
            # Remove temporary rectangle
            self.canvas.delete("temp_rect")
            
            # Draw permanent rectangle with label
            self.draw_label_on_canvas(label, len(self.current_labels))
            
            self.status_var.set(f"Added {label['type']} label. Total labels: {len(self.current_labels)}")
        else:
            self.canvas.delete("temp_rect")
            
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to original image coordinates"""
        if not hasattr(self, 'image_scale'):
            return canvas_x, canvas_y
            
        # Account for image offset and scaling
        img_x = (canvas_x - self.image_offset[0]) / self.image_scale
        img_y = (canvas_y - self.image_offset[1]) / self.image_scale
        
        return img_x, img_y
        
    def draw_label_on_canvas(self, label, label_num):
        """Draw a labeled bounding box on the canvas"""
        bbox = label['bbox']
        label_type = label['type']
        
        # Convert image coordinates back to canvas coordinates
        x1 = bbox[0] * self.image_scale + self.image_offset[0]
        y1 = bbox[1] * self.image_scale + self.image_offset[1]
        x2 = bbox[2] * self.image_scale + self.image_offset[0]
        y2 = bbox[3] * self.image_scale + self.image_offset[1]
        
        # Choose color based on object type
        colors = {
            'hp_bar': 'green', 'player_character': 'blue', 'pokemon_sprite': 'red',
            'npc_character': 'orange', 'menu_box': 'yellow', 'dialogue_box': 'purple',
            'item_icon': 'cyan', 'building': 'brown', 'tree': 'lime',
            'battle_menu': 'pink', 'text_area': 'white', 'status_icon': 'gray',
            'map_element': 'navy'
        }
        color = colors.get(label_type, 'white')
        
        # Draw rectangle
        rect_tag = f"label_{label_num}"
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags=rect_tag)
        
        # Draw label text
        self.canvas.create_text(x1, y1 - 10, anchor=tk.SW, text=f"{label_num}: {label_type}", 
                               fill=color, font=("Arial", 8), tags=rect_tag)
        
    def delete_last_label(self, event):
        """Delete the most recent label (right-click)"""
        if self.current_labels:
            removed_label = self.current_labels.pop()
            
            # Remove visual elements
            self.canvas.delete(f"label_{len(self.current_labels) + 1}")
            
            self.status_var.set(f"Removed {removed_label['type']} label. Total labels: {len(self.current_labels)}")
            
    def handle_keypress(self, event):
        """Handle keyboard shortcuts"""
        key = event.keysym.lower()
        
        if key == 's':
            self.save_current_labels()
        elif key == 'n':
            self.capture_screenshot()
        elif key == 'c':
            self.clear_labels()
            
    def save_current_labels(self):
        """Save current labels to file"""
        if not self.current_labels or not hasattr(self, 'current_screenshot_path'):
            messagebox.showwarning("No Labels", "No labels to save or no screenshot loaded!")
            return
            
        # Create metadata
        metadata = {
            'image_path': str(self.current_screenshot_path),
            'timestamp': int(time.time()),
            'labels': self.current_labels,
            'image_size': self.current_screenshot.shape[:2] if self.current_screenshot is not None else [0, 0]
        }
        
        # Save metadata
        base_name = self.current_screenshot_path.stem
        metadata_path = self.data_dir / "metadata" / f"{base_name}_labels.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Copy labeled image
        labeled_path = self.data_dir / "labeled" / f"{base_name}_labeled.png"
        
        if self.current_screenshot is not None:
            # Draw labels on copy of image
            labeled_image = self.current_screenshot.copy()
            
            for i, label in enumerate(self.current_labels):
                bbox = label['bbox']
                cv2.rectangle(labeled_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(labeled_image, f"{i+1}: {label['type']}", 
                           (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                           
            cv2.imwrite(str(labeled_path), labeled_image)
            
        # Update progress
        for screenshot in self.collected_screenshots:
            if screenshot['path'] == str(self.current_screenshot_path):
                screenshot['labeled'] = True
                break
                
        self.update_progress()
        self.status_var.set(f"Saved {len(self.current_labels)} labels to {metadata_path.name}")
        
        # Clear current labels for next image
        self.current_labels = []
        self.canvas.delete("all")
        
    def clear_labels(self):
        """Clear all current labels"""
        self.current_labels = []
        
        # Remove all label visuals but keep the image
        for i in range(20):  # Clear up to 20 labels
            self.canvas.delete(f"label_{i}")
            
        if self.current_screenshot is not None:
            self.display_screenshot(self.current_screenshot)
            
        self.status_var.set("All labels cleared")
        
    def start_auto_capture(self):
        """Start automatic screenshot capture every 5 seconds"""
        if not self.emulator_window:
            messagebox.showwarning("No Emulator", "Please find the emulator window first!")
            return
            
        def auto_capture_loop():
            for i in range(5, 0, -1):
                if not self.auto_capture_var.get():
                    break
                self.status_var.set(f"Auto capture in {i} seconds...")
                time.sleep(1)
                
            if self.auto_capture_var.get():
                self.capture_screenshot()
                if self.auto_capture_var.get():  # Still enabled
                    threading.Timer(5.0, auto_capture_loop).start()
                else:
                    self.status_var.set("Auto capture stopped")
                    
        self.auto_capture_var.set(True)
        threading.Thread(target=auto_capture_loop, daemon=True).start()
        
    def load_screenshots(self):
        """Load existing screenshots from directory"""
        screenshot_dir = filedialog.askdirectory(title="Select Screenshot Directory")
        if not screenshot_dir:
            return
            
        screenshot_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            screenshot_files.extend(Path(screenshot_dir).glob(ext))
            
        if not screenshot_files:
            messagebox.showinfo("No Screenshots", "No image files found in selected directory")
            return
            
        # Copy screenshots to our data directory
        copied_count = 0
        for src_path in screenshot_files:
            dst_path = self.data_dir / "screenshots" / src_path.name
            
            if not dst_path.exists():
                import shutil
                shutil.copy2(src_path, dst_path)
                
                self.collected_screenshots.append({
                    'path': str(dst_path),
                    'timestamp': int(time.time()),
                    'labeled': False
                })
                copied_count += 1
                
        self.update_progress()
        messagebox.showinfo("Import Complete", f"Imported {copied_count} screenshots")
        
    def generate_augmented_data(self):
        """Generate augmented versions of labeled data"""
        labeled_files = list((self.data_dir / "metadata").glob("*_labels.json"))
        
        if not labeled_files:
            messagebox.showwarning("No Labeled Data", "No labeled data found to augment!")
            return
            
        augmentation_count = 0
        
        for metadata_file in labeled_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Load original image
            original_image = cv2.imread(metadata['image_path'])
            if original_image is None:
                continue
                
            base_name = Path(metadata['image_path']).stem
            
            # Apply different augmentations
            augmentations = [
                ('bright', self.adjust_brightness, [1.2]),
                ('dark', self.adjust_brightness, [0.8]),
                ('contrast', self.adjust_contrast, [1.3]),
                ('blur', self.apply_blur, [1])
            ]
            
            for aug_name, aug_func, aug_params in augmentations:
                try:
                    augmented_image = aug_func(original_image, *aug_params)
                    
                    # Save augmented image
                    aug_path = self.data_dir / "augmented" / f"{base_name}_{aug_name}.png"
                    cv2.imwrite(str(aug_path), augmented_image)
                    
                    # Save augmented metadata
                    aug_metadata = metadata.copy()
                    aug_metadata['image_path'] = str(aug_path)
                    aug_metadata['augmentation'] = aug_name
                    
                    aug_metadata_path = self.data_dir / "metadata" / f"{base_name}_{aug_name}_labels.json"
                    with open(aug_metadata_path, 'w') as f:
                        json.dump(aug_metadata, f, indent=2)
                        
                    augmentation_count += 1
                    
                except Exception as e:
                    print(f"Augmentation failed for {base_name}: {e}")
                    
        messagebox.showinfo("Augmentation Complete", 
                           f"Generated {augmentation_count} augmented images")
        
    def adjust_brightness(self, image, factor):
        """Adjust image brightness"""
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
        
    def adjust_contrast(self, image, factor):
        """Adjust image contrast"""
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
        
    def apply_blur(self, image, kernel_size):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (kernel_size*2+1, kernel_size*2+1), 0)
        
    def update_progress(self):
        """Update progress information"""
        total_screenshots = len(self.collected_screenshots)
        labeled_count = sum(1 for s in self.collected_screenshots if s['labeled'])
        
        self.progress_var.set(f"Screenshots: {total_screenshots} | Labeled: {labeled_count}")
        
        if total_screenshots > 0:
            progress_percent = (labeled_count / total_screenshots) * 100
            self.progress_bar['value'] = progress_percent
            
    def show_collection_stats(self):
        """Show detailed collection statistics"""
        total_screenshots = len(self.collected_screenshots)
        labeled_count = sum(1 for s in self.collected_screenshots if s['labeled'])
        
        # Count labels by type
        label_counts = {}
        metadata_files = list((self.data_dir / "metadata").glob("*_labels.json"))
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                for label in metadata.get('labels', []):
                    label_type = label['type']
                    label_counts[label_type] = label_counts.get(label_type, 0) + 1
                    
            except Exception as e:
                print(f"Error reading {metadata_file}: {e}")
                
        # Show stats window
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Collection Statistics")
        stats_window.geometry("400x500")
        
        stats_text = tk.Text(stats_window, font=("Courier", 10))
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        stats_content = "POKEMON DATA COLLECTION STATISTICS\n"
        stats_content += "=" * 40 + "\n\n"
        stats_content += f"Total Screenshots: {total_screenshots}\n"
        stats_content += f"Labeled Screenshots: {labeled_count}\n"
        stats_content += f"Completion Rate: {(labeled_count/total_screenshots*100) if total_screenshots > 0 else 0:.1f}%\n\n"
        stats_content += "OBJECT LABELS BY TYPE:\n"
        stats_content += "-" * 25 + "\n"
        
        for obj_type, count in sorted(label_counts.items()):
            stats_content += f"{obj_type:18}: {count:3d}\n"
            
        total_labels = sum(label_counts.values())
        stats_content += "-" * 25 + "\n"
        stats_content += f"{'TOTAL LABELS':18}: {total_labels:3d}\n\n"
        
        stats_content += "RECOMMENDATIONS:\n"
        stats_content += "-" * 15 + "\n"
        
        if total_screenshots < 50:
            stats_content += "• Collect more screenshots (target: 50-100)\n"
        if labeled_count / total_screenshots < 0.8 if total_screenshots > 0 else True:
            stats_content += "• Label more screenshots\n"
        if max(label_counts.values()) if label_counts else 0 < 20:
            stats_content += "• Need more examples of each object type\n"
            
        stats_text.insert(tk.END, stats_content)
        stats_text.config(state=tk.DISABLED)
        
    def run(self):
        """Run the data collector GUI"""
        print("Pokemon Game Data Collector")
        print("=" * 40)
        print("Features:")
        print("- Automatic emulator window detection")
        print("- Interactive screenshot capture")  
        print("- Click-and-drag object labeling")
        print("- Automatic data augmentation")
        print("- Progress tracking and statistics")
        print("\nStarting GUI...")
        
        self.root.mainloop()

def main():
    """Main function"""
    collector = RealDataCollector()
    collector.run()

if __name__ == "__main__":
    main()