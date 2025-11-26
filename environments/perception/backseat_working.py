#!/usr/bin/env python3
"""Working backseat data collector - simplified version"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import json
import threading
import time
from pathlib import Path
from PIL import Image, ImageTk, ImageGrab
import win32gui
import win32ui
import win32con
from ctypes import windll

class WorkingBackseatCollector:
    """Working backseat data collector with minimal dependencies"""
    
    def __init__(self):
        print("[INIT] Creating working backseat collector...")
        
        # Create window
        self.root = tk.Tk()
        self.root.title("Working Backseat AI Trainer")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.current_screenshot = None
        self.emulator_window = None
        self.live_mode = False
        self.capture_thread = None
        
        # Data directory
        self.data_dir = Path("backseat_training_data")
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "screenshots").mkdir(exist_ok=True)
        
        # Create UI
        self.create_ui()
        
        print("[INIT] Working collector ready")
    
    def create_ui(self):
        """Create the user interface"""
        print("[UI] Creating interface...")
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - controls
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # Right panel - image display
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # === LEFT PANEL CONTROLS ===
        
        # Title
        title_label = ttk.Label(left_frame, text="Backseat AI Trainer", font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Window selection
        window_frame = ttk.LabelFrame(left_frame, text="Window Selection", padding=10)
        window_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(window_frame, text="Select Emulator Window", command=self.select_window).pack(fill=tk.X)
        
        self.window_label = ttk.Label(window_frame, text="No window selected", wraplength=250)
        self.window_label.pack(pady=(5, 0))
        
        # Capture controls
        capture_frame = ttk.LabelFrame(left_frame, text="Capture Controls", padding=10)
        capture_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(capture_frame, text="Take Screenshot", command=self.take_screenshot).pack(fill=tk.X, pady=(0, 5))
        
        self.live_button = ttk.Button(capture_frame, text="Start Live Mode", command=self.toggle_live_mode)
        self.live_button.pack(fill=tk.X, pady=(0, 5))
        
        # Save controls
        save_frame = ttk.LabelFrame(left_frame, text="Save Data", padding=10)
        save_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(save_frame, text="Save Screenshot", command=self.save_screenshot).pack(fill=tk.X)
        
        # Status
        status_frame = ttk.LabelFrame(left_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Ready", wraplength=250)
        self.status_label.pack()
        
        # === RIGHT PANEL - IMAGE DISPLAY ===
        
        # Image canvas
        canvas_frame = ttk.Frame(right_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='lightgray', width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = ttk.Label(right_frame, text="1. Select emulator window  2. Take screenshot or start live mode  3. Save data")
        instructions.pack(pady=10)
        
        print("[UI] Interface created")
    
    def select_window(self):
        """Select the emulator window"""
        print("[WINDOW] Selecting window...")
        
        def enum_windows_proc(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    windows.append((hwnd, title))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_proc, windows)
        
        # Create window selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Window")
        dialog.geometry("600x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Window list
        listbox = tk.Listbox(dialog)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for hwnd, title in windows:
            listbox.insert(tk.END, f"{title} (ID: {hwnd})")
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def select_window():
            selection = listbox.curselection()
            if selection:
                self.emulator_window = windows[selection[0]][0]
                title = windows[selection[0]][1]
                self.window_label.config(text=f"Selected: {title}")
                self.status_label.config(text="Window selected")
                dialog.destroy()
        
        ttk.Button(button_frame, text="Select", command=select_window).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
        
        dialog.center_on_parent()
    
    def take_screenshot(self):
        """Take a screenshot of the selected window"""
        if not self.emulator_window:
            messagebox.showwarning("Warning", "Please select a window first")
            return
        
        print("[CAPTURE] Taking screenshot...")
        
        try:
            # Get window dimensions
            rect = win32gui.GetWindowRect(self.emulator_window)
            x, y, x2, y2 = rect
            width, height = x2 - x, y2 - y
            
            # Capture window
            screenshot = ImageGrab.grab(bbox=(x, y, x2, y2))
            self.current_screenshot = screenshot
            
            # Display on canvas
            self.display_image(screenshot)
            
            self.status_label.config(text="Screenshot captured")
            print("[CAPTURE] Screenshot taken")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture screenshot: {e}")
            print(f"[ERROR] Screenshot failed: {e}")
    
    def display_image(self, image):
        """Display image on canvas"""
        if image is None:
            return
        
        # Resize image to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready yet
            self.root.after(100, lambda: self.display_image(image))
            return
        
        # Calculate scaling
        img_width, img_height = image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y, 1.0)  # Don't scale up
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize and display
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
        
        # Keep reference
        self.canvas.photo = photo
    
    def toggle_live_mode(self):
        """Toggle live capture mode"""
        if not self.emulator_window:
            messagebox.showwarning("Warning", "Please select a window first")
            return
        
        if self.live_mode:
            print("[LIVE] Stopping live mode...")
            self.live_mode = False
            self.live_button.config(text="Start Live Mode")
            self.status_label.config(text="Live mode stopped")
        else:
            print("[LIVE] Starting live mode...")
            self.live_mode = True
            self.live_button.config(text="Stop Live Mode")
            self.status_label.config(text="Live mode active")
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self.live_capture_loop, daemon=True)
            self.capture_thread.start()
    
    def live_capture_loop(self):
        """Live capture loop"""
        while self.live_mode:
            try:
                if self.emulator_window:
                    # Get window dimensions
                    rect = win32gui.GetWindowRect(self.emulator_window)
                    x, y, x2, y2 = rect
                    
                    # Capture
                    screenshot = ImageGrab.grab(bbox=(x, y, x2, y2))
                    self.current_screenshot = screenshot
                    
                    # Update display on main thread
                    self.root.after(0, lambda img=screenshot: self.display_image(img))
                    
                time.sleep(1/30)  # 30 FPS
                
            except Exception as e:
                print(f"[LIVE] Capture error: {e}")
                time.sleep(0.1)
    
    def save_screenshot(self):
        """Save current screenshot"""
        if self.current_screenshot is None:
            messagebox.showwarning("Warning", "No screenshot to save")
            return
        
        try:
            timestamp = int(time.time())
            filename = f"screenshot_{timestamp}.png"
            filepath = self.data_dir / "screenshots" / filename
            
            self.current_screenshot.save(filepath)
            
            self.status_label.config(text=f"Saved: {filename}")
            messagebox.showinfo("Saved", f"Screenshot saved as {filename}")
            print(f"[SAVE] Screenshot saved: {filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save screenshot: {e}")
            print(f"[ERROR] Save failed: {e}")
    
    def run(self):
        """Start the application"""
        print("[RUN] Starting application...")
        
        try:
            # Center window position methods for dialogs
            def center_on_parent(dialog):
                dialog.update_idletasks()
                parent_x = self.root.winfo_x()
                parent_y = self.root.winfo_y()
                parent_width = self.root.winfo_width()
                parent_height = self.root.winfo_height()
                
                dialog_width = dialog.winfo_width()
                dialog_height = dialog.winfo_height()
                
                x = parent_x + (parent_width - dialog_width) // 2
                y = parent_y + (parent_height - dialog_height) // 2
                
                dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
            
            # Add method to Toplevel class
            tk.Toplevel.center_on_parent = center_on_parent
            
            self.root.mainloop()
            
        except KeyboardInterrupt:
            print("[SHUTDOWN] Keyboard interrupt")
        finally:
            self.live_mode = False

def main():
    """Main function"""
    print("[MAIN] Starting working backseat collector...")
    
    try:
        app = WorkingBackseatCollector()
        app.run()
    except Exception as e:
        print(f"[ERROR] Application failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()