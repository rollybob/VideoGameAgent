#!/usr/bin/env python3
"""
📊 Real-time Training Visualizer
===============================

Live visualization of YOLO training progress with interactive plots
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import threading
import time
import json
from pathlib import Path
import re
from typing import Dict, List, Optional
import queue

class TrainingVisualizer:
    """Real-time training visualization GUI"""
    
    def __init__(self, training_dir: Path = None):
        self.root = tk.Tk()
        self.root.title("🔥 YOLO Training Visualizer")
        self.root.geometry("1200x800")
        
        # Training monitoring
        self.training_dir = training_dir
        self.monitoring = False
        self.data_queue = queue.Queue()
        
        # Training data storage
        self.training_data = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'precision': [],
            'recall': [],
            'mAP50': [],
            'mAP50_95': [],
            'learning_rate': []
        }
        
        self.setup_ui()
        self.setup_plots()
        
    def setup_ui(self):
        """Setup the visualization interface"""
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Training Monitor", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Training directory selection
        dir_frame = ttk.Frame(control_frame)
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(dir_frame, text="Training Directory:").pack(side=tk.LEFT)
        self.dir_var = tk.StringVar(value=str(self.training_dir) if self.training_dir else "No directory selected")
        self.dir_label = ttk.Label(dir_frame, textvariable=self.dir_var, foreground="blue")
        self.dir_label.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(dir_frame, text="Browse", command=self.browse_directory).pack(side=tk.RIGHT)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(button_frame, text="🔍 Start Monitoring", 
                                      command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="⏹️ Stop Monitoring", 
                                     command=self.stop_monitoring, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.clear_button = ttk.Button(button_frame, text="🔄 Clear Data", 
                                      command=self.clear_data)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Status display
        self.status_var = tk.StringVar(value="Ready to monitor training")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, font=('Arial', 10, 'bold'))
        self.status_label.pack(anchor=tk.W)
        
        # Current metrics display
        metrics_frame = ttk.LabelFrame(control_frame, text="Current Metrics", padding=5)
        metrics_frame.pack(fill=tk.X, pady=(10, 0))
        
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X)
        
        # Create metric labels
        self.metric_labels = {}
        metrics = ['Epoch', 'Train Loss', 'Val Loss', 'Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95', 'Learning Rate']
        
        for i, metric in enumerate(metrics):
            row = i // 4
            col = i % 4
            
            ttk.Label(metrics_grid, text=f"{metric}:", font=('Arial', 8, 'bold')).grid(
                row=row*2, column=col, sticky=tk.W, padx=5, pady=2)
            
            self.metric_labels[metric.lower().replace(' ', '_').replace('@', '').replace(':', '_')] = ttk.Label(
                metrics_grid, text="--", font=('Arial', 8), foreground="blue")
            self.metric_labels[metric.lower().replace(' ', '_').replace('@', '').replace(':', '_')].grid(
                row=row*2+1, column=col, sticky=tk.W, padx=5, pady=2)
        
        # Plot container
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
    def setup_plots(self):
        """Setup matplotlib plots"""
        
        # Create figure with subplots
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('YOLO Training Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss curves
        self.ax1.set_title('Training & Validation Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Precision & Recall
        self.ax2.set_title('Precision & Recall')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Score')
        self.ax2.grid(True, alpha=0.3)
        
        # Plot 3: mAP metrics
        self.ax3.set_title('Mean Average Precision')
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('mAP')
        self.ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning rate
        self.ax4.set_title('Learning Rate Schedule')
        self.ax4.set_xlabel('Epoch')
        self.ax4.set_ylabel('Learning Rate')
        self.ax4.grid(True, alpha=0.3)
        
        # Embed plots in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        plt.tight_layout()
        
    def browse_directory(self):
        """Browse for training directory"""
        from tkinter import filedialog
        
        directory = filedialog.askdirectory(title="Select YOLO Training Directory")
        if directory:
            self.training_dir = Path(directory)
            self.dir_var.set(str(self.training_dir))
            self.status_var.set(f"Directory selected: {self.training_dir.name}")
    
    def start_monitoring(self):
        """Start monitoring training progress"""
        if not self.training_dir or not self.training_dir.exists():
            self.status_var.set("❌ Please select a valid training directory")
            return
        
        self.monitoring = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_var.set("🔍 Monitoring training progress...")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_training, daemon=True)
        self.monitor_thread.start()
        
        # Start GUI update timer
        self.update_plots()
    
    def stop_monitoring(self):
        """Stop monitoring training progress"""
        self.monitoring = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("⏹️ Monitoring stopped")
    
    def clear_data(self):
        """Clear all training data"""
        for key in self.training_data:
            self.training_data[key].clear()
        
        # Clear plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        self.setup_plots()
        self.canvas.draw()
        
        # Clear metric labels
        for label in self.metric_labels.values():
            label.config(text="--")
        
        self.status_var.set("🔄 Data cleared")
    
    def monitor_training(self):
        """Monitor training files for updates"""
        
        results_file = self.training_dir / "results.csv"
        last_modified = 0
        
        while self.monitoring:
            try:
                if results_file.exists():
                    current_modified = results_file.stat().st_mtime
                    
                    if current_modified > last_modified:
                        # File was updated, read new data
                        self.read_training_data(results_file)
                        last_modified = current_modified
                
                # Also check for console output logs
                self.check_console_logs()
                
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(1)  # Check every second
    
    def read_training_data(self, results_file: Path):
        """Read training data from YOLO results.csv"""
        try:
            df = pd.read_csv(results_file)
            
            # Extract data (YOLO csv format)
            if len(df) > 0:
                latest_row = df.iloc[-1]
                
                # Add to our data storage
                epoch = len(df)
                
                # Queue the update for the main thread
                update_data = {
                    'epoch': epoch,
                    'train_loss': latest_row.get('train/box_loss', 0) + latest_row.get('train/cls_loss', 0),
                    'val_loss': latest_row.get('val/box_loss', 0) + latest_row.get('val/cls_loss', 0),
                    'precision': latest_row.get('metrics/precision(B)', 0),
                    'recall': latest_row.get('metrics/recall(B)', 0),
                    'mAP50': latest_row.get('metrics/mAP50(B)', 0),
                    'mAP50_95': latest_row.get('metrics/mAP50-95(B)', 0),
                    'learning_rate': latest_row.get('lr/pg0', 0)
                }
                
                self.data_queue.put(update_data)
                
        except Exception as e:
            print(f"Error reading training data: {e}")
    
    def check_console_logs(self):
        """Check for console logs and training status"""
        # Look for training log files
        log_files = list(self.training_dir.glob("*.log"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if "Epoch" in last_line:
                            # Update status with current epoch info
                            self.root.after(0, lambda: self.status_var.set(f"🔥 Training: {last_line}"))
            except:
                pass
    
    def update_plots(self):
        """Update plots with new data (called from main thread)"""
        
        # Process queued data updates
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                
                # Update training data
                self.training_data['epochs'].append(data['epoch'])
                self.training_data['train_loss'].append(data['train_loss'])
                self.training_data['val_loss'].append(data['val_loss'])
                self.training_data['precision'].append(data['precision'])
                self.training_data['recall'].append(data['recall'])
                self.training_data['mAP50'].append(data['mAP50'])
                self.training_data['mAP50_95'].append(data['mAP50_95'])
                self.training_data['learning_rate'].append(data['learning_rate'])
                
                # Update metric labels
                self.metric_labels['epoch'].config(text=f"{data['epoch']}")
                self.metric_labels['train_loss'].config(text=f"{data['train_loss']:.4f}")
                self.metric_labels['val_loss'].config(text=f"{data['val_loss']:.4f}")
                self.metric_labels['precision'].config(text=f"{data['precision']:.3f}")
                self.metric_labels['recall'].config(text=f"{data['recall']:.3f}")
                self.metric_labels['map05'].config(text=f"{data['mAP50']:.3f}")
                self.metric_labels['map05_095'].config(text=f"{data['mAP50_95']:.3f}")
                self.metric_labels['learning_rate'].config(text=f"{data['learning_rate']:.6f}")
                
            except queue.Empty:
                break
        
        # Update plots if we have data
        if self.training_data['epochs']:
            self.redraw_plots()
        
        # Schedule next update
        if self.monitoring:
            self.root.after(1000, self.update_plots)  # Update every second
    
    def redraw_plots(self):
        """Redraw all plots with current data"""
        
        epochs = self.training_data['epochs']
        
        # Clear and redraw plots
        self.ax1.clear()
        self.ax1.set_title('Training & Validation Loss')
        self.ax1.plot(epochs, self.training_data['train_loss'], 'b-', label='Train Loss', linewidth=2)
        self.ax1.plot(epochs, self.training_data['val_loss'], 'r-', label='Val Loss', linewidth=2)
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.clear()
        self.ax2.set_title('Precision & Recall')
        self.ax2.plot(epochs, self.training_data['precision'], 'g-', label='Precision', linewidth=2)
        self.ax2.plot(epochs, self.training_data['recall'], 'orange', label='Recall', linewidth=2)
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Score')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.clear()
        self.ax3.set_title('Mean Average Precision')
        self.ax3.plot(epochs, self.training_data['mAP50'], 'purple', label='mAP@0.5', linewidth=2)
        self.ax3.plot(epochs, self.training_data['mAP50_95'], 'brown', label='mAP@0.5:0.95', linewidth=2)
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('mAP')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.clear()
        self.ax4.set_title('Learning Rate Schedule')
        self.ax4.plot(epochs, self.training_data['learning_rate'], 'red', linewidth=2)
        self.ax4.set_xlabel('Epoch')
        self.ax4.set_ylabel('Learning Rate')
        self.ax4.grid(True, alpha=0.3)
        
        # Refresh canvas
        self.canvas.draw()
    
    def run(self):
        """Start the visualizer"""
        self.root.mainloop()

def main():
    """Run the training visualizer"""
    print("📊 Starting YOLO Training Visualizer...")
    
    visualizer = TrainingVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()