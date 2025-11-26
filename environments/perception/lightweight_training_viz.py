#!/usr/bin/env python3
"""
⚡ Lightweight Training Visualization Popup
==========================================

Minimal, automatic visualization that pops up during training
Integrates seamlessly with comp_model_trainer.py
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import pandas as pd
from pathlib import Path
import queue

class LightweightTrainingPopup:
    """Lightweight popup for training visualization"""
    
    def __init__(self, training_dir, parent_window=None):
        self.training_dir = Path(training_dir)
        self.parent = parent_window
        self.monitoring = True
        self.data_queue = queue.Queue()
        
        # Training data
        self.data = {
            'epochs': [],
            'loss': [],
            'precision': [],
            'recall': [],
            'mAP': []
        }
        
        self.setup_window()
        self.setup_plot()
        self.start_monitoring()
    
    def setup_window(self):
        """Create the popup window"""
        self.window = tk.Toplevel()
        self.window.title(f"🔥 Training: {self.training_dir.name}")
        self.window.geometry("600x400")
        
        # Position relative to parent
        if self.parent:
            x = self.parent.winfo_x() + 50
            y = self.parent.winfo_y() + 50
            self.window.geometry(f"600x400+{x}+{y}")
        
        # Stay on top but not always
        self.window.transient(self.parent)
        
        # Status bar
        status_frame = ttk.Frame(self.window)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="🔍 Monitoring training...")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=('Arial', 10))
        status_label.pack(side=tk.LEFT)
        
        # Current metrics
        self.metrics_var = tk.StringVar(value="Waiting for data...")
        metrics_label = ttk.Label(status_frame, textvariable=self.metrics_var, font=('Arial', 9))
        metrics_label.pack(side=tk.RIGHT)
        
        # Close button
        ttk.Button(status_frame, text="Close", command=self.close).pack(side=tk.RIGHT, padx=(10, 0))
    
    def setup_plot(self):
        """Create the training plot"""
        # Single plot with multiple metrics
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 5))
        self.fig.suptitle('Training Progress', fontsize=12, fontweight='bold')
        
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Metrics')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Embed in window
        canvas = FigureCanvasTkAgg(self.fig, self.window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        plt.tight_layout()
    
    def start_monitoring(self):
        """Start monitoring training files"""
        self.monitor_thread = threading.Thread(target=self.monitor_training, daemon=True)
        self.monitor_thread.start()
        
        # Start GUI updates
        self.update_display()
    
    def monitor_training(self):
        """Monitor training progress in background"""
        results_file = self.training_dir / "results.csv"
        last_size = 0
        
        while self.monitoring:
            try:
                if results_file.exists():
                    current_size = results_file.stat().st_size
                    
                    if current_size > last_size:
                        # Read new data
                        df = pd.read_csv(results_file)
                        if len(df) > 0:
                            latest = df.iloc[-1]
                            
                            # Extract key metrics
                            epoch = len(df)
                            train_loss = latest.get('train/box_loss', 0) + latest.get('train/cls_loss', 0)
                            precision = latest.get('metrics/precision(B)', 0)
                            recall = latest.get('metrics/recall(B)', 0)
                            mAP = latest.get('metrics/mAP50(B)', 0)
                            
                            # Queue update
                            self.data_queue.put({
                                'epoch': epoch,
                                'loss': train_loss,
                                'precision': precision,
                                'recall': recall,
                                'mAP': mAP
                            })
                        
                        last_size = current_size
                
            except Exception as e:
                print(f"Monitor error: {e}")
            
            time.sleep(2)
    
    def update_display(self):
        """Update the display with new data"""
        if not self.monitoring:
            return
        
        # Process queue
        updated = False
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                
                # Add to storage
                self.data['epochs'].append(data['epoch'])
                self.data['loss'].append(data['loss'])
                self.data['precision'].append(data['precision'])
                self.data['recall'].append(data['recall'])
                self.data['mAP'].append(data['mAP'])
                
                # Update status
                self.status_var.set(f"🔥 Training Epoch {data['epoch']}")
                self.metrics_var.set(f"Loss: {data['loss']:.3f} | P: {data['precision']:.3f} | R: {data['recall']:.3f} | mAP: {data['mAP']:.3f}")
                
                updated = True
                
            except queue.Empty:
                break
        
        # Redraw plot if updated
        if updated and self.data['epochs']:
            self.redraw_plot()
        
        # Schedule next update
        self.window.after(2000, self.update_display)
    
    def redraw_plot(self):
        """Redraw the training plot"""
        self.ax.clear()
        
        epochs = self.data['epochs']
        
        # Plot key metrics
        self.ax.plot(epochs, self.data['loss'], 'r-', label='Loss', linewidth=2)
        self.ax.plot(epochs, self.data['precision'], 'g-', label='Precision', linewidth=2)
        self.ax.plot(epochs, self.data['recall'], 'b-', label='Recall', linewidth=2)
        self.ax.plot(epochs, self.data['mAP'], 'purple', label='mAP@0.5', linewidth=2)
        
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Metrics')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Training Progress')
        
        # Auto-scale with some padding
        if len(epochs) > 1:
            self.ax.set_xlim(0, max(epochs) * 1.1)
        
        try:
            self.fig.canvas.draw()
        except:
            pass
    
    def close(self):
        """Close the popup"""
        self.monitoring = False
        if hasattr(self, 'window'):
            self.window.destroy()

# Integration functions for comp_model_trainer.py
def show_training_popup(training_dir, parent_window=None):
    """Show lightweight training popup"""
    return LightweightTrainingPopup(training_dir, parent_window)

# Test standalone
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    # Create test popup
    popup = LightweightTrainingPopup("models/test_training", root)
    
    root.mainloop()