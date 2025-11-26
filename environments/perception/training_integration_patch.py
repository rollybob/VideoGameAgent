#!/usr/bin/env python3
"""
🔧 Training Visualization Integration Patch
==========================================

Code to add to comp_model_trainer.py for built-in visualization
"""

# ADD THIS TO comp_model_trainer.py imports:
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import queue
"""

# ADD THIS TO CompetitiveModelTrainer.__init__():
"""
        # Training visualization
        self.training_plots = None
        self.visualization_enabled = tk.BooleanVar(value=True)
        self.training_data_queue = queue.Queue()
"""

# ADD THIS TO setup_control_panel() after the GPU toggle:
"""
        # Training visualization toggle
        viz_frame = ttk.Frame(session_frame)
        viz_frame.pack(fill=tk.X, pady=2)
        
        ttk.Checkbutton(viz_frame, text="Enable training visualization", 
                       variable=self.visualization_enabled).pack(side=tk.LEFT)
        
        ttk.Button(viz_frame, text="📊 Open Visualizer", 
                  command=self.open_training_visualizer).pack(side=tk.RIGHT)
"""

# ADD THESE METHODS TO CompetitiveModelTrainer:

def open_training_visualizer(self):
    """Open training visualization window"""
    if hasattr(self, 'viz_window') and self.viz_window.winfo_exists():
        self.viz_window.lift()
        return
    
    self.viz_window = tk.Toplevel(self.root)
    self.viz_window.title("🔥 Training Progress")
    self.viz_window.geometry("800x600")
    
    # Create plots
    self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 6))
    self.fig.suptitle('Real-time Training Progress', fontsize=14)
    
    # Setup subplots
    self.ax1.set_title('Loss')
    self.ax1.set_ylabel('Loss')
    self.ax1.grid(True, alpha=0.3)
    
    self.ax2.set_title('Precision & Recall')
    self.ax2.set_ylabel('Score')
    self.ax2.grid(True, alpha=0.3)
    
    self.ax3.set_title('mAP')
    self.ax3.set_ylabel('mAP')
    self.ax3.grid(True, alpha=0.3)
    
    self.ax4.set_title('Learning Rate')
    self.ax4.set_ylabel('LR')
    self.ax4.grid(True, alpha=0.3)
    
    # Embed in tkinter
    canvas = FigureCanvasTkAgg(self.fig, self.viz_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    plt.tight_layout()
    
    # Initialize data storage
    self.viz_data = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'precision': [],
        'recall': [],
        'mAP50': [],
        'learning_rate': []
    }

def monitor_training_progress(self, training_dir):
    """Monitor training progress in background thread"""
    results_file = Path(training_dir) / "results.csv"
    last_size = 0
    
    while self.training_active and hasattr(self, 'viz_window'):
        try:
            if results_file.exists():
                current_size = results_file.stat().st_size
                
                if current_size > last_size:
                    # New data available
                    df = pd.read_csv(results_file)
                    if len(df) > 0:
                        latest = df.iloc[-1]
                        
                        # Queue update for main thread
                        update = {
                            'epoch': len(df),
                            'train_loss': latest.get('train/box_loss', 0),
                            'val_loss': latest.get('val/box_loss', 0),
                            'precision': latest.get('metrics/precision(B)', 0),
                            'recall': latest.get('metrics/recall(B)', 0),
                            'mAP50': latest.get('metrics/mAP50(B)', 0),
                            'learning_rate': latest.get('lr/pg0', 0)
                        }
                        
                        self.training_data_queue.put(update)
                    
                    last_size = current_size
        
        except Exception as e:
            print(f"Monitoring error: {e}")
        
        time.sleep(2)

def update_training_plots(self):
    """Update training plots with new data"""
    if not (hasattr(self, 'viz_window') and self.viz_window.winfo_exists()):
        return
    
    # Process queued updates
    updated = False
    while not self.training_data_queue.empty():
        try:
            data = self.training_data_queue.get_nowait()
            
            # Add to visualization data
            self.viz_data['epochs'].append(data['epoch'])
            self.viz_data['train_loss'].append(data['train_loss'])
            self.viz_data['val_loss'].append(data['val_loss'])
            self.viz_data['precision'].append(data['precision'])
            self.viz_data['recall'].append(data['recall'])
            self.viz_data['mAP50'].append(data['mAP50'])
            self.viz_data['learning_rate'].append(data['learning_rate'])
            
            updated = True
            
        except queue.Empty:
            break
    
    if updated and self.viz_data['epochs']:
        # Redraw plots
        epochs = self.viz_data['epochs']
        
        self.ax1.clear()
        self.ax1.plot(epochs, self.viz_data['train_loss'], 'b-', label='Train')
        self.ax1.plot(epochs, self.viz_data['val_loss'], 'r-', label='Val')
        self.ax1.set_title('Loss')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.clear()
        self.ax2.plot(epochs, self.viz_data['precision'], 'g-', label='Precision')
        self.ax2.plot(epochs, self.viz_data['recall'], 'orange', label='Recall')
        self.ax2.set_title('Precision & Recall')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.clear()
        self.ax3.plot(epochs, self.viz_data['mAP50'], 'purple', linewidth=2)
        self.ax3.set_title('mAP@0.5')
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.clear()
        self.ax4.plot(epochs, self.viz_data['learning_rate'], 'red', linewidth=2)
        self.ax4.set_title('Learning Rate')
        self.ax4.grid(True, alpha=0.3)
        
        # Refresh
        if hasattr(self, 'viz_window'):
            try:
                self.fig.canvas.draw()
            except:
                pass
    
    # Schedule next update
    if self.training_active:
        self.root.after(2000, self.update_training_plots)

# MODIFY _train_models_thread() to start visualization:
"""
def _train_models_thread(self, top_models, yaml_path):
    # ... existing code ...
    
    # Start visualization monitoring if enabled
    if self.visualization_enabled.get() and hasattr(self, 'viz_window'):
        viz_thread = threading.Thread(
            target=self.monitor_training_progress, 
            args=(Path("models") / new_model_name,),
            daemon=True
        )
        viz_thread.start()
        
        # Start plot updates
        self.root.after(1000, self.update_training_plots)
    
    # ... rest of existing training code ...
"""

print("🔧 Integration patch ready!")
print("Add these code snippets to comp_model_trainer.py for built-in visualization")