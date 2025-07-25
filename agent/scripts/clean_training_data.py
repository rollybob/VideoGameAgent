#!/usr/bin/env python3
"""
Clean up training data with mismatched shapes
"""

import os
import numpy as np
import sys
sys.path.append('agent')

def clean_training_data():
    """Remove training files with incorrect shapes"""
    from ml_state_detector import GameStateMLDetector
    
    detector = GameStateMLDetector()
    expected_shape = detector.input_shape
    
    print(f"Expected shape: {expected_shape}")
    
    # Fix the path - training data is in agent/training_data
    data_path = os.path.join(os.getcwd(), 'agent', 'training_data')
    print(f"Looking in: {data_path}")
    if not os.path.exists(data_path):
        print(f"Training data path doesn't exist: {data_path}")
        return
    
    removed_count = 0
    total_count = 0
    
    for class_name in detector.class_names:
        safe_name = detector.safe_class_names[class_name] 
        class_dir = os.path.join(data_path, safe_name)
        
        if not os.path.exists(class_dir):
            continue
            
        for filename in os.listdir(class_dir):
            if filename.endswith('.npy'):
                filepath = os.path.join(class_dir, filename)
                total_count += 1
                
                try:
                    frame_data = np.load(filepath)
                    if frame_data.shape != expected_shape:
                        print(f"Removing {filepath}: shape {frame_data.shape}")
                        os.remove(filepath)
                        removed_count += 1
                    else:
                        print(f"Keeping {filepath}: correct shape {frame_data.shape}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    os.remove(filepath)
                    removed_count += 1
    
    print(f"\nCleaning complete:")
    print(f"  Total files: {total_count}")
    print(f"  Removed: {removed_count}")
    print(f"  Remaining: {total_count - removed_count}")

if __name__ == "__main__":
    clean_training_data()
