#!/usr/bin/env python3
import sys
import os
sys.path.append('../../agent')
from ml_state_detector import GameStateMLDetector

ml = GameStateMLDetector()
print(f"Data path: {ml.data_path}")
print(f"Full path: {os.path.abspath(ml.data_path)}")

# Check each class directory
for i, class_name in enumerate(ml.class_names):
    safe_name = ml.safe_class_names[class_name]
    class_dir = os.path.join(ml.data_path, safe_name)
    print(f"\nClass {i}: {class_name}")
    print(f"  Safe name: {safe_name}")
    print(f"  Directory: {class_dir}")
    print(f"  Exists: {os.path.exists(class_dir)}")
    
    if os.path.exists(class_dir):
        npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        print(f"  .npy files: {len(npy_files)}")
        if npy_files:
            print(f"    Files: {npy_files[:3]}")  # Show first 3

# Test loading
print(f"\nTesting load_training_data():")
X, y = ml.load_training_data()
print(f"Loaded X shape: {X.shape if len(X) > 0 else 'Empty'}")
print(f"Loaded y shape: {y.shape if len(y) > 0 else 'Empty'}")