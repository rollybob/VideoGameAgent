#!/usr/bin/env python3
import sys
sys.path.append('../../agent')
from ml_state_detector import GameStateMLDetector

ml = GameStateMLDetector()
print("Expected class names:")
for i, name in enumerate(ml.class_names):
    safe_name = ml.safe_class_names[name]
    print(f"{i}: '{name}' -> directory: '{safe_name}'")