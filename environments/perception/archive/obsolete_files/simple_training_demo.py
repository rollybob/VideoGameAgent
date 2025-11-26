#!/usr/bin/env python3
"""
TRAINING DEMONSTRATION: How Screenshots Train the Perception Layer
====================================================================

This script demonstrates exactly how each screenshot from the data
collector gets transformed and used to train the ML vision model.

User Flow:
1. Take screenshots with data collector
2. Confirm/correct object detections  
3. Save confirmed labels
4. **THIS SCRIPT**: Convert to ML training format
5. Train the neural network
6. Test improved model
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append('../../agent')

def check_data_status():
    """Check what training data exists"""
    print("=" * 70)
    print("PERCEPTION TRAINING DEMONSTRATION")
    print("=" * 70)
    
    # Check collector data
    collector_path = Path("assisted_training_data")
    ml_path = Path("../../training_data")
    
    print("\nSTEP 1: Checking Data Collector Output")
    print("-" * 50)
    
    labeled_count = 0
    if collector_path.exists():
        metadata_dir = collector_path / "metadata"
        if metadata_dir.exists():
            metadata_files = list(metadata_dir.glob("*_labels.json"))
            labeled_count = len(metadata_files)
            print(f"   Found {labeled_count} labeled screenshots")
            
            # Show first few examples
            for i, metadata_file in enumerate(metadata_files[:3]):
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                    labels = data.get('labels', [])
                    print(f"   Example {i+1}: {len(labels)} objects detected")
                    for label in labels:
                        obj_type = label['type']
                        bbox = label['bbox']
                        print(f"      * {obj_type} at ({bbox[0]:.0f}, {bbox[1]:.0f})")
                except Exception as e:
                    print(f"      Error reading file: {e}")
        else:
            print("   No metadata directory found")
    else:
        print("   No data collector output found")
    
    print("\nSTEP 2: Checking ML Training Data")
    print("-" * 50)
    
    ml_samples = 0
    if ml_path.exists():
        for state_dir in ml_path.iterdir():
            if state_dir.is_dir():
                npy_files = list(state_dir.glob("*.npy"))
                count = len(npy_files)
                ml_samples += count
                if count > 0:
                    print(f"   {state_dir.name}: {count} training samples")
    else:
        print("   No ML training data found")
    
    print("\nSTEP 3: Data Conversion Process")
    print("-" * 50)
    
    if labeled_count > 0:
        print("   How screenshots become training data:")
        print("   1. Load original screenshot (e.g., 240x160 pixels)")
        print("   2. Resize to 128x128 (neural network input size)")
        print("   3. Normalize pixels from [0-255] to [0-1] range")
        print("   4. Determine game state from labeled objects:")
        print("      * pokemon_sprite -> Battle state")
        print("      * player_character -> Overworld state")
        print("      * menu_box -> Dialogue state")
        print("   5. Save as .npy array for fast loading")
        
        if ml_samples == 0:
            print("   STATUS: Data collector has screenshots but they need conversion!")
            print("   RUN: python data_bridge.py to convert for ML training")
        else:
            print(f"   STATUS: {ml_samples} samples ready for training")
    else:
        print("   No screenshots found to convert")
        print("   USE: assisted_data_collector.py to collect labeled screenshots")
    
    print("\nSTEP 4: Neural Network Training")
    print("-" * 50)
    
    if ml_samples > 0:
        print("   Training process:")
        print("   1. Load all .npy training arrays")
        print("   2. Create neural network:")
        print("      * Input layer: 128x128x3 RGB images")
        print("      * Convolutional layers: extract visual features")
        print("      * Dense layers: classify game states")
        print("      * Output layer: probability for each state")
        print("   3. Training loop:")
        print("      * Show network thousands of your screenshots")
        print("      * Compare predictions to your labels")
        print("      * Adjust weights to minimize errors")
        print("      * Validate on unseen data")
        print(f"   4. Result: AI learns to recognize {ml_samples} visual patterns")
        
        if ml_samples >= 50:
            print("   STATUS: Enough data for good training!")
        elif ml_samples >= 10:
            print("   STATUS: Minimal data - training possible but limited")
        else:
            print("   STATUS: Need more data for reliable training")
    else:
        print("   No training data available")
        print("   Need to convert collector screenshots first")
    
    print("\nSTEP 5: Model Testing & Usage")
    print("-" * 50)
    
    model_path = Path("../../agent/models/game_state_classifier.h5")
    if model_path.exists():
        print("   Trained model found!")
        print("   Testing process:")
        print("   1. Load trained neural network")
        print("   2. Take new screenshot")
        print("   3. Preprocess (resize, normalize)")
        print("   4. Get AI prediction with confidence")
        print("   5. Compare to expected state")
        print("   USAGE: VGA.py uses this model for real-time detection")
    else:
        print("   No trained model found")
        print("   Need to train model with converted data first")
    
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    if labeled_count > 0 and ml_samples > 0:
        print("SUCCESS: Complete training pipeline!")
        print(f"* {labeled_count} screenshots collected")
        print(f"* {ml_samples} samples converted for training")
        print("* Ready to train improved AI model")
    elif labeled_count > 0:
        print("PARTIAL: Have screenshots, need conversion")
        print(f"* {labeled_count} screenshots ready")
        print("* Run data_bridge.py to convert")
    else:
        print("START HERE: No training data found")
        print("* Use assisted_data_collector.py first")
        print("* Label objects in screenshots")
        print("* Save confirmed labels")
        print("* Convert with data_bridge.py")
        print("* Train model")
    
    print("\nNext Steps:")
    if labeled_count == 0:
        print("1. Run: python assisted_data_collector.py")
        print("2. Collect and label 20+ diverse screenshots")
        print("3. Save confirmed labels")
    elif ml_samples == 0:
        print("1. Run: python data_bridge.py")
        print("2. Convert screenshots to ML format")
    elif ml_samples < 50:
        print("1. Collect more diverse screenshots")
        print("2. Aim for 50+ samples per game state")
    else:
        print("1. Train the model with existing data")
        print("2. Test in VGA.py")
        print("3. Collect more data if accuracy is low")

def main():
    """Run the training demonstration"""
    check_data_status()
    
    print("\nHow Each Screenshot Improves the AI:")
    print("* Screenshot shows game context (battle, overworld, etc.)")
    print("* Your labels teach AI what objects indicate each state")
    print("* Neural network learns visual patterns from pixels")
    print("* More diverse examples = better pattern recognition")
    print("* AI becomes more accurate at real-time state detection")

if __name__ == "__main__":
    main()