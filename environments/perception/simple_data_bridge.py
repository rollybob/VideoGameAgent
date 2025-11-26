#!/usr/bin/env python3
"""
Simple Data Bridge: Convert collector data to ML training format
================================================================

Converts AI-assisted data collector output to format expected by ML training.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append('../../agent')
from ml_state_detector import GameStateMLDetector

def convert_collector_data():
    """Convert data collector output to ML training format"""
    collector_path = Path("assisted_training_data")
    ml_training_path = Path("training_data")
    
    # State mapping from object detections to game states
    object_to_state_mapping = {
        'pokemon_sprite': 'Battle',
        'hp_bar': 'Battle',
        'player_character': 'Overworld',
        'npc_character': 'Overworld',
        'menu_box': 'Dialogue_Menu',
        'text_area': 'Dialogue_Menu',
        'shop_item': 'Shop',
        'pokemon_center_counter': 'Pokemon Center',
        'tree': 'Overworld',
        'building': 'Overworld',
    }
    
    print("=" * 60)
    print("DATA BRIDGE: Converting Screenshots")
    print("=" * 60)
    
    metadata_dir = collector_path / "metadata"
    screenshots_dir = collector_path / "screenshots"
    
    if not metadata_dir.exists():
        print("No metadata directory found!")
        return 0
    
    converted_count = 0
    state_counts = {}
    
    # Process each metadata file
    metadata_files = list(metadata_dir.glob("*_labels.json"))
    print(f"Found {len(metadata_files)} labeled screenshots to convert")
    
    for metadata_file in metadata_files:
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                labels_data = json.load(f)
            
            # Find corresponding screenshot
            screenshot_name = labels_data.get('image_path', '') or labels_data.get('screenshot_path', '')
            if not screenshot_name:
                continue
                
            screenshot_path = Path(screenshot_name)
            if not screenshot_path.exists():
                screenshot_path = screenshots_dir / screenshot_path.name
                if not screenshot_path.exists():
                    print(f"Warning: Screenshot not found for {metadata_file.name}")
                    continue
            
            # Load and preprocess image
            image = cv2.imread(str(screenshot_path))
            if image is None:
                print(f"Warning: Could not load image {screenshot_path}")
                continue
            
            # Determine game state from labels
            game_state = analyze_screenshot_state(labels_data, object_to_state_mapping)
            state_counts[game_state] = state_counts.get(game_state, 0) + 1
            
            # Preprocess for ML (same as ml_state_detector expects)
            target_size = (128, 128)  # Match detector input size
            resized = cv2.resize(image, target_size)
            normalized = resized.astype(np.float32) / 255.0
            
            # Save in ML training format
            state_dir = ml_training_path / game_state
            state_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = metadata_file.stem.replace('_labels', '')
            output_path = state_dir / f"{timestamp}.npy"
            np.save(output_path, normalized)
            
            converted_count += 1
            print(f"Converted: {screenshot_path.name} -> {game_state}")
            
        except Exception as e:
            print(f"Error processing {metadata_file}: {e}")
    
    print(f"\nConversion Summary:")
    print(f"Total samples converted: {converted_count}")
    for state, count in sorted(state_counts.items()):
        print(f"  {state}: {count} samples")
    
    return converted_count

def analyze_screenshot_state(labels_data, object_to_state_mapping):
    """Determine game state from detected objects"""
    if not labels_data.get('labels'):
        return 'Overworld'
    
    # Count object types
    object_counts = {}
    for label in labels_data['labels']:
        obj_type = label['type']
        object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
    
    # Determine state based on priority
    state_votes = {}
    for obj_type, count in object_counts.items():
        if obj_type in object_to_state_mapping:
            state = object_to_state_mapping[obj_type]
            state_votes[state] = state_votes.get(state, 0) + count
    
    # Return most voted state or default
    if state_votes:
        return max(state_votes.items(), key=lambda x: x[1])[0]
    else:
        return 'Overworld'

def main():
    """Run the data bridge conversion"""
    print("Simple Data Bridge")
    print("Converting data collector screenshots to ML training format...")
    
    converted = convert_collector_data()
    
    if converted > 0:
        print(f"\nSUCCESS: Converted {converted} screenshots!")
        print("\nNow you can train the model with:")
        print("  python quick_train_test.py")
    else:
        print("\nNo data was converted.")
        print("Make sure you have labeled screenshots in assisted_training_data/")

if __name__ == "__main__":
    main()