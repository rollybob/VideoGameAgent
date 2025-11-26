#!/usr/bin/env python3
"""
🌉 DATA BRIDGE: Connect AI-Assisted Data Collector to ML Training System
==========================================================================

This script converts the bounding box annotations from the data collector
into the format expected by the ML state detector for training.

Data Flow:
1. AI-Assisted Data Collector saves:
   - screenshots/game_timestamp.png
   - metadata/game_timestamp_labels.json
   
2. This bridge converts to:
   - training_data/StateClass/timestamp.npy
   - For use by ml_state_detector.py training
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append('../../agent')
from ml_state_detector import GameStateMLDetector

class DataBridge:
    """Bridge between data collector output and ML training input"""
    
    def __init__(self):
        # Paths
        self.collector_data_path = Path("assisted_training_data")
        self.real_data_path = Path("real_training_data")
        self.ml_training_path = Path("../../training_data")
        
        # State mapping from object detections to game states
        self.object_to_state_mapping = {
            # Objects that indicate specific game states
            'pokemon_sprite': 'Battle',  # Pokemon sprites = battle scene
            'hp_bar': 'Battle',          # HP bars = battle scene
            'player_character': 'Overworld',  # Player visible = overworld
            'npc_character': 'Overworld',      # NPCs = overworld
            'menu_box': 'Dialogue',           # Menu boxes = dialogue/menu
            'text_area': 'Dialogue',          # Text areas = dialogue
            'shop_item': 'Shop',              # Shop items = shop
            'pokemon_center_counter': 'Pokemon Center',  # PC counter = pokemon center
        }
        
        # Default state if no clear indicators
        self.default_state = 'Overworld'
        
        # ML detector setup
        self.ml_detector = GameStateMLDetector()
        
    def analyze_screenshot_state(self, labels_data):
        """Determine game state from detected objects"""
        if not labels_data.get('labels'):
            return self.default_state
        
        # Count object types
        object_counts = {}
        for label in labels_data['labels']:
            obj_type = label['type']
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # Determine state based on priority
        state_votes = {}
        for obj_type, count in object_counts.items():
            if obj_type in self.object_to_state_mapping:
                state = self.object_to_state_mapping[obj_type]
                state_votes[state] = state_votes.get(state, 0) + count
        
        # Return most voted state or default
        if state_votes:
            return max(state_votes.items(), key=lambda x: x[1])[0]
        else:
            return self.default_state
    
    def convert_collector_data(self, source_dir="assisted_training_data"):
        """Convert data collector output to ML training format"""
        source_path = Path(source_dir)
        metadata_dir = source_path / "metadata"
        screenshots_dir = source_path / "screenshots"
        
        if not metadata_dir.exists():
            print(f"No metadata directory found at {metadata_dir}")
            return 0
        
        converted_count = 0
        state_counts = {}
        
        print("🌉 Converting data collector output to ML training format...")
        print(f"Source: {source_path}")
        print(f"Target: {self.ml_training_path}")
        
        # Process each metadata file
        for metadata_file in metadata_dir.glob("*_labels.json"):
            try:
                # Load metadata
                with open(metadata_file, 'r') as f:
                    labels_data = json.load(f)
                
                # Find corresponding screenshot
                screenshot_name = labels_data.get('image_path')
                if not screenshot_name:
                    continue
                    
                screenshot_path = Path(screenshot_name)
                if not screenshot_path.exists():
                    # Try relative path
                    screenshot_path = screenshots_dir / screenshot_path.name
                    if not screenshot_path.exists():
                        print(f"Warning: Screenshot not found for {metadata_file}")
                        continue
                
                # Load and preprocess image
                image = cv2.imread(str(screenshot_path))
                if image is None:
                    print(f"Warning: Could not load image {screenshot_path}")
                    continue
                
                # Determine game state from labels
                game_state = self.analyze_screenshot_state(labels_data)
                state_counts[game_state] = state_counts.get(game_state, 0) + 1
                
                # Preprocess for ML (same as ml_state_detector expects)
                processed_frame = self.preprocess_frame(image)
                
                # Save in ML training format
                self.save_training_sample(processed_frame, game_state, metadata_file.stem)
                
                converted_count += 1
                print(f"✓ Converted: {screenshot_path.name} → {game_state}")
                
            except Exception as e:
                print(f"Error processing {metadata_file}: {e}")
        
        print(f"\n📊 Conversion Summary:")
        print(f"Total samples converted: {converted_count}")
        for state, count in sorted(state_counts.items()):
            print(f"  {state}: {count} samples")
        
        return converted_count
    
    def preprocess_frame(self, frame):
        """Preprocess frame same way as ML detector expects"""
        # Resize to model input size (same as ml_state_detector)
        target_size = (160, 144)  # GBA screen resolution
        resized = cv2.resize(frame, target_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def save_training_sample(self, processed_frame, game_state, sample_id):
        """Save sample in ML training format"""
        # Create state directory if needed
        state_dir = self.ml_training_path / game_state
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as .npy file (format expected by ml_state_detector)
        output_path = state_dir / f"{sample_id}.npy"
        np.save(output_path, processed_frame)
    
    def check_existing_training_data(self):
        """Check what training data already exists"""
        print("📋 Current ML Training Data Status:")
        
        if not self.ml_training_path.exists():
            print(f"No training data directory found at {self.ml_training_path}")
            return
        
        total_samples = 0
        for state_dir in self.ml_training_path.iterdir():
            if state_dir.is_dir():
                npy_files = list(state_dir.glob("*.npy"))
                count = len(npy_files)
                total_samples += count
                print(f"  {state_dir.name}: {count} samples")
        
        print(f"Total training samples: {total_samples}")
        
        if total_samples < 50:
            print("⚠️  Warning: Very few training samples. Need more data for reliable training.")
        elif total_samples < 200:
            print("⚠️  Warning: Limited training samples. Consider collecting more data.")
        else:
            print("✅ Good amount of training data!")
    
    def train_model_with_new_data(self):
        """Train the ML model with all available data"""
        print("\n🧠 Training ML Model with converted data...")
        
        # Initialize and train
        success = self.ml_detector.train_model(epochs=50, validation_split=0.2)
        
        if success:
            print("✅ Model training completed successfully!")
            print("Model saved and ready for use in VGA.py")
            return True
        else:
            print("❌ Model training failed!")
            return False
    
    def convert_and_train(self):
        """Full pipeline: convert data and train model"""
        print("🚀 Starting full data conversion and training pipeline...")
        
        # Step 1: Check existing data
        self.check_existing_training_data()
        
        # Step 2: Convert collector data
        converted = self.convert_collector_data()
        
        if converted > 0:
            print(f"\n✅ Successfully converted {converted} samples")
            
            # Step 3: Check updated data status
            print("\n📊 Updated training data:")
            self.check_existing_training_data()
            
            # Step 4: Train model
            success = self.train_model_with_new_data()
            
            if success:
                print("\n🎉 Complete! Your screenshots are now training the vision model!")
                print("\nNext steps:")
                print("1. Test the updated model in VGA.py")
                print("2. Collect more data if accuracy is low")
                print("3. Continue the training cycle")
                return True
        else:
            print("❌ No data was converted. Check that you have labeled screenshots.")
            return False

def main():
    """Run the data bridge"""
    print("=" * 60)
    print("🌉 DATA BRIDGE: Collector → ML Training")
    print("=" * 60)
    
    bridge = DataBridge()
    success = bridge.convert_and_train()
    
    if success:
        print("\n✅ SUCCESS: Your screenshots are now improving the AI!")
    else:
        print("\n❌ No data converted. Make sure you have:")
        print("   1. Taken screenshots with the data collector")
        print("   2. Confirmed/corrected the detections")
        print("   3. Saved the labels using 'Save Confirmed Labels'")

if __name__ == "__main__":
    main()