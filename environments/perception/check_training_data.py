#!/usr/bin/env python3
"""
Quick check of all available training data sources
"""

from pathlib import Path
import json

def check_training_data():
    """Check all training data sources"""
    print("TRAINING DATA INVENTORY")
    print("=" * 50)
    
    # Check competitive session training (YOLO format)
    comp_dir = Path("competitive_session_training")
    if comp_dir.exists():
        images = list((comp_dir / "images").glob("*.jpg"))
        labels = list((comp_dir / "labels").glob("*.txt"))
        paired = min(len(images), len(labels))
        print(f"Competitive training: {paired} image/label pairs")
        
        # Check a sample label file
        if labels:
            with open(labels[0], 'r') as f:
                sample_label = f.read().strip()
            print(f"  Sample label: {sample_label[:50]}...")
    else:
        print("Competitive training: 0 samples (directory not found)")
    
    # Check backseat training data
    backseat_dir = Path("backseat_training_data")
    if backseat_dir.exists():
        screenshots = list((backseat_dir / "screenshots").glob("*.png"))
        metadata = list((backseat_dir / "metadata").glob("*_labels.json"))
        print(f"Backseat training: {len(screenshots)} screenshots, {len(metadata)} metadata files")
        
        # Check a sample metadata file
        if metadata:
            with open(metadata[0], 'r') as f:
                sample_meta = json.load(f)
            print(f"  Sample metadata keys: {list(sample_meta.keys())}")
    else:
        print("Backseat training: 0 samples (directory not found)")
    
    # Check original training data
    original_dir = Path("../../training_data")
    if original_dir.exists():
        npy_files = list(original_dir.rglob("*.npy"))
        json_files = list(original_dir.rglob("*_meta.json"))
        print(f"Original training: {len(npy_files)} .npy files, {len(json_files)} metadata files")
    else:
        print("Original training: 0 samples (directory not found)")
    
    # Check synthetic data
    synthetic_dir = Path("../../training_data/training_data_legacy/synthetic")
    if synthetic_dir.exists():
        synthetic_images = list(synthetic_dir.glob("screen_*.png"))
        synthetic_json = list(synthetic_dir.glob("screen_*.json"))
        paired_synthetic = min(len(synthetic_images), len(synthetic_json))
        print(f"Synthetic training: {paired_synthetic} image/json pairs")
        
        # Check a sample synthetic annotation
        if synthetic_json:
            with open(synthetic_json[0], 'r') as f:
                sample_synthetic = json.load(f)
            print(f"  Sample synthetic keys: {list(sample_synthetic.keys())}")
    else:
        print("Synthetic training: 0 samples (directory not found)")
    
    # Summary
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS:")
    print("1. Competitive data is ready to use (YOLO format)")
    print("2. Backseat data needs conversion from metadata to YOLO labels")
    print("3. Original .npy data needs conversion pipeline")
    print("4. Synthetic data needs conversion from JSON to YOLO labels")
    print("\nConsolidated training would give you a much larger, more robust dataset!")

if __name__ == "__main__":
    check_training_data()