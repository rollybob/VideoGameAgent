#!/usr/bin/env python3
"""
Quick data status checker - see what training data exists
"""

import os
import json
from pathlib import Path

def check_collector_data():
    """Check data collector output"""
    print("📸 AI-Assisted Data Collector Output:")
    
    # Check different possible locations
    data_dirs = [
        "assisted_training_data",
        "real_training_data"
    ]
    
    total_screenshots = 0
    total_labeled = 0
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            print(f"\n  📁 {data_dir}:")
            
            # Check screenshots
            screenshots_dir = data_path / "screenshots"
            if screenshots_dir.exists():
                screenshots = list(screenshots_dir.glob("*.png"))
                print(f"    Screenshots: {len(screenshots)}")
                total_screenshots += len(screenshots)
            
            # Check metadata (labels)
            metadata_dir = data_path / "metadata"
            if metadata_dir.exists():
                metadata_files = list(metadata_dir.glob("*_labels.json"))
                print(f"    Labeled: {len(metadata_files)}")
                total_labeled += len(metadata_files)
                
                # Sample a few labels to see what's in them
                if metadata_files:
                    print(f"    Sample labels:")
                    for i, metadata_file in enumerate(metadata_files[:3]):
                        try:
                            with open(metadata_file, 'r') as f:
                                data = json.load(f)
                            labels = data.get('labels', [])
                            label_types = [l['type'] for l in labels]
                            print(f"      {metadata_file.name}: {len(labels)} objects - {set(label_types)}")
                        except Exception as e:
                            print(f"      {metadata_file.name}: Error reading - {e}")
                        if i >= 2:  # Limit to first 3
                            break
    
    print(f"\n📊 Collector Summary:")
    print(f"    Total Screenshots: {total_screenshots}")
    print(f"    Total Labeled: {total_labeled}")
    
    return total_screenshots, total_labeled

def check_ml_training_data():
    """Check ML training data"""
    print("\n🧠 ML Training Data (for actual model training):")
    
    ml_path = Path("../../training_data")
    if not ml_path.exists():
        print("    ❌ No ML training data directory found")
        return 0
    
    total_samples = 0
    for state_dir in ml_path.iterdir():
        if state_dir.is_dir():
            npy_files = list(state_dir.glob("*.npy"))
            json_files = list(state_dir.glob("*.json"))
            count = len(npy_files)
            total_samples += count
            if count > 0:
                print(f"    {state_dir.name}: {count} .npy samples, {len(json_files)} metadata")
    
    print(f"\n📊 ML Training Summary:")
    print(f"    Total ML Training Samples: {total_samples}")
    
    if total_samples == 0:
        print("    ❌ No .npy training files found!")
        print("    🔗 Need to convert collector data to ML format")
    elif total_samples < 50:
        print("    ⚠️  Very few samples - need more data")
    elif total_samples < 200:
        print("    ⚠️  Limited samples - could use more data")
    else:
        print("    ✅ Good amount of training data")
    
    return total_samples

def check_data_bridge_needed():
    """Check if data bridge conversion is needed"""
    collector_screenshots, collector_labeled = check_collector_data()
    ml_samples = check_ml_training_data()
    
    print("\n" + "="*50)
    print("🔍 DATA STATUS ANALYSIS")
    print("="*50)
    
    if collector_labeled > 0 and ml_samples == 0:
        print("🚨 ISSUE: You have labeled screenshots but they're NOT being used for ML training!")
        print("   📸 Collector Data: ✅ Available")
        print("   🧠 ML Training: ❌ Not converted")
        print("\n💡 SOLUTION: Run the data bridge to convert your screenshots")
        print("   Your labeled screenshots need to be converted to .npy format")
        print("   for the ML model to use them for training.")
        return True
        
    elif collector_labeled > 0 and ml_samples > 0:
        print("✅ GOOD: You have both collector data AND ML training data")
        print("   📸 Collector Data: ✅ Available") 
        print("   🧠 ML Training: ✅ Available")
        print("\n💡 You can collect more data or train with existing data")
        return False
        
    elif collector_labeled == 0 and ml_samples > 0:
        print("⚠️  You have ML training data but no recent collector data")
        print("   📸 Collector Data: ❌ None recent")
        print("   🧠 ML Training: ✅ Available")
        print("\n💡 You can train with existing data or collect more")
        return False
        
    else:
        print("❌ NO TRAINING DATA FOUND")
        print("   📸 Collector Data: ❌ None")
        print("   🧠 ML Training: ❌ None")
        print("\n💡 You need to:")
        print("   1. Use the AI-assisted data collector to take screenshots")
        print("   2. Label/confirm the detections")
        print("   3. Save the confirmed labels")
        print("   4. Run the data bridge to convert for ML training")
        return True

def main():
    """Main data status check"""
    print("🔍 TRAINING DATA STATUS CHECK")
    print("="*40)
    
    needs_bridge = check_data_bridge_needed()
    
    print("\n" + "="*50)
    print("📋 NEXT STEPS")
    print("="*50)
    
    if needs_bridge:
        print("1. ❗ Run data_bridge.py to convert your labeled screenshots")
        print("2. 🧠 Train the ML model with converted data")
        print("3. 🎮 Test the updated model in VGA.py")
    else:
        print("1. 📸 Collect more labeled screenshots (optional)")
        print("2. 🧠 Train/retrain the ML model")
        print("3. 🎮 Test the model in VGA.py")

if __name__ == "__main__":
    main()