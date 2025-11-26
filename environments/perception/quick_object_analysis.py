#!/usr/bin/env python3
"""
Quick analysis of object detection data vs game state data
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_object_detection_data():
    """Analyze the assisted_training_data (object detection)"""
    print("=" * 70)
    print("OBJECT DETECTION DATA (assisted_training_data)")
    print("=" * 70)
    
    data_dir = Path("assisted_training_data")
    metadata_dir = data_dir / "metadata"
    
    object_counts = defaultdict(int)
    total_objects = 0
    screenshot_count = 0
    
    for metadata_file in metadata_dir.glob("*.json"):
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        screenshot_count += 1
        for label in data.get('labels', []):
            obj_type = label['type']
            object_counts[obj_type] += 1
            total_objects += 1
    
    print(f"Screenshots: {screenshot_count}")
    print(f"Total object annotations: {total_objects}")
    print(f"Object types detected:")
    for obj_type, count in sorted(object_counts.items()):
        print(f"  • {obj_type}: {count} instances")
    
    return object_counts, total_objects

def analyze_game_state_data():
    """Analyze the training_data (game state classification)"""
    print("\n" + "=" * 70)
    print("GAME STATE CLASSIFICATION DATA (training_data)")
    print("=" * 70)
    
    data_dir = Path("training_data")
    state_counts = defaultdict(int)
    total_samples = 0
    
    for state_dir in data_dir.iterdir():
        if state_dir.is_dir() and state_dir.name != "synthetic":
            sample_count = len(list(state_dir.glob("*.npy")))
            state_counts[state_dir.name] = sample_count
            total_samples += sample_count
    
    print(f"Total samples: {total_samples}")
    print(f"Game states:")
    for state, count in sorted(state_counts.items()):
        print(f"  • {state}: {count} samples")
    
    return state_counts, total_samples

def compare_approaches():
    """Compare the two approaches"""
    print("\n" + "=" * 70)
    print("COMPARISON: Object Detection vs Game State Classification")
    print("=" * 70)
    
    obj_counts, obj_total = analyze_object_detection_data()
    state_counts, state_total = analyze_game_state_data()
    
    print(f"\n📊 DATA COMPARISON:")
    print(f"Object Detection: {len(obj_counts)} object types, {obj_total} annotations")
    print(f"Game State: {len(state_counts)} game states, {state_total} samples")
    
    print(f"\n🎯 WHAT EACH APPROACH TELLS THE AI:")
    print(f"\nObject Detection (BETTER for gameplay):")
    print(f"  ✓ 'There is a player_character at coordinates (913,439) to (1007,567)'")
    print(f"  ✓ 'There are 6 trees at specific locations on screen'")
    print(f"  ✓ 'There are 2 npc_characters the agent can interact with'")
    print(f"  ✓ Multiple objects detected simultaneously")
    print(f"  ✓ Precise bounding boxes for interaction")
    
    print(f"\nGame State Classification (LIMITED for gameplay):")
    print(f"  ~ 'This screen is probably an Overworld scene'")
    print(f"  ~ No specific object locations")
    print(f"  ~ No interaction targets")
    print(f"  ~ Only one classification per screen")
    
    print(f"\n🚀 WHY OBJECT DETECTION IS SUPERIOR:")
    print(f"  • AI knows WHAT objects exist")
    print(f"  • AI knows WHERE objects are located")
    print(f"  • AI can plan precise movements and interactions")
    print(f"  • Multiple detections per frame")
    print(f"  • Essential for autonomous gameplay")
    
    print(f"\n🎮 GAMEPLAY IMPLICATIONS:")
    print(f"Object Detection enables:")
    print(f"  • Walking to specific NPCs")
    print(f"  • Avoiding obstacles (trees)")
    print(f"  • Targeting specific UI elements")
    print(f"  • Understanding spatial relationships")
    
    print(f"\nGame State Classification only enables:")
    print(f"  • General context awareness")
    print(f"  • Very basic scene understanding")
    
    print(f"\n🏆 RECOMMENDATION:")
    print(f"Focus on Object Detection training using assisted_training_data")
    print(f"This is the foundation for intelligent game AI behavior!")

if __name__ == "__main__":
    compare_approaches()