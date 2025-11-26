#!/usr/bin/env python3
"""
Debug script to test individual model detection capabilities
"""

import cv2
import numpy as np
from pathlib import Path
from model_manager import ModelManager

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available - install with: pip install ultralytics")
    exit(1)

def test_individual_model(model_name, model_path):
    """Test an individual model's detection capabilities"""
    print(f"\n{'='*60}")
    print(f"TESTING MODEL: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")
    
    try:
        # Load model
        model = YOLO(str(model_path))
        print(f"✓ Model loaded successfully")
        print(f"Classes: {list(model.names.values())}")
        
        # Create test image with some basic shapes
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some colored rectangles to simulate game objects
        cv2.rectangle(test_image, (100, 50), (200, 150), (0, 255, 0), -1)  # Green
        cv2.rectangle(test_image, (300, 200), (400, 300), (255, 0, 0), -1)  # Blue  
        cv2.rectangle(test_image, (150, 350), (250, 420), (0, 0, 255), -1)  # Red
        
        # Add some text
        cv2.putText(test_image, "TEST DETECTION", (200, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        print(f"Test image shape: {test_image.shape}")
        
        # Run inference with different confidence thresholds
        for conf_threshold in [0.01, 0.1, 0.25, 0.5]:
            print(f"\n--- Testing with confidence threshold: {conf_threshold} ---")
            
            results = model(test_image, verbose=False, conf=conf_threshold)
            
            print(f"Number of result objects: {len(results)}")
            
            for i, result in enumerate(results):
                if result.boxes is not None and len(result.boxes) > 0:
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    
                    print(f"  Found {len(confidences)} detections:")
                    for j, (conf, cls, box) in enumerate(zip(confidences, classes, boxes)):
                        class_name = model.names[int(cls)]
                        print(f"    {j+1}. {class_name}: {conf:.3f} at {box}")
                else:
                    print(f"  No detections found")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing model: {e}")
        return False

def main():
    print("MODEL DETECTION DEBUGGING")
    print("========================")
    
    # Get available models
    manager = ModelManager()
    models = manager.get_all_model_versions()
    
    if not models:
        print("No models found!")
        return
    
    print(f"Found {len(models)} models:")
    for i, (name, path, time) in enumerate(models):
        print(f"{i+1}. {name} ({time})")
    
    # Test each model
    successful_models = []
    failed_models = []
    
    for name, path, time in models:
        if test_individual_model(name, path):
            successful_models.append(name)
        else:
            failed_models.append(name)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successful models: {len(successful_models)}")
    for model in successful_models:
        print(f"  - {model}")
    
    print(f"✗ Failed models: {len(failed_models)}")
    for model in failed_models:
        print(f"  - {model}")

def test_specific_models():
    """Test the competitive models specifically"""
    print("TESTING COMPETITIVE MODELS SPECIFICALLY")
    print("=" * 50)
    
    competitive_models = [
        ("pokemon_object_detector2_competitive_20250728_210432", "models/pokemon_object_detector2_competitive_20250728_210432/weights/best.pt"),
        ("pokemon_object_detector_competitive_20250728_201327", "models/pokemon_object_detector_competitive_20250728_201327/weights/best.pt"),
        ("pokemon_object_detector2", "models/pokemon_object_detector2/weights/best.pt"),  # For comparison
        ("pokemon_object_detector", "models/pokemon_object_detector/weights/best.pt"),   # For comparison
    ]
    
    for name, path in competitive_models:
        print(f"\n{'='*60}")
        print(f"TESTING: {name}")
        print(f"Path: {path}")
        
        if Path(path).exists():
            test_individual_model(name, Path(path))
        else:
            print(f"❌ Model file not found: {path}")

if __name__ == "__main__":
    # Test specific competitive models first
    test_specific_models()
    
    print("\n" + "="*60)
    print("FULL MODEL TESTING")
    print("="*60)
    
    # Then run full test
    main()