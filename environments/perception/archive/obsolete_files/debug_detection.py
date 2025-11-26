#!/usr/bin/env python3
"""
🔍 DEBUG: Detection System Troubleshooting
==========================================

Debug script to figure out why no detections are showing up.

Tests:
1. Model loading
2. Direct YOLO inference
3. Confidence thresholds
4. Detection pipeline
"""

import cv2
import numpy as np
from pathlib import Path
from model_manager import get_latest_model_path

def test_model_loading():
    """Test if the latest model loads correctly"""
    print("=" * 50)
    print("TEST 1: MODEL LOADING")
    print("=" * 50)
    
    try:
        model_path = get_latest_model_path()
        print(f"Latest model path: {model_path}")
        
        if not model_path or not model_path.exists():
            print(f"[ERROR] Model file not found!")
            return None
            
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        
        print(f"[OK] Model loaded successfully")
        print(f"Model classes: {list(model.names.values())}")
        print(f"Number of classes: {len(model.names)}")
        
        return model
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

def test_direct_inference(model):
    """Test direct YOLO inference on a training image"""
    print(f"\n" + "=" * 50)
    print("TEST 2: DIRECT YOLO INFERENCE")
    print("=" * 50)
    
    if not model:
        print("[SKIP] No model to test")
        return False
    
    # Find a test image from training data
    test_images = list(Path("assisted_training_data/screenshots").glob("*.png"))
    
    if not test_images:
        print("[ERROR] No test images found in assisted_training_data/screenshots/")
        return False
    
    test_img = test_images[0]
    print(f"Testing on: {test_img}")
    
    try:
        # Run inference
        results = model(str(test_img), conf=0.1, verbose=False)  # Very low confidence
        
        total_detections = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                total_detections += len(boxes)
                print(f"Found {len(boxes)} detections:")
                
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    cls_name = model.names[cls_id]
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    print(f"  {i+1}. {cls_name}: {conf:.3f} confidence at {bbox}")
            else:
                print("No detections found")
        
        print(f"\nTotal detections: {total_detections}")
        return total_detections > 0
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return False

def test_confidence_thresholds():
    """Test different confidence thresholds"""
    print(f"\n" + "=" * 50)
    print("TEST 3: CONFIDENCE THRESHOLDS")
    print("=" * 50)
    
    model_path = get_latest_model_path()
    if not model_path:
        print("[SKIP] No model available")
        return
    
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    
    # Find test image
    test_images = list(Path("assisted_training_data/screenshots").glob("*.png"))
    if not test_images:
        print("[ERROR] No test images")
        return
    
    test_img = test_images[0]
    
    # Test different confidence thresholds
    thresholds = [0.01, 0.1, 0.25, 0.5, 0.7, 0.9]
    
    for conf_thresh in thresholds:
        try:
            results = model(str(test_img), conf=conf_thresh, verbose=False)
            total_detections = 0
            
            for result in results:
                if result.boxes is not None:
                    total_detections += len(result.boxes)
            
            print(f"Confidence {conf_thresh:0.2f}: {total_detections} detections")
            
        except Exception as e:
            print(f"Confidence {conf_thresh:0.2f}: ERROR - {e}")

def test_backseat_collector_integration():
    """Test the actual backseat collector detection method"""
    print(f"\n" + "=" * 50)
    print("TEST 4: BACKSEAT COLLECTOR INTEGRATION")
    print("=" * 50)
    
    try:
        # Import without starting GUI
        from backseat_data_collector_fixed import BackseatDataCollector
        
        # Create collector and test detection
        collector = BackseatDataCollector()
        print(f"Detection method: {collector.detection_method}")
        print(f"AI model loaded: {collector.ai_model is not None}")
        
        if collector.ai_model is None:
            print("[ERROR] No AI model in collector!")
            return False
        
        # Test detection on a real image
        test_images = list(Path("assisted_training_data/screenshots").glob("*.png"))
        if not test_images:
            print("[ERROR] No test images")
            return False
            
        test_img = test_images[0]
        img = cv2.imread(str(test_img))
        
        if img is None:
            print(f"[ERROR] Could not load image: {test_img}")
            return False
        
        print(f"Testing detection on: {test_img}")
        print(f"Image shape: {img.shape}")
        
        # Call the collector's detection method directly
        detections = collector.detect_objects(img)
        
        print(f"Detected {len(detections)} objects:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det}")
        
        return len(detections) > 0
        
    except Exception as e:
        print(f"[ERROR] Backseat collector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("DETECTION SYSTEM DEBUG")
    print("=========================")
    
    # Test 1: Model loading
    model = test_model_loading()
    
    # Test 2: Direct inference  
    inference_ok = test_direct_inference(model)
    
    # Test 3: Confidence thresholds
    test_confidence_thresholds()
    
    # Test 4: Backseat collector integration
    integration_ok = test_backseat_collector_integration()
    
    # Summary
    print(f"\n" + "=" * 50)
    print("DEBUG SUMMARY")
    print("=" * 50)
    
    print(f"Model loads: {'OK' if model else 'FAIL'}")
    print(f"Direct inference works: {'OK' if inference_ok else 'FAIL'}")
    print(f"Collector integration works: {'OK' if integration_ok else 'FAIL'}")
    
    if model and inference_ok and integration_ok:
        print(f"\n[DIAGNOSIS] Detection system should be working!")
        print(f"If you're still not seeing detections, check:")
        print(f"  1. Game window capture is working")
        print(f"  2. Detection confidence thresholds in the GUI")
        print(f"  3. Object types match what was trained")
    elif model and inference_ok and not integration_ok:
        print(f"\n[DIAGNOSIS] Model works but collector integration broken!")
        print(f"Check the collector's detect_objects() method")
    elif model and not inference_ok:
        print(f"\n[DIAGNOSIS] Model loads but can't make detections!")
        print(f"The training may have failed or data is corrupted")
    else:
        print(f"\n[DIAGNOSIS] Model loading failed!")
        print(f"Check model file exists and YOLO is installed")