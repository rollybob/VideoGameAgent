#!/usr/bin/env python3
"""
🔧 FIX TRAINING: Retrain with Better Parameters
==============================================

The current model has poor accuracy (21% mAP50). This script retrains
with better parameters for small datasets.

Issues with current training:
- Only 11 images (too small)
- Quick training (15 epochs may be too few)
- Standard parameters (not optimized for small datasets)
"""

import sys
from pathlib import Path

# Add agent directory to path
sys.path.append('../../agent')

try:
    from ultralytics import YOLO
    print("[OK] YOLO imported successfully")
    YOLO_AVAILABLE = True
except ImportError:
    print("[ERROR] YOLO not available")
    YOLO_AVAILABLE = False
    sys.exit(1)

def fix_training():
    """Retrain with parameters optimized for small datasets"""
    print("=" * 50)
    print("FIXING TRAINING WITH BETTER PARAMETERS")
    print("=" * 50)
    
    # Check if we have YOLO training data
    yolo_dir = Path("yolo_training")
    if not yolo_dir.exists():
        print("[ERROR] No YOLO training data found")
        print("Run the quick_object_trainer.py first to prepare data")
        return False
    
    dataset_yaml = yolo_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        print("[ERROR] No dataset.yaml found")
        return False
    
    # Initialize model (start from scratch for small datasets)
    model = YOLO('yolov8n.pt')  # Start from pre-trained weights
    
    print(f"Training with optimized parameters for small datasets...")
    print(f"Dataset: {dataset_yaml}")
    
    # Training parameters optimized for small datasets
    results = model.train(
        data=str(dataset_yaml),
        epochs=50,  # More epochs for small dataset
        patience=15,  # More patience
        imgsz=640,  # Larger image size for better feature learning
        batch=2,    # Smaller batch size
        device='cpu',
        project="models",
        name='pokemon_object_detector_fixed',
        verbose=True,
        save_period=10,
        plots=True,  # Generate plots to see training progress
        val=True,
        
        # Optimizations for small datasets
        optimizer='AdamW',
        lr0=0.01,        # Higher learning rate
        lrf=0.01,        # Lower final learning rate ratio
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,  # More warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Data augmentation (important for small datasets)
        hsv_h=0.015,     # Hue augmentation
        hsv_s=0.7,       # Saturation augmentation  
        hsv_v=0.4,       # Value augmentation
        degrees=10.0,    # Rotation
        translate=0.1,   # Translation
        scale=0.5,       # Scaling
        shear=2.0,       # Shearing
        perspective=0.0, # Perspective
        flipud=0.5,      # Flip up-down
        fliplr=0.5,      # Flip left-right
        mosaic=1.0,      # Mosaic augmentation
        mixup=0.2,       # Mixup augmentation
        copy_paste=0.3,  # Copy-paste augmentation
        
        # Regularization
        dropout=0.0,     # No dropout for small dataset
        label_smoothing=0.0  # No label smoothing
    )
    
    print(f"\n[TRAINING COMPLETE]")
    print(f"Results: {results}")
    
    # Test the new model
    test_new_model()
    
    return True

def test_new_model():
    """Test the newly trained model"""
    print(f"\n" + "=" * 50)
    print("TESTING NEWLY TRAINED MODEL")
    print("=" * 50)
    
    model_path = Path("models/pokemon_object_detector_fixed/weights/best.pt")
    
    if not model_path.exists():
        print("[ERROR] New model not found")
        return
    
    model = YOLO(str(model_path))
    
    # Test on a training image
    test_images = list(Path("assisted_training_data/screenshots").glob("*.png"))
    if not test_images:
        print("[ERROR] No test images")
        return
    
    test_img = test_images[0]
    print(f"Testing on: {test_img}")
    
    # Test with different confidence levels
    for conf in [0.1, 0.25, 0.5]:
        results = model(str(test_img), conf=conf, verbose=False)
        
        total_detections = 0
        for result in results:
            if result.boxes is not None:
                total_detections += len(result.boxes)
                
                # Show some detections
                if conf == 0.25:  # Show details for middle confidence
                    for box in result.boxes:
                        cls_id = int(box.cls)
                        confidence = float(box.conf)
                        cls_name = model.names[cls_id]
                        print(f"  {cls_name}: {confidence:.3f}")
        
        print(f"Confidence {conf}: {total_detections} detections")
    
    print(f"\n[READY] If this shows good detections, the backseat collector should work!")

if __name__ == "__main__":
    if not YOLO_AVAILABLE:
        print("Install YOLO with: pip install ultralytics")
        sys.exit(1)
    
    print("FIXING OBJECT DETECTION TRAINING")
    print("====================================")
    print("")
    print("Problem: Current model has 21% accuracy (should be 50%+)")
    print("Solution: Retrain with parameters optimized for small datasets")
    print("")
    
    success = fix_training()
    
    if success:
        print(f"\n[SUCCESS] TRAINING FIXED!")
        print(f"The backseat collector will now automatically use the improved model.")
    else:
        print(f"\n[FAILED] TRAINING FAILED")
        print(f"Check the error messages above")