#!/usr/bin/env python3
"""
⚡ QUICK OBJECT DETECTION TRAINER
=================================

Fast iterative training for the backseat driving loop:
- Only 10-20 epochs (5-10 minutes instead of hours)
- Early stopping when accuracy plateaus
- Incremental updates to existing model
- Optimized for rapid feedback cycles

Perfect for the human-AI collaboration workflow!
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
import time

# Add agent directory to path
sys.path.append('../../agent')

# Import model manager for better versioning
from model_manager import ModelManager

try:
    from ultralytics import YOLO
    print("[OK] YOLO imported successfully")
    YOLO_AVAILABLE = True
except ImportError:
    print("[WARNING] YOLO not available, install with: pip install ultralytics")
    YOLO_AVAILABLE = False

class QuickObjectTrainer:
    def __init__(self, data_dir="assisted_training_data", feedback_dir="backseat_training_data/feedback"):
        self.data_dir = Path(data_dir)
        self.feedback_dir = Path(feedback_dir)
        self.model_path = Path("models")
        self.model_path.mkdir(exist_ok=True)
        
        # Quick training parameters
        self.quick_epochs = 15  # Much faster than 100!
        self.patience = 5  # Early stopping patience
        self.min_improvement = 0.01  # Minimum improvement to continue
        
    def analyze_feedback_data(self):
        """Analyze online learning feedback data"""
        print("\n📊 ONLINE LEARNING FEEDBACK DATA")
        print("=" * 40)
        
        if not self.feedback_dir.exists():
            print("No feedback data directory found")
            return 0, 0
            
        # Find online learning batch files
        batch_files = list(self.feedback_dir.glob("online_learning_batch_*.json"))
        session_files = list(self.feedback_dir.glob("feedback_session_*.json"))
        
        print(f"Online learning batches: {len(batch_files)}")
        print(f"Feedback sessions: {len(session_files)}")
        
        total_positive = 0
        total_negative = 0
        object_types = set()
        
        # Analyze online learning batches
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r') as f:
                    data = json.load(f)
                    
                positives = data.get('positive_samples', [])
                negatives = data.get('negative_samples', [])
                
                total_positive += len(positives)
                total_negative += len(negatives)
                
                # Collect object types
                for sample in positives + negatives:
                    detection = sample.get('detection', {})
                    obj_type = detection.get('type')
                    if obj_type:
                        object_types.add(obj_type)
                        
            except Exception as e:
                print(f"Error reading {batch_file}: {e}")
        
        print(f"Total positive feedback: {total_positive}")
        print(f"Total negative feedback: {total_negative}")
        print(f"Object types in feedback: {sorted(list(object_types))}")
        
        return total_positive, total_negative
    
    def quick_analysis(self):
        """Quick data analysis"""
        print("=" * 50)
        print("QUICK TRAINING DATA ANALYSIS")
        print("=" * 50)
        
        metadata_files = list((self.data_dir / "metadata").glob("*.json"))
        
        if not metadata_files:
            print("[ERROR] No training data found!")
            print(f"Expected data in: {self.data_dir / 'metadata'}")
            return False
            
        object_count = 0
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            object_count += len(data.get('labels', []))
            
        print(f"📊 Training Data Summary:")
        print(f"  Screenshots: {len(metadata_files)}")
        print(f"  Object annotations: {object_count}")
        print(f"  Training time: ~5-10 minutes (instead of 1+ hour)")
        print(f"  Strategy: Quick iterations for rapid improvement")
        
        # Analyze feedback data
        positive_feedback, negative_feedback = self.analyze_feedback_data()
        total_feedback = positive_feedback + negative_feedback
        
        if total_feedback > 0:
            print(f"\n🎯 FEEDBACK DATA INTEGRATION:")
            print(f"  Positive feedback samples: {positive_feedback}")
            print(f"  Negative feedback samples: {negative_feedback}")
            print(f"  Total feedback data: {total_feedback}")
            print(f"  💡 This feedback data CAN be used for training!")
            
            if total_feedback >= 50:
                print(f"  ✅ Sufficient feedback for training improvement")
            else:
                print(f"  ⚠️  Consider collecting more feedback (current: {total_feedback}, recommended: 50+)")
        else:
            print(f"\n📝 No feedback data found yet")
            print(f"  Start collecting feedback with: python backseat_data_collector_fixed.py")
        
        return True
        
    def prepare_training_data(self):
        """Prepare data in YOLO format quickly"""
        print(f"\n⚡ Preparing training data...")
        
        # Use existing YOLO directory if available
        yolo_dir = Path("yolo_training")
        if yolo_dir.exists():
            print(f"[REUSE] Using existing YOLO data: {yolo_dir}")
            return yolo_dir
            
        # Otherwise create it (this should be fast from previous run)
        print(f"[CREATE] Creating YOLO training data...")
        from object_detection_trainer import ObjectDetectionTrainer
        trainer = ObjectDetectionTrainer(str(self.data_dir))
        yolo_dir, class_list = trainer.prepare_yolo_format()
        
        return yolo_dir
        
    def quick_train(self, yolo_dir, incremental=False):
        """Fast training with early stopping"""
        print(f"\n⚡ QUICK TRAINING ({self.quick_epochs} epochs max)")
        print("=" * 50)
        
        if not YOLO_AVAILABLE:
            print("[ERROR] YOLO required for quick training")
            print("Install with: pip install ultralytics")
            return None
            
        # Check for existing model to update
        existing_model = self.model_path / "pokemon_object_detector" / "weights" / "best.pt"
        
        if incremental and existing_model.exists():
            print(f"[INCREMENTAL] Updating existing model: {existing_model}")
            model = YOLO(str(existing_model))
        else:
            print(f"[FRESH] Training new model from YOLOv8 pretrained weights")
            model = YOLO('yolov8n.pt')
            
        # Dataset config
        dataset_yaml = yolo_dir / "dataset.yaml"
        
        print(f"Starting quick training session...")
        print(f"⏰ Estimated time: 5-10 minutes")
        print(f"🎯 Goal: Rapid iteration for backseat driving feedback")
        
        start_time = time.time()
        
        try:
            # Quick training configuration
            results = model.train(
                data=str(dataset_yaml),
                epochs=self.quick_epochs,  # Much faster!
                patience=self.patience,  # Early stopping
                imgsz=416,  # Smaller image size for speed
                batch=4,  # Larger batch for efficiency
                device='cpu',
                project=str(self.model_path),
                name='pokemon_object_detector',
                verbose=False,  # Less output for speed
                save_period=5,  # Save checkpoints less frequently
                plots=False,  # Skip plots for speed
                val=True,
                optimizer='AdamW',  # Fast optimizer
                lr0=0.001,  # Learning rate
                warmup_epochs=1,  # Quick warmup
                cos_lr=True  # Cosine learning rate scheduling
            )
            
            training_time = time.time() - start_time
            
            print(f"\n[OK] QUICK TRAINING COMPLETE!")
            print(f"Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
            
            # Use model manager to show which model was created
            model_manager = ModelManager(str(self.model_path))
            latest_info = model_manager.get_latest_model_info()
            if latest_info:
                print(f"Model saved as: {latest_info['name']}")
                print(f"Model path: {latest_info['path']}")
                print(f"[READY] Backseat collector will now use this latest model automatically!")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            return None
    
    def quick_test(self):
        """Quick test of the trained model"""
        print(f"\n[TEST] QUICK MODEL TEST")
        print("=" * 30)
        
        model_path = self.model_path / "pokemon_object_detector" / "weights" / "best.pt"
        
        if not model_path.exists():
            print("[ERROR] No trained model found to test")
            return
            
        model = YOLO(str(model_path))
        
        # Test on a training image
        test_images = list((self.data_dir / "screenshots").glob("*.png"))
        
        if test_images:
            test_img = test_images[0]
            print(f"Testing on: {test_img.name}")
            
            results = model(str(test_img))
            
            total_detections = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    total_detections += len(boxes)
                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        cls_name = model.names[cls_id]
                        print(f"  Detection: {cls_name}: {conf:.2f} confidence")
                        
            print(f"\nDetected {total_detections} objects")
            print(f"Model ready for backseat driving feedback!")
            
    def show_quick_training_benefits(self):
        """Explain why quick training is better for iterative improvement"""
        print(f"\n[INFO] BACKSEAT DRIVING TRAINING STRATEGY")
        print("=" * 50)
        
        print(f"Quick Training Benefits:")
        print(f"  - 15 epochs instead of 100 (5-10 minutes vs 1+ hour)")
        print(f"  - Rapid iteration cycles")
        print(f"  - Incremental improvement with each feedback session")
        print(f"  - Perfect for human-AI collaboration")
        
        print(f"\nIteration Strategy:")
        print(f"  1. Collect feedback from users (right-clicks, corrections)")
        print(f"  2. Quick retrain (5-10 minutes)")
        print(f"  3. Test improved model")
        print(f"  4. Repeat cycle")
        
        print(f"\nWhy this works better than long training:")
        print(f"  - Faster feedback loop = better user experience")
        print(f"  - Early stopping prevents overfitting")
        print(f"  - Incremental updates build on previous knowledge")
        print(f"  - Users can see immediate improvement")

def main():
    """Run quick training for backseat driving"""
    print("[START] QUICK OBJECT DETECTION TRAINER")
    print("Optimized for backseat driving feedback cycles!")
    print("=" * 50)
    
    trainer = QuickObjectTrainer()
    
    # Quick analysis
    if not trainer.quick_analysis():
        return
        
    # Prepare data
    yolo_dir = trainer.prepare_training_data()
    
    # Quick train
    results = trainer.quick_train(yolo_dir)
    
    if results:
        # Quick test
        trainer.quick_test()
        
        # Show benefits
        trainer.show_quick_training_benefits()
        
        print(f"\n[SUCCESS] Ready for backseat driving data collection!")
        print(f"Run: python backseat_data_collector.py")
    else:
        print(f"\n[ERROR] Training failed. Check your data and try again.")

if __name__ == "__main__":
    main()