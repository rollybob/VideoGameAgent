#!/usr/bin/env python3
"""
🚀 QUICK BATCH TRAINER
=====================

Consolidates your existing YOLO-format competitive training data
and synthetic data for a larger, more robust training batch.

This focuses on data that can be easily converted and used immediately.
"""

import json
import shutil
import cv2
from pathlib import Path
from typing import List, Dict
import time
import random

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available - install with: pip install ultralytics")

class QuickBatchTrainer:
    """Quick consolidation of ready-to-use training data"""
    
    def __init__(self):
        self.output_dir = Path("quick_batch_training")
        self.shared_classes = self._load_shared_classes()
        
    def _load_shared_classes(self) -> List[str]:
        """Load the 16 shared classes"""
        shared_file = Path("shared_classes.json")
        if shared_file.exists():
            with open(shared_file, 'r') as f:
                data = json.load(f)
                return data['classes']
        
        return [
            "building", "dialogue_box", "hp_bar", "menu_box", 
            "npc_character", "player_character", "pokemon_sprite", 
            "text_area", "tree", "pokeball/item", "grass", "water", 
            "exp._bar", "level_indicator", "missed_object", "other"
        ]
    
    def create_batch_dataset(self) -> Path:
        """Create consolidated batch dataset"""
        print("Creating Quick Batch Dataset...")
        print("=" * 40)
        
        # Setup directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        total_samples = 0
        
        # 1. Copy competitive training data (already YOLO format)
        comp_samples = self._copy_competitive_data()
        total_samples += comp_samples
        
        # 2. Convert and add synthetic data
        synthetic_samples = self._convert_synthetic_data()
        total_samples += synthetic_samples
        
        print(f"Total consolidated samples: {total_samples}")
        
        if total_samples == 0:
            raise ValueError("No training data found!")
        
        # Split train/validation
        self._split_train_val()
        
        # Create dataset YAML
        yaml_path = self._create_yaml()
        
        print(f"Dataset ready: {yaml_path}")
        return yaml_path
    
    def _copy_competitive_data(self) -> int:
        """Copy competitive training data"""
        comp_dir = Path("competitive_session_training")
        if not comp_dir.exists():
            print("No competitive training data found")
            return 0
        
        images_src = comp_dir / "images"
        labels_src = comp_dir / "labels"
        
        count = 0
        for img_file in images_src.glob("*.jpg"):
            label_file = labels_src / f"{img_file.stem}.txt"
            if label_file.exists():
                # Copy to training folder
                dst_img = self.output_dir / "images" / "train" / f"comp_{count:04d}.jpg"
                dst_label = self.output_dir / "labels" / "train" / f"comp_{count:04d}.txt"
                
                shutil.copy2(img_file, dst_img)
                shutil.copy2(label_file, dst_label)
                count += 1
        
        print(f"Added {count} competitive samples")
        return count
    
    def _convert_synthetic_data(self) -> int:
        """Convert synthetic data to YOLO format"""
        synthetic_dir = Path("../../training_data/training_data_legacy/synthetic")
        if not synthetic_dir.exists():
            print("No synthetic data found")
            return 0
        
        count = 0
        for img_file in synthetic_dir.glob("screen_*.png"):
            json_file = synthetic_dir / f"{img_file.stem}.json"
            
            if json_file.exists():
                if self._convert_synthetic_sample(img_file, json_file, count):
                    count += 1
        
        print(f"Converted {count} synthetic samples")
        return count
    
    def _convert_synthetic_sample(self, img_path: Path, json_path: Path, index: int) -> bool:
        """Convert one synthetic sample to YOLO format"""
        try:
            # Load annotation
            with open(json_path, 'r') as f:
                annotation = json.load(f)
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            h, w = img.shape[:2]
            yolo_labels = []
            
            # Convert objects to YOLO format
            if 'objects_detected' in annotation:
                for obj in annotation['objects_detected']:
                    obj_type = obj.get('type', 'other')
                    
                    # Map object type to shared classes
                    if obj_type in self.shared_classes:
                        class_id = self.shared_classes.index(obj_type)
                    elif obj_type == 'character':
                        class_id = self.shared_classes.index('player_character')
                    elif obj_type == 'pokemon':
                        class_id = self.shared_classes.index('pokemon_sprite')
                    else:
                        class_id = self.shared_classes.index('other')
                    
                    # Get bounding box
                    bbox = obj.get('bbox', [])
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[:4]
                        
                        # Convert to YOLO format (normalized center x, y, width, height)
                        center_x = ((x1 + x2) / 2) / w
                        center_y = ((y1 + y2) / 2) / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        
                        # Ensure values are in valid range
                        center_x = max(0, min(1, center_x))
                        center_y = max(0, min(1, center_y))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            
            # Save converted sample
            dst_img = self.output_dir / "images" / "train" / f"synth_{index:04d}.jpg"
            dst_label = self.output_dir / "labels" / "train" / f"synth_{index:04d}.txt"
            
            # Convert PNG to JPG and save
            cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Save labels (even if empty - YOLO can handle empty label files)
            with open(dst_label, 'w') as f:
                f.write('\\n'.join(yolo_labels))
            
            return True
            
        except Exception as e:
            print(f"Error converting synthetic sample {img_path}: {e}")
            return False
    
    def _split_train_val(self, val_ratio: float = 0.2):
        """Split data into train/validation sets"""
        train_imgs = list((self.output_dir / "images" / "train").glob("*.jpg"))
        
        # Calculate split
        val_count = int(len(train_imgs) * val_ratio)
        
        # Randomly select validation samples
        random.shuffle(train_imgs)
        val_imgs = train_imgs[:val_count]
        
        # Move to validation folder
        for img_path in val_imgs:
            label_path = self.output_dir / "labels" / "train" / f"{img_path.stem}.txt"
            
            dst_img = self.output_dir / "images" / "val" / img_path.name
            dst_label = self.output_dir / "labels" / "val" / label_path.name
            
            shutil.move(str(img_path), str(dst_img))
            if label_path.exists():
                shutil.move(str(label_path), str(dst_label))
        
        train_count = len(train_imgs) - val_count
        print(f"Split: {train_count} train, {val_count} validation")
    
    def _create_yaml(self) -> Path:
        """Create dataset YAML file"""
        yaml_content = f'''# Quick Batch Training Dataset
path: {self.output_dir.absolute()}
train: images/train
val: images/val

nc: {len(self.shared_classes)}
names: {self.shared_classes}

# Consolidated from:
# - competitive_session_training/ (77 samples)
# - synthetic training data (100 samples)
'''
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        return yaml_path
    
    def train_batch_model(self, yaml_path: Path) -> bool:
        """Train model on batch dataset"""
        if not YOLO_AVAILABLE:
            print("YOLO not available!")
            return False
        
        print("\\nTraining Batch Model...")
        print("=" * 40)
        
        model = YOLO('yolov8n.pt')  # Start with pretrained weights
        
        # Training configuration optimized for ~150-200 samples
        results = model.train(
            data=str(yaml_path),
            epochs=80,  # Good for moderate dataset
            imgsz=640,
            batch=16,   # Good batch size for training
            save=True,
            project="models",
            name=f"batch_model_{int(time.time())}",
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.01,   # Standard learning rate
            lrf=0.001,  # Final learning rate
            patience=12, # Allow for good convergence
            save_period=10,
            close_mosaic=15,
            plots=True,
            verbose=True,
            device='cpu',  # Auto-detect GPU
            # Standard augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.2,
        )
        
        print("\\nTraining completed!")
        return True

def main():
    """Run quick batch training"""
    print("🚀 QUICK BATCH TRAINER")
    print("=" * 50)
    print("Consolidating competitive + synthetic training data")
    print("for a larger, more robust training batch.")
    print()
    
    try:
        trainer = QuickBatchTrainer()
        
        # Create consolidated dataset
        yaml_path = trainer.create_batch_dataset()
        
        # Train model
        if input("\\nStart training? (y/n): ").lower().startswith('y'):
            success = trainer.train_batch_model(yaml_path)
            
            if success:
                print("\\n🎉 BATCH TRAINING COMPLETE!")
                print("Your new model should perform much better with the larger dataset!")
            else:
                print("\\n❌ Training failed")
        else:
            print(f"\\nDataset ready at: {yaml_path}")
            print("Run training manually with: yolo train data={yaml_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()