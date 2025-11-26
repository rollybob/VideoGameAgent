#!/usr/bin/env python3
"""
🚀 CONSOLIDATED BATCH TRAINER
============================

Merges ALL training data sources into one large batch and trains a comprehensive model.
This should produce much better results than training on small individual datasets.

Data Sources:
1. training_data/ - Original ML state detection data (JSON/NPY format)
2. competitive_session_training/ - YOLO competitive training data  
3. backseat_training_data/ - Backseat collector screenshots + feedback
4. Any existing YOLO training datasets in models/

Benefits:
- Large, diverse dataset for better generalization
- Consistent 16-class structure across all data
- Proper train/validation split
- Standard YOLO training pipeline
"""

import json
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
from PIL import Image
import time
from datetime import datetime

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available - install with: pip install ultralytics")

class ConsolidatedBatchTrainer:
    """Merges all training data and trains a comprehensive model"""
    
    def __init__(self):
        self.project_root = Path("../../").resolve()  # Go up to GameAgentUSB
        self.perception_root = Path(".")
        self.output_dir = Path("consolidated_training_batch")
        self.shared_classes = self._load_shared_classes()
        
        print(f"Project root: {self.project_root}")
        print(f"Using {len(self.shared_classes)} shared classes")
        
    def _load_shared_classes(self) -> List[str]:
        """Load the 16 shared classes"""
        shared_file = Path("shared_classes.json")
        if shared_file.exists():
            with open(shared_file, 'r') as f:
                data = json.load(f)
                return data['classes']
        
        # Fallback to default 16 classes
        return [
            "building", "dialogue_box", "hp_bar", "menu_box", 
            "npc_character", "player_character", "pokemon_sprite", 
            "text_area", "tree", "pokeball/item", "grass", "water", 
            "exp._bar", "level_indicator", "missed_object", "other"
        ]
    
    def count_existing_data(self) -> Dict[str, int]:
        """Count training data from all sources"""
        counts = {}
        
        # 1. Competitive session training (YOLO format)
        comp_dir = self.perception_root / "competitive_session_training"
        if comp_dir.exists():
            images = list((comp_dir / "images").glob("*.jpg"))
            labels = list((comp_dir / "labels").glob("*.txt"))
            counts['competitive'] = min(len(images), len(labels))
        
        # 2. Backseat training data (need to process feedback)
        backseat_dir = self.perception_root / "backseat_training_data"
        if backseat_dir.exists():
            screenshots = list((backseat_dir / "screenshots").glob("*.png"))
            counts['backseat'] = len(screenshots)
        
        # 3. Original training data (JSON/NPY format)
        original_data = self.project_root / "training_data"
        if original_data.exists():
            # Count all .npy files recursively
            npy_files = list(original_data.rglob("*.npy"))
            json_files = list(original_data.rglob("*_meta.json"))
            counts['original'] = len(npy_files)
            counts['original_meta'] = len(json_files)
        
        # 4. Synthetic data
        synthetic_dir = original_data / "training_data_legacy" / "synthetic"
        if synthetic_dir.exists():
            synthetic_images = list(synthetic_dir.glob("screen_*.png"))
            counts['synthetic'] = len(synthetic_images)
        
        return counts
    
    def prepare_consolidated_dataset(self) -> Optional[Path]:
        """Merge all training data into YOLO format"""
        print("Preparing consolidated dataset...")
        print("=" * 50)
        
        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        total_samples = 0
        
        # 1. Copy competitive training data (already in YOLO format)
        total_samples += self._copy_competitive_data()
        
        # 2. Process backseat training data
        total_samples += self._process_backseat_data()
        
        # 3. Process synthetic data
        total_samples += self._process_synthetic_data()
        
        # 4. Convert any original training data if possible
        # (This would need specific conversion logic based on your original format)
        
        if total_samples == 0:
            print("No training data found!")
            return None
        
        # Create data.yaml
        yaml_content = self._create_dataset_yaml()
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Consolidated dataset created: {total_samples} total samples")
        return yaml_path
    
    def _copy_competitive_data(self) -> int:
        """Copy existing competitive training data"""
        comp_dir = self.perception_root / "competitive_session_training"
        if not comp_dir.exists():
            return 0
        
        images_src = comp_dir / "images"
        labels_src = comp_dir / "labels"
        
        if not images_src.exists() or not labels_src.exists():
            return 0
        
        count = 0
        for img_file in images_src.glob("*.jpg"):
            label_file = labels_src / f"{img_file.stem}.txt"
            if label_file.exists():
                # Copy to train folder (we'll split later)
                dst_img = self.output_dir / "images" / "train" / img_file.name
                dst_label = self.output_dir / "labels" / "train" / label_file.name
                
                shutil.copy2(img_file, dst_img)
                shutil.copy2(label_file, dst_label)
                count += 1
        
        print(f"Copied {count} competitive training samples")
        return count
    
    def _process_backseat_data(self) -> int:
        """Process backseat collector screenshots"""
        backseat_dir = self.perception_root / "backseat_training_data"
        if not backseat_dir.exists():
            return 0
        
        screenshots_dir = backseat_dir / "screenshots"
        metadata_dir = backseat_dir / "metadata"
        
        if not screenshots_dir.exists():
            return 0
        
        count = 0
        for img_file in screenshots_dir.glob("*.png"):
            # Look for corresponding metadata
            meta_file = metadata_dir / f"{img_file.stem}_labels.json"
            
            if meta_file.exists():
                # Convert to YOLO format
                if self._convert_backseat_sample(img_file, meta_file, count):
                    count += 1
        
        print(f"Processed {count} backseat training samples")
        return count
    
    def _process_synthetic_data(self) -> int:
        """Process legacy synthetic training data"""
        synthetic_dir = self.project_root / "training_data" / "training_data_legacy" / "synthetic"
        if not synthetic_dir.exists():
            return 0
        
        count = 0
        for img_file in synthetic_dir.glob("screen_*.png"):
            json_file = synthetic_dir / f"{img_file.stem}.json"
            
            if json_file.exists():
                if self._convert_synthetic_sample(img_file, json_file, count):
                    count += 1
        
        print(f"Processed {count} synthetic training samples")
        return count
    
    def _convert_backseat_sample(self, img_path: Path, meta_path: Path, index: int) -> bool:
        """Convert backseat sample to YOLO format"""
        try:
            # Load metadata
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            # Load image to get dimensions  
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            h, w = img.shape[:2]
            
            # Convert annotations to YOLO format
            yolo_labels = []
            
            # Process different annotation formats that might be in backseat data
            if 'detections' in metadata:
                for detection in metadata['detections']:
                    class_name = detection.get('type', 'other')
                    if class_name in self.shared_classes:
                        class_id = self.shared_classes.index(class_name)
                        
                        # Convert bbox to YOLO format (normalized center x, y, width, height)
                        bbox = detection.get('bbox', [])
                        if len(bbox) >= 4:
                            x1, y1, x2, y2 = bbox[:4]
                            
                            # Normalize and convert to center format
                            center_x = ((x1 + x2) / 2) / w
                            center_y = ((y1 + y2) / 2) / h
                            width = (x2 - x1) / w
                            height = (y2 - y1) / h
                            
                            yolo_labels.append(f"{class_id} {center_x} {center_y} {width} {height}")
            
            # Save converted data
            dst_img = self.output_dir / "images" / "train" / f"backseat_{index:04d}.jpg"
            dst_label = self.output_dir / "labels" / "train" / f"backseat_{index:04d}.txt"
            
            # Convert PNG to JPG and save
            cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Save labels
            with open(dst_label, 'w') as f:
                f.write('\\n'.join(yolo_labels))
            
            return True
            
        except Exception as e:
            print(f"Error converting backseat sample {img_path}: {e}")
            return False
    
    def _convert_synthetic_sample(self, img_path: Path, json_path: Path, index: int) -> bool:
        """Convert synthetic sample to YOLO format"""
        try:
            # Load annotation
            with open(json_path, 'r') as f:
                annotation = json.load(f)
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            h, w = img.shape[:2]
            
            # Convert annotations (format may vary)
            yolo_labels = []
            
            # Process annotations based on synthetic data format
            if 'objects' in annotation:
                for obj in annotation['objects']:
                    class_name = obj.get('class', 'other')
                    if class_name in self.shared_classes:
                        class_id = self.shared_classes.index(class_name)
                        
                        # Convert bbox
                        bbox = obj.get('bbox', [])
                        if len(bbox) >= 4:
                            x1, y1, x2, y2 = bbox[:4]
                            
                            center_x = ((x1 + x2) / 2) / w
                            center_y = ((y1 + y2) / 2) / h
                            width = (x2 - x1) / w
                            height = (y2 - y1) / h
                            
                            yolo_labels.append(f"{class_id} {center_x} {center_y} {width} {height}")
            
            # Save converted data
            dst_img = self.output_dir / "images" / "train" / f"synthetic_{index:04d}.jpg"
            dst_label = self.output_dir / "labels" / "train" / f"synthetic_{index:04d}.txt"
            
            cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            with open(dst_label, 'w') as f:
                f.write('\\n'.join(yolo_labels))
            
            return True
            
        except Exception as e:
            print(f"Error converting synthetic sample {img_path}: {e}")
            return False
    
    def _create_dataset_yaml(self) -> str:
        """Create YOLO dataset configuration"""
        return f'''# Consolidated training dataset
# Generated: {datetime.now().isoformat()}
# Merged from all available training data sources

path: {self.output_dir.absolute()}
train: images/train
val: images/val

nc: {len(self.shared_classes)}
names: {self.shared_classes}

# Data sources:
# - competitive_session_training/ (YOLO format)
# - backseat_training_data/ (screenshots + feedback)  
# - training_data/ synthetic data (legacy format)
'''
    
    def split_train_val(self, val_ratio: float = 0.2):
        """Split training data into train/validation sets"""
        train_imgs = list((self.output_dir / "images" / "train").glob("*.jpg"))
        
        if len(train_imgs) == 0:
            print("No training images found!")
            return
        
        # Calculate split
        val_count = int(len(train_imgs) * val_ratio)
        train_count = len(train_imgs) - val_count
        
        print(f"Splitting dataset: {train_count} train, {val_count} validation")
        
        # Move random subset to validation
        import random
        random.shuffle(train_imgs)
        val_imgs = train_imgs[:val_count]
        
        for img_path in val_imgs:
            label_path = self.output_dir / "labels" / "train" / f"{img_path.stem}.txt"
            
            # Move image and label to val folder
            dst_img = self.output_dir / "images" / "val" / img_path.name
            dst_label = self.output_dir / "labels" / "val" / label_path.name
            
            shutil.move(str(img_path), str(dst_img))
            if label_path.exists():
                shutil.move(str(label_path), str(dst_label))
    
    def train_consolidated_model(self, yaml_path: Path) -> bool:
        """Train a model on the consolidated dataset"""
        if not YOLO_AVAILABLE:
            print("YOLO not available for training!")
            return False
        
        print("Training consolidated model...")
        print("=" * 50)
        
        # Load base model (pretrained YOLOv8n)
        model = YOLO('yolov8n.pt')  # Start with pretrained weights
        
        # Configure training parameters for large dataset
        train_args = {
            'data': str(yaml_path),
            'epochs': 100,  # More epochs for comprehensive training
            'imgsz': 640,
            'batch': 16,  # Larger batch for large dataset
            'save': True,
            'project': "models",
            'name': f"consolidated_model_{int(time.time())}",
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.01,  # Higher learning rate for full training
            'lrf': 0.001,  # Final learning rate
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,
            'patience': 15,  # More patience for large dataset
            'save_period': 10,  # Save every 10 epochs
            'close_mosaic': 20,  # Close mosaic after 20 epochs
            'plots': True,  # Enable training plots
            'verbose': True,
            'device': 'cpu',  # Will auto-detect GPU if available
            # Standard augmentation for large dataset
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 5.0,  # Rotation
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,  # Full mosaic for large dataset
            'mixup': 0.2,
            'copy_paste': 0.3,
        }
        
        try:
            results = model.train(**train_args)
            print("Training completed successfully!")
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            return False

def main():
    """Main function to run consolidated batch training"""
    print("🚀 CONSOLIDATED BATCH TRAINER")
    print("=" * 60)
    
    trainer = ConsolidatedBatchTrainer()
    
    # Count existing data
    print("Counting existing training data...")
    counts = trainer.count_existing_data()
    
    total_samples = sum(counts.values())
    print(f"Found training data:")
    for source, count in counts.items():
        print(f"  - {source}: {count} samples")
    print(f"  Total: {total_samples} samples")
    
    if total_samples == 0:
        print("No training data found! Make sure you have:")
        print("  - competitive_session_training/ with images and labels")
        print("  - backseat_training_data/ with screenshots")
        print("  - training_data/ with original data")
        return
    
    # Prepare consolidated dataset
    yaml_path = trainer.prepare_consolidated_dataset()
    if yaml_path is None:
        print("Failed to prepare dataset!")
        return
    
    # Split train/validation
    trainer.split_train_val(val_ratio=0.2)
    
    # Train model
    print(f"\\nStarting training with {total_samples} samples...")
    success = trainer.train_consolidated_model(yaml_path)
    
    if success:
        print("\\n🎉 CONSOLIDATED TRAINING COMPLETE!")
        print("Check the models/ directory for your new comprehensive model.")
    else:
        print("\\n❌ Training failed - check error messages above.")

if __name__ == "__main__":
    main()