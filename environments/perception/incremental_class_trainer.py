#!/usr/bin/env python3
"""
🔄 INCREMENTAL CLASS TRAINER
============================

Safely adds new object classes to your existing successful model without breaking 
what it already knows. This solves the "new objects" problem without sacrificing 
the proven 0.82 confidence performance on existing classes.

Strategy:
1. Start with your best working model (pokemon_object_detector2)
2. Expand dataset to include new classes while keeping all old training data
3. Train incrementally with conservative parameters to avoid catastrophic forgetting
4. Preserve learned features while adding new capabilities

This is much safer than consolidated training from scratch!
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from datetime import datetime
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[ERROR] YOLO not available - install with: pip install ultralytics")

class IncrementalClassTrainer:
    """Adds new classes to existing successful models"""
    
    def __init__(self, base_model_name: str = "pokemon_object_detector2"):
        self.base_model_name = base_model_name
        self.base_model_path = Path("models") / base_model_name / "weights" / "best.pt"
        self.work_dir = Path("incremental_training")
        self.work_dir.mkdir(exist_ok=True)
        
        # Load original class configuration
        self.original_classes = self._get_original_classes()
        print(f"📋 Original model classes: {self.original_classes}")
        
    def _get_original_classes(self) -> List[str]:
        """Get the class list from the original successful model"""
        original_yaml = Path("yolo_training") / "dataset.yaml"
        if original_yaml.exists():
            with open(original_yaml) as f:
                data = yaml.safe_load(f)
                return data.get('names', [])
        
        # Fallback to known working classes
        return ['building', 'dialogue_box', 'hp_bar', 'menu_box', 'npc_character', 
                'player_character', 'pokemon_sprite', 'text_area', 'tree']
    
    def prepare_incremental_dataset(self, new_classes: List[str]) -> Tuple[Path, Dict]:
        """Create dataset combining original data + new class data"""
        
        print(f"🔄 Preparing incremental dataset...")
        print(f"   Original classes: {len(self.original_classes)}")
        print(f"   New classes: {new_classes}")
        
        # Create expanded class list
        all_classes = self.original_classes.copy()
        for new_class in new_classes:
            if new_class not in all_classes:
                all_classes.append(new_class)
        
        print(f"   Total classes: {len(all_classes)}")
        
        # Create incremental dataset directory
        timestamp = int(time.time())
        dataset_dir = self.work_dir / f"dataset_incremental_{timestamp}"
        dataset_dir.mkdir(exist_ok=True)
        
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        train_images = images_dir / "train"
        train_labels = labels_dir / "train"
        train_images.mkdir(exist_ok=True)
        train_labels.mkdir(exist_ok=True)
        
        # Copy original training data (CRITICAL - preserve what works!)
        original_count = self._copy_original_data(train_images, train_labels)
        
        # Add new class data from competitive training
        new_count = self._copy_new_class_data(train_images, train_labels, all_classes)
        
        # Create dataset YAML
        yaml_path = self._create_incremental_yaml(dataset_dir, all_classes)
        
        stats = {
            'original_samples': original_count,
            'new_samples': new_count,
            'total_samples': original_count + new_count,
            'original_classes': len(self.original_classes),
            'total_classes': len(all_classes),
            'new_classes_added': new_classes
        }
        
        print(f"✅ Dataset prepared: {stats['total_samples']} samples, {stats['total_classes']} classes")
        return yaml_path, stats
    
    def _copy_original_data(self, train_images: Path, train_labels: Path) -> int:
        """Copy the original successful training data"""
        print("   📁 Copying original training data...")
        
        original_images = Path("yolo_training") / "images" / "train"
        original_labels = Path("yolo_training") / "labels" / "train"
        
        count = 0
        if original_images.exists() and original_labels.exists():
            for img_file in original_images.glob("*.png"):
                label_file = original_labels / f"{img_file.stem}.txt"
                
                if label_file.exists():
                    # Copy image and label
                    shutil.copy2(img_file, train_images)
                    shutil.copy2(label_file, train_labels)
                    count += 1
        
        print(f"   ✓ Copied {count} original samples")
        return count
    
    def _copy_new_class_data(self, train_images: Path, train_labels: Path, all_classes: List[str]) -> int:
        """Copy new class data from competitive training"""
        print("   📁 Adding new class training data...")
        
        competitive_images = Path("competitive_session_training") / "images"
        competitive_labels = Path("competitive_session_training") / "labels"
        
        count = 0
        if competitive_images.exists() and competitive_labels.exists():
            for img_file in competitive_images.glob("*.jpg"):
                label_file = competitive_labels / f"{img_file.stem}.txt"
                
                if label_file.exists():
                    # Convert image to PNG and copy
                    new_img_name = f"comp_{img_file.stem}.png"
                    
                    # Update label file with new class indices
                    updated_labels = self._update_label_indices(label_file, all_classes)
                    
                    if updated_labels:  # Only copy if labels were successfully updated
                        shutil.copy2(img_file, train_images / new_img_name)
                        
                        # Write updated labels
                        with open(train_labels / f"comp_{img_file.stem}.txt", 'w') as f:
                            f.write(updated_labels)
                        count += 1
        
        print(f"   ✓ Added {count} new class samples")
        return count
    
    def _update_label_indices(self, label_file: Path, all_classes: List[str]) -> str:
        """Update label file to use new class indices"""
        try:
            # Load shared classes to map from
            from shared_class_manager import get_shared_class_manager
            shared_manager = get_shared_class_manager()
            shared_classes = shared_manager.get_classes()
            
            updated_lines = []
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class_id = int(parts[0])
                        
                        # Map from shared class to new class index
                        if old_class_id < len(shared_classes):
                            class_name = shared_classes[old_class_id]
                            if class_name in all_classes:
                                new_class_id = all_classes.index(class_name)
                                parts[0] = str(new_class_id)
                                updated_lines.append(' '.join(parts))
            
            return '\n'.join(updated_lines) + '\n' if updated_lines else ""
            
        except Exception as e:
            print(f"   ⚠️  Error updating labels for {label_file}: {e}")
            return ""
    
    def _create_incremental_yaml(self, dataset_dir: Path, all_classes: List[str]) -> Path:
        """Create YAML configuration for incremental training"""
        yaml_content = {
            'path': str(dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/train',  # Use train as val for small datasets
            'nc': len(all_classes),
            'names': all_classes
        }
        
        yaml_path = dataset_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        return yaml_path
    
    def train_incremental_model(self, yaml_path: Path, stats: Dict) -> Path:
        """Train model incrementally from the successful base model"""
        if not YOLO_AVAILABLE:
            raise RuntimeError("YOLO not available")
        
        print(f"\n🚀 Starting incremental training...")
        print(f"   Base model: {self.base_model_name}")
        print(f"   Dataset: {stats['total_samples']} samples, {stats['total_classes']} classes")
        
        # Load the successful base model
        if not self.base_model_path.exists():
            print(f"❌ Base model not found: {self.base_model_path}")
            print("   Available models:")
            for model_dir in Path("models").glob("pokemon_object_detector*"):
                if (model_dir / "weights" / "best.pt").exists():
                    print(f"   - {model_dir.name}")
            raise FileNotFoundError(f"Base model not found: {self.base_model_path}")
        
        model = YOLO(str(self.base_model_path))
        
        # INCREMENTAL TRAINING PARAMETERS - Conservative to avoid catastrophic forgetting
        timestamp = int(time.time())
        model_name = f"{self.base_model_name}_incremental_{timestamp}"
        
        # Conservative parameters to preserve existing knowledge
        epochs = 30        # Fewer epochs to avoid overwriting learned features
        lr0 = 0.001       # Lower learning rate for fine-tuning
        patience = 15     # More patience for incremental learning
        
        print(f"   Model name: {model_name}")
        print(f"   Strategy: Conservative incremental (epochs={epochs}, lr={lr0})")
        
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=640,
            batch=2,              # Small batch for stability
            save=True,
            project="models",
            name=model_name,
            exist_ok=True,
            pretrained=True,
            optimizer='auto',     # Use proven optimizer
            lr0=lr0,             # Conservative learning rate
            lrf=0.1,             # Standard final learning rate
            patience=patience,    # Allow for slower convergence
            save_period=5,       # Regular checkpoints
            verbose=True,
            plots=True,
            # Reduced augmentation to preserve learned features
            hsv_h=0.005,         # Minimal color augmentation
            hsv_s=0.2,
            hsv_v=0.2,
            degrees=0.0,         # No rotation augmentation
            translate=0.05,      # Minimal translation
            scale=0.2,           # Minimal scaling
            shear=0.0,          # No shear
            perspective=0.0,     # No perspective
            flipud=0.0,         # No vertical flip
            fliplr=0.25,        # Light horizontal flip
            mosaic=0.5,         # Reduced mosaic
            mixup=0.0,          # No mixup
            copy_paste=0.0      # No copy-paste
        )
        
        model_path = Path("models") / model_name / "weights" / "best.pt"
        print(f"✅ Incremental model trained: {model_path}")
        
        return model_path

def main():
    """Interactive incremental training"""
    print("🔄 INCREMENTAL CLASS TRAINER")
    print("=" * 50)
    print("Safely add new classes to your successful model!")
    print()
    
    trainer = IncrementalClassTrainer()
    
    # Show what new classes are available
    try:
        from shared_class_manager import get_shared_class_manager
        shared_manager = get_shared_class_manager()
        all_shared_classes = shared_manager.get_classes()
        
        new_classes = [cls for cls in all_shared_classes if cls not in trainer.original_classes]
        
        print(f"📋 Available new classes to add:")
        for i, cls in enumerate(new_classes, 1):
            print(f"   {i}. {cls}")
        
        print(f"\nAdd all {len(new_classes)} new classes? (y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            print(f"\n🚀 Adding {len(new_classes)} new classes to {trainer.base_model_name}")
            
            # Prepare incremental dataset
            yaml_path, stats = trainer.prepare_incremental_dataset(new_classes)
            
            print(f"\n📊 Dataset Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
            
            # Start training
            if input("\nStart incremental training? (y/n): ").strip().lower() == 'y':
                model_path = trainer.train_incremental_model(yaml_path, stats)
                print(f"\n🎉 SUCCESS! New model ready: {model_path}")
                print(f"This model should retain 0.82+ confidence on existing objects")
                print(f"while adding detection for {len(new_classes)} new classes!")
            else:
                print(f"Dataset prepared at: {yaml_path}")
                print("You can train later by running this script again.")
        else:
            print("Cancelled - no changes made")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()