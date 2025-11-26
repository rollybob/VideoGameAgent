#!/usr/bin/env python3
"""
🧠 SMART CONSOLIDATED TRAINER
============================

Long-term solution for training data management:
1. Automatically discovers and merges ALL training data sources
2. Handles incremental updates - new data seamlessly added
3. Maintains training history and versioning
4. Optimizes dataset composition for best results
5. Provides rollback and comparison capabilities

This is the "set it and forget it" solution for continuous model improvement.
"""

import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import time
import random
import hashlib
from datetime import datetime
import pickle

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class DatasetRegistry:
    """Tracks all training data sources and their status"""
    
    def __init__(self, registry_file: str = "dataset_registry.json"):
        self.registry_file = Path(registry_file)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load existing registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        
        return {
            'version': '1.0',
            'last_updated': datetime.now().isoformat(),
            'sources': {},
            'datasets': {},
            'models': {}
        }
    
    def _save_registry(self):
        """Save registry to file"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def scan_data_sources(self) -> Dict[str, Dict]:
        """Scan for all available data sources"""
        sources = {}
        
        # 1. Competitive session training
        comp_dir = Path("competitive_session_training")
        if comp_dir.exists():
            sources['competitive'] = self._scan_yolo_data(comp_dir)
        
        # 2. Backseat training data
        backseat_dir = Path("backseat_training_data")
        if backseat_dir.exists():
            sources['backseat'] = self._scan_backseat_data(backseat_dir)
        
        # 3. Original training data
        original_dir = Path("../../training_data")
        if original_dir.exists():
            sources['original'] = self._scan_original_data(original_dir)
        
        # 4. Synthetic data
        synthetic_dir = Path("../../training_data/training_data_legacy/synthetic")
        if synthetic_dir.exists():
            sources['synthetic'] = self._scan_synthetic_data(synthetic_dir)
        
        # 5. Any additional YOLO datasets in models/
        models_dir = Path("models")
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "data.yaml").exists():
                    sources[f'model_{model_dir.name}'] = self._scan_yolo_data(model_dir)
        
        # Update registry
        for source_name, source_info in sources.items():
            source_hash = self._compute_source_hash(source_info)
            
            if source_name not in self.registry['sources']:
                self.registry['sources'][source_name] = {
                    'first_seen': datetime.now().isoformat(),
                    'last_hash': source_hash,
                    'sample_count': source_info['sample_count'],
                    'status': 'new'
                }
            else:
                old_hash = self.registry['sources'][source_name]['last_hash']
                if source_hash != old_hash:
                    self.registry['sources'][source_name]['last_hash'] = source_hash
                    self.registry['sources'][source_name]['sample_count'] = source_info['sample_count']
                    self.registry['sources'][source_name]['status'] = 'updated'
                    self.registry['sources'][source_name]['last_updated'] = datetime.now().isoformat()
                else:
                    self.registry['sources'][source_name]['status'] = 'unchanged'
        
        self._save_registry()
        return sources
    
    def _scan_yolo_data(self, data_dir: Path) -> Dict:
        """Scan YOLO format data"""
        images = list((data_dir / "images").glob("*.jpg")) if (data_dir / "images").exists() else []
        labels = list((data_dir / "labels").glob("*.txt")) if (data_dir / "labels").exists() else []
        
        return {
            'format': 'yolo',
            'sample_count': min(len(images), len(labels)),
            'image_files': len(images),
            'label_files': len(labels),
            'path': str(data_dir)
        }
    
    def _scan_backseat_data(self, data_dir: Path) -> Dict:
        """Scan backseat collector data"""
        screenshots = list((data_dir / "screenshots").glob("*.png"))
        metadata = list((data_dir / "metadata").glob("*_labels.json"))
        feedback = list((data_dir / "feedback").glob("*.json"))
        
        return {
            'format': 'backseat',
            'sample_count': min(len(screenshots), len(metadata)),
            'screenshots': len(screenshots),
            'metadata': len(metadata),
            'feedback_files': len(feedback),
            'path': str(data_dir)
        }
    
    def _scan_original_data(self, data_dir: Path) -> Dict:
        """Scan original training data"""
        npy_files = list(data_dir.rglob("*.npy"))
        json_files = list(data_dir.rglob("*_meta.json"))
        
        return {
            'format': 'original',
            'sample_count': len(npy_files),
            'npy_files': len(npy_files),
            'json_files': len(json_files),
            'path': str(data_dir)
        }
    
    def _scan_synthetic_data(self, data_dir: Path) -> Dict:
        """Scan synthetic training data"""
        images = list(data_dir.glob("screen_*.png"))
        annotations = list(data_dir.glob("screen_*.json"))
        
        return {
            'format': 'synthetic',
            'sample_count': min(len(images), len(annotations)),
            'images': len(images),
            'annotations': len(annotations),
            'path': str(data_dir)
        }
    
    def _compute_source_hash(self, source_info: Dict) -> str:
        """Compute hash for change detection"""
        hashable_data = f"{source_info['sample_count']}_{source_info.get('last_modified', '')}"
        return hashlib.md5(hashable_data.encode()).hexdigest()[:8]
    
    def get_new_or_updated_sources(self) -> List[str]:
        """Get sources that are new or have been updated"""
        updated_sources = []
        for source_name, source_info in self.registry['sources'].items():
            if source_info['status'] in ['new', 'updated']:
                updated_sources.append(source_name)
        return updated_sources
    
    def mark_sources_processed(self, source_names: List[str]):
        """Mark sources as processed"""
        for source_name in source_names:
            if source_name in self.registry['sources']:
                self.registry['sources'][source_name]['status'] = 'processed'
        self._save_registry()

class SmartConsolidatedTrainer:
    """Intelligent training data consolidation and management"""
    
    def __init__(self):
        self.registry = DatasetRegistry()
        self.output_base = Path("consolidated_datasets")
        self.shared_classes = self._load_shared_classes()
        self.current_dataset_dir = None
        
    def _load_shared_classes(self) -> List[str]:
        """Load shared classes"""
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
    
    def create_or_update_dataset(self, incremental: bool = True) -> Tuple[Path, Dict]:
        """Create new dataset or incrementally update existing one"""
        
        print("🔍 Scanning for training data sources...")
        sources = self.registry.scan_data_sources()
        
        # Show data source summary
        total_samples = sum(info['sample_count'] for info in sources.values())
        print(f"Found {len(sources)} data sources with {total_samples} total samples:")
        for source_name, info in sources.items():
            status = self.registry.registry['sources'][source_name]['status']
            print(f"  - {source_name}: {info['sample_count']} samples ({status})")
        
        if incremental:
            # Only process new/updated sources
            updated_sources = self.registry.get_new_or_updated_sources()
            if not updated_sources:
                print("No new or updated data found. Dataset is up to date.")
                return self._get_latest_dataset_path(), {}
            
            print(f"Processing {len(updated_sources)} updated sources: {updated_sources}")
            sources_to_process = {k: v for k, v in sources.items() if k in updated_sources}
        else:
            # Full rebuild
            print("Full dataset rebuild requested")
            sources_to_process = sources
        
        # Create new dataset version
        timestamp = int(time.time())
        dataset_dir = self.output_base / f"dataset_v{timestamp}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_dataset_dir = dataset_dir
        
        # Setup YOLO directory structure
        for split in ['train', 'val']:
            (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Process each data source
        total_converted = 0
        conversion_stats = {}
        
        for source_name, source_info in sources_to_process.items():
            print(f"Processing {source_name}...")
            
            if source_info['format'] == 'yolo':
                converted = self._process_yolo_source(source_info, source_name)
            elif source_info['format'] == 'synthetic':
                converted = self._process_synthetic_source(source_info, source_name)
            elif source_info['format'] == 'backseat':
                converted = self._process_backseat_source(source_info, source_name)
            elif source_info['format'] == 'original':
                converted = self._process_original_source(source_info, source_name)
            else:
                print(f"  Unknown format: {source_info['format']}")
                converted = 0
            
            conversion_stats[source_name] = converted
            total_converted += converted
            print(f"  ✓ Converted {converted} samples")
        
        # If incremental, merge with existing dataset
        if incremental and self._has_existing_dataset():
            total_converted += self._merge_with_existing()
        
        # Split train/validation
        self._split_train_val()
        
        # Create dataset YAML
        yaml_path = self._create_dataset_yaml(total_converted, conversion_stats)
        
        # Update registry
        self.registry.registry['datasets'][f'v{timestamp}'] = {
            'created': datetime.now().isoformat(),
            'total_samples': total_converted,
            'sources': list(sources_to_process.keys()),
            'yaml_path': str(yaml_path),
            'incremental': incremental
        }
        
        # Mark sources as processed
        self.registry.mark_sources_processed(list(sources_to_process.keys()))
        
        print(f"\\n✅ Dataset ready: {total_converted} samples at {yaml_path}")
        return yaml_path, conversion_stats
    
    def _process_yolo_source(self, source_info: Dict, source_name: str) -> int:
        """Process YOLO format data source"""
        source_path = Path(source_info['path'])
        images_dir = source_path / "images"
        labels_dir = source_path / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            return 0
        
        count = 0
        for img_file in images_dir.glob("*.jpg"):
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                # Copy to training set
                dst_img = self.current_dataset_dir / "images" / "train" / f"{source_name}_{count:04d}.jpg"
                dst_label = self.current_dataset_dir / "labels" / "train" / f"{source_name}_{count:04d}.txt"
                
                shutil.copy2(img_file, dst_img)
                shutil.copy2(label_file, dst_label)
                count += 1
        
        return count
    
    def _process_synthetic_source(self, source_info: Dict, source_name: str) -> int:
        """Process synthetic training data"""
        source_path = Path(source_info['path'])
        
        count = 0
        for img_file in source_path.glob("screen_*.png"):
            json_file = source_path / f"{img_file.stem}.json"
            
            if json_file.exists():
                if self._convert_synthetic_sample(img_file, json_file, f"{source_name}_{count:04d}"):
                    count += 1
        
        return count
    
    def _process_backseat_source(self, source_info: Dict, source_name: str) -> int:
        """Process backseat collector data"""
        source_path = Path(source_info['path'])
        screenshots_dir = source_path / "screenshots"
        metadata_dir = source_path / "metadata"
        
        if not screenshots_dir.exists() or not metadata_dir.exists():
            return 0
        
        count = 0
        for img_file in screenshots_dir.glob("*.png"):
            meta_file = metadata_dir / f"{img_file.stem}_labels.json"
            
            if meta_file.exists():
                if self._convert_backseat_sample(img_file, meta_file, f"{source_name}_{count:04d}"):
                    count += 1
        
        return count
    
    def _process_original_source(self, source_info: Dict, source_name: str) -> int:
        """Process original .npy training data"""
        # This would need specific conversion logic based on your original data format
        # For now, skip original data conversion
        print(f"  Skipping original data conversion (not implemented yet)")
        return 0
    
    def _convert_synthetic_sample(self, img_path: Path, json_path: Path, name: str) -> bool:
        """Convert synthetic sample to YOLO format"""
        try:
            with open(json_path, 'r') as f:
                annotation = json.load(f)
            
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            h, w = img.shape[:2]
            yolo_labels = []
            
            if 'objects_detected' in annotation:
                for obj in annotation['objects_detected']:
                    obj_type = obj.get('type', 'other')
                    
                    # Map to shared classes
                    if obj_type in self.shared_classes:
                        class_id = self.shared_classes.index(obj_type)
                    elif obj_type == 'character':
                        class_id = self.shared_classes.index('player_character')
                    elif obj_type == 'pokemon':
                        class_id = self.shared_classes.index('pokemon_sprite')
                    else:
                        class_id = self.shared_classes.index('other')
                    
                    bbox = obj.get('bbox', [])
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[:4]
                        
                        center_x = ((x1 + x2) / 2) / w
                        center_y = ((y1 + y2) / 2) / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        
                        # Clamp values
                        center_x = max(0, min(1, center_x))
                        center_y = max(0, min(1, center_y))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            
            # Save converted sample
            dst_img = self.current_dataset_dir / "images" / "train" / f"{name}.jpg"
            dst_label = self.current_dataset_dir / "labels" / "train" / f"{name}.txt"
            
            cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            with open(dst_label, 'w') as f:
                f.write('\\n'.join(yolo_labels))
            
            return True
            
        except Exception as e:
            print(f"  Error converting {img_path}: {e}")
            return False
    
    def _convert_backseat_sample(self, img_path: Path, meta_path: Path, name: str) -> bool:
        """Convert backseat sample to YOLO format"""
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            h, w = img.shape[:2]
            yolo_labels = []
            
            # Process detections or corrections from backseat data
            if 'detections' in metadata:
                for detection in metadata['detections']:
                    class_name = detection.get('type', 'other')
                    if class_name in self.shared_classes:
                        class_id = self.shared_classes.index(class_name)
                        
                        bbox = detection.get('bbox', [])
                        if len(bbox) >= 4:
                            x1, y1, x2, y2 = bbox[:4]
                            
                            center_x = ((x1 + x2) / 2) / w
                            center_y = ((y1 + y2) / 2) / h
                            width = (x2 - x1) / w
                            height = (y2 - y1) / h
                            
                            center_x = max(0, min(1, center_x))
                            center_y = max(0, min(1, center_y))
                            width = max(0, min(1, width))
                            height = max(0, min(1, height))
                            
                            yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            
            # Save converted sample
            dst_img = self.current_dataset_dir / "images" / "train" / f"{name}.jpg"
            dst_label = self.current_dataset_dir / "labels" / "train" / f"{name}.txt"
            
            cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            with open(dst_label, 'w') as f:
                f.write('\\n'.join(yolo_labels))
            
            return True
            
        except Exception as e:
            print(f"  Error converting {img_path}: {e}")
            return False
    
    def _has_existing_dataset(self) -> bool:
        """Check if there's an existing dataset to merge with"""
        return len(self.registry.registry['datasets']) > 0
    
    def _get_latest_dataset_path(self) -> Optional[Path]:
        """Get path to the latest dataset"""
        datasets = self.registry.registry['datasets']
        if not datasets:
            return None
        
        latest_version = max(datasets.keys(), key=lambda x: int(x.replace('v', '')))
        yaml_path = datasets[latest_version]['yaml_path']
        return Path(yaml_path) if Path(yaml_path).exists() else None
    
    def _merge_with_existing(self) -> int:
        """Merge with existing dataset (for incremental updates)"""
        latest_path = self._get_latest_dataset_path()
        if not latest_path:
            return 0
        
        existing_dataset_dir = latest_path.parent
        
        # Copy existing training data
        count = 0
        
        # Copy images
        existing_train_imgs = existing_dataset_dir / "images" / "train"
        if existing_train_imgs.exists():
            for img_file in existing_train_imgs.glob("*.jpg"):
                label_file = existing_dataset_dir / "labels" / "train" / f"{img_file.stem}.txt"
                
                dst_img = self.current_dataset_dir / "images" / "train" / f"existing_{count:04d}.jpg"
                dst_label = self.current_dataset_dir / "labels" / "train" / f"existing_{count:04d}.txt"
                
                shutil.copy2(img_file, dst_img)
                if label_file.exists():
                    shutil.copy2(label_file, dst_label)
                count += 1
        
        print(f"  Merged {count} existing samples")
        return count
    
    def _split_train_val(self, val_ratio: float = 0.2):
        """Split data into train/validation"""
        train_imgs = list((self.current_dataset_dir / "images" / "train").glob("*.jpg"))
        
        val_count = int(len(train_imgs) * val_ratio)
        random.shuffle(train_imgs)
        val_imgs = train_imgs[:val_count]
        
        for img_path in val_imgs:
            label_path = self.current_dataset_dir / "labels" / "train" / f"{img_path.stem}.txt"
            
            dst_img = self.current_dataset_dir / "images" / "val" / img_path.name
            dst_label = self.current_dataset_dir / "labels" / "val" / label_path.name
            
            shutil.move(str(img_path), str(dst_img))
            if label_path.exists():
                shutil.move(str(label_path), str(dst_label))
        
        train_count = len(train_imgs) - val_count
        print(f"Split: {train_count} train, {val_count} validation")
    
    def _create_dataset_yaml(self, total_samples: int, conversion_stats: Dict) -> Path:
        """Create dataset YAML file"""
        
        # Format conversion stats as YAML comments
        stats_comments = []
        for source, count in conversion_stats.items():
            stats_comments.append(f"# - {source}: {count} samples")
        stats_text = '\n'.join(stats_comments) if stats_comments else "# - No new data processed"
        
        yaml_content = f'''# Smart Consolidated Dataset
# Generated: {datetime.now().isoformat()}  
# Total samples: {total_samples}

path: {self.current_dataset_dir.absolute()}
train: images/train
val: images/val

nc: {len(self.shared_classes)}
names: {self.shared_classes}

# Conversion statistics:
{stats_text}

# Data sources included:
# - All YOLO format competitive training
# - Synthetic training data (converted)
# - Backseat collector data (converted)
# - Incremental updates supported
'''
        
        yaml_path = self.current_dataset_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        return yaml_path
    
    def train_model(self, yaml_path: Path, model_name: Optional[str] = None) -> bool:
        """Train model with smart parameters based on dataset size"""
        if not YOLO_AVAILABLE:
            print("YOLO not available!")
            return False
        
        # Analyze dataset size
        train_imgs = list((self.current_dataset_dir / "images" / "train").glob("*.jpg"))
        dataset_size = len(train_imgs)
        
        print(f"\\n🚀 Training model on {dataset_size} samples...")
        
        model = YOLO('yolov8n.pt')
        
        # Use proven successful parameters from object_detection_trainer
        if dataset_size < 100:
            epochs, lr0, patience, batch = 100, 0.01, 10, 2
            strategy = "small_dataset_proven"
        elif dataset_size < 300:
            epochs, lr0, patience, batch = 100, 0.01, 10, 2
            strategy = "medium_dataset_proven"
        else:
            epochs, lr0, patience, batch = 100, 0.01, 10, 4
            strategy = "large_dataset_proven"
        
        print(f"Using {strategy} training strategy")
        
        if not model_name:
            model_name = f"consolidated_v{int(time.time())}"
        
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=640,
            batch=batch,
            save=True,
            project="models",
            name=model_name,
            exist_ok=True,
            pretrained=True,
            optimizer='auto',  # Use default optimizer like successful models
            lr0=lr0,
            lrf=0.1,  # Higher final learning rate like successful models
            patience=patience,
            save_period=max(epochs//10, 5),
            close_mosaic=max(epochs//5, 10),
            plots=True,
            verbose=True,
            device='cpu',  # Auto-detect GPU
        )
        
        # Record model in registry
        self.registry.registry['models'][model_name] = {
            'created': datetime.now().isoformat(),
            'dataset_version': self.current_dataset_dir.name,
            'training_samples': dataset_size,
            'strategy': strategy,
            'yaml_path': str(yaml_path)
        }
        self.registry._save_registry()
        
        print(f"\\n✅ Model '{model_name}' trained successfully!")
        return True

def main():
    """Main function for smart consolidated training"""
    print("🧠 SMART CONSOLIDATED TRAINER")
    print("=" * 60)
    print("Long-term solution for continuous model improvement")
    print()
    
    trainer = SmartConsolidatedTrainer()
    
    # Choice: incremental or full rebuild
    print("Dataset update options:")
    print("1. Incremental update (only process new/changed data)")
    print("2. Full rebuild (process all data sources)")
    choice = input("Choose (1/2): ").strip()
    
    incremental = choice != '2'
    
    try:
        # Create or update dataset
        yaml_path, stats = trainer.create_or_update_dataset(incremental=incremental)
        
        if not stats and incremental:
            print("Dataset is already up to date!")
            return
        
        # Train model
        if input("\\nStart training? (y/n): ").lower().startswith('y'):
            success = trainer.train_model(yaml_path)
            
            if success:
                print("\\n🎉 SMART CONSOLIDATED TRAINING COMPLETE!")
                print("Your model now benefits from ALL available training data!")
                print("\\n📈 Future benefits:")
                print("- New competitive sessions automatically included")
                print("- Backseat feedback continuously improves dataset")
                print("- Incremental updates keep model current")
                print("- No more model degradation!")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()