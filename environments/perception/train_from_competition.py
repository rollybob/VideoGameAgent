#!/usr/bin/env python3
"""
COMPETITIVE SESSION TRAINER
============================

Converts competitive session data into YOLO training data and trains models.
This script takes the session JSON file and creates training data from the 
bounding boxes and correct answers you provided.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime
import os
import argparse

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available - install with: pip install ultralytics")

def load_session_data(session_file):
    """Load competitive session data from JSON file"""
    with open(session_file, 'r') as f:
        data = json.load(f)
    return data

def create_yolo_dataset_from_session(session_data, output_dir="competitive_training_data"):
    """Convert session data to YOLO training format"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create YOLO directory structure
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all unique classes from the session
    all_classes = set()
    for competition in session_data['competitions']:
        all_classes.add(competition['correct_answer'])
    
    class_list = sorted(list(all_classes))
    class_to_id = {cls: idx for idx, cls in enumerate(class_list)}
    
    print(f"Found {len(class_list)} classes: {class_list}")
    
    # Create data.yaml file
    yaml_content = f"""
# Competitive training dataset from session {session_data.get('start_time', 'unknown')}
path: {output_dir.absolute()}
train: images/train
val: images/train  # Using same data for validation for now

nc: {len(class_list)}
names: {class_list}
"""
    
    with open(output_dir / "data.yaml", 'w') as f:
        f.write(yaml_content.strip())
    
    print(f"Created data.yaml with {len(class_list)} classes")
    
    # Track statistics
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'class_counts': {cls: 0 for cls in class_list},
        'skipped_competitions': 0
    }
    
    # We need to have access to the original images that were captured
    # Since we don't have them, we'll create instructions for manual setup
    print("\n" + "="*60)
    print("IMPORTANT: MANUAL SETUP REQUIRED")
    print("="*60)
    print("This script cannot automatically create training images because")
    print("the competitive trainer doesn't save the captured screenshots.")
    print("\nTo complete the training setup:")
    print("1. Manually save screenshots from your emulator during competition")
    print("2. Name them sequentially: game_001.jpg, game_002.jpg, etc.")
    print("3. Place them in:", images_dir.absolute())
    print("4. Run this script again to generate the label files")
    print("\nAlternatively, modify the competitive trainer to save screenshots.")
    print("="*60)
    
    # Generate label files based on the competitions
    for i, competition in enumerate(session_data['competitions']):
        image_filename = f"game_{i+1:03d}.jpg"
        label_filename = f"game_{i+1:03d}.txt"
        
        image_path = images_dir / image_filename
        label_path = labels_dir / label_filename
        
        # Get bounding box and class
        bbox = competition['bounding_box']
        correct_class = competition['correct_answer']
        class_id = class_to_id[correct_class]
        
        # We need image dimensions to normalize the bounding box
        # Since we don't have the actual images, we'll assume standard dimensions
        # This should be updated when real images are available
        img_width = 1920  # Assumed emulator width
        img_height = 1080  # Assumed emulator height
        
        # Convert absolute bbox to YOLO format (normalized center x, y, width, height)
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2 / img_width
        center_y = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Create label file
        label_content = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\\n"
        
        with open(label_path, 'w') as f:
            f.write(label_content)
        
        stats['total_annotations'] += 1
        stats['class_counts'][correct_class] += 1
        
        if i < 5:  # Show first few examples
            print(f"Example {i+1}: {image_filename} -> {correct_class} at bbox {bbox}")
    
    stats['total_images'] = len(session_data['competitions'])
    
    # Print statistics
    print(f"\nDATASET STATISTICS:")
    print(f"Total images expected: {stats['total_images']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Class distribution:")
    for cls, count in stats['class_counts'].items():
        print(f"  {cls}: {count} instances")
    
    return output_dir, class_list, stats

def analyze_model_performance(session_data):
    """Analyze which models performed best in the session"""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    performances = session_data['model_performances']
    
    # Sort models by accuracy
    sorted_models = sorted(performances.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("Models ranked by accuracy:")
    for i, (model_name, perf) in enumerate(sorted_models):
        print(f"{i+1}. {model_name}")
        print(f"   Accuracy: {perf['accuracy']:.1%}")
        print(f"   Score: {perf['final_score']}")
        print(f"   Correct/Total: {perf['correct_predictions']}/{perf['total_predictions']}")
        print(f"   Avg Confidence: {perf['average_confidence']:.3f}")
        print()
    
    # Analyze common mistakes
    print("COMMON MISTAKES ANALYSIS:")
    mistake_patterns = {}
    
    for competition in session_data['competitions']:
        correct_answer = competition['correct_answer']
        for pred in competition['predictions']:
            if not pred['correct']:
                mistake_key = (pred['prediction'], correct_answer)
                if mistake_key not in mistake_patterns:
                    mistake_patterns[mistake_key] = []
                mistake_patterns[mistake_key].append(pred['confidence'])
    
    # Sort by frequency
    sorted_mistakes = sorted(mistake_patterns.items(), key=lambda x: len(x[1]), reverse=True)
    
    print("Most common mistakes (predicted -> actual):")
    for i, ((predicted, actual), confidences) in enumerate(sorted_mistakes[:10]):
        avg_conf = sum(confidences) / len(confidences)
        print(f"{i+1}. Predicted '{predicted}' instead of '{actual}': {len(confidences)} times (avg conf: {avg_conf:.3f})")
    
    return sorted_models[0][0] if sorted_models else None

def train_improved_model(dataset_dir, base_model_name=None, model_name_suffix="competitive"):
    """Train a new model using the competitive session data"""
    if not YOLO_AVAILABLE:
        print("Cannot train - YOLO not available")
        return None
    
    dataset_dir = Path(dataset_dir)
    data_yaml = dataset_dir / "data.yaml"
    
    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found")
        return None
    
    # Determine base model
    if base_model_name:
        print(f"Starting training from {base_model_name}")
        # Find the model file
        base_model_path = Path(f"models/{base_model_name}/weights/best.pt")
        if base_model_path.exists():
            model = YOLO(str(base_model_path))
        else:
            print(f"Base model not found at {base_model_path}, using yolov8n.pt")
            model = YOLO('yolov8n.pt')
    else:
        print("Starting training from yolov8n.pt")
        model = YOLO('yolov8n.pt')
    
    # Create output directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"pokemon_object_detector_{model_name_suffix}_{timestamp}"
    
    print(f"Training new model: {output_name}")
    print(f"Using dataset: {data_yaml}")
    
    # Training parameters optimized for competitive feedback data
    results = model.train(
        data=str(data_yaml),
        epochs=100,  # More epochs since we have human-verified labels
        imgsz=640,
        batch=16,
        save=True,
        project="models",
        name=output_name,
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment='randaugment',
        erasing=0.4,
        crop_fraction=1.0,
        plots=True,
        verbose=True
    )
    
    print(f"Training complete! Model saved to models/{output_name}")
    return output_name

def main():
    parser = argparse.ArgumentParser(description='Train models from competitive session data')
    parser.add_argument('session_file', help='Path to the competitive session JSON file')
    parser.add_argument('--base-model', help='Base model to start training from (e.g., pokemon_object_detector2)')
    parser.add_argument('--no-train', action='store_true', help='Only create dataset, don\'t train')
    parser.add_argument('--output-dir', default='competitive_training_data', help='Output directory for training data')
    
    args = parser.parse_args()
    
    session_file = Path(args.session_file)
    if not session_file.exists():
        print(f"Error: Session file {session_file} not found")
        return
    
    print(f"Loading session data from: {session_file}")
    session_data = load_session_data(session_file)
    
    print(f"Session from {session_data['start_time']} to {session_data['end_time']}")
    print(f"Total competitions: {session_data['total_competitions']}")
    
    # Analyze model performance
    best_model = analyze_model_performance(session_data)
    
    # Create training dataset
    dataset_dir, classes, stats = create_yolo_dataset_from_session(session_data, args.output_dir)
    
    if not args.no_train and stats['total_annotations'] > 0:
        # Use the best performing model as base (or user specified)
        base_model = args.base_model or best_model
        
        print(f"\nStarting training with base model: {base_model}")
        new_model = train_improved_model(dataset_dir, base_model, "competitive")
        
        if new_model:
            print(f"\n🏆 SUCCESS! New model trained: {new_model}")
            print("This model incorporates your competitive feedback and should perform better!")
    
    print(f"\nDataset created in: {dataset_dir.absolute()}")
    print("Remember to add actual screenshot images to complete the training setup!")

if __name__ == "__main__":
    main()