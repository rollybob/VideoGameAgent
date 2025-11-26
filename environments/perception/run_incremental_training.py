#!/usr/bin/env python3
"""
🚀 Auto-run incremental training 
"""

from incremental_class_trainer import IncrementalClassTrainer
from shared_class_manager import get_shared_class_manager

def main():
    print("🔄 AUTO INCREMENTAL CLASS TRAINER")
    print("=" * 50)
    
    trainer = IncrementalClassTrainer()
    
    # Get new classes to add
    shared_manager = get_shared_class_manager()
    all_shared_classes = shared_manager.get_classes()
    new_classes = [cls for cls in all_shared_classes if cls not in trainer.original_classes]
    
    print(f"📋 Adding {len(new_classes)} new classes:")
    for cls in new_classes:
        print(f"   - {cls}")
    
    # Prepare incremental dataset
    print(f"\n🔄 Preparing incremental dataset...")
    yaml_path, stats = trainer.prepare_incremental_dataset(new_classes)
    
    print(f"\n📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Start training
    print(f"\n🚀 Starting incremental training...")
    try:
        model_path = trainer.train_incremental_model(yaml_path, stats)
        print(f"\n🎉 SUCCESS! New model ready: {model_path}")
        print(f"This model should retain 0.82+ confidence on existing objects")
        print(f"while adding detection for {len(new_classes)} new classes!")
        
        # Show model info
        from model_manager import ModelManager
        manager = ModelManager()
        manager.print_model_status()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()