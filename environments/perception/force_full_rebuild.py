#!/usr/bin/env python3
"""
🔄 FORCE FULL REBUILD
====================
Forces a complete rebuild of the consolidated dataset with all available data sources.
"""

import sys
from smart_consolidated_trainer import SmartConsolidatedTrainer

def main():
    print("🔄 FORCING FULL CONSOLIDATED DATASET REBUILD")
    print("=" * 60)
    print("This will include ALL available training data sources")
    print()
    
    trainer = SmartConsolidatedTrainer()
    
    try:
        # Force full rebuild (incremental=False)
        yaml_path, stats = trainer.create_or_update_dataset(incremental=False)
        
        if not stats:
            print("❌ No data sources found!")
            return
        
        print(f"\n📊 Dataset Statistics:")
        for source, count in stats.items():
            print(f"  {source}: {count} samples")
        
        total_samples = sum(stats.values())
        print(f"  TOTAL: {total_samples} samples")
        
        if total_samples < 100:
            print(f"\n⚠️  WARNING: Only {total_samples} samples for 16 classes!")
            print("This may result in poor model performance.")
            print("Recommended minimum: 50+ samples per class (800+ total)")
        
        # Ask about training
        print(f"\n🎯 Ready to train model with {total_samples} samples")
        train_choice = input("Start training now? (y/n): ").strip().lower()
        
        if train_choice == 'y':
            print("\n🚀 Starting model training...")
            model_path = trainer.train_model(yaml_path)
            print(f"✅ Model trained and saved: {model_path}")
        else:
            print(f"✅ Dataset ready at: {yaml_path}")
            print("You can train later using the smart_consolidated_trainer.py")
            
    except KeyboardInterrupt:
        print("\n⛔ Training cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()