#!/usr/bin/env python3
"""
🔧 MODEL MANAGER: Automatic Latest Model Discovery
=================================================

Solves the versioning problem where trainers create pokemon_object_detector, 
pokemon_object_detector2, pokemon_object_detector3, etc. but the data collector
doesn't know which one to use.

Usage:
- get_latest_model_path() - Returns path to newest trained model
- get_all_model_versions() - Lists all available model versions
- create_model_symlink() - Creates a "latest" symlink for easy access
"""

import os
import re
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

class ModelManager:
    """Manages automatic discovery of the latest trained models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.base_model_name = "pokemon_object_detector"
        
    def get_all_model_versions(self) -> List[Tuple[str, Path, datetime]]:
        """Get all model versions sorted by creation time (newest first)"""
        if not self.models_dir.exists():
            return []
            
        models = []
        
        # Scan ALL directories for models with weights/best.pt
        for item in self.models_dir.iterdir():
            if item.is_dir():
                weights_path = item / "weights" / "best.pt"
                if weights_path.exists():
                    # Get creation time of the model directory
                    creation_time = datetime.fromtimestamp(item.stat().st_ctime)
                    models.append((item.name, weights_path, creation_time))
        
        # Also check for any .pt files directly in the models directory
        for item in self.models_dir.iterdir():
            if item.is_file() and item.suffix == '.pt':
                # Get creation time of the model file
                creation_time = datetime.fromtimestamp(item.stat().st_ctime)
                models.append((item.stem, item, creation_time))
        
        # Sort by creation time (newest first)
        models.sort(key=lambda x: x[2], reverse=True)
        return models
    
    def get_latest_model_path(self) -> Optional[Path]:
        """Get the path to the most recently trained model"""
        models = self.get_all_model_versions()
        if models:
            return models[0][1]  # Return the weights path of the newest model
        return None
    
    def get_latest_model_info(self) -> Optional[dict]:
        """Get detailed info about the latest model"""
        models = self.get_all_model_versions()
        if models:
            name, path, creation_time = models[0]
            return {
                'name': name,
                'path': path,
                'creation_time': creation_time,
                'age_minutes': (datetime.now() - creation_time).total_seconds() / 60
            }
        return None
    
    def create_latest_symlink(self) -> bool:
        """Create a 'latest' symlink pointing to the newest model"""
        latest_path = self.get_latest_model_path()
        if not latest_path:
            return False
            
        symlink_path = self.models_dir / "latest_model.pt"
        
        try:
            # Remove existing symlink if it exists
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            
            # Create new symlink (relative path for portability)
            relative_path = os.path.relpath(latest_path, symlink_path.parent)
            symlink_path.symlink_to(relative_path)
            return True
        except Exception as e:
            print(f"Warning: Could not create symlink: {e}")
            return False
    
    def print_model_status(self):
        """Print a summary of available models"""
        models = self.get_all_model_versions()
        
        print("\n[MODEL MANAGER STATUS]")
        print("=" * 30)
        
        if not models:
            print("[ERROR] No trained models found")
            print(f"Expected location: {self.models_dir}")
            return
        
        print(f"Found {len(models)} trained model(s):")
        
        for i, (name, path, creation_time) in enumerate(models):
            age_minutes = (datetime.now() - creation_time).total_seconds() / 60
            status = "[LATEST]" if i == 0 else "[OLDER]"
            
            print(f"  {status} {name}")
            print(f"     Path: {path}")
            print(f"     Created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')} ({age_minutes:.1f} min ago)")
            print()
        
        # Show what will be used
        latest = self.get_latest_model_info()
        if latest:
            print(f"[ACTIVE MODEL]: {latest['name']}")
            print(f"   Age: {latest['age_minutes']:.1f} minutes")
            print(f"   Path: {latest['path']}")

# Convenience functions for easy import
def get_latest_model_path() -> Optional[Path]:
    """Quick function to get the latest model path"""
    manager = ModelManager()
    return manager.get_latest_model_path()

def print_model_status():
    """Quick function to print model status"""
    manager = ModelManager()
    manager.print_model_status()

if __name__ == "__main__":
    # CLI usage
    manager = ModelManager()
    manager.print_model_status()
    
    # Try to create symlink
    if manager.create_latest_symlink():
        print("\n[OK] Created latest_model.pt symlink")
    else:
        print("\n[WARNING] Could not create symlink (this is optional)")