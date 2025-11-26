#!/usr/bin/env python3
"""
🔄 CLASS REGISTRY: Dynamic Class Management System
=================================================

Solves the problem of rigid class architectures by providing:
1. Dynamic class discovery from all models
2. Backward/forward compatibility between models
3. Automatic class mapping and inference adaptation
4. Extensible class registration system

This allows adding new object types without breaking existing models.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ClassRegistry:
    """Manages dynamic class registration and compatibility between models"""
    
    def __init__(self, registry_file: str = "class_registry.json"):
        self.registry_file = Path(registry_file)
        self.classes: Dict[str, dict] = {}
        self.models_dir = Path("models")
        
        # Load existing registry
        self.load_registry()
        
        # Scan models and update registry
        self.scan_and_update_models()
    
    def load_registry(self):
        """Load existing class registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    self.classes = data.get('classes', {})
                    print(f"Loaded {len(self.classes)} classes from registry")
            except Exception as e:
                print(f"Warning: Could not load registry: {e}")
                self.classes = {}
    
    def save_registry(self):
        """Save class registry to file"""
        try:
            registry_data = {
                'last_updated': datetime.now().isoformat(),
                'total_classes': len(self.classes),
                'classes': self.classes
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            print(f"Saved registry with {len(self.classes)} classes")
        except Exception as e:
            print(f"Warning: Could not save registry: {e}")
    
    def scan_and_update_models(self):
        """Scan all models and update the class registry"""
        if not YOLO_AVAILABLE or not self.models_dir.exists():
            return
        
        updated = False
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                weights_path = model_dir / "weights" / "best.pt"
                if weights_path.exists():
                    try:
                        # Load model to get class names
                        model = YOLO(str(weights_path))
                        model_classes = list(model.names.values())
                        model_name = model_dir.name
                        
                        # Register each class
                        for class_name in model_classes:
                            if self.register_class(class_name, model_name, str(weights_path)):
                                updated = True
                        
                        print(f"Scanned {model_name}: {len(model_classes)} classes")
                        
                    except Exception as e:
                        print(f"Warning: Could not scan {model_dir.name}: {e}")
        
        if updated:
            self.save_registry()
    
    def register_class(self, class_name: str, model_name: str, model_path: str) -> bool:
        """Register a class with the model that supports it"""
        if class_name not in self.classes:
            self.classes[class_name] = {
                'first_seen': datetime.now().isoformat(),
                'models': [],
                'aliases': [],
                'description': f'Auto-discovered from {model_name}'
            }
            updated = True
        else:
            updated = False
        
        # Add model to the class's supported models list
        model_info = {
            'name': model_name,
            'path': model_path,
            'added': datetime.now().isoformat()
        }
        
        # Check if model already registered for this class
        existing_models = [m['name'] for m in self.classes[class_name]['models']]
        if model_name not in existing_models:
            self.classes[class_name]['models'].append(model_info)
            updated = True
        
        return updated
    
    def get_all_classes(self) -> List[str]:
        """Get all registered classes"""
        return sorted(list(self.classes.keys()))
    
    def get_model_classes(self, model_path: str) -> List[str]:
        """Get classes supported by a specific model"""
        if not YOLO_AVAILABLE:
            return []
        
        try:
            model = YOLO(str(model_path))
            return list(model.names.values())
        except Exception as e:
            print(f"Warning: Could not get classes for {model_path}: {e}")
            return []
    
    def find_compatible_models(self, required_classes: List[str]) -> List[Tuple[str, str, float]]:
        """Find models that support the required classes"""
        compatible_models = []
        
        for class_name in required_classes:
            if class_name in self.classes:
                for model_info in self.classes[class_name]['models']:
                    model_name = model_info['name']
                    model_path = model_info['path']
                    
                    # Calculate compatibility score
                    model_classes = self.get_model_classes(model_path)
                    if model_classes:
                        supported_count = sum(1 for cls in required_classes if cls in model_classes)
                        compatibility = supported_count / len(required_classes)
                        
                        compatible_models.append((model_name, model_path, compatibility))
        
        # Remove duplicates and sort by compatibility
        unique_models = {}
        for name, path, comp in compatible_models:
            if name not in unique_models or comp > unique_models[name][1]:
                unique_models[name] = (path, comp)
        
        result = [(name, path, comp) for name, (path, comp) in unique_models.items()]
        return sorted(result, key=lambda x: x[2], reverse=True)
    
    def create_class_mapping(self, source_classes: List[str], target_classes: List[str]) -> Dict[str, str]:
        """Create mapping between different class sets for compatibility"""
        mapping = {}
        
        # Direct matches first
        for source_class in source_classes:
            if source_class in target_classes:
                mapping[source_class] = source_class
        
        # Semantic mapping for similar classes
        semantic_mappings = {
            'pokeball/item': 'pokemon_sprite',
            'exp._bar': 'hp_bar',
            'level_indicator': 'text_area',
            'grass': 'tree',
            'water': 'tree',
            'bush': 'tree',
            'rock': 'building',
            'sign': 'building',
            'door': 'building',
            'trainer': 'npc_character',
            'rival': 'npc_character',
        }
        
        # Apply semantic mappings
        for source_class in source_classes:
            if source_class not in mapping and source_class in semantic_mappings:
                target = semantic_mappings[source_class]
                if target in target_classes:
                    mapping[source_class] = target
        
        # Fallback: map unmapped classes to most similar target
        unmapped = [cls for cls in source_classes if cls not in mapping]
        for source_class in unmapped:
            # Simple heuristic: find target with most similar name
            best_match = None
            best_score = 0
            
            for target_class in target_classes:
                # Calculate similarity score (simple string matching)
                score = len(set(source_class.lower()) & set(target_class.lower()))
                if score > best_score:
                    best_score = score
                    best_match = target_class
            
            if best_match:
                mapping[source_class] = best_match
            else:
                # Ultimate fallback to first target class
                mapping[source_class] = target_classes[0] if target_classes else 'unknown'
        
        return mapping
    
    def adapt_inference_results(self, results, source_model_classes: List[str], target_classes: List[str]):
        """Adapt inference results from one class set to another"""
        if not results or source_model_classes == target_classes:
            return results
        
        # Create mapping
        class_mapping = self.create_class_mapping(source_model_classes, target_classes)
        
        # Adapt results
        adapted_results = []
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                # Get class indices and map them
                classes = result.boxes.cls.cpu().numpy()
                
                # Map class indices
                mapped_classes = []
                for cls_idx in classes:
                    source_class = source_model_classes[int(cls_idx)]
                    target_class = class_mapping.get(source_class, source_class)
                    
                    # Find target class index
                    if target_class in target_classes:
                        mapped_classes.append(target_classes.index(target_class))
                    else:
                        mapped_classes.append(0)  # Fallback to first class
                
                # Update result with mapped classes
                import torch
                result.boxes.cls = torch.tensor(mapped_classes, device=result.boxes.cls.device)
        
        return results
    
    def print_registry_status(self):
        """Print current registry status"""
        print("\n[CLASS REGISTRY STATUS]")
        print("=" * 40)
        print(f"Total registered classes: {len(self.classes)}")
        print(f"Registry file: {self.registry_file}")
        
        if self.classes:
            print("\nRegistered classes:")
            for class_name, info in sorted(self.classes.items()):
                model_count = len(info['models'])
                print(f"  • {class_name} (supported by {model_count} models)")
                
                # Show which models support this class
                for model_info in info['models'][:3]:  # Show first 3 models
                    print(f"    - {model_info['name']}")
                if len(info['models']) > 3:
                    print(f"    - ... and {len(info['models']) - 3} more")
        
        print(f"\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# Convenience functions
def get_registry() -> ClassRegistry:
    """Get the global class registry instance"""
    return ClassRegistry()

def get_all_classes() -> List[str]:
    """Quick function to get all registered classes"""
    registry = get_registry()
    return registry.get_all_classes()

def find_best_model_for_classes(required_classes: List[str]) -> Optional[Tuple[str, str]]:
    """Find the best model that supports the required classes"""
    registry = get_registry()
    compatible = registry.find_compatible_models(required_classes)
    
    if compatible:
        name, path, compatibility = compatible[0]
        return name, path
    return None


if __name__ == "__main__":
    # CLI usage
    registry = ClassRegistry()
    registry.print_registry_status()