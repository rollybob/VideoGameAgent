#!/usr/bin/env python3
"""
🧠 ADAPTIVE MODEL WRAPPER: Universal Model Interface
===================================================

Provides a unified interface for models with different class architectures.
Handles automatic class mapping, inference adaptation, and result translation.

Key features:
- Load any YOLO model regardless of class count
- Automatic class compatibility mapping
- Result translation between different class systems
- Confidence preservation during adaptation
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from class_registry import ClassRegistry

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available - install with: pip install ultralytics torch")


class AdaptiveModelWrapper:
    """Wrapper for YOLO models that handles different class architectures"""
    
    def __init__(self, model_path: str, target_classes: Optional[List[str]] = None):
        """
        Initialize adaptive model wrapper
        
        Args:
            model_path: Path to the YOLO model
            target_classes: Desired output class names (if None, uses model's native classes)
        """
        self.model_path = Path(model_path)
        self.model = None
        self.native_classes = []
        self.target_classes = target_classes or []
        self.class_mapping = {}
        self.registry = ClassRegistry()
        
        # Load model and setup class mapping
        self.load_model()
        self.setup_class_mapping()
    
    def load_model(self) -> bool:
        """Load the YOLO model and extract its class information"""
        if not YOLO_AVAILABLE:
            print("Error: YOLO not available")
            return False
        
        if not self.model_path.exists():
            print(f"Error: Model file not found: {self.model_path}")
            return False
        
        try:
            self.model = YOLO(str(self.model_path))
            self.native_classes = list(self.model.names.values())
            print(f"Loaded model with {len(self.native_classes)} classes: {self.native_classes}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def setup_class_mapping(self):
        """Setup class mapping between native and target classes"""
        if not self.target_classes:
            # No target specified, use native classes
            self.target_classes = self.native_classes.copy()
            self.class_mapping = {cls: cls for cls in self.native_classes}
        else:
            # Create mapping from native to target classes
            self.class_mapping = self.registry.create_class_mapping(
                self.native_classes, self.target_classes
            )
        
        print(f"Class mapping: {self.class_mapping}")
    
    def predict(self, image, conf: float = 0.25, **kwargs):
        """
        Run inference and adapt results to target class system
        
        Args:
            image: Input image (numpy array, PIL Image, or path)
            conf: Confidence threshold
            **kwargs: Additional YOLO inference parameters
        
        Returns:
            Adapted results with target class system
        """
        if self.model is None:
            print("Error: Model not loaded")
            return None
        
        try:
            # Run native inference
            results = self.model(image, conf=conf, verbose=False, **kwargs)
            
            # If no adaptation needed, return native results
            if set(self.native_classes) == set(self.target_classes):
                return results
            
            # Adapt results to target class system
            adapted_results = self.adapt_results(results)
            return adapted_results
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
    
    def adapt_results(self, results):
        """Adapt inference results to target class system"""
        if not results:
            return results
        
        adapted_results = []
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                # Get detection data
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # Adapt classes
                adapted_boxes = []
                adapted_confidences = []
                adapted_classes = []
                
                for i, cls_idx in enumerate(classes):
                    native_class = self.native_classes[int(cls_idx)]
                    target_class = self.class_mapping.get(native_class, native_class)
                    
                    # Find target class index
                    if target_class in self.target_classes:
                        target_idx = self.target_classes.index(target_class)
                        
                        adapted_boxes.append(boxes[i])
                        adapted_confidences.append(confidences[i])
                        adapted_classes.append(target_idx)
                
                # Create adapted result
                if adapted_boxes:
                    import torch
                    
                    # Convert to tensors
                    adapted_boxes_tensor = torch.tensor(adapted_boxes, device=result.boxes.xyxy.device)
                    adapted_conf_tensor = torch.tensor(adapted_confidences, device=result.boxes.conf.device)  
                    adapted_cls_tensor = torch.tensor(adapted_classes, device=result.boxes.cls.device)
                    
                    # Create new data tensor combining all information
                    # Format: [x1, y1, x2, y2, conf, cls]
                    adapted_data = torch.cat([
                        adapted_boxes_tensor,
                        adapted_conf_tensor.unsqueeze(1),
                        adapted_cls_tensor.unsqueeze(1)
                    ], dim=1)
                    
                    # Create new Boxes object with adapted data
                    from ultralytics.engine.results import Boxes
                    result.boxes = Boxes(adapted_data, result.orig_shape)
                    
                    # Update names to target classes
                    adapted_names = {i: name for i, name in enumerate(self.target_classes)}
                    result.names = adapted_names
                else:
                    # No valid detections after adaptation
                    result.boxes = None
            
            adapted_results.append(result)
        
        return adapted_results
    
    def get_native_classes(self) -> List[str]:
        """Get the model's native class names"""
        return self.native_classes.copy()
    
    def get_target_classes(self) -> List[str]:
        """Get the target class names"""
        return self.target_classes.copy()
    
    def get_class_mapping(self) -> Dict[str, str]:
        """Get the class mapping dictionary"""
        return self.class_mapping.copy()
    
    def set_target_classes(self, target_classes: List[str]):
        """Update target classes and rebuild mapping"""
        self.target_classes = target_classes
        self.setup_class_mapping()
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name by ID (for backward compatibility)"""
        try:
            if 0 <= class_id < len(self.target_classes):
                return self.target_classes[class_id]
            elif 0 <= class_id < len(self.native_classes):
                return self.native_classes[class_id]
            else:
                return f"unknown_class_{class_id}"
        except (IndexError, TypeError):
            return f"unknown_class_{class_id}"
    
    def info(self):
        """Print model information"""
        print(f"\n[ADAPTIVE MODEL INFO]")
        print(f"Model: {self.model_path.name}")
        print(f"Native classes ({len(self.native_classes)}): {self.native_classes}")
        print(f"Target classes ({len(self.target_classes)}): {self.target_classes}")
        print(f"Class mapping:")
        for native, target in self.class_mapping.items():
            if native != target:
                print(f"  {native} -> {target}")
        print()


class ModelManager:
    """Manages multiple adaptive models and provides unified interface"""
    
    def __init__(self):
        self.models: Dict[str, AdaptiveModelWrapper] = {}
        self.registry = ClassRegistry()
        self.active_classes = self.registry.get_all_classes()
    
    def load_model(self, name: str, model_path: str, target_classes: Optional[List[str]] = None) -> bool:
        """Load a model with adaptive wrapper"""
        try:
            wrapper = AdaptiveModelWrapper(model_path, target_classes or self.active_classes)
            if wrapper.model is not None:
                self.models[name] = wrapper
                print(f"Loaded adaptive model: {name}")
                return True
        except Exception as e:
            print(f"Failed to load model {name}: {e}")
        return False
    
    def predict_with_model(self, model_name: str, image, **kwargs):
        """Run inference with a specific model"""
        if model_name in self.models:
            return self.models[model_name].predict(image, **kwargs)
        else:
            print(f"Model {model_name} not loaded")
            return None
    
    def predict_with_best_model(self, image, required_classes: Optional[List[str]] = None, **kwargs):
        """Find and use the best model for the required classes"""
        if required_classes is None:
            required_classes = self.active_classes
        
        # Find best model from registry
        compatible_models = self.registry.find_compatible_models(required_classes)
        
        if not compatible_models:
            print("No compatible models found")
            return None
        
        best_model_name, best_model_path, compatibility = compatible_models[0]
        
        # Load model if not already loaded
        if best_model_name not in self.models:
            if not self.load_model(best_model_name, best_model_path, required_classes):
                print(f"Failed to load best model: {best_model_name}")
                return None
        
        print(f"Using best model: {best_model_name} (compatibility: {compatibility:.2f})")
        return self.predict_with_model(best_model_name, image, **kwargs)
    
    def set_active_classes(self, classes: List[str]):
        """Set the active class system for all models"""
        self.active_classes = classes
        
        # Update all loaded models to use new class system
        for model in self.models.values():
            model.set_target_classes(classes)
    
    def list_models(self):
        """List all loaded models"""
        print(f"\n[LOADED ADAPTIVE MODELS]")
        if not self.models:
            print("No models loaded")
        else:
            for name, wrapper in self.models.items():
                print(f"• {name}: {len(wrapper.get_native_classes())} -> {len(wrapper.get_target_classes())} classes")


if __name__ == "__main__":
    # Test the adaptive system
    registry = ClassRegistry()
    registry.print_registry_status()
    
    # Test model manager
    manager = ModelManager()
    
    # Try to load a model adaptively
    from model_manager import ModelManager as BasicModelManager
    basic_manager = BasicModelManager()
    latest_model = basic_manager.get_latest_model_path()
    
    if latest_model:
        print(f"\nTesting adaptive wrapper with: {latest_model}")
        manager.load_model("test_model", str(latest_model))
        manager.list_models()