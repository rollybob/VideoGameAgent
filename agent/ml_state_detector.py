import numpy as np
import cv2
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from dataclasses import dataclass

@dataclass
class StateDetectionResult:
    predicted_state: str
    confidence: float
    all_predictions: Dict[str, float]
    timestamp: float

class GameStateMLDetector:
    def __init__(self, model_path: str = "models", data_path: str = "training_data"):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.class_names = [
            "Overworld", "Battle", 
            "Pokemon Center", "Shop", "Gym", "Dialogue/Menu", 
            "Water/Flying", "Indoor", "Cave"
        ]
        # Create filesystem-safe versions of class names
        self.safe_class_names = {
            name: name.replace("/", "_").replace("\\", "_").replace(":", "_") 
            for name in self.class_names
        }
        self.input_shape = (128, 128, 3)  # Match neural_game_agent for compatibility
        self.confidence_threshold = 0.7  # Minimum confidence for state changes
        self.state_history = []
        self.last_confident_state = "Overworld"
        self.last_state_change = time.time()
        
        # Autonomous training parameters
        self.auto_collection_enabled = False
        self.auto_collection_threshold = 0.95  # High confidence for auto-labeling
        self.uncertainty_threshold = 0.3  # Low confidence triggers manual review
        self.min_samples_per_class = 50
        self.max_auto_samples_per_session = 100
        self.auto_samples_collected = 0
        self.pending_uncertain_samples = []
        
        # Manual data collection review
        self.manual_review_enabled = False  # Allow manual data to be reviewed
        self.pending_manual_samples = []  # Queue for manually collected samples
        
        # Create directories
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        for class_name in self.class_names:
            safe_name = self.safe_class_names[class_name]
            os.makedirs(os.path.join(self.data_path, safe_name), exist_ok=True)
    
    def preprocess_frame(self, frame) -> np.ndarray:
        """Preprocess frame for ML model input"""
        if frame is None:
            return np.zeros(self.input_shape)
        
        # Resize to model input size
        resized = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def collect_training_data(self, frame, manual_label: str, auto_save: bool = True, manual_collection: bool = False):
        """Collect labeled data for training"""
        if manual_label not in self.class_names:
            print(f"Invalid label: {manual_label}. Valid labels: {self.class_names}")
            return False
        
        # If manual review is enabled and this is manual collection, add to review queue
        if manual_collection and self.manual_review_enabled:
            # Create a dummy result for consistency with auto-collected samples
            dummy_result = type('obj', (object,), {
                'predicted_state': manual_label,
                'confidence': 1.0,  # Manual collection assumed to be confident
                'all_predictions': {manual_label: 1.0},
                'timestamp': time.time()
            })()
            
            self.pending_manual_samples.append({
                'frame': frame.copy() if frame is not None else None,
                'result': dummy_result,
                'timestamp': time.time(),
                'original_label': manual_label,
                'source': 'manual'
            })
            
            print(f"Manual sample queued for review: {manual_label}")
            return True
        
        # Validate frame shape before saving
        if frame is not None:
            original_shape = frame.shape
            print(f"DEBUG: Input frame shape: {original_shape}")
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Validate preprocessed frame shape
        if processed_frame.shape != self.input_shape:
            print(f"ERROR: Preprocessed frame shape {processed_frame.shape} doesn't match expected {self.input_shape}")
            return False
        
        print(f"DEBUG: Preprocessed frame shape: {processed_frame.shape} - OK")
        
        # Use safe name for directory but keep original label in filename
        safe_name = self.safe_class_names[manual_label]
        safe_label = manual_label.replace("/", "_").replace("\\", "_").replace(":", "_")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{safe_label}_{timestamp}.npy"
        filepath = os.path.join(self.data_path, safe_name, filename)
        
        # Save the processed frame
        np.save(filepath, processed_frame)
        
        if auto_save:
            # Also save metadata
            metadata = {
                "timestamp": timestamp,
                "label": manual_label,  # Keep original label in metadata
                "original_shape": frame.shape if frame is not None else None,
                "processed_shape": processed_frame.shape,
                "collection_method": "auto" if not manual_collection else "manual"
            }
            
            metadata_file = os.path.join(self.data_path, safe_name, f"{safe_label}_{timestamp}_meta.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Collected training sample: {manual_label} -> {filename}")
        return True
    
    def create_data_augmentation(self):
        """Create data augmentation pipeline for better training"""
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
        ])
    
    def create_cnn_model(self) -> keras.Model:
        """Create enhanced CNN architecture for game state classification"""
        # Data augmentation
        data_augmentation = self.create_data_augmentation()
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Data augmentation (only applied during training)
            data_augmentation,
            
            # First convolutional block with residual connection
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth convolutional block for more capacity
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),  # Better than Flatten for generalization
            
            # Dense layers with stronger regularization
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']  # Removed top_3_categorical_accuracy for compatibility
        )
        
        return model
    
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all collected training data"""
        X, y = [], []
        
        for i, class_name in enumerate(self.class_names):
            safe_name = self.safe_class_names[class_name]
            class_dir = os.path.join(self.data_path, safe_name)
            if not os.path.exists(class_dir):
                continue
            
            # Load all .npy files for this class
            for filename in os.listdir(class_dir):
                if filename.endswith('.npy'):
                    filepath = os.path.join(class_dir, filename)
                    try:
                        frame_data = np.load(filepath)
                        
                        # Check if shape matches expected input shape
                        if frame_data.shape != self.input_shape:
                            print(f"Skipping {filepath}: incompatible shape {frame_data.shape}, expected {self.input_shape}")
                            continue
                            
                        X.append(frame_data)
                        y.append(i)  # Class index
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
        
        if len(X) == 0:
            print("No training data found!")
            return np.array([]), np.array([])
        
        try:
            X = np.array(X)
        except ValueError as e:
            print(f"Error creating training array: {e}")
            print("This usually means training data has inconsistent shapes.")
            print("Attempting to filter and fix...")
            
            # Filter out any remaining inconsistent data
            filtered_X, filtered_y = [], []
            for i, frame_data in enumerate(X):
                if frame_data.shape == self.input_shape:
                    filtered_X.append(frame_data)
                    filtered_y.append(y[i])
                else:
                    print(f"Filtering out data with shape {frame_data.shape}")
            
            if len(filtered_X) == 0:
                print("No valid training data after filtering!")
                return np.array([]), np.array([])
                
            X = np.array(filtered_X)
            y = filtered_y
            
        y = keras.utils.to_categorical(y, len(self.class_names))
        
        print(f"Loaded {len(X)} training samples across {len(self.class_names)} classes")
        return X, y
    
    def train_model(self, epochs: int = 30, validation_split: float = 0.2) -> bool:
        """Train the CNN model"""
        print("Loading training data...")
        X, y = self.load_training_data()
        
        if len(X) == 0:
            print("Cannot train: No training data available")
            return False
        
        if len(X) < 10:
            print(f"Warning: Only {len(X)} samples available. Need more data for reliable training.")
            
        # Check for class imbalance
        stats = self.get_data_collection_stats()
        class_counts = [stats.get(name, 0) for name in self.class_names]
        max_samples = max(class_counts)
        min_samples = min(class_counts)
        empty_classes = sum(1 for count in class_counts if count == 0)
        
        if empty_classes > 0:
            print(f"WARNING: {empty_classes} classes have no training data!")
            print("This will cause poor performance. Consider collecting more balanced data.")
            
        if max_samples > 0 and min_samples >= 1:
            imbalance_ratio = max_samples / min_samples
            if imbalance_ratio > 10:
                print(f"WARNING: Severe class imbalance detected (ratio: {imbalance_ratio:.1f}x)")
                print("Model may be biased toward overrepresented classes.")
                print("Consider collecting more samples for underrepresented classes.")
        
        print("Creating model...")
        self.model = self.create_cnn_model()
        
        print("Model architecture:")
        self.model.summary()
        
        # Callbacks for training
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_path, 'best_model.h5'),
                save_best_only=True
            )
        ]
        
        # Calculate class weights to handle imbalance
        from sklearn.utils import class_weight
        
        # Get class labels for each sample
        y_labels = np.argmax(y, axis=1)
        unique_classes = np.unique(y_labels)
        
        if len(unique_classes) > 1:  # Only calculate if we have multiple classes
            class_weights_array = class_weight.compute_class_weight(
                'balanced', 
                classes=unique_classes, 
                y=y_labels
            )
            class_weights_dict = {unique_classes[i]: class_weights_array[i] 
                                for i in range(len(unique_classes))}
            print(f"Using class weights to handle imbalance: {class_weights_dict}")
        else:
            class_weights_dict = None
            print("Only one class present - no class weights needed")
        
        print("Starting training...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=16,  # Reduced batch size for stability with small dataset
            callbacks=callbacks,
            class_weight=class_weights_dict,
            verbose=1
        )
        
        # Save the final model
        model_file = os.path.join(self.model_path, 'game_state_classifier.h5')
        self.model.save(model_file)
        
        # Save class names
        class_names_file = os.path.join(self.model_path, 'class_names.pkl')
        with open(class_names_file, 'wb') as f:
            pickle.dump(self.class_names, f)
        
        print(f"Model training completed and saved to {model_file}")
        return True
    
    def load_model(self) -> bool:
        """Load trained model with input shape compatibility check"""
        model_file = os.path.join(self.model_path, 'game_state_classifier.h5')
        class_names_file = os.path.join(self.model_path, 'class_names.pkl')
        
        if not os.path.exists(model_file):
            print(f"No trained model found at {model_file}")
            return False
        
        try:
            # Load model
            loaded_model = keras.models.load_model(model_file)
            
            # Check input shape compatibility
            model_input_shape = loaded_model.input_shape[1:]  # Remove batch dimension
            expected_shape = self.input_shape
            
            if model_input_shape != expected_shape:
                print(f"Model input shape mismatch!")
                print(f"  Loaded model expects: {model_input_shape}")
                print(f"  Current detector uses: {expected_shape}")
                print(f"  Backing up incompatible model and creating new one...")
                
                # Backup the incompatible model
                backup_name = f"game_state_classifier_shape_{model_input_shape[0]}x{model_input_shape[1]}.h5"
                backup_path = os.path.join(self.model_path, backup_name)
                os.rename(model_file, backup_path)
                print(f"  Old model backed up as: {backup_name}")
                
                return False
            
            self.model = loaded_model
            
            if os.path.exists(class_names_file):
                with open(class_names_file, 'rb') as f:
                    self.class_names = pickle.load(f)
            
            print(f"Model loaded successfully from {model_file}")
            print(f"Input shape: {model_input_shape}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Model may be corrupted or incompatible")
            return False
    
    def predict_state(self, frame) -> StateDetectionResult:
        """Predict game state with confidence scores"""
        if self.model is None:
            # Fallback to last confident state if no model
            return StateDetectionResult(
                predicted_state=self.last_confident_state,
                confidence=0.0,
                all_predictions={},
                timestamp=time.time()
            )
        
        # Validate frame shape before saving
        if frame is not None:
            original_shape = frame.shape
            print(f"DEBUG: Input frame shape: {original_shape}")
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Validate preprocessed frame shape
        if processed_frame.shape != self.input_shape:
            print(f"ERROR: Preprocessed frame shape {processed_frame.shape} doesn't match expected {self.input_shape}")
            return False
        
        print(f"DEBUG: Preprocessed frame shape: {processed_frame.shape} - OK")
        
        # Add batch dimension
        input_batch = np.expand_dims(processed_frame, axis=0)
        
        # Get predictions
        predictions = self.model.predict(input_batch, verbose=0)[0]
        
        # Create prediction dictionary
        all_predictions = {
            self.class_names[i]: float(predictions[i])
            for i in range(len(self.class_names))
        }
        
        # Get the most confident prediction
        max_confidence_idx = np.argmax(predictions)
        predicted_state = self.class_names[max_confidence_idx]
        confidence = float(predictions[max_confidence_idx])
        
        result = StateDetectionResult(
            predicted_state=predicted_state,
            confidence=confidence,
            all_predictions=all_predictions,
            timestamp=time.time()
        )
        
        # Update state history
        self.state_history.append(result)
        if len(self.state_history) > 10:  # Keep last 10 predictions
            self.state_history.pop(0)
        
        # Autonomous training: auto-collect high-confidence samples
        self.auto_collect_if_appropriate(frame, result)
        
        return result
    
    def get_confident_state(self, frame) -> str:
        """Get state with confidence-based filtering"""
        result = self.predict_state(frame)
        current_time = time.time()
        
        # If confidence is high enough, accept the prediction
        if result.confidence >= self.confidence_threshold:
            # Also check if state has been consistent recently
            if len(self.state_history) >= 3:
                recent_states = [r.predicted_state for r in self.state_history[-3:]]
                if recent_states.count(result.predicted_state) >= 2:  # 2 out of 3 recent predictions agree
                    self.last_confident_state = result.predicted_state
                    self.last_state_change = current_time
                    return result.predicted_state
        
        # If not confident enough or inconsistent, check for state change cooldown
        time_since_change = current_time - self.last_state_change
        if time_since_change < 2.0:  # Don't change state for 2 seconds
            return self.last_confident_state
        
        # If it's been a while and we're still not confident, 
        # maybe accept a lower confidence prediction
        if result.confidence >= 0.5 and time_since_change > 5.0:
            self.last_confident_state = result.predicted_state
            self.last_state_change = current_time
            return result.predicted_state
        
        # Default to last confident state
        return self.last_confident_state
    
    def get_data_collection_stats(self) -> Dict[str, int]:
        """Get statistics about collected training data"""
        stats = {}
        total = 0
        
        # Get all existing directories in training_data
        existing_dirs = []
        if os.path.exists(self.data_path):
            existing_dirs = [d for d in os.listdir(self.data_path) 
                           if os.path.isdir(os.path.join(self.data_path, d))]
        
        # Count samples for current class names
        for class_name in self.class_names:
            safe_name = self.safe_class_names[class_name]
            class_dir = os.path.join(self.data_path, safe_name)
            count = 0
            
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
            
            stats[class_name] = count
            total += count
        
        # Handle legacy data directories that might exist
        legacy_mappings = {
            'Wild_Battle': 'Battle',
            'Trainer_Battle': 'Battle', 
            'Indoor_Cave': None  # This should have been migrated manually
        }
        
        for legacy_dir, new_class in legacy_mappings.items():
            if legacy_dir in existing_dirs and new_class:
                legacy_path = os.path.join(self.data_path, legacy_dir)
                if os.path.exists(legacy_path):
                    legacy_count = len([f for f in os.listdir(legacy_path) if f.endswith('.npy')])
                    if legacy_count > 0:
                        stats[new_class] = stats.get(new_class, 0) + legacy_count
                        total += legacy_count
                        print(f"Warning: Found {legacy_count} samples in legacy directory {legacy_dir}")
                        print(f"These are counted toward '{new_class}' but should be migrated.")
        
        stats['total'] = total
        return stats
    
    def reset_state_history(self):
        """Reset state history and confidence tracking"""
        self.state_history = []
        self.last_confident_state = "Overworld"
        self.last_state_change = time.time()
    
    def get_uncertainty_score(self, predictions: np.ndarray) -> float:
        """Calculate uncertainty score using entropy"""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        entropy = -np.sum(predictions * np.log(predictions + epsilon))
        # Normalize by max possible entropy
        max_entropy = -np.log(1.0 / len(predictions))
        return entropy / max_entropy
    
    def should_auto_collect(self, result: StateDetectionResult) -> bool:
        """Determine if this sample should be automatically collected"""
        if not self.auto_collection_enabled:
            return False
        
        if self.auto_samples_collected >= self.max_auto_samples_per_session:
            return False
        
        # Check if confidence is high enough
        if result.confidence < self.auto_collection_threshold:
            return False
        
        # Check if this class needs more data
        stats = self.get_data_collection_stats()
        current_count = stats.get(result.predicted_state, 0)
        
        # Prioritize classes with fewer samples
        if current_count < self.min_samples_per_class:
            return True
        
        # For well-represented classes, be more selective
        return result.confidence > 0.98 and len(self.state_history) >= 3
    
    def should_flag_for_review(self, result: StateDetectionResult) -> bool:
        """Determine if this sample needs manual review"""
        uncertainty = self.get_uncertainty_score(
            np.array(list(result.all_predictions.values()))
        )
        
        # Flag high uncertainty predictions
        if uncertainty > self.uncertainty_threshold:
            return True
        
        # Flag if predictions are inconsistent with recent history
        if len(self.state_history) >= 3:
            recent_states = [r.predicted_state for r in self.state_history[-3:]]
            if len(set(recent_states)) >= 3:  # All different states
                return True
        
        return False
    
    def auto_collect_if_appropriate(self, frame, result: StateDetectionResult) -> bool:
        """Automatically collect high-confidence samples"""
        if self.should_auto_collect(result):
            success = self.collect_training_data(
                frame, 
                result.predicted_state, 
                auto_save=True
            )
            if success:
                self.auto_samples_collected += 1
                print(f"Auto-collected sample for {result.predicted_state} "
                      f"(confidence: {result.confidence:.3f}) "
                      f"[{self.auto_samples_collected}/{self.max_auto_samples_per_session}]")
                return True
        
        # Flag uncertain samples for manual review
        if self.should_flag_for_review(result):
            self.pending_uncertain_samples.append({
                'frame': frame.copy() if frame is not None else None,
                'result': result,
                'timestamp': time.time()
            })
            # Limit queue size
            if len(self.pending_uncertain_samples) > 20:
                self.pending_uncertain_samples.pop(0)
        
        return False
    
    def get_pending_review_count(self) -> int:
        """Get number of samples pending manual review (both uncertain and manual)"""
        return len(self.pending_uncertain_samples) + len(self.pending_manual_samples)
    
    def add_to_pending_review(self, frame) -> int:
        """Add a frame to pending manual review and return total count"""
        # Add frame with timestamp for manual review
        sample_data = {
            'frame': frame,
            'timestamp': time.time(),
            'type': 'manual_collection'
        }
        self.pending_manual_samples.append(sample_data)
        return self.get_pending_review_count()
    
    def get_next_uncertain_sample(self):
        """Get the next sample that needs manual review (prioritize uncertain over manual)"""
        # Prioritize uncertain samples first (these need immediate attention)
        if self.pending_uncertain_samples:
            return self.pending_uncertain_samples.pop(0)
        # Then review manual samples if no uncertain ones
        elif self.pending_manual_samples:
            return self.pending_manual_samples.pop(0)
        return None
    
    def enable_manual_review(self, enabled: bool = True):
        """Enable/disable manual data collection review"""
        self.manual_review_enabled = enabled
        if enabled:
            print("Manual data review enabled - collect data will queue for review")
        else:
            print("Manual data review disabled - collect data saves immediately")
            # Process any pending manual samples immediately
            if self.pending_manual_samples:
                print(f"Processing {len(self.pending_manual_samples)} pending manual samples...")
                while self.pending_manual_samples:
                    sample = self.pending_manual_samples.pop(0)
                    self.collect_training_data(
                        sample['frame'], 
                        sample['original_label'], 
                        auto_save=True, 
                        manual_collection=False
                    )
    
    def enable_autonomous_training(self, enabled: bool = True):
        """Enable/disable autonomous data collection"""
        self.auto_collection_enabled = enabled
        if enabled:
            print("Autonomous training enabled - will auto-collect high-confidence samples")
        else:
            print("Autonomous training disabled")
    
    def reset_auto_collection_session(self):
        """Reset auto-collection counters for new session"""
        self.auto_samples_collected = 0
        self.pending_uncertain_samples = []