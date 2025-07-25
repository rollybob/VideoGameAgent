import numpy as np
import cv2
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pickle
from dataclasses import dataclass

@dataclass
class StateDetectionResult:
    predicted_state: str
    confidence: float
    all_predictions: Dict[str, float]
    timestamp: float

class GameStateSklearnDetector:
    def __init__(self, model_path: str = "models", data_path: str = "training_data"):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.class_names = [
            "Overworld", "Battle", "Wild Battle", "Trainer Battle", 
            "Pokemon Center", "Shop", "Gym", "Dialogue/Menu", 
            "Water/Flying", "Indoor/Cave"
        ]
        self.feature_size = 64 * 64 * 3  # Flattened image features
        self.confidence_threshold = 0.6  # Lower threshold for sklearn
        self.state_history = []
        self.last_confident_state = "Overworld"
        self.last_state_change = time.time()
        
        # Create directories
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        for class_name in self.class_names:
            os.makedirs(os.path.join(self.data_path, class_name), exist_ok=True)
    
    def extract_features(self, frame) -> np.ndarray:
        """Extract features from frame for sklearn models"""
        if frame is None:
            return np.zeros(self.feature_size)
        
        # Resize to consistent size
        resized = cv2.resize(frame, (64, 64))
        
        # Method 1: Simple flattened pixels
        flattened = resized.flatten().astype(np.float32) / 255.0
        
        # Method 2: Add some hand-crafted features
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Color statistics
        color_features = []
        for channel in range(3):
            channel_data = resized[:, :, channel]
            color_features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data)
            ])
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (64 * 64)
        
        # Texture features (simple)
        texture_features = [
            np.mean(np.abs(np.gradient(gray.astype(float))[0])),
            np.mean(np.abs(np.gradient(gray.astype(float))[1]))
        ]
        
        # Combine all features
        additional_features = np.array(color_features + [edge_density] + texture_features)
        
        # Combine flattened pixels with hand-crafted features
        combined_features = np.concatenate([flattened, additional_features])
        
        return combined_features
    
    def collect_training_data(self, frame, manual_label: str, auto_save: bool = True):
        """Collect labeled data for training"""
        if manual_label not in self.class_names:
            print(f"Invalid label: {manual_label}. Valid labels: {self.class_names}")
            return False
        
        # Extract features
        features = self.extract_features(frame)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{manual_label}_{timestamp}.npy"
        filepath = os.path.join(self.data_path, manual_label, filename)
        
        # Save the features
        np.save(filepath, features)
        
        if auto_save:
            # Also save metadata
            metadata = {
                "timestamp": timestamp,
                "label": manual_label,
                "original_shape": frame.shape if frame is not None else None,
                "feature_size": len(features)
            }
            
            metadata_file = os.path.join(self.data_path, manual_label, f"{manual_label}_{timestamp}_meta.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Collected training sample: {manual_label} -> {filename}")
        return True
    
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all collected training data"""
        X, y = [], []
        
        for i, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_dir):
                continue
            
            # Load all .npy files for this class
            for filename in os.listdir(class_dir):
                if filename.endswith('.npy'):
                    filepath = os.path.join(class_dir, filename)
                    try:
                        feature_data = np.load(filepath)
                        X.append(feature_data)
                        y.append(i)  # Class index
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
        
        if len(X) == 0:
            print("No training data found!")
            return np.array([]), np.array([])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Loaded {len(X)} training samples across {len(self.class_names)} classes")
        return X, y
    
    def train_model(self, model_type: str = "random_forest") -> bool:
        """Train the sklearn model"""
        print("Loading training data...")
        X, y = self.load_training_data()
        
        if len(X) == 0:
            print("Cannot train: No training data available")
            return False
        
        if len(X) < 10:
            print(f"Warning: Only {len(X)} samples available. Need more data for reliable training.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Choose model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "svm":
            self.model = SVC(
                kernel='rbf',
                probability=True,  # Enable probability estimation
                random_state=42
            )
        else:
            print(f"Unknown model type: {model_type}")
            return False
        
        print(f"Training {model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.3f}")
        
        # Detailed classification report
        target_names = [self.class_names[i] for i in np.unique(y)]
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Save model and scaler
        model_file = os.path.join(self.model_path, f'sklearn_classifier_{model_type}.joblib')
        scaler_file = os.path.join(self.model_path, 'feature_scaler.joblib')
        class_names_file = os.path.join(self.model_path, 'class_names.pkl')
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        
        with open(class_names_file, 'wb') as f:
            pickle.dump(self.class_names, f)
        
        print(f"Model saved to {model_file}")
        return True
    
    def load_model(self, model_type: str = "random_forest") -> bool:
        """Load trained model"""
        model_file = os.path.join(self.model_path, f'sklearn_classifier_{model_type}.joblib')
        scaler_file = os.path.join(self.model_path, 'feature_scaler.joblib')
        class_names_file = os.path.join(self.model_path, 'class_names.pkl')
        
        if not os.path.exists(model_file) or not os.path.exists(scaler_file):
            print(f"No trained model found at {model_file}")
            return False
        
        try:
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            
            if os.path.exists(class_names_file):
                with open(class_names_file, 'rb') as f:
                    self.class_names = pickle.load(f)
            
            print(f"Model loaded successfully from {model_file}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
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
        
        # Extract features
        features = self.extract_features(frame)
        features_scaled = self.scaler.transform([features])
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            # Models with probability support
            probabilities = self.model.predict_proba(features_scaled)[0]
        else:
            # Fallback for models without probability
            prediction = self.model.predict(features_scaled)[0]
            probabilities = np.zeros(len(self.class_names))
            probabilities[prediction] = 1.0
        
        # Create prediction dictionary
        all_predictions = {
            self.class_names[i]: float(probabilities[i])
            for i in range(len(self.class_names))
        }
        
        # Get the most confident prediction
        max_confidence_idx = np.argmax(probabilities)
        predicted_state = self.class_names[max_confidence_idx]
        confidence = float(probabilities[max_confidence_idx])
        
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
        if result.confidence >= 0.4 and time_since_change > 5.0:  # Lower threshold for sklearn
            self.last_confident_state = result.predicted_state
            self.last_state_change = current_time
            return result.predicted_state
        
        # Default to last confident state
        return self.last_confident_state
    
    def get_data_collection_stats(self) -> Dict[str, int]:
        """Get statistics about collected training data"""
        stats = {}
        total = 0
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_path, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
                stats[class_name] = count
                total += count
            else:
                stats[class_name] = 0
        
        stats['total'] = total
        return stats
    
    def reset_state_history(self):
        """Reset state history and confidence tracking"""
        self.state_history = []
        self.last_confident_state = "Overworld"
        self.last_state_change = time.time()