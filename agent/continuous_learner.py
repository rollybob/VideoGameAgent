"""
Continuous Learning System - Learn while playing
"""

import time
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

class ContinuousLearner:
    """
    Implements continuous learning during gameplay
    Automatically collects and labels data based on outcomes
    """
    
    def __init__(self, ml_detector):
        self.ml_detector = ml_detector
        self.experience_buffer = deque(maxlen=1000)  # Store recent experiences
        self.last_state = None
        self.last_action = None
        self.last_frame = None
        self.last_confidence = 0.0
        
        # Learning parameters
        self.min_confidence_for_auto_label = 0.9  # Only auto-label high confidence
        self.max_auto_samples_per_session = 50   # Limit auto collection
        self.auto_samples_collected = 0
        
    def observe_action(self, frame: np.ndarray, predicted_state: str, 
                      action: str, confidence: float):
        """Observe an action being taken"""
        experience = {
            'frame': frame.copy() if frame is not None else None,
            'predicted_state': predicted_state,
            'action': action,
            'confidence': confidence,
            'timestamp': time.time()
        }
        
        self.experience_buffer.append(experience)
        
        # Store for next observation
        self.last_state = predicted_state
        self.last_action = action
        self.last_frame = frame
        self.last_confidence = confidence
    
    def observe_outcome(self, new_state: str, success: bool = True):
        """Observe the outcome of the last action"""
        if not self.experience_buffer:
            return
        
        last_experience = self.experience_buffer[-1]
        
        # If the prediction was correct and confidence was high, use for training
        if (success and 
            last_experience['confidence'] > self.min_confidence_for_auto_label and
            self.auto_samples_collected < self.max_auto_samples_per_session):
            
            # Auto-collect this as training data
            if last_experience['frame'] is not None:
                success = self.ml_detector.collect_training_data(
                    last_experience['frame'],
                    last_experience['predicted_state'],
                    auto_save=True,
                    manual_collection=False
                )
                
                if success:
                    self.auto_samples_collected += 1
                    print(f"Auto-collected training sample: {last_experience['predicted_state']} "
                          f"({self.auto_samples_collected}/{self.max_auto_samples_per_session})")
    
    def should_retrain(self) -> bool:
        """Determine if we should retrain the model"""
        # Retrain every 20 auto-collected samples
        return self.auto_samples_collected > 0 and self.auto_samples_collected % 20 == 0
    
    def get_learning_stats(self) -> Dict:
        """Get current learning statistics"""
        return {
            'experiences_stored': len(self.experience_buffer),
            'auto_samples_collected': self.auto_samples_collected,
            'max_auto_samples': self.max_auto_samples_per_session,
            'should_retrain': self.should_retrain()
        }
