"""
Fast Movement Agent - Lightweight decision making for movement
"""

import random
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

class FastMovementAgent:
    """
    Lightweight agent for fast movement decisions
    Bypasses heavy perception/text analysis for simple exploration
    """
    
    def __init__(self):
        self.last_direction = None
        self.direction_counter = 0
        self.max_same_direction = 10  # Change direction after 10 moves
        self.last_decision_time = 0
        self.min_decision_time = 0.01  # 10ms minimum between decisions
        
    def should_use_fast_mode(self, current_state: str, confidence: float = 0.0) -> bool:
        """Determine if we should use fast mode for this situation"""
        # Use fast mode for basic exploration in known safe states
        # Include Battle in case state detection is incorrect
        exploration_states = ["Overworld", "Water/Flying", "Indoor/Cave", "Indoor", "Cave", "Battle"]
        
        # Use fast mode if:
        # 1. We're in an exploration state
        # 2. No high-confidence specific action needed
        return current_state in exploration_states and confidence < 0.8
    
    def make_fast_decision(self, current_state: str, screen_analysis: Dict = None) -> Tuple[str, str, float]:
        """Make a fast movement decision without heavy AI processing"""
        current_time = time.time()
        
        # Respect minimum decision time only for actual use, not testing
        if hasattr(self, '_in_game_use') and current_time - self.last_decision_time < self.min_decision_time:
            return "wait", "Fast mode cooldown", 0.9
        
        self.last_decision_time = current_time
        
        # Simple movement logic
        if self.last_direction is None or self.direction_counter >= self.max_same_direction:
            # Choose new random direction
            self.last_direction = random.choice(["up", "down", "left", "right"])
            self.direction_counter = 0
        
        self.direction_counter += 1
        
        reasoning = f"Fast exploration: {self.last_direction} ({self.direction_counter}/{self.max_same_direction})"
        confidence = 0.7  # Good confidence for simple movement
        
        return self.last_direction, reasoning, confidence
    
    def reset_direction(self):
        """Reset direction counter (call when agent gets stuck or changes context)"""
        self.last_direction = None
        self.direction_counter = 0
