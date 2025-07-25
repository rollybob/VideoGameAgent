"""
Simple Neural Agent - Lightweight fallback when TensorFlow is not available
Uses basic heuristics and rule-based logic to provide intelligent behavior
"""
import numpy as np
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Any
from collections import deque

@dataclass
class SimpleAgentObservation:
    """Simple observation for lightweight agent"""
    visual_state: Any  # Screen frame
    game_state: str
    state_confidence: float
    action_history: List[str]
    timestamp: float

@dataclass 
class SimpleAgentAction:
    """Simple action output"""
    action_type: str
    action_value: str
    confidence: float
    reasoning: str
    expected_outcome: str

class SimpleNeuralAgent:
    """
    Lightweight rule-based agent that provides intelligent behavior
    without requiring TensorFlow. Uses patterns and heuristics.
    """
    
    def __init__(self):
        self.action_space = [
            'up', 'down', 'left', 'right', 
            'a', 'b', 'start', 'select',
            'wait'
        ]
        
        self.state_space = [
            "Overworld", "Battle", 
            "Pokemon Center", "Shop", "Gym", "Dialogue/Menu", 
            "Water/Flying", "Indoor", "Cave"
        ]
        
        # Simple memory systems
        self.action_history = deque(maxlen=50)
        self.state_history = deque(maxlen=20)
        self.movement_patterns = deque(maxlen=10)
        self.success_patterns = {}
        self.last_successful_action = {}
        
        # Strategy preferences based on state
        self.state_strategies = {
            "Overworld": {"explore": 0.7, "interact": 0.3},
            "Battle": {"attack": 0.7, "strategy": 0.3}, 
            "Pokemon Center": {"heal": 0.9, "explore": 0.1},
            "Shop": {"browse": 0.6, "buy": 0.3, "exit": 0.1},
            "Gym": {"battle": 0.7, "navigate": 0.3},
            "Dialogue/Menu": {"advance": 0.8, "navigate": 0.2},
            "Water/Flying": {"explore": 0.9, "interact": 0.1},
            "Indoor": {"explore": 0.8, "interact": 0.2},
            "Cave": {"explore": 0.7, "interact": 0.3}  # Caves might need more interaction
        }
        
        # Pattern recognition counters
        self.stuck_counter = 0
        self.repetition_penalty = {}
        self.exploration_bonus = {}
        
        print("Simple Neural Agent initialized (TensorFlow-free mode)")
    
    def process_observation(self, frame, game_state: str, reward: float = 0.0) -> SimpleAgentObservation:
        """Process game frame into simple observation"""
        return SimpleAgentObservation(
            visual_state=frame,
            game_state=game_state,
            state_confidence=1.0,
            action_history=list(self.action_history),
            timestamp=time.time()
        )
    
    def select_action(self, observation: SimpleAgentObservation) -> SimpleAgentAction:
        """Select action using intelligent rule-based logic"""
        state = observation.game_state
        
        # Update internal state
        self.state_history.append(state)
        self.update_patterns()
        
        # Get action based on current state strategy
        action, reasoning, confidence = self.get_strategic_action(state, observation)
        
        # Apply learning adjustments
        action, reasoning = self.apply_learning_adjustments(action, reasoning, state)
        
        # Update history
        self.action_history.append(action)
        
        return SimpleAgentAction(
            action_type='intelligent_rule',
            action_value=action,
            confidence=confidence,
            reasoning=reasoning,
            expected_outcome=f"Strategic action for {state}"
        )
    
    def get_strategic_action(self, state: str, observation) -> tuple:
        """Get strategic action based on game state"""
        
        if state in ["Overworld", "Water/Flying", "Indoor", "Cave"]:
            return self.handle_exploration_state(state)
            
        elif state == "Battle":
            return self.handle_battle_state(state)
            
        elif state in ["Dialogue/Menu"]:
            return self.handle_dialogue_state()
            
        elif state == "Pokemon Center":
            return self.handle_pokemon_center()
            
        elif state == "Shop":
            return self.handle_shop()
            
        elif state == "Gym":
            return self.handle_gym()
            
        else:
            # Default exploration
            return random.choice(['up', 'down', 'left', 'right']), "Default exploration", 0.5
    
    def handle_exploration_state(self, state: str) -> tuple:
        """Smart exploration logic"""
        # Check if we're stuck
        if self.is_stuck():
            action = self.get_unstuck_action()
            return action, f"Unstuck maneuver in {state}", 0.8
        
        # Prefer unexplored directions
        movement_actions = ['up', 'down', 'left', 'right']
        recent_moves = list(self.action_history)[-5:]
        
        # Calculate direction preferences
        direction_scores = {}
        for direction in movement_actions:
            score = 1.0
            
            # Penalty for recent repetition
            recent_count = recent_moves.count(direction)
            score -= recent_count * 0.3
            
            # Bonus for unexplored directions
            if direction not in recent_moves:
                score += 0.5
                
            # Bonus for breaking patterns
            if len(self.movement_patterns) > 2:
                if direction not in self.movement_patterns:
                    score += 0.3
            
            direction_scores[direction] = max(0.1, score)
        
        # Select direction based on scores
        total_score = sum(direction_scores.values())
        rand = random.random() * total_score
        cumulative = 0
        
        for direction, score in direction_scores.items():
            cumulative += score
            if rand <= cumulative:
                confidence = min(0.9, score / max(direction_scores.values()))
                return direction, f"Smart exploration - avoiding repetition", confidence
        
        # Fallback
        return random.choice(movement_actions), "Random exploration fallback", 0.3
    
    def handle_battle_state(self, state: str) -> tuple:
        """Intelligent battle decisions - unified for all battle types"""
        battle_actions = ['a', 'down', 'b']  # Attack, different move, run/back
        
        # Unified battle strategy (works for both wild and trainer battles)
        strategies = [
            ('a', 'Attack Pokemon', 0.7),
            ('down', 'Try different move', 0.2),
            ('b', 'Defensive action/run', 0.1)
        ]
        
        # Select strategy based on recent success
        for action, reason, base_conf in strategies:
            success_rate = self.success_patterns.get(f"Battle_{action}", 0.5)
            adjusted_conf = base_conf * (0.5 + success_rate)
            
            if random.random() < adjusted_conf:
                return action, reason, adjusted_conf
        
        # Default to attack
        return 'a', 'Default battle action', 0.5
    
    def handle_dialogue_state(self) -> tuple:
        """Handle dialogue and menus"""
        # Usually advance with A, sometimes navigate with directional keys
        recent_actions = list(self.action_history)[-3:]
        
        if recent_actions.count('a') > 2:
            # Too many A presses, try navigation
            nav_action = random.choice(['up', 'down'])
            return nav_action, "Menu navigation - avoiding A spam", 0.7
        else:
            return 'a', "Advance dialogue", 0.8
    
    def handle_pokemon_center(self) -> tuple:
        """Pokemon Center strategy"""
        # Move up to nurse counter, then interact
        center_sequence = ['up', 'up', 'a', 'a']
        recent_count = len([a for a in self.action_history if a in ['up', 'a']])
        
        if recent_count < 4:
            action = 'up' if recent_count % 2 == 0 else 'a'
            return action, "Pokemon Center healing sequence", 0.9
        else:
            # Exit after healing
            return 'down', "Exit Pokemon Center", 0.7
    
    def handle_shop(self) -> tuple:
        """Shop interaction strategy"""
        shop_actions = [('down', 'Browse items', 0.4),
                       ('a', 'Select/Buy', 0.3),
                       ('b', 'Back/Exit', 0.3)]
        
        action, reason, conf = random.choices(shop_actions, weights=[s[2] for s in shop_actions])[0]
        return action, reason, conf
    
    def handle_gym(self) -> tuple:
        """Gym navigation and battle"""
        # Mix of exploration and battle actions
        gym_actions = [('up', 'Navigate gym', 0.3),
                      ('down', 'Navigate gym', 0.3),
                      ('left', 'Navigate gym', 0.3),
                      ('right', 'Navigate gym', 0.3),
                      ('a', 'Interact/Battle', 0.4)]
        
        action, reason, conf = random.choices(gym_actions, weights=[s[2] for s in gym_actions])[0]
        return action, f"Gym strategy: {reason}", conf
    
    def is_stuck(self) -> bool:
        """Detect if agent is stuck in repetitive behavior"""
        if len(self.action_history) < 6:
            return False
        
        recent_actions = list(self.action_history)[-6:]
        
        # Check for alternating patterns
        if (len(set(recent_actions[-4:])) == 2 and 
            recent_actions[-4] == recent_actions[-2] and
            recent_actions[-3] == recent_actions[-1]):
            self.stuck_counter += 1
            return self.stuck_counter > 2
        
        # Check for repetitive single action
        if len(set(recent_actions)) == 1:
            self.stuck_counter += 1
            return self.stuck_counter > 3
        
        # Reset counter if not stuck
        self.stuck_counter = max(0, self.stuck_counter - 1)
        return False
    
    def get_unstuck_action(self) -> str:
        """Get action to break out of stuck patterns"""
        recent_actions = list(self.action_history)[-5:]
        movement_actions = ['up', 'down', 'left', 'right']
        escape_actions = ['b', 'start', 'select']
        
        # Try movement not recently used
        unused_movements = [a for a in movement_actions if a not in recent_actions]
        if unused_movements:
            return random.choice(unused_movements)
        
        # Try escape actions
        if random.random() < 0.3:
            return random.choice(escape_actions)
        
        # Random perpendicular movement
        if recent_actions:
            last_move = recent_actions[-1]
            if last_move in ['up', 'down']:
                return random.choice(['left', 'right'])
            elif last_move in ['left', 'right']:
                return random.choice(['up', 'down'])
        
        return random.choice(movement_actions)
    
    def apply_learning_adjustments(self, action: str, reasoning: str, state: str) -> tuple:
        """Apply learning-based adjustments to actions"""
        # Track success patterns (simplified)
        pattern_key = f"{state}_{action}"
        
        # Update repetition penalties
        if action in self.repetition_penalty:
            self.repetition_penalty[action] += 1
        else:
            self.repetition_penalty[action] = 1
        
        # Decay penalties over time
        for key in self.repetition_penalty:
            self.repetition_penalty[key] *= 0.95
        
        # High repetition penalty - try alternative
        if self.repetition_penalty.get(action, 0) > 5:
            alternatives = [a for a in self.action_space if a != action]
            if alternatives:
                alt_action = random.choice(alternatives)
                return alt_action, f"Learning adjustment: avoiding {action} repetition", 0.6
        
        return action, reasoning
    
    def update_patterns(self):
        """Update movement and behavior patterns"""
        if len(self.action_history) >= 3:
            recent_pattern = tuple(list(self.action_history)[-3:])
            self.movement_patterns.append(recent_pattern)
    
    def save_model(self, path: str):
        """Save learning data (patterns and success rates)"""
        import json
        import os
        
        os.makedirs(path, exist_ok=True)
        
        save_data = {
            'success_patterns': self.success_patterns,
            'repetition_penalty': dict(self.repetition_penalty),
            'exploration_bonus': self.exploration_bonus,
            'timestamp': time.time()
        }
        
        with open(f"{path}/simple_neural_agent.json", 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Simple neural agent data saved to {path}")
    
    def load_model(self, path: str) -> bool:
        """Load learning data"""
        import json
        import os
        
        model_file = f"{path}/simple_neural_agent.json"
        if os.path.exists(model_file):
            try:
                with open(model_file, 'r') as f:
                    data = json.load(f)
                
                self.success_patterns = data.get('success_patterns', {})
                self.repetition_penalty = data.get('repetition_penalty', {})
                self.exploration_bonus = data.get('exploration_bonus', {})
                
                print(f"Simple neural agent data loaded from {path}")
                return True
            except Exception as e:
                print(f"Error loading simple agent data: {e}")
                return False
        return False
    
    def get_model_summary(self):
        """Get summary of the simple agent"""
        print("\n=== SIMPLE NEURAL AGENT (TensorFlow-free) ===")
        print(f"Action Space: {len(self.action_space)} actions")
        print(f"State Space: {len(self.state_space)} states")
        print(f"Success Patterns Learned: {len(self.success_patterns)}")
        print(f"Strategy States: {len(self.state_strategies)}")
        print("Architecture: Rule-based with pattern recognition")
        print("Memory: Action history, success patterns, anti-repetition")
        print("Learning: Adaptive strategy selection based on experience")