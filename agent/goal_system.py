import time
import random
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

class GoalPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class GoalStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class Goal:
    id: str
    name: str
    description: str
    priority: GoalPriority
    goal_type: str
    target_data: Dict[str, Any] = field(default_factory=dict)
    status: GoalStatus = GoalStatus.ACTIVE
    created_time: float = field(default_factory=time.time)
    progress: float = 0.0  # 0.0 to 1.0
    max_attempts: int = 10
    attempts: int = 0
    timeout: float = 300.0  # 5 minutes default timeout
    prerequisites: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        return time.time() - self.created_time > self.timeout
    
    def increment_attempt(self):
        self.attempts += 1
        if self.attempts >= self.max_attempts:
            self.status = GoalStatus.FAILED

class GoalSystem:
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.goal_queue: List[str] = []  # Ordered by priority
        self.completed_goals: List[str] = []
        self.failed_goals: List[str] = []
        
        # Goal state tracking
        self.current_goal_id: Optional[str] = None
        self.last_goal_evaluation = time.time()
        self.evaluation_interval = 5.0  # Evaluate goals every 5 seconds
        
        # Game state tracking for goal conditions
        self.current_badges = 0
        self.current_pokemon_count = 0
        self.current_health_status = "healthy"
        self.current_money = 0
        self.last_area_change = time.time()
        self.stuck_counter = 0
        
    def reset(self):
        """Reset the goal system to initial state"""
        self.goals.clear()
        self.goal_queue.clear()
        self.completed_goals.clear()
        self.failed_goals.clear()
        
        # Reset goal state tracking
        self.current_goal_id = None
        self.last_goal_evaluation = time.time()
        
        # Reset game state tracking
        self.current_badges = 0
        self.current_pokemon_count = 0
        self.current_health_status = "healthy"
        self.current_money = 0
        self.last_area_change = time.time()
        self.stuck_counter = 0
        
    def create_goal(self, goal_id: str, name: str, description: str, 
                   priority: GoalPriority, goal_type: str, 
                   target_data: Dict[str, Any] = None) -> Goal:
        """Create a new goal"""
        if target_data is None:
            target_data = {}
            
        goal = Goal(
            id=goal_id,
            name=name,
            description=description,
            priority=priority,
            goal_type=goal_type,
            target_data=target_data
        )
        
        self.goals[goal_id] = goal
        self._insert_goal_by_priority(goal_id)
        return goal
    
    def _insert_goal_by_priority(self, goal_id: str):
        """Insert goal into queue based on priority"""
        goal = self.goals[goal_id]
        priority_value = goal.priority.value
        
        # Find insertion point
        insert_index = len(self.goal_queue)
        for i, existing_id in enumerate(self.goal_queue):
            existing_goal = self.goals[existing_id]
            if existing_goal.priority.value > priority_value:
                insert_index = i
                break
        
        self.goal_queue.insert(insert_index, goal_id)
    
    def get_current_goal(self) -> Optional[Goal]:
        """Get the highest priority active goal"""
        self._cleanup_goals()
        
        for goal_id in self.goal_queue:
            goal = self.goals.get(goal_id)
            if goal and goal.status == GoalStatus.ACTIVE:
                return goal
        return None
    
    def complete_goal(self, goal_id: str, progress: float = 1.0):
        """Mark a goal as completed"""
        if goal_id in self.goals:
            self.goals[goal_id].status = GoalStatus.COMPLETED
            self.goals[goal_id].progress = progress
            self.completed_goals.append(goal_id)
            if goal_id in self.goal_queue:
                self.goal_queue.remove(goal_id)
    
    def fail_goal(self, goal_id: str, reason: str = ""):
        """Mark a goal as failed"""
        if goal_id in self.goals:
            self.goals[goal_id].status = GoalStatus.FAILED
            self.goals[goal_id].target_data["failure_reason"] = reason
            self.failed_goals.append(goal_id)
            if goal_id in self.goal_queue:
                self.goal_queue.remove(goal_id)
    
    def _cleanup_goals(self):
        """Remove expired or invalid goals"""
        to_remove = []
        for goal_id in self.goal_queue:
            goal = self.goals.get(goal_id)
            if not goal or goal.status != GoalStatus.ACTIVE or goal.is_expired():
                to_remove.append(goal_id)
        
        for goal_id in to_remove:
            if goal_id in self.goal_queue:
                self.goal_queue.remove(goal_id)
    
    def update_game_state(self, badges: int = None, pokemon_count: int = None, 
                         health_status: str = None, money: int = None,
                         stuck_counter: int = None):
        """Update tracked game state for goal evaluation"""
        if badges is not None:
            self.current_badges = badges
        if pokemon_count is not None:
            self.current_pokemon_count = pokemon_count
        if health_status is not None:
            self.current_health_status = health_status
        if money is not None:
            self.current_money = money
        if stuck_counter is not None:
            self.stuck_counter = stuck_counter
    
    def evaluate_and_create_goals(self, game_state: str, memory_data: Dict[str, Any] = None):
        """Evaluate current situation and create appropriate goals"""
        if time.time() - self.last_goal_evaluation < self.evaluation_interval:
            return
        
        self.last_goal_evaluation = time.time()
        
        # Critical goals (highest priority)
        self._evaluate_critical_goals(game_state, memory_data)
        
        # High priority goals (story progression)
        self._evaluate_high_priority_goals(game_state, memory_data)
        
        # Medium priority goals (pokemon and battles)
        self._evaluate_medium_priority_goals(game_state, memory_data)
        
        # Low priority goals (exploration and items)
        self._evaluate_low_priority_goals(game_state, memory_data)
    
    def _evaluate_critical_goals(self, game_state: str, memory_data: Dict[str, Any]):
        """Critical goals that need immediate attention"""
        
        # Health emergency
        if self.current_health_status == "low" and not self._has_goal_type("heal_pokemon"):
            self.create_goal(
                f"heal_emergency_{time.time()}",
                "Emergency Healing",
                "Team health is critical - find Pokemon Center immediately",
                GoalPriority.CRITICAL,
                "heal_pokemon",
                {"urgency": "critical", "target": "pokemon_center"}
            )
        
        # Stuck resolution
        if self.stuck_counter > 20 and not self._has_goal_type("unstuck"):
            self.create_goal(
                f"unstuck_{time.time()}",
                "Resolve Stuck Situation", 
                "Agent appears stuck - try different strategy",
                GoalPriority.CRITICAL,
                "unstuck",
                {"stuck_count": self.stuck_counter, "strategy": "escape"}
            )
    
    def _evaluate_high_priority_goals(self, game_state: str, memory_data: Dict[str, Any]):
        """High priority story progression goals"""
        
        # Gym challenge
        if not self._has_goal_type("gym_challenge"):
            if memory_data and "landmarks" in memory_data.get("world_map", {}):
                gyms = memory_data["world_map"]["landmarks"].get("gyms", [])
                if gyms and self.current_badges < 8:
                    nearest_gym = gyms[0]  # Simplification - pick first gym
                    self.create_goal(
                        f"gym_challenge_{self.current_badges + 1}",
                        f"Challenge Gym #{self.current_badges + 1}",
                        "Defeat gym leader to earn badge",
                        GoalPriority.HIGH,
                        "gym_challenge",
                        {"gym_location": nearest_gym, "badge_number": self.current_badges + 1}
                    )
        
        # Story progression (talk to key NPCs)
        if game_state == "Dialogue/Menu" and not self._has_goal_type("story_npc"):
            self.create_goal(
                f"story_npc_{time.time()}",
                "Important Conversation",
                "Engage with NPC for story progression", 
                GoalPriority.HIGH,
                "story_npc",
                {"interaction_type": "dialogue"}
            )
    
    def _evaluate_medium_priority_goals(self, game_state: str, memory_data: Dict[str, Any]):
        """Medium priority pokemon and battle goals"""
        
        # Pokemon collection
        if game_state == "Wild Battle" and not self._has_goal_type("catch_pokemon"):
            self.create_goal(
                f"catch_pokemon_{time.time()}",
                "Catch Wild Pokemon",
                "Attempt to catch new Pokemon species",
                GoalPriority.MEDIUM,
                "catch_pokemon",
                {"battle_type": "wild", "strategy": "catch"}
            )
        
        # Trainer battle
        if game_state == "Trainer Battle" and not self._has_goal_type("trainer_battle"):
            self.create_goal(
                f"trainer_battle_{time.time()}",
                "Defeat Trainer",
                "Win trainer battle for experience and money",
                GoalPriority.MEDIUM,
                "trainer_battle",
                {"battle_type": "trainer", "strategy": "win"}
            )
        
        # Team healing (non-critical)
        if self.current_health_status == "medium" and not self._has_goal_type("heal_pokemon"):
            self.create_goal(
                f"heal_routine_{time.time()}",
                "Routine Team Healing",
                "Visit Pokemon Center for team maintenance",
                GoalPriority.MEDIUM,
                "heal_pokemon",
                {"urgency": "routine", "target": "pokemon_center"}
            )
    
    def _evaluate_low_priority_goals(self, game_state: str, memory_data: Dict[str, Any]):
        """Low priority exploration and resource goals"""
        
        # Area exploration
        if not self._has_goal_type("explore_area") and self.stuck_counter < 5:
            unexplored_areas = self._find_unexplored_areas(memory_data)
            if unexplored_areas:
                self.create_goal(
                    f"explore_{time.time()}",
                    "Explore New Area",
                    "Discover unvisited locations",
                    GoalPriority.LOW,
                    "explore_area",
                    {"target_areas": unexplored_areas, "exploration_type": "systematic"}
                )
        
        # Item collection
        if game_state == "Shop" and not self._has_goal_type("shop_items"):
            self.create_goal(
                f"shop_{time.time()}",
                "Browse Shop Items",
                "Check for useful items and supplies",
                GoalPriority.LOW,
                "shop_items",
                {"action": "browse", "budget": self.current_money}
            )
    
    def _has_goal_type(self, goal_type: str) -> bool:
        """Check if we already have an active goal of this type"""
        for goal_id in self.goal_queue:
            goal = self.goals.get(goal_id)
            if goal and goal.goal_type == goal_type and goal.status == GoalStatus.ACTIVE:
                return True
        return False
    
    def _find_unexplored_areas(self, memory_data: Dict[str, Any]) -> List[str]:
        """Find areas that haven't been fully explored"""
        if not memory_data:
            return ["unknown_areas"]
        
        areas = memory_data.get("areas", {})
        unexplored = []
        
        for area_hash, area_data in areas.items():
            visit_count = area_data.get("visit_count", 0)
            if visit_count < 3:  # Areas visited less than 3 times
                unexplored.append(area_data.get("name", f"area_{area_hash[:6]}"))
        
        return unexplored
    
    def get_goal_action(self, current_goal: Goal, game_state: str, 
                       screen_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get the recommended action for the current goal"""
        
        if not current_goal:
            return {"action": "explore", "direction": "random"}
        
        goal_type = current_goal.goal_type
        
        if goal_type == "heal_pokemon":
            return self._get_healing_action(current_goal, game_state, screen_analysis)
        elif goal_type == "gym_challenge":
            return self._get_gym_action(current_goal, game_state, screen_analysis)
        elif goal_type == "catch_pokemon":
            return self._get_catch_action(current_goal, game_state, screen_analysis)
        elif goal_type == "trainer_battle":
            return self._get_battle_action(current_goal, game_state, screen_analysis)
        elif goal_type == "explore_area":
            return self._get_exploration_action(current_goal, game_state, screen_analysis)
        elif goal_type == "shop_items":
            return self._get_shop_action(current_goal, game_state, screen_analysis)
        elif goal_type == "unstuck":
            return self._get_unstuck_action(current_goal, game_state, screen_analysis)
        elif goal_type == "story_npc":
            return self._get_dialogue_action(current_goal, game_state, screen_analysis)
        else:
            return {"action": "explore", "direction": "smart"}
    
    def _get_healing_action(self, goal: Goal, game_state: str, screen_analysis: Dict[str, Any]) -> Dict[str, Any]:
        if game_state == "Pokemon Center":
            goal.progress = 1.0
            return {"action": "interact", "button": "A", "context": "heal_team"}
        else:
            return {"action": "navigate", "target": "pokemon_center", "priority": "high"}
    
    def _get_gym_action(self, goal: Goal, game_state: str, screen_analysis: Dict[str, Any]) -> Dict[str, Any]:
        if game_state == "Gym":
            return {"action": "navigate_gym", "strategy": "challenge_leader"}
        elif game_state in ["Trainer Battle", "Battle"]:
            return {"action": "battle", "strategy": "aggressive"}
        else:
            gym_location = goal.target_data.get("gym_location")
            return {"action": "navigate", "target": gym_location, "priority": "high"}
    
    def _get_catch_action(self, goal: Goal, game_state: str, screen_analysis: Dict[str, Any]) -> Dict[str, Any]:
        if game_state == "Wild Battle":
            return {"action": "catch_sequence", "strategy": "weaken_then_catch"}
        else:
            goal.progress = 1.0  # Battle ended
            return {"action": "continue", "result": "battle_ended"}
    
    def _get_battle_action(self, goal: Goal, game_state: str, screen_analysis: Dict[str, Any]) -> Dict[str, Any]:
        if game_state == "Trainer Battle":
            return {"action": "battle", "strategy": "win"}
        else:
            goal.progress = 1.0  # Battle ended
            return {"action": "continue", "result": "battle_ended"}
    
    def _get_exploration_action(self, goal: Goal, game_state: str, screen_analysis: Dict[str, Any]) -> Dict[str, Any]:
        target_areas = goal.target_data.get("target_areas", [])
        return {"action": "explore", "targets": target_areas, "strategy": "systematic"}
    
    def _get_shop_action(self, goal: Goal, game_state: str, screen_analysis: Dict[str, Any]) -> Dict[str, Any]:
        if game_state == "Shop":
            return {"action": "browse_shop", "budget": goal.target_data.get("budget", 0)}
        else:
            goal.progress = 1.0  # Left shop
            return {"action": "continue", "result": "shop_visit_complete"}
    
    def _get_unstuck_action(self, goal: Goal, game_state: str, screen_analysis: Dict[str, Any]) -> Dict[str, Any]:
        strategies = ["random_movement", "backtrack", "menu_escape", "direction_change"]
        strategy = random.choice(strategies)
        return {"action": "unstuck_strategy", "method": strategy}
    
    def _get_dialogue_action(self, goal: Goal, game_state: str, screen_analysis: Dict[str, Any]) -> Dict[str, Any]:
        if game_state == "Dialogue/Menu":
            return {"action": "dialogue", "approach": "advance"}
        else:
            goal.progress = 1.0  # Dialogue ended
            return {"action": "continue", "result": "dialogue_complete"}
    
    def get_goals_summary(self) -> Dict[str, Any]:
        """Get summary of current goals for display"""
        active_goals = [self.goals[gid] for gid in self.goal_queue 
                       if gid in self.goals and self.goals[gid].status == GoalStatus.ACTIVE]
        
        return {
            "total_goals": len(self.goals),
            "active_goals": len(active_goals),
            "completed_goals": len(self.completed_goals),
            "failed_goals": len(self.failed_goals),
            "current_goal": active_goals[0] if active_goals else None,
            "goal_queue": [{"name": self.goals[gid].name, "priority": self.goals[gid].priority.name} 
                          for gid in self.goal_queue[:5] if gid in self.goals]
        }