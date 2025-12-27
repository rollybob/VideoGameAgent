"""
Tiered Decision System - Real-time game agent architecture
===========================================================

Implements a three-tier decision system for real-time gameplay:

FAST LAYER (every frame, <20ms):
  - Behavior trees for immediate responses
  - No heavy computation
  - Handles: movement, button presses, anti-stuck

MEDIUM LAYER (every 500ms):
  - State analysis and goal evaluation
  - Memory updates
  - Handles: context switching, goal progress, memory

SLOW LAYER (every 5s+, async):
  - LLM strategic reasoning (when hardware available)
  - Long-term planning
  - Handles: quest understanding, strategy adaptation

The key insight: most game decisions don't need AI reasoning!
Walking, fighting, menu navigation - these are pattern matching,
not deep thinking. Save the LLM for actual strategic decisions.
"""

import time
import threading
from queue import Queue, Empty
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from behavior_tree import (
    BehaviorTree, GameState, NodeStatus,
    create_exploration_tree, create_battle_tree, create_dialogue_tree
)
from debug_system import get_debugger, info, debug, warning, error, monitor_performance
from config import get_config


class DecisionLayer(Enum):
    """The three decision layers"""
    FAST = "fast"      # Behavior trees, <20ms
    MEDIUM = "medium"  # State analysis, <200ms
    SLOW = "slow"      # LLM reasoning, async


@dataclass
class LayerDecision:
    """Decision from a layer"""
    layer: DecisionLayer
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class StrategicGoal:
    """Long-term goal from slow layer"""
    goal_type: str
    description: str
    priority: int = 1
    sub_goals: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None


class TieredDecisionSystem:
    """
    Orchestrates the three decision layers.

    Fast layer runs every frame and makes immediate decisions.
    Medium layer runs periodically for state analysis.
    Slow layer runs asynchronously for strategic planning.
    """

    def __init__(self):
        self.config = get_config()
        self.debugger = get_debugger()

        # Game state (shared across layers)
        self.game_state = GameState()

        # Layer timing
        self.last_medium_tick = 0.0
        self.last_slow_tick = 0.0
        self.medium_interval = self.config.reasoning.medium_layer_interval_ms / 1000.0
        self.slow_interval = self.config.reasoning.slow_layer_interval_ms / 1000.0

        # Behavior trees for fast layer
        self.behavior_trees: Dict[str, BehaviorTree] = {
            'exploration': create_exploration_tree(),
            'battle': create_battle_tree(),
            'dialogue': create_dialogue_tree(),
        }
        self.active_tree: Optional[BehaviorTree] = None

        # Strategic goals from slow layer
        self.strategic_goals: List[StrategicGoal] = []
        self.current_strategy: Optional[str] = None

        # Async slow layer
        self.slow_layer_queue: Queue = Queue()
        self.slow_layer_results: Queue = Queue()
        self.slow_layer_thread: Optional[threading.Thread] = None
        self.slow_layer_running = False

        # Performance tracking
        self.stats = {
            'fast_ticks': 0,
            'medium_ticks': 0,
            'slow_ticks': 0,
            'avg_fast_ms': 0.0,
            'avg_medium_ms': 0.0,
            'tree_switches': 0,
        }

        # Callbacks for external systems
        self.on_state_change: Optional[Callable[[str], None]] = None
        self.on_goal_update: Optional[Callable[[StrategicGoal], None]] = None

    def start(self):
        """Start the decision system (including async slow layer)"""
        info("tiered_decision", "Starting tiered decision system")

        # Start slow layer thread
        self.slow_layer_running = True
        self.slow_layer_thread = threading.Thread(
            target=self._slow_layer_worker,
            daemon=True
        )
        self.slow_layer_thread.start()

        # Select initial behavior tree
        self._select_behavior_tree("exploration")

    def stop(self):
        """Stop the decision system"""
        self.slow_layer_running = False
        if self.slow_layer_thread:
            self.slow_layer_thread.join(timeout=1.0)
        info("tiered_decision", "Tiered decision system stopped")

    # =========================================================================
    # MAIN TICK - Called every frame
    # =========================================================================

    @monitor_performance("tiered_decision", "main_tick")
    def tick(self, frame=None) -> Optional[LayerDecision]:
        """
        Main tick - called every frame.
        Runs fast layer every time, medium/slow as needed.
        """
        current_time = time.time()

        # Always run fast layer
        fast_decision = self._tick_fast_layer()

        # Run medium layer if interval elapsed
        if current_time - self.last_medium_tick >= self.medium_interval:
            self._tick_medium_layer(frame)
            self.last_medium_tick = current_time

        # Queue slow layer if interval elapsed (runs async)
        if current_time - self.last_slow_tick >= self.slow_interval:
            self._queue_slow_layer()
            self.last_slow_tick = current_time

        # Check for slow layer results
        self._process_slow_layer_results()

        return fast_decision

    # =========================================================================
    # FAST LAYER - Behavior trees, <20ms
    # =========================================================================

    @monitor_performance("tiered_decision", "fast_layer")
    def _tick_fast_layer(self) -> Optional[LayerDecision]:
        """
        Fast layer: Execute behavior tree for immediate action.
        Target: <20ms per tick
        """
        start = time.perf_counter()

        if not self.active_tree:
            return None

        try:
            result = self.active_tree.tick(self.game_state)

            elapsed_ms = (time.perf_counter() - start) * 1000
            self._update_fast_stats(elapsed_ms)

            if elapsed_ms > 20:
                warning("tiered_decision", f"Fast layer exceeded 20ms: {elapsed_ms:.2f}ms")

            # Extract action from blackboard if available
            action = self.active_tree.blackboard.get('last_action', 'continue')

            return LayerDecision(
                layer=DecisionLayer.FAST,
                action=action,
                parameters=dict(self.active_tree.blackboard),
                confidence=1.0 if result == NodeStatus.SUCCESS else 0.5,
            )

        except Exception as e:
            error("tiered_decision", "Fast layer error", exception=e)
            return None

    def _update_fast_stats(self, elapsed_ms: float):
        """Update fast layer statistics"""
        self.stats['fast_ticks'] += 1
        n = self.stats['fast_ticks']
        self.stats['avg_fast_ms'] = (
            self.stats['avg_fast_ms'] * (n - 1) + elapsed_ms
        ) / n

    # =========================================================================
    # MEDIUM LAYER - State analysis, <200ms
    # =========================================================================

    @monitor_performance("tiered_decision", "medium_layer")
    def _tick_medium_layer(self, frame=None):
        """
        Medium layer: Analyze state and update context.
        Target: <200ms, runs every 500ms
        """
        start = time.perf_counter()

        try:
            # Update game state from frame
            if frame is not None:
                self._update_game_state_from_frame(frame)

            # Check for context changes
            new_context = self._determine_context()
            if new_context != self.game_state.current_context:
                self._handle_context_change(new_context)

            # Evaluate stuck detection
            self._evaluate_stuck_state()

            # Update goal progress
            self._update_goal_progress()

            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stats['medium_ticks'] += 1

            if elapsed_ms > 200:
                warning("tiered_decision", f"Medium layer exceeded 200ms: {elapsed_ms:.2f}ms")

            debug("tiered_decision", f"Medium tick: context={new_context}, elapsed={elapsed_ms:.2f}ms")

        except Exception as e:
            error("tiered_decision", "Medium layer error", exception=e)

    def _update_game_state_from_frame(self, frame):
        """Update game state based on frame analysis"""
        try:
            from screen_reader import detect_game_state, analyze_screen_regions

            # Detect current game state
            state = detect_game_state(frame)
            self.game_state.in_battle = "Battle" in state
            self.game_state.dialogue_active = state == "Dialogue/Menu"

            # Update frame counter
            self.game_state.frames_in_state += 1
            self.game_state.last_update = time.time()

        except Exception as e:
            debug("tiered_decision", f"Frame analysis error: {e}")

    def _determine_context(self) -> str:
        """Determine current game context"""
        if self.game_state.in_battle:
            return "battle"
        elif self.game_state.dialogue_active:
            return "dialogue"
        elif self.game_state.menu_active:
            return "menu"
        else:
            return "overworld"

    def _handle_context_change(self, new_context: str):
        """Handle a change in game context"""
        old_context = self.game_state.current_context
        self.game_state.current_context = new_context
        self.game_state.frames_in_state = 0

        info("tiered_decision", f"Context change: {old_context} -> {new_context}")

        # Select appropriate behavior tree
        self._select_behavior_tree(new_context)

        # Notify callback
        if self.on_state_change:
            self.on_state_change(new_context)

    def _select_behavior_tree(self, context: str):
        """Select behavior tree for context"""
        tree_map = {
            'battle': 'battle',
            'dialogue': 'dialogue',
            'menu': 'dialogue',  # Use dialogue tree for menus too
            'overworld': 'exploration',
        }

        tree_name = tree_map.get(context, 'exploration')

        if tree_name in self.behavior_trees:
            if self.active_tree != self.behavior_trees[tree_name]:
                self.active_tree = self.behavior_trees[tree_name]
                self.active_tree.reset()
                self.stats['tree_switches'] += 1
                debug("tiered_decision", f"Switched to {tree_name} tree")

    def _evaluate_stuck_state(self):
        """Evaluate if agent is stuck"""
        # This would use frame similarity from the agent
        # For now, just track via game state
        if self.game_state.frames_in_state > 60:  # ~2 seconds at 30fps
            self.game_state.stuck_count += 1
            self.game_state.is_stuck = self.game_state.stuck_count > 10
        else:
            self.game_state.stuck_count = max(0, self.game_state.stuck_count - 1)
            self.game_state.is_stuck = False

    def _update_goal_progress(self):
        """Update progress on strategic goals"""
        # Remove expired goals
        current_time = time.time()
        self.strategic_goals = [
            g for g in self.strategic_goals
            if g.expires_at is None or g.expires_at > current_time
        ]

    # =========================================================================
    # SLOW LAYER - LLM reasoning, async
    # =========================================================================

    def _queue_slow_layer(self):
        """Queue a slow layer analysis (runs async)"""
        context = {
            'game_state': self.game_state.current_context,
            'in_battle': self.game_state.in_battle,
            'stuck_count': self.game_state.stuck_count,
            'current_goals': [g.description for g in self.strategic_goals],
            'timestamp': time.time(),
        }

        try:
            self.slow_layer_queue.put_nowait(context)
        except Exception:
            pass  # Queue full, skip this tick

    def _slow_layer_worker(self):
        """Worker thread for slow layer processing"""
        while self.slow_layer_running:
            try:
                context = self.slow_layer_queue.get(timeout=1.0)
                result = self._process_slow_layer(context)
                if result:
                    self.slow_layer_results.put(result)
                self.stats['slow_ticks'] += 1
            except Empty:
                continue
            except Exception as e:
                error("tiered_decision", "Slow layer worker error", exception=e)

    def _process_slow_layer(self, context: Dict[str, Any]) -> Optional[StrategicGoal]:
        """
        Process slow layer reasoning.
        This is where LLM reasoning would go when hardware is available.
        For now, uses simple heuristics.
        """
        try:
            # Simple heuristic-based strategy (placeholder for LLM)
            goal = self._heuristic_strategy(context)
            return goal

        except Exception as e:
            error("tiered_decision", "Slow layer processing error", exception=e)
            return None

    def _heuristic_strategy(self, context: Dict[str, Any]) -> Optional[StrategicGoal]:
        """
        Simple heuristic strategy (placeholder for LLM reasoning).
        Replace this with actual LLM calls when hardware is available.
        """
        game_state = context.get('game_state', 'unknown')

        # Generate simple goals based on context
        if context.get('stuck_count', 0) > 20:
            return StrategicGoal(
                goal_type="escape",
                description="Agent is stuck, try alternative routes",
                priority=1,
                sub_goals=["rotate_direction", "backtrack", "try_menu"],
                expires_at=time.time() + 30,
            )

        if game_state == 'overworld':
            return StrategicGoal(
                goal_type="explore",
                description="Explore the current area thoroughly",
                priority=2,
                sub_goals=["check_npcs", "find_items", "battle_wild"],
                expires_at=time.time() + 60,
            )

        if game_state == 'battle':
            return StrategicGoal(
                goal_type="win_battle",
                description="Win the current battle efficiently",
                priority=1,
                sub_goals=["use_effective_moves", "heal_if_needed"],
                expires_at=time.time() + 120,
            )

        return None

    def _process_slow_layer_results(self):
        """Process any completed slow layer results"""
        try:
            while True:
                goal = self.slow_layer_results.get_nowait()
                if goal:
                    self._apply_strategic_goal(goal)
        except Empty:
            pass

    def _apply_strategic_goal(self, goal: StrategicGoal):
        """Apply a strategic goal from slow layer"""
        # Check if we already have a similar goal
        for existing in self.strategic_goals:
            if existing.goal_type == goal.goal_type:
                # Update instead of add
                existing.description = goal.description
                existing.sub_goals = goal.sub_goals
                existing.expires_at = goal.expires_at
                return

        # Add new goal
        self.strategic_goals.append(goal)
        self.strategic_goals.sort(key=lambda g: g.priority)

        info("tiered_decision", f"New strategic goal: {goal.goal_type} - {goal.description}")

        if self.on_goal_update:
            self.on_goal_update(goal)

    # =========================================================================
    # STATE UPDATES (called by external systems)
    # =========================================================================

    def update_battle_state(self, in_battle: bool, hp_percent: float = 1.0):
        """Update battle state from external system"""
        self.game_state.in_battle = in_battle
        self.game_state.current_hp_percent = hp_percent

    def update_stuck_counter(self, count: int):
        """Update stuck counter from frame similarity"""
        self.game_state.stuck_count = count
        self.game_state.is_stuck = count > 10

    def update_visible_text(self, texts: List[str]):
        """Update visible text from OCR"""
        self.game_state.visible_text = texts

    def set_detected_buttons(self, buttons: List[str]):
        """Set detected UI buttons"""
        self.game_state.detected_buttons = buttons

    # =========================================================================
    # CUSTOM BEHAVIOR TREES
    # =========================================================================

    def register_behavior_tree(self, name: str, tree: BehaviorTree):
        """Register a custom behavior tree"""
        self.behavior_trees[name] = tree
        info("tiered_decision", f"Registered behavior tree: {name}")

    def get_current_tree(self) -> Optional[str]:
        """Get name of current active tree"""
        for name, tree in self.behavior_trees.items():
            if tree is self.active_tree:
                return name
        return None

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = dict(self.stats)
        stats['current_context'] = self.game_state.current_context
        stats['current_tree'] = self.get_current_tree()
        stats['active_goals'] = len(self.strategic_goals)
        stats['is_stuck'] = self.game_state.is_stuck

        if self.active_tree:
            stats['tree_stats'] = self.active_tree.get_stats()

        return stats


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Tiered Decision System Test")
    print("=" * 50)

    # Create and start system
    system = TieredDecisionSystem()

    # Set up callbacks
    def on_state_change(new_state):
        print(f"  [Callback] State changed to: {new_state}")

    def on_goal_update(goal):
        print(f"  [Callback] New goal: {goal.goal_type}")

    system.on_state_change = on_state_change
    system.on_goal_update = on_goal_update

    system.start()

    print("\nSimulating ticks:")
    try:
        for i in range(10):
            decision = system.tick()
            if decision:
                print(f"  Tick {i+1}: {decision.action} (confidence: {decision.confidence})")
            else:
                print(f"  Tick {i+1}: No decision")

            # Simulate time passing
            time.sleep(0.1)

            # Simulate state changes
            if i == 3:
                print("\n  [Simulating battle start]")
                system.update_battle_state(True, 0.8)

            if i == 6:
                print("\n  [Simulating battle end]")
                system.update_battle_state(False)

    finally:
        system.stop()

    print("\nStatistics:")
    stats = system.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nTiered Decision System Test Complete!")
