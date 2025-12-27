"""
Behavior Tree System - Fast, deterministic decision making for real-time gameplay
=================================================================================

Provides a hierarchical behavior tree implementation for making game decisions
in milliseconds rather than seconds. This replaces LLM reasoning for real-time
decisions while still allowing LLM for strategic planning.

Key concepts:
- Nodes return SUCCESS, FAILURE, or RUNNING
- Composite nodes (Selector, Sequence) control flow
- Decorator nodes modify child behavior
- Action nodes perform actual game actions
- Condition nodes check game state
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import time


class NodeStatus(Enum):
    """Result of a behavior tree node execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


@dataclass
class GameState:
    """
    Lightweight game state for fast decision making.
    Updated by perception layer, consumed by behavior tree.
    """
    # Core state
    current_context: str = "unknown"  # battle, overworld, menu, dialogue

    # Battle state
    in_battle: bool = False
    battle_menu_visible: bool = False
    move_menu_visible: bool = False
    party_menu_visible: bool = False
    current_hp_percent: float = 1.0
    opponent_hp_percent: float = 1.0

    # Movement state
    is_stuck: bool = False
    stuck_count: int = 0
    last_direction: str = "up"

    # Menu/Dialogue state
    dialogue_active: bool = False
    menu_active: bool = False

    # Detected elements (from OCR/perception)
    visible_text: List[str] = field(default_factory=list)
    detected_buttons: List[str] = field(default_factory=list)

    # Timing
    last_update: float = field(default_factory=time.time)
    frames_in_state: int = 0


class BehaviorNode(ABC):
    """Base class for all behavior tree nodes"""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self.status = NodeStatus.FAILURE

    @abstractmethod
    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        """Execute this node. Override in subclasses."""
        pass

    def reset(self):
        """Reset node state for next tree traversal"""
        self.status = NodeStatus.FAILURE


# =============================================================================
# COMPOSITE NODES - Control flow
# =============================================================================

class Selector(BehaviorNode):
    """
    Tries children in order until one succeeds.
    Like an OR gate - succeeds if ANY child succeeds.

    Use for: "Try action A, if that fails try action B, etc."
    """

    def __init__(self, name: str = "", children: List[BehaviorNode] = None):
        super().__init__(name)
        self.children = children or []

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        for child in self.children:
            result = child.tick(state, blackboard)
            if result == NodeStatus.SUCCESS:
                self.status = NodeStatus.SUCCESS
                return NodeStatus.SUCCESS
            elif result == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return NodeStatus.RUNNING

        self.status = NodeStatus.FAILURE
        return NodeStatus.FAILURE

    def reset(self):
        super().reset()
        for child in self.children:
            child.reset()


class Sequence(BehaviorNode):
    """
    Runs children in order, stops on first failure.
    Like an AND gate - succeeds only if ALL children succeed.

    Use for: "Do A, then B, then C - all must succeed"
    """

    def __init__(self, name: str = "", children: List[BehaviorNode] = None):
        super().__init__(name)
        self.children = children or []
        self.current_child = 0

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        while self.current_child < len(self.children):
            result = self.children[self.current_child].tick(state, blackboard)

            if result == NodeStatus.FAILURE:
                self.current_child = 0
                self.status = NodeStatus.FAILURE
                return NodeStatus.FAILURE
            elif result == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return NodeStatus.RUNNING

            self.current_child += 1

        self.current_child = 0
        self.status = NodeStatus.SUCCESS
        return NodeStatus.SUCCESS

    def reset(self):
        super().reset()
        self.current_child = 0
        for child in self.children:
            child.reset()


class Parallel(BehaviorNode):
    """
    Runs all children simultaneously.
    Succeeds based on success_threshold (default: all must succeed).
    """

    def __init__(self, name: str = "", children: List[BehaviorNode] = None,
                 success_threshold: int = None):
        super().__init__(name)
        self.children = children or []
        self.success_threshold = success_threshold or len(self.children)

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        success_count = 0
        failure_count = 0

        for child in self.children:
            result = child.tick(state, blackboard)
            if result == NodeStatus.SUCCESS:
                success_count += 1
            elif result == NodeStatus.FAILURE:
                failure_count += 1

        if success_count >= self.success_threshold:
            self.status = NodeStatus.SUCCESS
            return NodeStatus.SUCCESS
        elif failure_count > len(self.children) - self.success_threshold:
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE
        else:
            self.status = NodeStatus.RUNNING
            return NodeStatus.RUNNING


# =============================================================================
# DECORATOR NODES - Modify child behavior
# =============================================================================

class Inverter(BehaviorNode):
    """Inverts child result: SUCCESS -> FAILURE, FAILURE -> SUCCESS"""

    def __init__(self, child: BehaviorNode, name: str = ""):
        super().__init__(name or f"Not({child.name})")
        self.child = child

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        result = self.child.tick(state, blackboard)
        if result == NodeStatus.SUCCESS:
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE
        elif result == NodeStatus.FAILURE:
            self.status = NodeStatus.SUCCESS
            return NodeStatus.SUCCESS
        else:
            self.status = NodeStatus.RUNNING
            return NodeStatus.RUNNING


class Repeater(BehaviorNode):
    """Repeats child N times (or until failure if repeat_until_fail=True)"""

    def __init__(self, child: BehaviorNode, times: int = 1,
                 repeat_until_fail: bool = False, name: str = ""):
        super().__init__(name or f"Repeat({child.name})")
        self.child = child
        self.times = times
        self.repeat_until_fail = repeat_until_fail
        self.count = 0

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        result = self.child.tick(state, blackboard)

        if self.repeat_until_fail:
            if result == NodeStatus.FAILURE:
                self.count = 0
                self.status = NodeStatus.SUCCESS
                return NodeStatus.SUCCESS
            else:
                self.status = NodeStatus.RUNNING
                return NodeStatus.RUNNING
        else:
            if result == NodeStatus.SUCCESS:
                self.count += 1
                if self.count >= self.times:
                    self.count = 0
                    self.status = NodeStatus.SUCCESS
                    return NodeStatus.SUCCESS
            elif result == NodeStatus.FAILURE:
                self.count = 0
                self.status = NodeStatus.FAILURE
                return NodeStatus.FAILURE

            self.status = NodeStatus.RUNNING
            return NodeStatus.RUNNING

    def reset(self):
        super().reset()
        self.count = 0
        self.child.reset()


class Succeeder(BehaviorNode):
    """Always returns SUCCESS regardless of child result"""

    def __init__(self, child: BehaviorNode, name: str = ""):
        super().__init__(name or f"AlwaysSucceed({child.name})")
        self.child = child

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        self.child.tick(state, blackboard)
        self.status = NodeStatus.SUCCESS
        return NodeStatus.SUCCESS


# =============================================================================
# CONDITION NODES - Check game state
# =============================================================================

class Condition(BehaviorNode):
    """Generic condition check using a callable"""

    def __init__(self, check: Callable[[GameState, Dict], bool], name: str = ""):
        super().__init__(name or "Condition")
        self.check = check

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        if self.check(state, blackboard):
            self.status = NodeStatus.SUCCESS
            return NodeStatus.SUCCESS
        else:
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE


class InBattle(Condition):
    """Check if currently in battle"""
    def __init__(self):
        super().__init__(lambda s, b: s.in_battle, "InBattle?")


class IsStuck(Condition):
    """Check if agent appears stuck"""
    def __init__(self, threshold: int = 10):
        super().__init__(
            lambda s, b: s.stuck_count >= threshold,
            f"IsStuck(>{threshold})"
        )


class LowHealth(Condition):
    """Check if HP is below threshold"""
    def __init__(self, threshold: float = 0.3):
        super().__init__(
            lambda s, b: s.current_hp_percent < threshold,
            f"LowHealth(<{threshold})"
        )


class DialogueActive(Condition):
    """Check if dialogue is on screen"""
    def __init__(self):
        super().__init__(lambda s, b: s.dialogue_active, "DialogueActive?")


class MenuVisible(Condition):
    """Check if a menu is visible"""
    def __init__(self):
        super().__init__(lambda s, b: s.menu_active, "MenuVisible?")


class HasText(Condition):
    """Check if specific text is visible"""
    def __init__(self, text: str, case_sensitive: bool = False):
        if case_sensitive:
            check = lambda s, b: any(text in t for t in s.visible_text)
        else:
            text_lower = text.lower()
            check = lambda s, b: any(text_lower in t.lower() for t in s.visible_text)
        super().__init__(check, f"HasText('{text}')")


# =============================================================================
# ACTION NODES - Perform game actions
# =============================================================================

class Action(BehaviorNode):
    """Generic action node using a callable"""

    def __init__(self, action: Callable[[GameState, Dict], NodeStatus], name: str = ""):
        super().__init__(name or "Action")
        self.action = action

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        self.status = self.action(state, blackboard)
        return self.status


class SetBlackboard(BehaviorNode):
    """Set a value in the blackboard"""

    def __init__(self, key: str, value: Any, name: str = ""):
        super().__init__(name or f"Set({key})")
        self.key = key
        self.value = value

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        blackboard[self.key] = self.value
        self.status = NodeStatus.SUCCESS
        return NodeStatus.SUCCESS


class Wait(BehaviorNode):
    """Wait for specified duration"""

    def __init__(self, duration: float, name: str = ""):
        super().__init__(name or f"Wait({duration}s)")
        self.duration = duration
        self.start_time = None

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time
        if elapsed >= self.duration:
            self.start_time = None
            self.status = NodeStatus.SUCCESS
            return NodeStatus.SUCCESS
        else:
            self.status = NodeStatus.RUNNING
            return NodeStatus.RUNNING

    def reset(self):
        super().reset()
        self.start_time = None


# =============================================================================
# GAME-SPECIFIC ACTION NODES
# =============================================================================

class PressButton(BehaviorNode):
    """Press a game controller button"""

    def __init__(self, button: str, duration: float = 0.1, name: str = ""):
        super().__init__(name or f"Press({button})")
        self.button = button
        self.duration = duration

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        try:
            from controllers.gba_controller import press
            success = press(self.button, self.duration)
            self.status = NodeStatus.SUCCESS if success else NodeStatus.FAILURE
            return self.status
        except Exception as e:
            blackboard['last_error'] = str(e)
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE


class MashButton(BehaviorNode):
    """Mash a button repeatedly"""

    def __init__(self, button: str, count: int = 3, delay: float = 0.1, name: str = ""):
        super().__init__(name or f"Mash({button}x{count})")
        self.button = button
        self.count = count
        self.delay = delay

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        try:
            from controllers.gba_controller import mash
            success = mash(self.button, self.count, self.delay)
            self.status = NodeStatus.SUCCESS if success else NodeStatus.FAILURE
            return self.status
        except Exception as e:
            blackboard['last_error'] = str(e)
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE


class MoveDirection(BehaviorNode):
    """Move in a direction"""

    def __init__(self, direction: str = None, duration: float = 0.2, name: str = ""):
        super().__init__(name or f"Move({direction or 'dynamic'})")
        self.direction = direction
        self.duration = duration

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        direction = self.direction or blackboard.get('move_direction', 'up')
        try:
            from controllers.gba_controller import press
            success = press(direction, self.duration)
            if success:
                blackboard['last_direction'] = direction
            self.status = NodeStatus.SUCCESS if success else NodeStatus.FAILURE
            return self.status
        except Exception as e:
            blackboard['last_error'] = str(e)
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE


class RotateDirection(BehaviorNode):
    """Rotate to next direction (for unstuck behavior)"""

    DIRECTIONS = ["up", "right", "down", "left"]

    def __init__(self, name: str = ""):
        super().__init__(name or "RotateDirection")

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        current = blackboard.get('direction_index', 0)
        next_idx = (current + 1) % len(self.DIRECTIONS)
        blackboard['direction_index'] = next_idx
        blackboard['move_direction'] = self.DIRECTIONS[next_idx]
        self.status = NodeStatus.SUCCESS
        return NodeStatus.SUCCESS


class RandomDirection(BehaviorNode):
    """Pick a random direction"""

    DIRECTIONS = ["up", "right", "down", "left"]

    def __init__(self, name: str = ""):
        super().__init__(name or "RandomDirection")

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        import random
        direction = random.choice(self.DIRECTIONS)
        blackboard['move_direction'] = direction
        self.status = NodeStatus.SUCCESS
        return NodeStatus.SUCCESS


# =============================================================================
# BEHAVIOR TREE - Main container
# =============================================================================

class BehaviorTree:
    """
    Main behavior tree container.
    Manages the root node and provides tick/reset interface.
    """

    def __init__(self, root: BehaviorNode, name: str = "BehaviorTree"):
        self.name = name
        self.root = root
        self.blackboard: Dict[str, Any] = {}
        self.tick_count = 0
        self.last_tick_time = 0.0

    def tick(self, state: GameState) -> NodeStatus:
        """Execute one tick of the behavior tree"""
        start = time.perf_counter()
        result = self.root.tick(state, self.blackboard)
        self.last_tick_time = time.perf_counter() - start
        self.tick_count += 1
        return result

    def reset(self):
        """Reset tree state"""
        self.root.reset()
        self.tick_count = 0

    def clear_blackboard(self):
        """Clear the blackboard"""
        self.blackboard.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get tree statistics"""
        return {
            'name': self.name,
            'tick_count': self.tick_count,
            'last_tick_ms': self.last_tick_time * 1000,
            'blackboard_keys': list(self.blackboard.keys())
        }


# =============================================================================
# PRESET BEHAVIOR TREES
# =============================================================================

def create_exploration_tree() -> BehaviorTree:
    """Create a behavior tree for overworld exploration"""

    # Unstuck behavior
    unstuck = Sequence("Unstuck", [
        IsStuck(threshold=10),
        RotateDirection(),
        MoveDirection(duration=0.3),
    ])

    # Normal exploration
    explore = Sequence("Explore", [
        MoveDirection(duration=0.2),
        Wait(0.05),
        MashButton("A", count=1, delay=0.1),  # Interact with things
    ])

    # Main exploration selector
    root = Selector("ExplorationRoot", [
        unstuck,
        explore,
    ])

    return BehaviorTree(root, "Exploration")


def create_battle_tree() -> BehaviorTree:
    """Create a behavior tree for Pokemon battles"""

    # Select Fight option
    select_fight = Sequence("SelectFight", [
        PressButton("up", 0.05),
        PressButton("left", 0.05),
        PressButton("A", 0.1),
        Wait(0.15),
    ])

    # Select a move (cycles through)
    select_move = Sequence("SelectMove", [
        PressButton("A", 0.1),
        Wait(0.2),
        MashButton("A", count=3, delay=0.1),  # Confirm and advance dialogue
    ])

    # Full battle turn
    battle_turn = Sequence("BattleTurn", [
        select_fight,
        Wait(0.1),
        select_move,
    ])

    return BehaviorTree(battle_turn, "Battle")


def create_dialogue_tree() -> BehaviorTree:
    """Create a behavior tree for advancing dialogue"""

    root = Sequence("DialogueRoot", [
        MashButton("A", count=2, delay=0.15),
        Wait(0.1),
    ])

    return BehaviorTree(root, "Dialogue")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Behavior Tree System Test")
    print("=" * 50)

    # Create test state
    state = GameState(
        current_context="overworld",
        in_battle=False,
        stuck_count=0,
    )

    # Test exploration tree
    exploration = create_exploration_tree()

    print("\nTesting Exploration Tree:")
    for i in range(5):
        result = exploration.tick(state)
        print(f"  Tick {i+1}: {result.value} ({exploration.last_tick_time*1000:.2f}ms)")
        state.stuck_count += 1  # Simulate getting stuck

    # Simulate getting really stuck
    state.stuck_count = 15
    result = exploration.tick(state)
    print(f"  Stuck tick: {result.value} - direction now: {exploration.blackboard.get('move_direction')}")

    # Test battle tree
    print("\nTesting Battle Tree:")
    battle = create_battle_tree()
    state.in_battle = True
    state.current_context = "battle"

    # Note: This will try to actually press keys, so just show structure
    print(f"  Battle tree ready: {battle.name}")
    print(f"  Root node: {battle.root.name}")

    print("\nBehavior Tree System Test Complete!")
