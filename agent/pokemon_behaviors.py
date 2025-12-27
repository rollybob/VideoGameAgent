"""
Pokemon-Specific Behavior Trees - Specialized behaviors for Pokemon games
==========================================================================

Contains behavior trees optimized for Pokemon Fire Red/Leaf Green gameplay:
- Battle management (with PP tracking, switching, type awareness)
- Exploration (grass walking, NPC interaction, door/cave navigation)
- Menu navigation (healing, saving, item management)
- Dialogue advancement
"""

from typing import Dict, Any, List, Optional
from behavior_tree import (
    BehaviorTree, BehaviorNode, NodeStatus, GameState,
    Selector, Sequence, Parallel, Inverter, Repeater,
    Condition, Action, Wait, SetBlackboard,
    PressButton, MashButton, MoveDirection, RotateDirection
)


# =============================================================================
# POKEMON-SPECIFIC CONDITIONS
# =============================================================================

class InPokemonBattle(Condition):
    """Check if in a Pokemon battle"""
    def __init__(self):
        super().__init__(lambda s, b: s.in_battle, "InPokemonBattle?")


class BattleMenuVisible(Condition):
    """Check if main battle menu is visible (Fight/Pokemon/Bag/Run)"""
    def __init__(self):
        super().__init__(lambda s, b: s.battle_menu_visible, "BattleMenuVisible?")


class MoveMenuVisible(Condition):
    """Check if move selection menu is visible"""
    def __init__(self):
        super().__init__(lambda s, b: s.move_menu_visible, "MoveMenuVisible?")


class PartyMenuVisible(Condition):
    """Check if party selection menu is visible"""
    def __init__(self):
        super().__init__(lambda s, b: s.party_menu_visible, "PartyMenuVisible?")


class HasLowHP(Condition):
    """Check if current Pokemon has low HP"""
    def __init__(self, threshold: float = 0.25):
        super().__init__(
            lambda s, b: s.current_hp_percent < threshold,
            f"HasLowHP(<{threshold*100:.0f}%)"
        )
        self.threshold = threshold


class OpponentLowHP(Condition):
    """Check if opponent Pokemon has low HP"""
    def __init__(self, threshold: float = 0.25):
        super().__init__(
            lambda s, b: s.opponent_hp_percent < threshold,
            f"OpponentLowHP(<{threshold*100:.0f}%)"
        )


class AllMovesExhausted(Condition):
    """Check if all moves are out of PP"""
    def __init__(self):
        super().__init__(
            lambda s, b: b.get('pp_empty', set()) == {1, 2, 3, 4},
            "AllMovesExhausted?"
        )


class MoveHasPP(Condition):
    """Check if current move has PP"""
    def __init__(self):
        def check(state, bb):
            current_move = bb.get('current_move', 1)
            pp_empty = bb.get('pp_empty', set())
            return current_move not in pp_empty
        super().__init__(check, "MoveHasPP?")


class IsWildBattle(Condition):
    """Check if this is a wild battle (can run)"""
    def __init__(self):
        super().__init__(
            lambda s, b: b.get('battle_type', 'wild') == 'wild',
            "IsWildBattle?"
        )


class NearPokemonCenter(Condition):
    """Check if near a Pokemon Center"""
    def __init__(self):
        def check(state, bb):
            for text in state.visible_text:
                if 'pokemon center' in text.lower() or 'pokecenter' in text.lower():
                    return True
            return False
        super().__init__(check, "NearPokemonCenter?")


class NearTallGrass(Condition):
    """Check if near tall grass (for wild encounters)"""
    def __init__(self):
        # This would use perception in a full implementation
        super().__init__(
            lambda s, b: b.get('near_grass', True),
            "NearTallGrass?"
        )


# =============================================================================
# POKEMON-SPECIFIC ACTIONS
# =============================================================================

class NavigateToBattleMenuOption(BehaviorNode):
    """Navigate to a specific battle menu option"""

    # Menu positions: FIGHT=top-left, POKEMON=bottom-left, BAG=top-right, RUN=bottom-right
    MENU_POSITIONS = {
        'fight': (0, 0),
        'pokemon': (0, 1),
        'bag': (1, 0),
        'run': (1, 1),
    }

    def __init__(self, option: str, name: str = ""):
        super().__init__(name or f"GoTo({option.upper()})")
        self.option = option.lower()

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        if self.option not in self.MENU_POSITIONS:
            return NodeStatus.FAILURE

        try:
            from controllers.gba_controller import press
            import time

            target_col, target_row = self.MENU_POSITIONS[self.option]

            # Reset to top-left first
            press("up", 0.05)
            time.sleep(0.05)
            press("left", 0.05)
            time.sleep(0.05)

            # Navigate to target
            if target_col == 1:
                press("right", 0.05)
                time.sleep(0.05)
            if target_row == 1:
                press("down", 0.05)
                time.sleep(0.05)

            blackboard['last_menu_option'] = self.option
            return NodeStatus.SUCCESS

        except Exception as e:
            blackboard['last_error'] = str(e)
            return NodeStatus.FAILURE


class NavigateToMoveSlot(BehaviorNode):
    """Navigate to a specific move slot (1-4)"""

    def __init__(self, slot: int = None, name: str = ""):
        super().__init__(name or f"GoToMove({slot or 'dynamic'})")
        self.slot = slot

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        slot = self.slot or blackboard.get('current_move', 1)

        try:
            from controllers.gba_controller import press
            import time

            # Move slots: 1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right
            # Reset to slot 1 first
            press("up", 0.05)
            time.sleep(0.05)
            press("up", 0.05)
            time.sleep(0.05)
            press("left", 0.05)
            time.sleep(0.05)

            # Navigate to target slot
            if slot == 2:
                press("right", 0.05)
            elif slot == 3:
                press("down", 0.05)
            elif slot == 4:
                press("down", 0.05)
                time.sleep(0.05)
                press("right", 0.05)

            time.sleep(0.05)
            return NodeStatus.SUCCESS

        except Exception as e:
            blackboard['last_error'] = str(e)
            return NodeStatus.FAILURE


class NavigateToPartySlot(BehaviorNode):
    """Navigate to a party member slot (1-6)"""

    def __init__(self, slot: int = None, name: str = ""):
        super().__init__(name or f"GoToParty({slot or 'dynamic'})")
        self.slot = slot

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        slot = self.slot or blackboard.get('target_party_slot', 2)

        try:
            from controllers.gba_controller import press
            import time

            # Reset to top
            for _ in range(3):
                press("up", 0.05)
                time.sleep(0.05)

            # Navigate down to slot
            for _ in range(slot - 1):
                press("down", 0.05)
                time.sleep(0.05)

            return NodeStatus.SUCCESS

        except Exception as e:
            blackboard['last_error'] = str(e)
            return NodeStatus.FAILURE


class CycleMoveSlot(BehaviorNode):
    """Cycle to next available move slot"""

    def __init__(self, name: str = ""):
        super().__init__(name or "CycleMove")

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        current = blackboard.get('current_move', 1)
        pp_empty = blackboard.get('pp_empty', set())

        # Try to find next available move
        for i in range(4):
            next_move = ((current - 1 + i) % 4) + 1
            if next_move not in pp_empty:
                blackboard['current_move'] = next_move
                return NodeStatus.SUCCESS

        # All moves exhausted
        return NodeStatus.FAILURE


class MarkMoveEmpty(BehaviorNode):
    """Mark current move as out of PP"""

    def __init__(self, name: str = ""):
        super().__init__(name or "MarkMoveEmpty")

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        current = blackboard.get('current_move', 1)
        pp_empty = blackboard.get('pp_empty', set())
        pp_empty.add(current)
        blackboard['pp_empty'] = pp_empty
        return NodeStatus.SUCCESS


class CyclePartySlot(BehaviorNode):
    """Cycle to next party slot for switching"""

    def __init__(self, name: str = ""):
        super().__init__(name or "CycleParty")

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        current = blackboard.get('current_party_slot', 1)
        tried = blackboard.get('party_tried', set())

        # Try slots 2-6 first (avoid current active), then 1
        order = list(range(2, 7)) + [1]

        for slot in order:
            if slot not in tried:
                blackboard['target_party_slot'] = slot
                tried.add(slot)
                blackboard['party_tried'] = tried
                return NodeStatus.SUCCESS

        # All party members tried
        return NodeStatus.FAILURE


class ResetBattleMemory(BehaviorNode):
    """Reset battle-specific memory for new battle"""

    def __init__(self, name: str = ""):
        super().__init__(name or "ResetBattleMemory")

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        blackboard['current_move'] = 1
        blackboard['pp_empty'] = set()
        blackboard['party_tried'] = set()
        blackboard['turn_count'] = 0
        blackboard['battle_type'] = 'wild'
        return NodeStatus.SUCCESS


class IncrementTurnCount(BehaviorNode):
    """Increment the turn counter"""

    def __init__(self, name: str = ""):
        super().__init__(name or "IncrementTurn")

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        blackboard['turn_count'] = blackboard.get('turn_count', 0) + 1
        return NodeStatus.SUCCESS


class AdvanceBattleDialogue(BehaviorNode):
    """Advance through battle dialogue/animations"""

    def __init__(self, taps: int = 3, name: str = ""):
        super().__init__(name or f"AdvanceDialogue(x{taps})")
        self.taps = taps

    def tick(self, state: GameState, blackboard: Dict[str, Any]) -> NodeStatus:
        try:
            from controllers.gba_controller import mash
            mash("A", count=self.taps, delay=0.12)
            return NodeStatus.SUCCESS
        except Exception as e:
            blackboard['last_error'] = str(e)
            return NodeStatus.FAILURE


# =============================================================================
# POKEMON BATTLE BEHAVIOR TREES
# =============================================================================

def create_pokemon_battle_tree() -> BehaviorTree:
    """
    Create comprehensive Pokemon battle behavior tree.

    Flow:
    1. Open Fight menu
    2. Try moves in order (1-4)
    3. If move fails (no PP), mark and try next
    4. If all moves fail, try switching
    5. If no switch possible, use Struggle or Run
    """

    # === SUB-TREES ===

    # Attempt a single move
    try_current_move = Sequence("TryCurrentMove", [
        NavigateToMoveSlot(),
        PressButton("A", 0.1),
        Wait(0.2),
        AdvanceBattleDialogue(taps=2),
        # TODO: Add frame similarity check for move success
        IncrementTurnCount(),
        CycleMoveSlot(),  # Prepare next move for next turn
    ])

    # Attempt to switch Pokemon
    try_switch = Sequence("TrySwitchPokemon", [
        NavigateToBattleMenuOption("pokemon"),
        PressButton("A", 0.1),
        Wait(0.15),
        CyclePartySlot(),
        NavigateToPartySlot(),
        PressButton("A", 0.1),
        Wait(0.15),
        PressButton("A", 0.1),  # Confirm switch
        Wait(2.0),  # Switch animation
        AdvanceBattleDialogue(taps=3),
        ResetBattleMemory(),  # New Pokemon, reset moves
    ])

    # Attempt to run (wild battles only)
    try_run = Sequence("TryRun", [
        IsWildBattle(),
        NavigateToBattleMenuOption("run"),
        PressButton("A", 0.1),
        Wait(0.3),
        AdvanceBattleDialogue(taps=3),
    ])

    # Use Struggle (when all PP exhausted)
    use_struggle = Sequence("UseStruggle", [
        NavigateToBattleMenuOption("fight"),
        PressButton("A", 0.1),
        Wait(0.1),
        PressButton("A", 0.1),  # Struggle is auto-selected
        Wait(0.3),
        AdvanceBattleDialogue(taps=4),
    ])

    # === MAIN BATTLE FLOW ===

    # Normal attack sequence
    attack_sequence = Sequence("AttackSequence", [
        NavigateToBattleMenuOption("fight"),
        PressButton("A", 0.1),
        Wait(0.12),
        Selector("TryMoveOrSwitch", [
            Sequence("TryMoveIfHasPP", [
                MoveHasPP(),
                try_current_move,
            ]),
            Sequence("HandleNoPP", [
                MarkMoveEmpty(),
                CycleMoveSlot(),
                # Try another move or fallback
                Selector("RetryOrFallback", [
                    Sequence("RetryWithNewMove", [
                        Inverter(AllMovesExhausted()),
                        try_current_move,
                    ]),
                    try_switch,
                    use_struggle,
                    try_run,
                ]),
            ]),
        ]),
    ])

    # Complete battle turn
    battle_turn = Selector("BattleTurn", [
        attack_sequence,
        try_switch,
        use_struggle,
        try_run,
    ])

    return BehaviorTree(battle_turn, "PokemonBattle")


def create_pokemon_exploration_tree() -> BehaviorTree:
    """
    Create Pokemon exploration behavior tree.

    Handles:
    - Random movement with anti-stuck
    - Grass walking for encounters
    - NPC interaction
    - Door/cave entry
    """

    # Unstuck behavior - rotate direction when stuck
    unstuck = Sequence("Unstuck", [
        Condition(lambda s, b: s.stuck_count >= 10, "IsStuck?"),
        RotateDirection(),
        MoveDirection(duration=0.3),
        SetBlackboard("stuck_count", 0),
    ])

    # Normal exploration movement
    explore_move = Sequence("ExploreMove", [
        MoveDirection(duration=0.2),
        Wait(0.05),
    ])

    # Try to interact with things
    try_interact = Sequence("TryInteract", [
        PressButton("A", 0.1),
        Wait(0.1),
    ])

    # Occasional interaction attempt
    explore_with_interact = Selector("ExploreWithInteract", [
        Sequence("InteractCheck", [
            Condition(lambda s, b: b.get('move_count', 0) % 5 == 0, "ShouldInteract?"),
            try_interact,
        ]),
        explore_move,
    ])

    # Track move count
    count_move = Action(
        lambda s, b: (b.update({'move_count': b.get('move_count', 0) + 1}), NodeStatus.SUCCESS)[1],
        "CountMove"
    )

    # Main exploration
    main_exploration = Sequence("MainExploration", [
        count_move,
        Selector("MoveOrUnstuck", [
            unstuck,
            explore_with_interact,
        ]),
    ])

    return BehaviorTree(main_exploration, "PokemonExploration")


def create_pokemon_dialogue_tree() -> BehaviorTree:
    """
    Create Pokemon dialogue handling behavior tree.

    Handles:
    - Advancing text
    - Yes/No choices (defaults to Yes)
    - Menu selections
    """

    # Advance dialogue
    advance = Sequence("AdvanceDialogue", [
        PressButton("A", 0.15),
        Wait(0.1),
    ])

    # Handle Yes/No prompts (choose Yes by default)
    handle_choice = Sequence("HandleYesNo", [
        Condition(
            lambda s, b: any('yes' in t.lower() for t in s.visible_text),
            "HasYesNoChoice?"
        ),
        PressButton("up", 0.05),  # Ensure on Yes
        Wait(0.05),
        PressButton("A", 0.15),
    ])

    # Main dialogue handler
    dialogue_handler = Selector("DialogueHandler", [
        handle_choice,
        advance,
    ])

    return BehaviorTree(dialogue_handler, "PokemonDialogue")


def create_pokemon_healing_tree() -> BehaviorTree:
    """
    Create Pokemon Center healing behavior tree.

    Handles navigating to nurse and healing party.
    """

    # Walk to counter
    approach_counter = Sequence("ApproachCounter", [
        MoveDirection("up", 0.3),
        Wait(0.1),
        MoveDirection("up", 0.3),
        Wait(0.1),
    ])

    # Talk to nurse
    talk_to_nurse = Sequence("TalkToNurse", [
        PressButton("A", 0.15),
        Wait(0.5),
    ])

    # Confirm healing
    confirm_healing = Sequence("ConfirmHealing", [
        PressButton("A", 0.15),
        Wait(0.3),
        PressButton("A", 0.15),
        Wait(2.0),  # Healing animation
        PressButton("A", 0.15),
        Wait(0.3),
    ])

    # Leave center
    leave_center = Sequence("LeaveCenter", [
        MoveDirection("down", 0.5),
        Wait(0.2),
    ])

    # Full healing sequence
    heal_sequence = Sequence("HealSequence", [
        approach_counter,
        talk_to_nurse,
        confirm_healing,
        leave_center,
    ])

    return BehaviorTree(heal_sequence, "PokemonHealing")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_pokemon_behavior_trees() -> Dict[str, BehaviorTree]:
    """Create all Pokemon-specific behavior trees"""
    return {
        'battle': create_pokemon_battle_tree(),
        'exploration': create_pokemon_exploration_tree(),
        'dialogue': create_pokemon_dialogue_tree(),
        'healing': create_pokemon_healing_tree(),
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Pokemon Behavior Trees Test")
    print("=" * 50)

    # Create all trees
    trees = create_pokemon_behavior_trees()

    for name, tree in trees.items():
        print(f"\n{name.upper()} Tree:")
        print(f"  Name: {tree.name}")
        print(f"  Root: {tree.root.name}")

    # Test battle tree with mock state
    print("\n" + "=" * 50)
    print("Testing Battle Tree:")

    battle_tree = trees['battle']
    state = GameState(
        current_context="battle",
        in_battle=True,
        battle_menu_visible=True,
    )

    # Initialize blackboard
    battle_tree.blackboard['current_move'] = 1
    battle_tree.blackboard['pp_empty'] = set()
    battle_tree.blackboard['battle_type'] = 'wild'

    print(f"  Initial move: {battle_tree.blackboard.get('current_move')}")
    print(f"  PP empty: {battle_tree.blackboard.get('pp_empty')}")

    # Simulate some turns (without actually pressing keys)
    print("\n  Simulating structure traversal...")
    print(f"  Tree tick count: {battle_tree.tick_count}")

    print("\nPokemon Behavior Trees Test Complete!")
