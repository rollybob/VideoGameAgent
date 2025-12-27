"""
Strategy Engine - LLM-ready strategic reasoning for game agents
================================================================

This module provides strategic reasoning capabilities that can:
1. Use heuristic rules (works on any hardware)
2. Use local LLM when GPU is available
3. Use API-based LLM (Claude, GPT-4, etc.) for best quality

The key insight: strategic decisions don't need to be real-time.
We can take 1-5 seconds to think about "what should I do next?" while
the fast layer keeps the agent moving.

Usage:
    strategy = StrategyEngine()

    # Get strategic advice (async-safe)
    advice = await strategy.analyze_situation(game_context)

    # Or synchronous
    advice = strategy.analyze_situation_sync(game_context)
"""

import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from debug_system import info, debug, warning, error
from config import get_config


class StrategyBackend(Enum):
    """Available strategy backends"""
    HEURISTIC = "heuristic"      # Rule-based, no AI
    LOCAL_LLM = "local_llm"      # Local model (Mistral, Llama, etc.)
    API_LLM = "api_llm"          # API-based (Claude, GPT-4)


@dataclass
class GameContext:
    """Context information for strategic analysis"""
    current_state: str                    # battle, overworld, menu, etc.
    location_info: str = ""               # Current area/location
    party_status: Dict[str, Any] = field(default_factory=dict)  # Pokemon team status
    inventory: Dict[str, int] = field(default_factory=dict)     # Items
    progress: Dict[str, Any] = field(default_factory=dict)      # Badges, story progress
    recent_events: List[str] = field(default_factory=list)      # What happened recently
    visible_text: List[str] = field(default_factory=list)       # OCR results
    stuck_count: int = 0
    time_in_state: float = 0.0


@dataclass
class StrategicAdvice:
    """Strategic advice from the engine"""
    primary_goal: str
    reasoning: str
    suggested_actions: List[str]
    priority: int = 1  # 1=highest
    confidence: float = 1.0
    expires_in_seconds: float = 60.0
    context_requirements: List[str] = field(default_factory=list)


class StrategyProvider(ABC):
    """Abstract base for strategy providers"""

    @abstractmethod
    def analyze(self, context: GameContext) -> StrategicAdvice:
        """Analyze context and provide strategic advice"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available"""
        pass


class HeuristicStrategy(StrategyProvider):
    """
    Rule-based strategy using game knowledge.
    Works on any hardware, provides reasonable advice.
    """

    def __init__(self):
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        """Load strategy rules"""
        return {
            'battle_priorities': [
                ('low_hp', 0.2, 'heal_or_switch', "Low HP - consider healing or switching"),
                ('type_disadvantage', 0.0, 'switch_pokemon', "Type disadvantage - switch if possible"),
                ('normal', 1.0, 'attack', "Use best available attack"),
            ],
            'exploration_priorities': [
                ('stuck', 15, 'try_escape', "Stuck for too long - try alternative routes"),
                ('near_pokecenter', 0, 'heal_first', "Near Pokemon Center - heal before proceeding"),
                ('unexplored_area', 0, 'explore', "Explore unexplored areas"),
                ('default', 0, 'systematic_explore', "Systematically explore current area"),
            ],
            'dialogue_priorities': [
                ('quest_giver', 0, 'accept_quest', "NPC offering quest - accept it"),
                ('story_npc', 0, 'listen_carefully', "Story NPC - pay attention"),
                ('default', 0, 'advance', "Advance dialogue"),
            ],
        }

    def is_available(self) -> bool:
        return True  # Always available

    def analyze(self, context: GameContext) -> StrategicAdvice:
        """Analyze context using heuristic rules"""

        if context.current_state == 'battle':
            return self._analyze_battle(context)
        elif context.current_state == 'dialogue':
            return self._analyze_dialogue(context)
        else:
            return self._analyze_exploration(context)

    def _analyze_battle(self, context: GameContext) -> StrategicAdvice:
        """Analyze battle situation"""
        hp_percent = context.party_status.get('current_hp_percent', 1.0)

        if hp_percent < 0.2:
            return StrategicAdvice(
                primary_goal="survive",
                reasoning="HP is critically low. Need to heal or switch to preserve Pokemon.",
                suggested_actions=["use_potion", "switch_pokemon", "run_if_wild"],
                priority=1,
                confidence=0.9,
                expires_in_seconds=30,
            )
        elif hp_percent < 0.5:
            return StrategicAdvice(
                primary_goal="cautious_attack",
                reasoning="HP is moderate. Attack but be ready to heal.",
                suggested_actions=["attack", "monitor_hp", "prepare_heal"],
                priority=2,
                confidence=0.8,
                expires_in_seconds=30,
            )
        else:
            return StrategicAdvice(
                primary_goal="aggressive_attack",
                reasoning="HP is healthy. Attack aggressively to end battle quickly.",
                suggested_actions=["attack_strongest", "use_type_advantage"],
                priority=2,
                confidence=0.9,
                expires_in_seconds=30,
            )

    def _analyze_exploration(self, context: GameContext) -> StrategicAdvice:
        """Analyze exploration situation"""

        if context.stuck_count > 15:
            return StrategicAdvice(
                primary_goal="escape_stuck",
                reasoning="Agent has been stuck for too long. Need to try different approach.",
                suggested_actions=["rotate_direction", "backtrack", "try_menu", "look_for_door"],
                priority=1,
                confidence=0.8,
                expires_in_seconds=20,
            )

        # Check for important locations in visible text
        text_lower = " ".join(context.visible_text).lower()

        if "pokemon center" in text_lower or "pokecenter" in text_lower:
            hp_percent = context.party_status.get('current_hp_percent', 1.0)
            if hp_percent < 0.7:
                return StrategicAdvice(
                    primary_goal="heal_team",
                    reasoning="Pokemon Center detected and team could use healing.",
                    suggested_actions=["enter_pokecenter", "heal_all", "save_game"],
                    priority=1,
                    confidence=0.9,
                    expires_in_seconds=60,
                )

        if "gym" in text_lower:
            return StrategicAdvice(
                primary_goal="prepare_gym",
                reasoning="Gym detected. Ensure team is ready before challenging.",
                suggested_actions=["check_team_health", "ensure_items", "challenge_gym"],
                priority=2,
                confidence=0.7,
                expires_in_seconds=120,
            )

        # Default exploration
        return StrategicAdvice(
            primary_goal="explore",
            reasoning="Continue exploring the current area systematically.",
            suggested_actions=["move_unexplored", "talk_to_npcs", "check_items"],
            priority=3,
            confidence=0.6,
            expires_in_seconds=60,
        )

    def _analyze_dialogue(self, context: GameContext) -> StrategicAdvice:
        """Analyze dialogue situation"""
        text_lower = " ".join(context.visible_text).lower()

        # Check for important dialogue patterns
        if "yes" in text_lower and "no" in text_lower:
            return StrategicAdvice(
                primary_goal="make_choice",
                reasoning="Dialogue presents a choice. Analyze context to decide.",
                suggested_actions=["analyze_question", "choose_wisely"],
                priority=1,
                confidence=0.7,
                expires_in_seconds=30,
            )

        if any(word in text_lower for word in ["quest", "mission", "help", "need"]):
            return StrategicAdvice(
                primary_goal="accept_quest",
                reasoning="NPC appears to be offering a quest or request.",
                suggested_actions=["accept", "note_objective"],
                priority=2,
                confidence=0.7,
                expires_in_seconds=30,
            )

        return StrategicAdvice(
            primary_goal="advance_dialogue",
            reasoning="Continue through dialogue to gather information.",
            suggested_actions=["press_a", "read_text"],
            priority=3,
            confidence=0.9,
            expires_in_seconds=10,
        )


class LocalLLMStrategy(StrategyProvider):
    """
    Local LLM strategy using transformers.
    Requires GPU for reasonable performance.
    """

    def __init__(self, model_name: str = None):
        self.config = get_config()
        self.model_name = model_name or self.config.reasoning.strategy_model
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def is_available(self) -> bool:
        """Check if local LLM is available"""
        if self._initialized:
            return self.model is not None

        try:
            import torch
            if not torch.cuda.is_available():
                debug("strategy", "CUDA not available for local LLM")
                return False

            # Check VRAM (need at least 4GB for small models)
            total_vram = torch.cuda.get_device_properties(0).total_memory
            if total_vram < 4 * 1024 * 1024 * 1024:  # 4GB
                debug("strategy", f"Insufficient VRAM: {total_vram / 1e9:.1f}GB")
                return False

            return True

        except ImportError:
            return False

    def _initialize(self):
        """Lazy initialization of LLM"""
        if self._initialized:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            debug("strategy", f"Loading local LLM: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

            info("strategy", f"Local LLM loaded: {self.model_name}")

        except Exception as e:
            error("strategy", f"Failed to load local LLM: {e}")
            self.model = None

        self._initialized = True

    def analyze(self, context: GameContext) -> StrategicAdvice:
        """Analyze using local LLM"""
        if not self.is_available():
            raise RuntimeError("Local LLM not available")

        self._initialize()

        if self.model is None:
            raise RuntimeError("Local LLM failed to initialize")

        prompt = self._build_prompt(context)
        response = self._generate_response(prompt)
        return self._parse_response(response, context)

    def _build_prompt(self, context: GameContext) -> str:
        """Build prompt for LLM"""
        return f"""You are a Pokemon game strategy advisor. Analyze the current situation and provide strategic advice.

Current State: {context.current_state}
Location: {context.location_info or "Unknown"}
Party Status: {json.dumps(context.party_status)}
Recent Events: {", ".join(context.recent_events[-5:]) if context.recent_events else "None"}
Visible Text: {", ".join(context.visible_text[:3]) if context.visible_text else "None"}
Stuck Count: {context.stuck_count}

Provide strategic advice in this format:
GOAL: [primary objective]
REASONING: [why this goal]
ACTIONS: [comma-separated list of suggested actions]
PRIORITY: [1-5, 1 being highest]
CONFIDENCE: [0.0-1.0]

Your advice:"""

    def _generate_response(self, prompt: str) -> str:
        """Generate response from LLM"""
        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.config.reasoning.strategy_temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

    def _parse_response(self, response: str, context: GameContext) -> StrategicAdvice:
        """Parse LLM response into StrategicAdvice"""
        lines = response.split('\n')

        goal = "continue"
        reasoning = response
        actions = ["observe"]
        priority = 3
        confidence = 0.5

        for line in lines:
            line = line.strip()
            if line.startswith("GOAL:"):
                goal = line[5:].strip()
            elif line.startswith("REASONING:"):
                reasoning = line[10:].strip()
            elif line.startswith("ACTIONS:"):
                actions = [a.strip() for a in line[8:].split(",")]
            elif line.startswith("PRIORITY:"):
                try:
                    priority = int(line[9:].strip())
                except ValueError:
                    pass
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line[11:].strip())
                except ValueError:
                    pass

        return StrategicAdvice(
            primary_goal=goal,
            reasoning=reasoning,
            suggested_actions=actions,
            priority=priority,
            confidence=confidence,
            expires_in_seconds=60,
        )


class StrategyEngine:
    """
    Main strategy engine that manages providers.
    Automatically selects best available backend.
    """

    def __init__(self, preferred_backend: StrategyBackend = None):
        self.config = get_config()

        # Initialize providers
        self.providers: Dict[StrategyBackend, StrategyProvider] = {
            StrategyBackend.HEURISTIC: HeuristicStrategy(),
        }

        # Try to add local LLM provider
        local_llm = LocalLLMStrategy()
        if local_llm.is_available():
            self.providers[StrategyBackend.LOCAL_LLM] = local_llm

        # Select backend
        if preferred_backend and preferred_backend in self.providers:
            self.active_backend = preferred_backend
        elif StrategyBackend.LOCAL_LLM in self.providers:
            self.active_backend = StrategyBackend.LOCAL_LLM
        else:
            self.active_backend = StrategyBackend.HEURISTIC

        info("strategy", f"Strategy engine using: {self.active_backend.value}")

        # Cache for advice
        self._cached_advice: Optional[StrategicAdvice] = None
        self._cache_context_hash: Optional[str] = None
        self._cache_time: float = 0.0

    def get_available_backends(self) -> List[StrategyBackend]:
        """Get list of available backends"""
        return list(self.providers.keys())

    def set_backend(self, backend: StrategyBackend) -> bool:
        """Set active backend"""
        if backend in self.providers:
            self.active_backend = backend
            info("strategy", f"Switched to backend: {backend.value}")
            return True
        return False

    def _context_hash(self, context: GameContext) -> str:
        """Create hash of context for caching"""
        key_parts = [
            context.current_state,
            str(context.stuck_count // 5),  # Group by 5s
            str(len(context.visible_text)),
        ]
        return "|".join(key_parts)

    def analyze_situation(self, context: GameContext) -> StrategicAdvice:
        """
        Analyze current game situation and provide strategic advice.
        Uses caching to avoid redundant analysis.
        """
        # Check cache
        context_hash = self._context_hash(context)
        cache_age = time.time() - self._cache_time

        if (self._cached_advice and
            self._cache_context_hash == context_hash and
            cache_age < self._cached_advice.expires_in_seconds):
            debug("strategy", "Returning cached advice")
            return self._cached_advice

        # Get fresh advice
        try:
            provider = self.providers[self.active_backend]
            advice = provider.analyze(context)

            # Cache it
            self._cached_advice = advice
            self._cache_context_hash = context_hash
            self._cache_time = time.time()

            debug("strategy", f"Generated advice: {advice.primary_goal}")
            return advice

        except Exception as e:
            error("strategy", f"Strategy analysis failed: {e}")

            # Fall back to heuristic
            if self.active_backend != StrategyBackend.HEURISTIC:
                warning("strategy", "Falling back to heuristic")
                return self.providers[StrategyBackend.HEURISTIC].analyze(context)

            # Return safe default
            return StrategicAdvice(
                primary_goal="observe",
                reasoning="Strategy analysis failed - observing",
                suggested_actions=["wait", "observe"],
                priority=5,
                confidence=0.3,
            )

    def analyze_situation_sync(self, context: GameContext) -> StrategicAdvice:
        """Synchronous wrapper for analyze_situation"""
        return self.analyze_situation(context)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Strategy Engine Test")
    print("=" * 50)

    # Create engine
    engine = StrategyEngine()
    print(f"Active backend: {engine.active_backend.value}")
    print(f"Available backends: {[b.value for b in engine.get_available_backends()]}")

    # Test contexts
    contexts = [
        GameContext(
            current_state="battle",
            party_status={'current_hp_percent': 0.15},
        ),
        GameContext(
            current_state="battle",
            party_status={'current_hp_percent': 0.8},
        ),
        GameContext(
            current_state="overworld",
            stuck_count=20,
        ),
        GameContext(
            current_state="overworld",
            visible_text=["Pokemon Center", "Nurse Joy"],
            party_status={'current_hp_percent': 0.4},
        ),
        GameContext(
            current_state="dialogue",
            visible_text=["Would you like to battle?", "Yes", "No"],
        ),
    ]

    print("\nAnalyzing situations:")
    for i, context in enumerate(contexts):
        print(f"\n--- Situation {i+1}: {context.current_state} ---")
        advice = engine.analyze_situation(context)
        print(f"Goal: {advice.primary_goal}")
        print(f"Reasoning: {advice.reasoning}")
        print(f"Actions: {advice.suggested_actions}")
        print(f"Priority: {advice.priority}, Confidence: {advice.confidence}")

    print("\nStrategy Engine Test Complete!")
