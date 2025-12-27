"""
Knowledge Acquisition System - Learning game mechanics from multiple sources
=============================================================================

The agent needs to learn how to play games it has never seen before.
This module provides multiple strategies for acquiring game knowledge:

1. TUTORIAL LEARNING - Parse and understand in-game tutorials
2. USER ASSISTANCE - Ask the human operator for help
3. WEB SEARCH - Find walkthroughs, guides, wikis online
4. EXPERIMENTATION - Trial and error with outcome tracking
5. TRANSFER LEARNING - Apply knowledge from similar games

Priority order when stuck:
1. Check if we've seen this situation before (memory)
2. Try experimentation (safe actions first)
3. Ask user if available
4. Search web for guidance
5. Fall back to random exploration
"""

import time
import json
import threading
from queue import Queue, Empty
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod

from debug_system import info, debug, warning, error


class KnowledgeSource(Enum):
    """Sources of game knowledge"""
    TUTORIAL = "tutorial"           # In-game tutorial text
    USER = "user"                   # Human operator
    WEB = "web"                     # Online guides/wikis
    EXPERIMENTATION = "experiment"  # Trial and error
    TRANSFER = "transfer"           # From similar games
    MEMORY = "memory"               # Previously learned


@dataclass
class GameKnowledge:
    """A piece of learned game knowledge"""
    topic: str                      # What this knowledge is about
    content: str                    # The actual knowledge
    source: KnowledgeSource         # Where it came from
    confidence: float = 1.0         # How confident we are (0-1)
    game_id: str = ""               # Which game this applies to
    genre: str = ""                 # Genre for transfer learning
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    used_count: int = 0             # How often this has been used
    success_rate: float = 0.0       # How often it led to success


@dataclass
class HelpRequest:
    """A request for help from the user"""
    question: str
    context: Dict[str, Any]
    screenshot_path: Optional[str] = None
    options: List[str] = field(default_factory=list)  # Multiple choice if applicable
    timeout_seconds: float = 300.0   # 5 minute default timeout
    created_at: float = field(default_factory=time.time)
    answered: bool = False
    answer: Optional[str] = None


@dataclass
class ExperimentResult:
    """Result of a trial-and-error experiment"""
    action: str
    context_before: str
    context_after: str
    outcome: str  # 'positive', 'negative', 'neutral', 'unknown'
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class KnowledgeProvider(ABC):
    """Abstract base for knowledge providers"""

    @abstractmethod
    def can_help(self, topic: str, context: Dict[str, Any]) -> bool:
        """Check if this provider can help with a topic"""
        pass

    @abstractmethod
    def get_knowledge(self, topic: str, context: Dict[str, Any]) -> Optional[GameKnowledge]:
        """Get knowledge about a topic"""
        pass


class TutorialLearner(KnowledgeProvider):
    """
    Learn from in-game tutorials by parsing detected text.

    Looks for patterns like:
    - "Press A to jump"
    - "Use the sword with B"
    - "Talk to NPCs by pressing A"
    - Tutorial screens with instructions
    """

    # Common tutorial patterns across many games
    INSTRUCTION_PATTERNS = [
        # Direct button instructions
        (r"press\s+(\w+)\s+to\s+(.+)", "button_action"),
        (r"use\s+(\w+)\s+to\s+(.+)", "button_action"),
        (r"(\w+)\s+button\s*[-:]\s*(.+)", "button_action"),

        # Movement instructions
        (r"move\s+(up|down|left|right|around)\s+(.+)", "movement"),
        (r"(d-pad|control pad|stick)\s*[-:]\s*(.+)", "movement"),

        # Game mechanics
        (r"(health|hp|life)\s*[-:]\s*(.+)", "mechanic"),
        (r"(attack|defend|block|dodge)\s*[-:]\s*(.+)", "mechanic"),
        (r"(save|load|pause)\s*[-:]\s*(.+)", "mechanic"),

        # Goals and objectives
        (r"(objective|goal|mission)\s*[-:]\s*(.+)", "objective"),
        (r"(find|collect|defeat|reach)\s+(.+)", "objective"),
    ]

    def __init__(self):
        import re
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), pattern_type)
            for pattern, pattern_type in self.INSTRUCTION_PATTERNS
        ]
        self.learned_instructions: List[GameKnowledge] = []

    def can_help(self, topic: str, context: Dict[str, Any]) -> bool:
        # Can help if we have relevant learned instructions
        topic_lower = topic.lower()
        for knowledge in self.learned_instructions:
            if topic_lower in knowledge.topic.lower() or topic_lower in knowledge.content.lower():
                return True
        return False

    def get_knowledge(self, topic: str, context: Dict[str, Any]) -> Optional[GameKnowledge]:
        topic_lower = topic.lower()
        best_match = None
        best_score = 0

        for knowledge in self.learned_instructions:
            score = 0
            if topic_lower in knowledge.topic.lower():
                score += 2
            if topic_lower in knowledge.content.lower():
                score += 1
            if any(tag.lower() in topic_lower for tag in knowledge.tags):
                score += 1

            if score > best_score:
                best_score = score
                best_match = knowledge

        return best_match

    def parse_tutorial_text(self, texts: List[str], game_id: str = "") -> List[GameKnowledge]:
        """Parse tutorial text and extract knowledge"""
        new_knowledge = []

        for text in texts:
            for pattern, pattern_type in self.compiled_patterns:
                match = pattern.search(text)
                if match:
                    knowledge = GameKnowledge(
                        topic=pattern_type,
                        content=text,
                        source=KnowledgeSource.TUTORIAL,
                        confidence=0.9,
                        game_id=game_id,
                        tags=[pattern_type, match.group(1) if match.groups() else ""]
                    )
                    new_knowledge.append(knowledge)
                    self.learned_instructions.append(knowledge)
                    info("knowledge", f"Learned from tutorial: {text[:50]}...")

        return new_knowledge


class UserAssistant(KnowledgeProvider):
    """
    Ask the human operator for help when stuck.

    Features:
    - Queue-based async questions
    - Timeout handling
    - Screenshot attachment
    - Multiple choice options
    - Learning from answers for future
    """

    def __init__(self):
        self.question_queue: Queue = Queue()
        self.answer_queue: Queue = Queue()
        self.pending_requests: Dict[str, HelpRequest] = {}
        self.answered_knowledge: List[GameKnowledge] = []

        # Callbacks for UI integration
        self.on_question_asked: Optional[Callable[[HelpRequest], None]] = None
        self.user_available = True  # Can be toggled by UI

    def can_help(self, topic: str, context: Dict[str, Any]) -> bool:
        return self.user_available

    def get_knowledge(self, topic: str, context: Dict[str, Any]) -> Optional[GameKnowledge]:
        # Check if we've asked this before
        for knowledge in self.answered_knowledge:
            if topic.lower() in knowledge.topic.lower():
                return knowledge

        # Need to ask user - this is async
        return None

    def ask_user(self, question: str, context: Dict[str, Any],
                 screenshot_path: Optional[str] = None,
                 options: List[str] = None) -> HelpRequest:
        """Ask user a question (async)"""
        request = HelpRequest(
            question=question,
            context=context,
            screenshot_path=screenshot_path,
            options=options or []
        )

        request_id = f"q_{int(time.time()*1000)}"
        self.pending_requests[request_id] = request

        info("knowledge", f"Asking user: {question}")

        # Notify UI if callback set
        if self.on_question_asked:
            self.on_question_asked(request)

        return request

    def provide_answer(self, request_id: str, answer: str):
        """Called by UI when user provides an answer"""
        if request_id in self.pending_requests:
            request = self.pending_requests[request_id]
            request.answered = True
            request.answer = answer

            # Convert to knowledge
            knowledge = GameKnowledge(
                topic=request.question,
                content=answer,
                source=KnowledgeSource.USER,
                confidence=1.0,
                tags=["user_provided"]
            )
            self.answered_knowledge.append(knowledge)

            info("knowledge", f"User answered: {answer[:50]}...")

            del self.pending_requests[request_id]
            return knowledge

        return None

    def check_pending_timeouts(self):
        """Check for timed out requests"""
        current_time = time.time()
        timed_out = []

        for req_id, request in self.pending_requests.items():
            if current_time - request.created_at > request.timeout_seconds:
                timed_out.append(req_id)

        for req_id in timed_out:
            warning("knowledge", f"User help request timed out: {req_id}")
            del self.pending_requests[req_id]


class WebSearcher(KnowledgeProvider):
    """
    Search the web for game guides and walkthroughs.

    Strategies:
    - Search for "[game name] walkthrough"
    - Search for "[game name] how to [topic]"
    - Parse wiki pages for mechanics
    - Find video guides (extract key moments)
    """

    def __init__(self):
        self.cache: Dict[str, GameKnowledge] = {}
        self.search_enabled = True  # Can be disabled

    def can_help(self, topic: str, context: Dict[str, Any]) -> bool:
        return self.search_enabled

    def get_knowledge(self, topic: str, context: Dict[str, Any]) -> Optional[GameKnowledge]:
        # Check cache first
        cache_key = f"{context.get('game_name', 'unknown')}:{topic}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Would perform actual web search here
        # For now, return None (to be implemented with actual web scraping)
        return None

    def search_walkthrough(self, game_name: str, topic: str) -> Optional[GameKnowledge]:
        """Search for walkthrough information"""
        query = f"{game_name} {topic} guide walkthrough"

        # This would use actual web search
        # Options:
        # 1. Use a search API (Google, Bing, DuckDuckGo)
        # 2. Scrape specific wiki sites (GameFAQs, IGN, game-specific wikis)
        # 3. Use an LLM API that has web access

        info("knowledge", f"Would search web for: {query}")

        # Placeholder - return None until implemented
        return None

    def parse_wiki_page(self, url: str) -> List[GameKnowledge]:
        """Parse a wiki page for game knowledge"""
        # Would use BeautifulSoup or similar to extract structured info
        return []


class Experimenter(KnowledgeProvider):
    """
    Learn through trial and error.

    Strategy:
    1. Try safe actions first (non-destructive)
    2. Observe outcomes
    3. Remember what worked and what didn't
    4. Build up action->outcome mappings
    """

    # Actions generally safe to try
    SAFE_ACTIONS = [
        ("A", "interact/confirm"),
        ("B", "cancel/back"),
        ("start", "menu/pause"),
        ("select", "secondary menu"),
        ("up", "move/navigate"),
        ("down", "move/navigate"),
        ("left", "move/navigate"),
        ("right", "move/navigate"),
    ]

    def __init__(self):
        self.experiments: List[ExperimentResult] = []
        self.action_outcomes: Dict[str, Dict[str, int]] = {}  # action -> {outcome -> count}

    def can_help(self, topic: str, context: Dict[str, Any]) -> bool:
        # Can always try experimenting
        return True

    def get_knowledge(self, topic: str, context: Dict[str, Any]) -> Optional[GameKnowledge]:
        # Look for relevant experiment results
        topic_lower = topic.lower()

        relevant = [
            exp for exp in self.experiments
            if topic_lower in exp.action.lower() or
               topic_lower in exp.context_before.lower()
        ]

        if relevant:
            # Find most successful action
            positive_experiments = [e for e in relevant if e.outcome == 'positive']
            if positive_experiments:
                best = positive_experiments[-1]  # Most recent success
                return GameKnowledge(
                    topic=topic,
                    content=f"Try action '{best.action}' - worked before in similar context",
                    source=KnowledgeSource.EXPERIMENTATION,
                    confidence=0.6,
                    tags=["experiment", best.action]
                )

        return None

    def suggest_experiment(self, context: str) -> str:
        """Suggest an action to try"""
        # Find least-tried action for this context
        context_experiments = [
            e for e in self.experiments
            if context.lower() in e.context_before.lower()
        ]

        action_counts = {}
        for action, _ in self.SAFE_ACTIONS:
            action_counts[action] = sum(
                1 for e in context_experiments if e.action == action
            )

        # Return least-tried action
        least_tried = min(action_counts.items(), key=lambda x: x[1])
        return least_tried[0]

    def record_experiment(self, action: str, context_before: str,
                         context_after: str, outcome: str,
                         details: Dict[str, Any] = None):
        """Record the result of an experiment"""
        result = ExperimentResult(
            action=action,
            context_before=context_before,
            context_after=context_after,
            outcome=outcome,
            details=details or {}
        )
        self.experiments.append(result)

        # Update action->outcome mapping
        if action not in self.action_outcomes:
            self.action_outcomes[action] = {}
        if outcome not in self.action_outcomes[action]:
            self.action_outcomes[action][outcome] = 0
        self.action_outcomes[action][outcome] += 1

        debug("knowledge", f"Recorded experiment: {action} -> {outcome}")


class TransferLearner(KnowledgeProvider):
    """
    Apply knowledge from similar games.

    Strategy:
    1. Identify game genre
    2. Load knowledge from games in same genre
    3. Apply with lower confidence (needs verification)
    """

    # Common knowledge by genre
    GENRE_KNOWLEDGE = {
        'rpg': [
            ("battle", "Usually turn-based. Select actions from menu."),
            ("healing", "Use items or visit designated healing locations."),
            ("leveling", "Defeat enemies to gain experience and level up."),
            ("saving", "Save at designated points or through menu."),
        ],
        'platformer': [
            ("jumping", "Usually A or B button. Hold longer to jump higher."),
            ("enemies", "Jump on them or avoid them."),
            ("collectibles", "Gather coins/items for points or power-ups."),
            ("goal", "Reach the end of each level, usually on the right."),
        ],
        'puzzle': [
            ("mechanics", "Each puzzle type has specific rules to discover."),
            ("progression", "Solve puzzles to advance."),
            ("hints", "Look for visual cues about what to do."),
        ],
        'action': [
            ("combat", "Attack button for main attack, dodge/block for defense."),
            ("health", "Monitor health bar, heal when low."),
            ("progression", "Defeat enemies and bosses to advance."),
        ],
    }

    def __init__(self):
        self.detected_genre: Optional[str] = None
        self.game_specific_knowledge: Dict[str, List[GameKnowledge]] = {}

    def can_help(self, topic: str, context: Dict[str, Any]) -> bool:
        return self.detected_genre is not None

    def get_knowledge(self, topic: str, context: Dict[str, Any]) -> Optional[GameKnowledge]:
        if not self.detected_genre:
            return None

        genre_info = self.GENRE_KNOWLEDGE.get(self.detected_genre, [])
        topic_lower = topic.lower()

        for info_topic, info_content in genre_info:
            if info_topic in topic_lower or topic_lower in info_topic:
                return GameKnowledge(
                    topic=info_topic,
                    content=info_content,
                    source=KnowledgeSource.TRANSFER,
                    confidence=0.5,  # Lower confidence - needs verification
                    genre=self.detected_genre,
                    tags=["transfer", self.detected_genre]
                )

        return None

    def detect_genre(self, visual_cues: List[str], text_cues: List[str]) -> str:
        """Detect game genre from visual and text cues"""
        scores = {genre: 0 for genre in self.GENRE_KNOWLEDGE.keys()}

        # RPG indicators
        rpg_keywords = ['hp', 'mp', 'exp', 'level', 'attack', 'defense', 'magic',
                       'party', 'equipment', 'inventory']
        for keyword in rpg_keywords:
            for text in text_cues:
                if keyword in text.lower():
                    scores['rpg'] += 1

        # Platformer indicators
        platform_keywords = ['lives', 'coins', 'score', 'time', 'stage', 'world']
        for keyword in platform_keywords:
            for text in text_cues:
                if keyword in text.lower():
                    scores['platformer'] += 1

        # Visual cues for platformer (side-scrolling, platforms visible)
        if any('platform' in cue.lower() for cue in visual_cues):
            scores['platformer'] += 2

        # Pick highest scoring genre
        best_genre = max(scores.items(), key=lambda x: x[1])
        if best_genre[1] > 0:
            self.detected_genre = best_genre[0]
            info("knowledge", f"Detected genre: {self.detected_genre}")
            return self.detected_genre

        return "unknown"


# =============================================================================
# MAIN KNOWLEDGE ACQUISITION SYSTEM
# =============================================================================

class KnowledgeAcquisitionSystem:
    """
    Orchestrates multiple knowledge sources to help the agent learn.

    Priority when needing information:
    1. Memory (previously learned)
    2. Tutorial text (if visible)
    3. Transfer from similar games
    4. Experimentation (safe actions)
    5. Ask user (if available)
    6. Web search (if enabled)
    """

    def __init__(self):
        # Initialize providers
        self.tutorial_learner = TutorialLearner()
        self.user_assistant = UserAssistant()
        self.web_searcher = WebSearcher()
        self.experimenter = Experimenter()
        self.transfer_learner = TransferLearner()

        # Knowledge base
        self.knowledge_base: List[GameKnowledge] = []

        # Current game info
        self.current_game: str = ""
        self.current_genre: str = ""

        # Settings
        self.user_help_enabled = True
        self.web_search_enabled = False  # Disabled by default
        self.auto_experiment = True

        # Persistence
        self.knowledge_file: Optional[Path] = None

    def set_game(self, game_name: str, genre: str = ""):
        """Set the current game being played"""
        self.current_game = game_name
        if genre:
            self.current_genre = genre
            self.transfer_learner.detected_genre = genre

        info("knowledge", f"Set game: {game_name} (genre: {genre or 'unknown'})")

        # Load any existing knowledge for this game
        self._load_game_knowledge(game_name)

    def process_screen_text(self, texts: List[str]):
        """Process text from screen for potential learning"""
        # Try to learn from tutorial-like text
        new_knowledge = self.tutorial_learner.parse_tutorial_text(
            texts,
            self.current_game
        )

        for knowledge in new_knowledge:
            self._add_knowledge(knowledge)

        # Try to detect genre if not known
        if not self.current_genre:
            detected = self.transfer_learner.detect_genre([], texts)
            if detected != "unknown":
                self.current_genre = detected

    def get_help(self, topic: str, context: Dict[str, Any]) -> Optional[GameKnowledge]:
        """
        Get help on a topic using all available sources.
        Returns the best knowledge found, or None if nothing available.
        """
        # Add current game to context
        context['game_name'] = self.current_game
        context['genre'] = self.current_genre

        # 1. Check existing knowledge base
        knowledge = self._search_knowledge_base(topic)
        if knowledge and knowledge.confidence > 0.7:
            knowledge.used_count += 1
            return knowledge

        # 2. Try tutorial learner
        knowledge = self.tutorial_learner.get_knowledge(topic, context)
        if knowledge:
            return knowledge

        # 3. Try transfer learning
        knowledge = self.transfer_learner.get_knowledge(topic, context)
        if knowledge:
            return knowledge

        # 4. Try experimentation results
        knowledge = self.experimenter.get_knowledge(topic, context)
        if knowledge:
            return knowledge

        # 5. Ask user if enabled and available
        if self.user_help_enabled and self.user_assistant.user_available:
            # This is async - will return None immediately
            # but queues the question
            self.user_assistant.ask_user(
                f"I'm stuck on: {topic}. What should I do?",
                context
            )

        # 6. Web search if enabled
        if self.web_search_enabled:
            knowledge = self.web_searcher.get_knowledge(topic, context)
            if knowledge:
                self._add_knowledge(knowledge)
                return knowledge

        return None

    def suggest_action_when_stuck(self, context: str) -> str:
        """Suggest an action when the agent is stuck"""
        # First check if we have knowledge about this situation
        knowledge = self.get_help("stuck", {'situation': context})
        if knowledge:
            return knowledge.content

        # Otherwise, suggest an experiment
        if self.auto_experiment:
            action = self.experimenter.suggest_experiment(context)
            return f"Try pressing {action}"

        return "Press A to interact, or move in a different direction"

    def record_action_outcome(self, action: str, context_before: str,
                             context_after: str, success: bool):
        """Record the outcome of an action for learning"""
        outcome = 'positive' if success else 'negative'

        # Check if context changed at all
        if context_before == context_after:
            outcome = 'neutral'

        self.experimenter.record_experiment(
            action, context_before, context_after, outcome
        )

        # If positive, create knowledge
        if success:
            knowledge = GameKnowledge(
                topic=f"action_in_{context_before[:20]}",
                content=f"Action '{action}' works in this context",
                source=KnowledgeSource.EXPERIMENTATION,
                confidence=0.7,
                game_id=self.current_game,
                tags=["learned", action]
            )
            self._add_knowledge(knowledge)

    def _add_knowledge(self, knowledge: GameKnowledge):
        """Add knowledge to the base"""
        # Check for duplicates
        for existing in self.knowledge_base:
            if (existing.topic == knowledge.topic and
                existing.content == knowledge.content):
                # Update confidence if new is higher
                if knowledge.confidence > existing.confidence:
                    existing.confidence = knowledge.confidence
                return

        self.knowledge_base.append(knowledge)
        debug("knowledge", f"Added knowledge: {knowledge.topic}")

    def _search_knowledge_base(self, topic: str) -> Optional[GameKnowledge]:
        """Search knowledge base for relevant info"""
        topic_lower = topic.lower()

        candidates = []
        for knowledge in self.knowledge_base:
            score = 0
            if topic_lower in knowledge.topic.lower():
                score += 2
            if topic_lower in knowledge.content.lower():
                score += 1
            if knowledge.game_id == self.current_game:
                score += 1
            if knowledge.genre == self.current_genre:
                score += 0.5

            if score > 0:
                candidates.append((knowledge, score))

        if candidates:
            # Return highest scoring
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    def _load_game_knowledge(self, game_name: str):
        """Load previously saved knowledge for a game"""
        if self.knowledge_file and self.knowledge_file.exists():
            try:
                with open(self.knowledge_file, 'r') as f:
                    data = json.load(f)
                    game_data = data.get(game_name, [])
                    for item in game_data:
                        knowledge = GameKnowledge(
                            topic=item['topic'],
                            content=item['content'],
                            source=KnowledgeSource[item['source']],
                            confidence=item.get('confidence', 0.5),
                            game_id=game_name,
                            tags=item.get('tags', [])
                        )
                        self.knowledge_base.append(knowledge)

                info("knowledge", f"Loaded {len(game_data)} knowledge items for {game_name}")
            except Exception as e:
                warning("knowledge", f"Failed to load knowledge: {e}")

    def save_knowledge(self):
        """Save knowledge to file"""
        if not self.knowledge_file:
            return

        try:
            # Load existing data
            existing_data = {}
            if self.knowledge_file.exists():
                with open(self.knowledge_file, 'r') as f:
                    existing_data = json.load(f)

            # Add/update current game's knowledge
            game_knowledge = [
                {
                    'topic': k.topic,
                    'content': k.content,
                    'source': k.source.name,
                    'confidence': k.confidence,
                    'tags': k.tags,
                    'used_count': k.used_count,
                }
                for k in self.knowledge_base
                if k.game_id == self.current_game
            ]

            existing_data[self.current_game] = game_knowledge

            with open(self.knowledge_file, 'w') as f:
                json.dump(existing_data, f, indent=2)

            info("knowledge", f"Saved knowledge to {self.knowledge_file}")

        except Exception as e:
            error("knowledge", f"Failed to save knowledge: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge system statistics"""
        return {
            'total_knowledge': len(self.knowledge_base),
            'current_game': self.current_game,
            'current_genre': self.current_genre,
            'experiments_run': len(self.experimenter.experiments),
            'tutorial_items': len(self.tutorial_learner.learned_instructions),
            'pending_questions': len(self.user_assistant.pending_requests),
            'user_help_enabled': self.user_help_enabled,
            'web_search_enabled': self.web_search_enabled,
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Knowledge Acquisition System Test")
    print("=" * 50)

    # Create system
    kas = KnowledgeAcquisitionSystem()
    kas.set_game("Pokemon Fire Red", "rpg")

    # Simulate tutorial text
    tutorial_texts = [
        "Press A to talk to people",
        "Press B to cancel",
        "Use the D-pad to move around",
        "Press START to open the menu",
        "HP shows your Pokemon's health",
    ]

    print("\nProcessing tutorial text:")
    kas.process_screen_text(tutorial_texts)

    # Try to get help
    print("\nGetting help on topics:")
    topics = ["talk", "menu", "battle", "healing", "unknown topic"]

    for topic in topics:
        knowledge = kas.get_help(topic, {})
        if knowledge:
            print(f"  {topic}: {knowledge.content[:50]}... (source: {knowledge.source.value})")
        else:
            print(f"  {topic}: No knowledge found")

    # Test experimentation
    print("\nTesting experimentation:")
    suggested = kas.experimenter.suggest_experiment("battle menu")
    print(f"  Suggested action: {suggested}")

    kas.record_action_outcome("A", "battle menu", "move selection", True)
    print("  Recorded successful action")

    # Get stats
    print("\nSystem stats:")
    stats = kas.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nKnowledge Acquisition System Test Complete!")
