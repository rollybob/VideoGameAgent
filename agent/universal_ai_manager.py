"""
Universal AI Manager - Modular Game Agent
========================================

Coordinates all AI modules to create a versatile, platform-agnostic game agent.
Integrates perception, text understanding, and reasoning layers into a unified
decision-making system that works across all gaming platforms.

Designed to replace platform-specific hardcoded logic with intelligent,
context-aware decision making.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import our debugging system
from .debug_system import (
    get_debugger,
    monitor_performance,
    info,
    debug,
    warning,
    error,
)

# Import modular AI components
from .perception import PerceptionEngine, GamePerception
from .nlp import TextAnalyzer, TextAnalysis
from .reasoning import RuleBasedReasoningEngine, GameDecision


@dataclass
class UniversalGameContext:
    """Complete game understanding from all AI modules"""
    timestamp: float
    raw_frame: np.ndarray
    perception: GamePerception
    text_analysis: TextAnalysis
    decision: GameDecision
    confidence: float
    platform_hints: Dict[str, Any]  # Platform-specific context hints
    processing_time: float


@dataclass
class ActionResult:
    """Result of an AI-driven action"""
    action: str                     # Primary action to execute
    parameters: Dict[str, Any]      # Action parameters
    confidence: float               # Confidence in this action
    reasoning: str                  # Human-readable reasoning
    fallback_actions: List[str]     # Alternative actions if primary fails
    expected_outcome: str           # What we expect to happen
    platform_specific: bool = False # Whether this is platform-specific


class UniversalAIManager:
    """
    Universal AI Manager that coordinates all modular components
    
    Features:
    - Platform-agnostic decision making
    - Intelligent context understanding  
    - Modular component coordination
    - Comprehensive debugging and monitoring
    - Fallback handling and error recovery
    - Performance optimization
    """
    
    def __init__(self, 
                 enable_perception: bool = True,
                 enable_text_analysis: bool = True,
                 enable_reasoning: bool = True,
                 debug_level: str = "INFO"):
        
        self.debugger = get_debugger()
        self.layer_name = "universal_ai"
        
        info(self.layer_name, "Initializing Universal AI Manager...")
        
        # Configuration
        self.enable_perception = enable_perception
        self.enable_text_analysis = enable_text_analysis
        self.enable_reasoning = enable_reasoning
        
        # Initialize AI modules
        self.perception_engine = None
        self.text_analyzer = None
        self.reasoning_engine = None
        self.initialize_ai_modules()
        
        # Universal action mappings (platform-agnostic)
        self.universal_actions = self.load_universal_actions()
        
        # Context history for learning and adaptation
        self.context_history = []
        self.max_history = 50
        
        # Performance tracking
        self.stats = {
            'frames_processed': 0,
            'decisions_made': 0,
            'avg_processing_time': 0.0,
            'successful_actions': 0,
            'failed_actions': 0,
            'avg_confidence': 0.0,
            'module_performance': {}
        }
        
        # Platform adaptation
        self.platform_context = {}
        self.learned_patterns = {}
        
        info(self.layer_name, "Universal AI Manager initialized successfully")
    
    @monitor_performance("universal_ai", "module_initialization")
    def initialize_ai_modules(self):
        """Initialize all AI modules with error handling"""
        try:
            if self.enable_perception:
                debug(self.layer_name, "Initializing Perception Engine...")
                self.perception_engine = PerceptionEngine()
                info(self.layer_name, "Perception Engine ready")
            
            if self.enable_text_analysis:
                debug(self.layer_name, "Initializing Text Analyzer...")
                self.text_analyzer = TextAnalyzer()
                info(self.layer_name, "Text Analyzer ready")
            
            if self.enable_reasoning:
                debug(self.layer_name, "Initializing Reasoning Engine...")
                self.reasoning_engine = RuleBasedReasoningEngine()
                info(self.layer_name, "Reasoning Engine ready")
                
        except Exception as e:
            error(self.layer_name, "Failed to initialize AI modules", exception=e)
            # Continue with partial initialization
            warning(self.layer_name, "Running with partial AI capabilities")
    
    def load_universal_actions(self) -> Dict[str, Dict[str, Any]]:
        """Load universal action mappings that work across platforms"""
        debug(self.layer_name, "Loading universal action mappings")
        
        return {
            # Universal navigation actions
            'navigate_up': {
                'type': 'movement',
                'direction': 'up',
                'platforms': {'gameboy': 'up', 'pc': 'w', 'console': 'up_arrow'},
                'confidence_threshold': 0.7
            },
            'navigate_down': {
                'type': 'movement', 
                'direction': 'down',
                'platforms': {'gameboy': 'down', 'pc': 's', 'console': 'down_arrow'},
                'confidence_threshold': 0.7
            },
            'navigate_left': {
                'type': 'movement',
                'direction': 'left', 
                'platforms': {'gameboy': 'left', 'pc': 'a', 'console': 'left_arrow'},
                'confidence_threshold': 0.7
            },
            'navigate_right': {
                'type': 'movement',
                'direction': 'right',
                'platforms': {'gameboy': 'right', 'pc': 'd', 'console': 'right_arrow'},
                'confidence_threshold': 0.7
            },
            
            # Universal interaction actions
            'primary_action': {
                'type': 'interaction',
                'purpose': 'primary_interact',
                'platforms': {'gameboy': 'A', 'pc': 'space', 'console': 'x'},
                'confidence_threshold': 0.8
            },
            'secondary_action': {
                'type': 'interaction',
                'purpose': 'secondary_interact',
                'platforms': {'gameboy': 'B', 'pc': 'escape', 'console': 'circle'},
                'confidence_threshold': 0.8
            },
            'menu_action': {
                'type': 'interface',
                'purpose': 'open_menu',
                'platforms': {'gameboy': 'start', 'pc': 'tab', 'console': 'options'},
                'confidence_threshold': 0.9
            },
            
            # Universal response actions
            'confirm_action': {
                'type': 'response',
                'purpose': 'confirm',
                'platforms': {'gameboy': 'A', 'pc': 'enter', 'console': 'x'},
                'confidence_threshold': 0.9
            },
            'cancel_action': {
                'type': 'response', 
                'purpose': 'cancel',
                'platforms': {'gameboy': 'B', 'pc': 'escape', 'console': 'circle'},
                'confidence_threshold': 0.9
            },
            
            # Universal strategic actions
            'wait_observe': {
                'type': 'strategy',
                'purpose': 'observe',
                'platforms': {'gameboy': 'wait', 'pc': 'wait', 'console': 'wait'},
                'confidence_threshold': 0.5
            },
            'explore_area': {
                'type': 'strategy',
                'purpose': 'exploration',
                'platforms': {'gameboy': 'up', 'pc': 'w', 'console': 'up_arrow'},
                'confidence_threshold': 0.6
            },
            'move_randomly': {
                'type': 'movement',
                'purpose': 'random_exploration',
                'platforms': {'gameboy': 'up', 'pc': 'w', 'console': 'up_arrow'},
                'confidence_threshold': 0.4
            },
            'explore_new_area': {
                'type': 'movement',
                'purpose': 'directed_exploration',
                'platforms': {'gameboy': 'up', 'pc': 'w', 'console': 'up_arrow'},
                'confidence_threshold': 0.7
            }
        }
    
    @monitor_performance("universal_ai", "frame_analysis")
    def analyze_frame(self, frame: np.ndarray, platform_hint: str = "unknown") -> UniversalGameContext:
        """
        Comprehensive frame analysis using all AI modules
        
        Args:
            frame: Raw game frame to analyze
            platform_hint: Optional hint about the gaming platform
            
        Returns:
            UniversalGameContext with complete understanding
        """
        start_time = time.time()
        
        debug(self.layer_name, f"Analyzing frame for platform: {platform_hint}")
        
        try:
            # Step 1: Perception Analysis
            perception = None
            if self.perception_engine:
                perception = self.perception_engine.analyze_frame(frame)
                debug(self.layer_name, f"Perception: {len(perception.detected_objects)} objects, {len(perception.detected_text)} texts")
            else:
                # Create minimal perception for compatibility
                from .perception import GamePerception
                perception = GamePerception(
                    timestamp=time.time(),
                    detected_objects=[],
                    detected_text=[], 
                    game_context="unknown",
                    dominant_colors=[],
                    screen_regions={},
                    confidence=0.3,
                    processing_time=0.0
                )
            
            # Step 2: Text Understanding
            text_analysis = None
            if self.text_analyzer and perception.detected_text:
                text_analysis = self.text_analyzer.analyze_texts(perception.detected_text)
                debug(self.layer_name, f"Text Analysis: {len(text_analysis.intents)} intents, {len(text_analysis.game_commands)} commands")
            else:
                # Create minimal text analysis for compatibility
                from .nlp import TextAnalysis
                text_analysis = TextAnalysis(
                    timestamp=time.time(),
                    original_texts=[],
                    processed_texts=[],
                    combined_context="",
                    intents=[],
                    semantic_matches=[],
                    game_commands=[],
                    dialogue_content=[],
                    ui_elements=[],
                    stats_info={},
                    confidence=0.3,
                    processing_time=0.0
                )
            
            # Step 3: Reasoning and Decision Making
            decision = None
            if self.reasoning_engine:
                # Add platform context to reasoning
                current_state = self._infer_universal_state(perception, text_analysis, platform_hint)
                decision = self.reasoning_engine.make_decision(perception, text_analysis, current_state)
                debug(self.layer_name, f"Decision: {decision.primary_action} (confidence: {decision.confidence:.2f})")
            else:
                # Create minimal decision for compatibility
                from .reasoning import GameDecision, ReasoningStep
                decision = GameDecision(
                    timestamp=time.time(),
                    decision_type='wait',
                    primary_action='wait_observe',
                    action_parameters={},
                    reasoning_chain=[ReasoningStep('fallback', 'No reasoning engine available', 0.3, [], [])],
                    confidence=0.3,
                    fallback_actions=['wait_observe'],
                    expected_outcome='Wait and observe',
                    processing_time=0.0
                )
            
            # Step 4: Create Universal Context
            processing_time = time.time() - start_time
            
            # Calculate overall confidence
            confidences = [perception.confidence, text_analysis.confidence, decision.confidence]
            overall_confidence = sum(confidences) / len(confidences)
            
            # Create platform hints
            platform_hints = self._create_platform_hints(perception, text_analysis, platform_hint)
            
            context = UniversalGameContext(
                timestamp=start_time,
                raw_frame=frame,
                perception=perception,
                text_analysis=text_analysis,
                decision=decision,
                confidence=overall_confidence,
                platform_hints=platform_hints,
                processing_time=processing_time
            )
            
            # Update context history
            self._update_context_history(context)
            
            # Update statistics
            self._update_stats(context)
            
            info(self.layer_name, f"Frame analyzed: {decision.primary_action} (confidence: {overall_confidence:.2f})", 
                 details={'processing_time': processing_time, 'platform': platform_hint})
            
            return context
            
        except Exception as e:
            error(self.layer_name, "Frame analysis failed", exception=e)
            return self._create_fallback_context(frame)
    
    def _infer_universal_state(self, perception: GamePerception, text_analysis: TextAnalysis, platform_hint: str) -> str:
        """Infer universal game state that works across platforms"""
        
        # Start with perception context if available
        if perception.game_context and perception.game_context != "unknown":
            base_state = perception.game_context
        else:
            base_state = "exploration"
        
        # Enhance with text analysis
        if text_analysis.intents:
            intent_contexts = [intent.context_category for intent in text_analysis.intents]
            if 'battle' in intent_contexts:
                base_state = 'combat'
            elif 'dialogue' in intent_contexts:
                base_state = 'interaction'
            elif 'menu' in intent_contexts:
                base_state = 'interface'
        
        # Normalize to universal states
        universal_states = {
            'battle': 'combat', 'combat': 'combat', 'fight': 'combat',
            'dialogue': 'interaction', 'interaction': 'interaction', 'npc': 'interaction',
            'menu': 'interface', 'interface': 'interface', 'ui': 'interface',
            'overworld': 'exploration', 'exploration': 'exploration', 'world': 'exploration',
            'shop': 'transaction', 'store': 'transaction', 'buy': 'transaction',
            'inventory': 'management', 'items': 'management', 'equipment': 'management'
        }
        
        return universal_states.get(base_state.lower(), 'exploration')
    
    def _create_platform_hints(self, perception: GamePerception, text_analysis: TextAnalysis, platform_hint: str) -> Dict[str, Any]:
        """Create platform-specific context hints"""
        hints = {
            'platform': platform_hint,
            'ui_style': 'unknown',
            'input_method': 'unknown',
            'game_genre': 'unknown'
        }
        
        # Infer UI style from text and objects
        if text_analysis.ui_elements:
            ui_elements = [elem.lower() for elem in text_analysis.ui_elements]
            if any(elem in ui_elements for elem in ['fight', 'pokemon', 'bag', 'run']):
                hints['ui_style'] = 'jrpg'
                hints['game_genre'] = 'rpg'
            elif any(elem in ui_elements for elem in ['attack', 'defend', 'magic', 'item']):
                hints['ui_style'] = 'classic_rpg'
                hints['game_genre'] = 'rpg'
            elif any(elem in ui_elements for elem in ['yes', 'no', 'ok', 'cancel']):
                hints['ui_style'] = 'dialog_box'
        
        # Infer input method from platform
        input_methods = {
            'gameboy': 'dpad_buttons',
            'nintendo': 'dpad_buttons', 
            'pc': 'keyboard_mouse',
            'playstation': 'analog_buttons',
            'xbox': 'analog_buttons'
        }
        hints['input_method'] = input_methods.get(platform_hint.lower(), 'unknown')
        
        return hints
    
    @monitor_performance("universal_ai", "action_translation")
    def translate_to_platform_action(self, 
                                   universal_action: str, 
                                   platform: str = "gameboy",
                                   confidence: float = 0.8) -> ActionResult:
        """
        Translate universal action to platform-specific action
        
        Args:
            universal_action: Universal action from reasoning engine
            platform: Target platform for action translation
            confidence: Confidence in the action decision
            
        Returns:
            ActionResult with platform-specific action details
        """
        debug(self.layer_name, f"Translating '{universal_action}' for platform '{platform}'")
        
        try:
            # Direct mapping if action exists in universal actions
            if universal_action in self.universal_actions:
                action_def = self.universal_actions[universal_action]
                platform_action = action_def['platforms'].get(platform.lower(), action_def['platforms'].get('gameboy', universal_action))
                
                return ActionResult(
                    action=platform_action,
                    parameters={'universal_action': universal_action},
                    confidence=min(confidence, action_def.get('confidence_threshold', 0.8)),
                    reasoning=f"Universal action '{universal_action}' mapped to '{platform_action}' for {platform}",
                    fallback_actions=[self._get_safe_fallback(platform)],
                    expected_outcome=f"Execute {action_def['type']} action on {platform}",
                    platform_specific=True
                )
            
            # Intelligent mapping for common action patterns
            action_mappings = self._get_intelligent_mappings(universal_action, platform)
            if action_mappings:
                return action_mappings
            
            # Fallback to safe action
            return self._create_safe_fallback_action(universal_action, platform, confidence)
            
        except Exception as e:
            warning(self.layer_name, f"Action translation failed for '{universal_action}'", 
                   details={'platform': platform, 'error': str(e)})
            return self._create_safe_fallback_action(universal_action, platform, 0.3)
    
    def _get_intelligent_mappings(self, action: str, platform: str) -> Optional[ActionResult]:
        """Get intelligent action mappings based on action patterns"""
        
        action_lower = action.lower()
        
        # Movement actions
        if 'up' in action_lower or 'north' in action_lower:
            return self.translate_to_platform_action('navigate_up', platform)
        elif 'down' in action_lower or 'south' in action_lower:
            return self.translate_to_platform_action('navigate_down', platform)
        elif 'left' in action_lower or 'west' in action_lower:
            return self.translate_to_platform_action('navigate_left', platform)
        elif 'right' in action_lower or 'east' in action_lower:
            return self.translate_to_platform_action('navigate_right', platform)
        
        # Interaction actions
        elif 'interact' in action_lower or 'talk' in action_lower or 'examine' in action_lower:
            return self.translate_to_platform_action('primary_action', platform)
        elif 'back' in action_lower or 'cancel' in action_lower or 'exit' in action_lower:
            return self.translate_to_platform_action('cancel_action', platform)
        elif 'confirm' in action_lower or 'select' in action_lower or 'accept' in action_lower:
            return self.translate_to_platform_action('confirm_action', platform)
        elif 'menu' in action_lower or 'inventory' in action_lower:
            return self.translate_to_platform_action('menu_action', platform)
        
        # Strategic actions
        elif 'wait' in action_lower or 'observe' in action_lower:
            return self.translate_to_platform_action('wait_observe', platform)
        elif 'explore' in action_lower or 'wander' in action_lower:
            return self.translate_to_platform_action('explore_area', platform)
        
        return None
    
    def _get_safe_fallback(self, platform: str) -> str:
        """Get safe fallback action for platform"""
        safe_actions = {
            'gameboy': 'wait',
            'pc': 'wait',
            'nintendo': 'wait',
            'playstation': 'wait', 
            'xbox': 'wait'
        }
        return safe_actions.get(platform.lower(), 'wait')
    
    def _create_safe_fallback_action(self, original_action: str, platform: str, confidence: float) -> ActionResult:
        """Create safe fallback action when translation fails"""
        safe_action = self._get_safe_fallback(platform)
        
        return ActionResult(
            action=safe_action,
            parameters={'original_action': original_action, 'fallback': True},
            confidence=max(0.3, confidence * 0.5),
            reasoning=f"Fallback to safe action '{safe_action}' for unmapped action '{original_action}'",
            fallback_actions=['wait', 'observe'],
            expected_outcome="Safe wait action to prevent issues",
            platform_specific=False
        )
    
    def _update_context_history(self, context: UniversalGameContext):
        """Update context history for learning and adaptation"""
        self.context_history.append(context)
        
        # Keep history manageable
        if len(self.context_history) > self.max_history:
            self.context_history = self.context_history[-self.max_history//2:]
    
    def _update_stats(self, context: UniversalGameContext):
        """Update performance statistics"""
        self.stats['frames_processed'] += 1
        self.stats['decisions_made'] += 1
        
        # Update average processing time
        current_avg = self.stats['avg_processing_time']
        count = self.stats['frames_processed']
        self.stats['avg_processing_time'] = (current_avg * (count - 1) + context.processing_time) / count
        
        # Update average confidence
        current_conf_avg = self.stats['avg_confidence']
        self.stats['avg_confidence'] = (current_conf_avg * (count - 1) + context.confidence) / count
    
    def _create_fallback_context(self, frame: np.ndarray) -> UniversalGameContext:
        """Create fallback context when analysis fails"""
        from .perception import GamePerception
        from .nlp import TextAnalysis
        from .reasoning import GameDecision, ReasoningStep
        
        return UniversalGameContext(
            timestamp=time.time(),
            raw_frame=frame,
            perception=GamePerception(
                timestamp=time.time(),
                detected_objects=[],
                detected_text=[],
                game_context="unknown",
                dominant_colors=[],
                screen_regions={},
                confidence=0.2,
                processing_time=0.0
            ),
            text_analysis=TextAnalysis(
                timestamp=time.time(),
                original_texts=[],
                processed_texts=[],
                combined_context="",
                intents=[],
                semantic_matches=[],
                game_commands=[],
                dialogue_content=[],
                ui_elements=[],
                stats_info={},
                confidence=0.2,
                processing_time=0.0
            ),
            decision=GameDecision(
                timestamp=time.time(),
                decision_type='wait',
                primary_action='wait_observe',
                action_parameters={},
                reasoning_chain=[ReasoningStep('fallback', 'Analysis failed, using safe fallback', 0.2, [], [])],
                confidence=0.2,
                fallback_actions=['wait_observe'],
                expected_outcome='Safe fallback action',
                processing_time=0.0
            ),
            confidence=0.2,
            platform_hints={'platform': 'unknown'},
            processing_time=0.0
        )
    
    def get_adaptation_insights(self) -> Dict[str, Any]:
        """Get insights for platform adaptation and learning"""
        if not self.context_history:
            return {}
        
        recent_contexts = self.context_history[-10:]
        
        insights = {
            'common_states': {},
            'successful_actions': {},
            'confidence_trends': [],
            'platform_patterns': {}
        }
        
        # Analyze common states
        for context in recent_contexts:
            state = context.decision.decision_type
            insights['common_states'][state] = insights['common_states'].get(state, 0) + 1
        
        # Confidence trends
        insights['confidence_trends'] = [ctx.confidence for ctx in recent_contexts]
        
        # Platform patterns
        for context in recent_contexts:
            platform = context.platform_hints.get('platform', 'unknown')
            if platform not in insights['platform_patterns']:
                insights['platform_patterns'][platform] = {'actions': {}, 'avg_confidence': 0.0}
            
            action = context.decision.primary_action
            insights['platform_patterns'][platform]['actions'][action] = \
                insights['platform_patterns'][platform]['actions'].get(action, 0) + 1
        
        return insights
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = self.stats.copy()
        
        # Add module-specific stats
        if self.perception_engine:
            stats['perception_stats'] = self.perception_engine.get_stats()
        if self.text_analyzer:
            stats['text_analysis_stats'] = self.text_analyzer.get_stats()
        if self.reasoning_engine:
            stats['reasoning_stats'] = self.reasoning_engine.get_stats()
        
        # Add adaptation insights
        stats['adaptation_insights'] = self.get_adaptation_insights()
        
        return stats
    
    def create_context_summary(self, context: UniversalGameContext) -> str:
        """Create human-readable summary of game context"""
        summary_parts = []
        
        # Perception summary
        if context.perception.detected_objects:
            summary_parts.append(f"{len(context.perception.detected_objects)} objects detected")
        if context.perception.detected_text:
            summary_parts.append(f"{len(context.perception.detected_text)} text elements")
        
        # Text analysis summary
        if context.text_analysis.intents:
            intent_types = [intent.intent_type for intent in context.text_analysis.intents]
            summary_parts.append(f"Intents: {', '.join(set(intent_types))}")
        
        # Decision summary
        summary_parts.append(f"Action: {context.decision.primary_action}")
        summary_parts.append(f"Confidence: {context.confidence:.2f}")
        summary_parts.append(f"Processing: {context.processing_time:.3f}s")
        
        return " | ".join(summary_parts)


# Testing and example usage
if __name__ == "__main__":
    # Initialize debugging
    from .debug_system import init_debugging
    init_debugging(log_level="DEBUG")
    
    print("Testing Universal AI Manager...")
    
    try:
        # Initialize Universal AI Manager
        ai_manager = UniversalAIManager(
            enable_perception=True,
            enable_text_analysis=True, 
            enable_reasoning=True
        )
        
        # Create mock game frame
        mock_frame = np.zeros((320, 480, 3), dtype=np.uint8)
        mock_frame[100:200, 100:200] = [255, 0, 0]  # Red square
        
        # Analyze frame
        print(f"\nAnalyzing mock game frame...")
        context = ai_manager.analyze_frame(mock_frame, platform_hint="gameboy")
        
        # Display results
        print(f"\nAnalysis Results:")
        print(f"   Decision: {context.decision.primary_action}")
        print(f"   Confidence: {context.confidence:.3f}")
        print(f"   Processing time: {context.processing_time:.3f}s")
        
        # Test action translation
        print(f"\nTesting Action Translation:")
        platforms = ['gameboy', 'pc', 'playstation']
        actions = ['navigate_up', 'primary_action', 'wait_observe']
        
        for platform in platforms:
            print(f"\n   Platform: {platform}")
            for action in actions:
                result = ai_manager.translate_to_platform_action(action, platform)
                print(f"     {action} -> {result.action} (confidence: {result.confidence:.2f})")
        
        # Show comprehensive summary
        summary = ai_manager.create_context_summary(context)
        print(f"\nContext Summary: {summary}")
        
        # Show performance stats
        stats = ai_manager.get_stats()
        print(f"\nPerformance Stats:")
        print(f"   Frames processed: {stats['frames_processed']}")
        print(f"   Average processing time: {stats['avg_processing_time']:.3f}s")
        print(f"   Average confidence: {stats['avg_confidence']:.3f}")
        
        print("\nUniversal AI Manager test completed!")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()