#!/usr/bin/env python3
"""
🧠 Simple Reasoning Engine - Lightweight Decision Making
=======================================================

Rule-based reasoning system for game decisions without heavy LLM requirements.
Perfect for testing and fast decision making.
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Import debug system
from debug_system import get_debugger, monitor_performance, info, debug, warning, error

# Import data structures from main reasoning engine
from reasoning_engine import GameContext, ReasoningStep, GameDecision

# Import other layer data structures
from perception_engine import GamePerception, DetectedObject, DetectedText
from text_analyzer import TextAnalysis, TextIntent


class SimpleReasoningEngine:
    """
    Lightweight rule-based reasoning engine for game decisions
    
    Features:
    - Fast rule-based decision making
    - No heavy model dependencies
    - Context-aware game logic
    - Battle strategy patterns
    - Menu navigation logic
    """
    
    def __init__(self):
        self.debugger = get_debugger()
        self.layer_name = "simple_reasoning"
        
        info(self.layer_name, "Initializing Simple Reasoning Engine...")
        
        # Load rule-based knowledge
        self.game_rules = self._load_game_rules()
        self.decision_patterns = self._load_decision_patterns()
        
        # Performance tracking
        self.stats = {
            'decisions_made': 0,
            'avg_reasoning_time': 0.0,
            'successful_actions': 0,
            'failed_actions': 0,
            'confidence_scores': []
        }
        
        info(self.layer_name, "SUCCESS: Simple Reasoning Engine initialized")
    
    def _load_game_rules(self) -> Dict[str, Any]:
        """Load game-specific rules and strategies"""
        return {
            'battle_rules': {
                'priorities': [
                    ('low_hp', 'heal_or_switch'),
                    ('type_advantage', 'attack'),
                    ('strong_opponent', 'switch_pokemon'),
                    ('default_battle', 'fight')
                ],
                'hp_thresholds': {
                    'critical': 0.25,
                    'low': 0.5,
                    'healthy': 0.75
                }
            },
            
            'exploration_rules': {
                'priorities': [
                    ('npc_nearby', 'interact'),
                    ('item_visible', 'collect'),
                    ('door_or_entrance', 'enter'),
                    ('unexplored_area', 'explore'),
                    ('default_exploration', 'move_forward')
                ]
            },
            
            'menu_rules': {
                'priorities': [
                    ('battle_menu', 'select_fight_if_healthy'),
                    ('dialogue_choice', 'select_positive_option'),
                    ('shop_menu', 'buy_useful_items'),
                    ('pokemon_menu', 'select_strongest'),
                    ('default_menu', 'select_first_option')
                ]
            }
        }
    
    def _load_decision_patterns(self) -> Dict[str, List[str]]:
        """Load common decision patterns"""
        return {
            'battle_actions': ['fight', 'select_fight', 'attack', 'use_move'],
            'defensive_actions': ['pokemon', 'select_pokemon', 'switch', 'bag', 'use_item'],
            'escape_actions': ['run', 'select_run', 'escape'],
            'navigation_actions': ['move_up', 'move_down', 'move_left', 'move_right'],
            'interaction_actions': ['interact', 'talk', 'examine', 'select'],
            'menu_actions': ['select', 'confirm', 'back', 'cancel'],
            'wait_actions': ['wait', 'observe', 'pause']
        }
    
    @monitor_performance("simple_reasoning", "decision_making")
    def make_decision(self, 
                     perception: GamePerception, 
                     text_analysis: TextAnalysis,
                     current_state: Optional[str] = None) -> GameDecision:
        """
        Make fast rule-based decision
        
        Args:
            perception: Visual perception data
            text_analysis: Text understanding results
            current_state: Optional explicit state override
            
        Returns:
            GameDecision with reasoning chain
        """
        start_time = time.time()
        
        debug(self.layer_name, "Starting rule-based decision making")
        
        try:
            # 1. Create game context
            game_context = self._create_game_context(perception, text_analysis, current_state)
            
            # 2. Apply rule-based reasoning
            reasoning_chain = self._apply_rule_based_reasoning(game_context)
            
            # 3. Make final decision
            decision = self._make_rule_based_decision(game_context, reasoning_chain)
            
            # 4. Validate decision
            validated_decision = self._validate_decision(decision, game_context)
            
            processing_time = time.time() - start_time
            validated_decision.processing_time = processing_time
            
            # Update statistics
            self._update_stats(validated_decision)
            
            info(self.layer_name, f"Rule-based decision: {validated_decision.primary_action}", 
                 details={'confidence': validated_decision.confidence, 
                         'processing_time': processing_time,
                         'state': game_context.current_state})
            
            return validated_decision
            
        except Exception as e:
            error(self.layer_name, "Rule-based decision making failed", exception=e)
            return self._create_fallback_decision()
    
    def _create_game_context(self, 
                           perception: GamePerception, 
                           text_analysis: TextAnalysis,
                           current_state: Optional[str]) -> GameContext:
        """Create unified game context"""
        debug(self.layer_name, "Creating game context")
        
        # Determine current state
        if current_state is None:
            current_state = self._infer_game_state(perception, text_analysis)
        
        # Extract key entities
        key_entities = self._extract_key_entities(perception, text_analysis)
        
        # Determine available actions
        available_actions = self._determine_available_actions(perception, text_analysis, current_state)
        
        # Create spatial information
        spatial_info = self._create_spatial_info(perception)
        
        # Calculate confidence
        confidence = (perception.confidence + text_analysis.confidence) / 2.0
        
        context = GameContext(
            timestamp=time.time(),
            perception=perception,
            text_analysis=text_analysis,
            current_state=current_state,
            confidence=confidence,
            key_entities=key_entities,
            available_actions=available_actions,
            spatial_info=spatial_info
        )
        
        debug(self.layer_name, f"Context created: state={current_state}, entities={len(key_entities)}")
        return context
    
    def _infer_game_state(self, perception: GamePerception, text_analysis: TextAnalysis) -> str:
        """Infer current game state from available data"""
        
        # Check UI elements for battle indicators
        ui_elements_lower = [elem.lower() for elem in text_analysis.ui_elements]
        
        if any(elem in ui_elements_lower for elem in ['fight', 'pokemon', 'bag', 'run']):
            return 'battle'
        elif any(elem in ui_elements_lower for elem in ['yes', 'no']):
            return 'dialogue'
        elif any(elem in ui_elements_lower for elem in ['buy', 'sell', 'exit']):
            return 'shop'
        elif len(text_analysis.ui_elements) > 2:
            return 'menu'
        else:
            return 'exploration'
    
    def _extract_key_entities(self, perception: GamePerception, text_analysis: TextAnalysis) -> List[str]:
        """Extract important entities"""
        entities = []
        
        # From text analysis
        for intent in text_analysis.intents:
            if intent.entities:
                for entity_type, entity_list in intent.entities.items():
                    entities.extend(entity_list)
        
        # From perception objects
        for obj in perception.detected_objects:
            if hasattr(obj, 'object_class') and obj.object_class:
                entities.append(obj.object_class)
        
        # From UI elements
        entities.extend(text_analysis.ui_elements)
        entities.extend(text_analysis.game_commands)
        
        return list(set(entities))[:10]  # Unique, limited
    
    def _determine_available_actions(self, 
                                   perception: GamePerception, 
                                   text_analysis: TextAnalysis,
                                   current_state: str) -> List[str]:
        """Determine available actions based on context"""
        actions = ['wait', 'observe']  # Always available
        
        if current_state == 'battle':
            # Battle-specific actions based on UI
            ui_lower = [elem.lower() for elem in text_analysis.ui_elements]
            if 'fight' in ui_lower:
                actions.append('select_fight')
            if 'pokemon' in ui_lower:
                actions.append('select_pokemon')
            if 'bag' in ui_lower:
                actions.append('select_bag')
            if 'run' in ui_lower:
                actions.append('select_run')
                
        elif current_state == 'dialogue':
            actions.extend(['respond_yes', 'respond_no', 'continue_dialogue'])
            
        elif current_state == 'exploration':
            actions.extend(['move_up', 'move_down', 'move_left', 'move_right', 'interact'])
            
        elif current_state == 'menu':
            for element in text_analysis.ui_elements:
                actions.append(f'select_{element.lower()}')
        
        return actions
    
    def _create_spatial_info(self, perception: GamePerception) -> Dict[str, Any]:
        """Create spatial mapping of screen elements"""
        spatial = {
            'screen_regions': {'top': [], 'middle': [], 'bottom': []},
            'object_positions': [],
            'text_positions': []
        }
        
        # Categorize objects by screen region
        for obj in perception.detected_objects:
            y_center = obj.center[1] if hasattr(obj, 'center') and obj.center else 0
            
            if y_center < 150:
                spatial['screen_regions']['top'].append(obj)
            elif y_center < 350:
                spatial['screen_regions']['middle'].append(obj)
            else:
                spatial['screen_regions']['bottom'].append(obj)
            
            spatial['object_positions'].append({
                'object': getattr(obj, 'object_class', 'unknown'),
                'position': getattr(obj, 'center', (0, 0)),
                'confidence': getattr(obj, 'confidence', 0.0)
            })
        
        # Add text positions
        for text in perception.detected_texts:
            spatial['text_positions'].append({
                'text': text.text,
                'position': text.center,
                'confidence': text.confidence
            })
        
        return spatial
    
    def _apply_rule_based_reasoning(self, game_context: GameContext) -> List[ReasoningStep]:
        """Apply rule-based reasoning for current context"""
        debug(self.layer_name, "Applying rule-based reasoning")
        
        reasoning_steps = []
        
        # Step 1: Analyze current situation
        analysis_step = self._analyze_situation(game_context)
        reasoning_steps.append(analysis_step)
        
        # Step 2: Apply state-specific rules
        planning_step = self._apply_state_rules(game_context)
        reasoning_steps.append(planning_step)
        
        # Step 3: Make decision based on rules
        decision_step = self._apply_decision_rules(game_context)
        reasoning_steps.append(decision_step)
        
        # Step 4: Validate decision
        validation_step = self._validate_rule_decision(game_context)
        reasoning_steps.append(validation_step)
        
        debug(self.layer_name, f"Generated {len(reasoning_steps)} reasoning steps")
        return reasoning_steps
    
    def _analyze_situation(self, game_context: GameContext) -> ReasoningStep:
        """Analyze current game situation"""
        situation_factors = []
        
        # State analysis
        situation_factors.append(f"Current state: {game_context.current_state}")
        
        # Entity analysis
        if game_context.key_entities:
            situation_factors.append(f"Key entities: {', '.join(game_context.key_entities[:3])}")
        
        # UI analysis
        if game_context.text_analysis.ui_elements:
            situation_factors.append(f"UI options: {', '.join(game_context.text_analysis.ui_elements[:3])}")
        
        content = ". ".join(situation_factors)
        
        return ReasoningStep(
            step_type='analysis',
            content=content,
            confidence=0.9,
            evidence=[f"State: {game_context.current_state}", f"Entities: {len(game_context.key_entities)}"],
            alternatives=[]
        )
    
    def _apply_state_rules(self, game_context: GameContext) -> ReasoningStep:
        """Apply rules specific to current game state"""
        state = game_context.current_state
        
        if state == 'battle':
            return self._apply_battle_rules(game_context)
        elif state == 'exploration':
            return self._apply_exploration_rules(game_context)
        elif state == 'dialogue':
            return self._apply_dialogue_rules(game_context)
        elif state == 'menu':
            return self._apply_menu_rules(game_context)
        else:
            return ReasoningStep(
                step_type='planning',
                content=f"No specific rules for state {state}, using general approach",
                confidence=0.6,
                evidence=[],
                alternatives=[]
            )
    
    def _apply_battle_rules(self, game_context: GameContext) -> ReasoningStep:
        """Apply battle-specific decision rules"""
        ui_elements = [elem.lower() for elem in game_context.text_analysis.ui_elements]
        
        # Battle rule priorities
        if 'fight' in ui_elements:
            strategy = "Attack is available - prioritize offensive action"
            confidence = 0.8
        elif 'pokemon' in ui_elements:
            strategy = "Pokemon switch available - consider tactical switch"
            confidence = 0.7
        elif 'bag' in ui_elements:
            strategy = "Items available - consider healing or support items"
            confidence = 0.6
        elif 'run' in ui_elements:
            strategy = "Escape available - consider retreat if outmatched"
            confidence = 0.5
        else:
            strategy = "Limited battle options - use default attack strategy"
            confidence = 0.4
        
        return ReasoningStep(
            step_type='planning',
            content=f"Battle strategy: {strategy}",
            confidence=confidence,
            evidence=[f"UI options: {ui_elements}"],
            alternatives=['fight', 'pokemon', 'bag', 'run']
        )
    
    def _apply_exploration_rules(self, game_context: GameContext) -> ReasoningStep:
        """Apply exploration-specific rules"""
        objects = [obj.object_class for obj in game_context.perception.detected_objects 
                  if hasattr(obj, 'object_class')]
        
        if any('npc' in obj.lower() or 'person' in obj.lower() for obj in objects):
            strategy = "NPC detected - prioritize interaction for information"
            confidence = 0.8
        elif any('item' in obj.lower() for obj in objects):
            strategy = "Item detected - prioritize collection"
            confidence = 0.7
        else:
            strategy = "Open exploration - move forward to discover new areas"
            confidence = 0.6
        
        return ReasoningStep(
            step_type='planning',
            content=f"Exploration strategy: {strategy}",
            confidence=confidence,
            evidence=[f"Objects: {objects[:3]}"],
            alternatives=['interact', 'move_forward', 'examine']
        )
    
    def _apply_dialogue_rules(self, game_context: GameContext) -> ReasoningStep:
        """Apply dialogue-specific rules"""
        ui_elements = [elem.lower() for elem in game_context.text_analysis.ui_elements]
        
        if 'yes' in ui_elements and 'no' in ui_elements:
            strategy = "Yes/No choice - default to positive response for story progression"
            confidence = 0.7
        else:
            strategy = "Continue dialogue to gather information"
            confidence = 0.8
        
        return ReasoningStep(
            step_type='planning',
            content=f"Dialogue strategy: {strategy}",
            confidence=confidence,
            evidence=[f"Options: {ui_elements}"],
            alternatives=['respond_yes', 'respond_no', 'continue']
        )
    
    def _apply_menu_rules(self, game_context: GameContext) -> ReasoningStep:
        """Apply menu navigation rules"""
        ui_elements = game_context.text_analysis.ui_elements
        
        if ui_elements:
            strategy = f"Select most relevant menu option from available choices"
            confidence = 0.7
        else:
            strategy = "No clear menu options - use default selection"
            confidence = 0.4
        
        return ReasoningStep(
            step_type='planning',
            content=f"Menu strategy: {strategy}",
            confidence=confidence,
            evidence=[f"Menu options: {ui_elements[:3]}"],
            alternatives=[f'select_{elem.lower()}' for elem in ui_elements[:3]]
        )
    
    def _apply_decision_rules(self, game_context: GameContext) -> ReasoningStep:
        """Make final decision based on all rules"""
        state = game_context.current_state
        available_actions = game_context.available_actions
        
        # Select best action based on state and available options
        if state == 'battle' and 'select_fight' in available_actions:
            decision = "select_fight"
            rationale = "Battle context - choose fight option"
        elif state == 'dialogue' and 'respond_yes' in available_actions:
            decision = "respond_yes"
            rationale = "Dialogue context - choose positive response"
        elif state == 'exploration' and 'interact' in available_actions:
            decision = "interact"
            rationale = "Exploration context - interact with environment"
        elif available_actions:
            decision = available_actions[0]  # First available non-wait action
            rationale = f"Default to first available action: {decision}"
        else:
            decision = "wait"
            rationale = "No clear actions available - wait for more information"
        
        return ReasoningStep(
            step_type='decision',
            content=f"Decision: {decision}. Rationale: {rationale}",
            confidence=0.8,
            evidence=[f"State: {state}", f"Available: {available_actions[:3]}"],
            alternatives=available_actions[:3]
        )
    
    def _validate_rule_decision(self, game_context: GameContext) -> ReasoningStep:
        """Validate the rule-based decision"""
        return ReasoningStep(
            step_type='validation',
            content="Rule-based decision validated against game context and available actions",
            confidence=0.8,
            evidence=[f"Context confidence: {game_context.confidence:.2f}"],
            alternatives=[]
        )
    
    def _make_rule_based_decision(self, game_context: GameContext, reasoning_chain: List[ReasoningStep]) -> GameDecision:
        """Create final decision from reasoning chain"""
        debug(self.layer_name, "Creating final decision")
        
        # Extract action from decision step
        primary_action = 'wait'
        for step in reasoning_chain:
            if step.step_type == 'decision':
                # Extract action from step content
                if 'Decision:' in step.content:
                    action_part = step.content.split('Decision:')[1].split('.')[0].strip()
                    primary_action = action_part
                    break
                    
        # Ensure action is available
        if primary_action not in game_context.available_actions:
            primary_action = game_context.available_actions[0] if game_context.available_actions else 'wait'
        
        # Determine decision type
        if 'move' in primary_action.lower():
            decision_type = 'action'
        elif 'respond' in primary_action.lower():
            decision_type = 'dialogue_response'
        elif primary_action == 'wait':
            decision_type = 'wait'
        else:
            decision_type = 'action'
        
        # Calculate confidence
        avg_confidence = sum(step.confidence for step in reasoning_chain) / len(reasoning_chain)
        context_boost = game_context.confidence * 0.2
        total_confidence = min(1.0, avg_confidence + context_boost)
        
        # Generate fallback actions
        fallback_actions = [action for action in game_context.available_actions 
                           if action != primary_action][:3]
        if 'wait' not in fallback_actions:
            fallback_actions.append('wait')
        
        # Create decision
        decision = GameDecision(
            timestamp=time.time(),
            decision_type=decision_type,
            primary_action=primary_action,
            action_parameters={'context': game_context.current_state},
            reasoning_chain=reasoning_chain,
            confidence=total_confidence,
            fallback_actions=fallback_actions,
            expected_outcome=f"Execute {primary_action} in {game_context.current_state} context",
            processing_time=0.0
        )
        
        debug(self.layer_name, f"Rule-based decision created: {primary_action}")
        return decision
    
    def _validate_decision(self, decision: GameDecision, game_context: GameContext) -> GameDecision:
        """Validate decision against context"""
        # Ensure action is available
        if decision.primary_action not in game_context.available_actions:
            warning(self.layer_name, f"Action {decision.primary_action} not available")
            if decision.fallback_actions:
                for fallback in decision.fallback_actions:
                    if fallback in game_context.available_actions:
                        decision.primary_action = fallback
                        decision.confidence *= 0.9
                        break
        
        return decision
    
    def _create_fallback_decision(self) -> GameDecision:
        """Create safe fallback decision"""
        return GameDecision(
            timestamp=time.time(),
            decision_type='wait',
            primary_action='wait',
            action_parameters={},
            reasoning_chain=[
                ReasoningStep(
                    step_type='fallback',
                    content='Using safe fallback due to reasoning error',
                    confidence=0.3,
                    evidence=[],
                    alternatives=[]
                )
            ],
            confidence=0.3,
            fallback_actions=['observe', 'wait'],
            expected_outcome='Wait for clearer information',
            processing_time=0.0
        )
    
    def _update_stats(self, decision: GameDecision):
        """Update performance statistics"""
        self.stats['decisions_made'] += 1
        
        # Update average reasoning time
        current_avg = self.stats['avg_reasoning_time']
        count = self.stats['decisions_made']
        self.stats['avg_reasoning_time'] = (current_avg * (count - 1) + decision.processing_time) / count
        
        # Track confidence scores
        self.stats['confidence_scores'].append(decision.confidence)
        if len(self.stats['confidence_scores']) > 100:
            self.stats['confidence_scores'].pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.stats.copy()
        if stats['confidence_scores']:
            stats['avg_confidence'] = sum(stats['confidence_scores']) / len(stats['confidence_scores'])
        else:
            stats['avg_confidence'] = 0.0
        return stats


# Testing
if __name__ == "__main__":
    print("🧠 Testing Simple Reasoning Engine...")
    
    # Initialize debugging
    from debug_system import init_debugging
    init_debugging(log_level="DEBUG")
    
    try:
        # Initialize simple reasoning engine
        simple_reasoning = SimpleReasoningEngine()
        
        # Create mock test data
        from perception_engine import GamePerception, DetectedObject, DetectedText
        from text_analyzer import TextAnalysis, TextIntent
        
        # Mock battle scenario
        mock_perception = GamePerception(
            timestamp=time.time(),
            detected_objects=[
                DetectedObject(
                    object_class="pokemon",
                    confidence=0.9,
                    bbox=(100, 100, 200, 200),
                    center=(150, 150)
                )
            ],
            detected_texts=[
                DetectedText(
                    text="FIGHT",
                    confidence=0.95,
                    bbox=(50, 400, 120, 430),
                    center=(85, 415),
                    is_menu_item=True
                )
            ],
            scene_description="Battle scene with menu",
            confidence=0.85
        )
        
        mock_text_analysis = TextAnalysis(
            timestamp=time.time(),
            original_texts=["FIGHT"],
            processed_texts=["fight"],
            combined_context="Battle menu",
            intents=[
                TextIntent(
                    intent_type="battle_menu",
                    confidence=0.9,
                    entities={'commands': ['fight']},
                    keywords=['fight'],
                    sentiment='neutral',
                    action_required=True,
                    context_category='battle'
                )
            ],
            semantic_matches=[],
            game_commands=['fight'],
            dialogue_content=[],
            ui_elements=['fight'],
            stats_info={},
            confidence=0.8,
            processing_time=0.01
        )
        
        # Make decision
        decision = simple_reasoning.make_decision(mock_perception, mock_text_analysis)
        
        print(f"\n✅ Simple Reasoning Results:")
        print(f"   Decision: {decision.primary_action}")
        print(f"   Type: {decision.decision_type}")
        print(f"   Confidence: {decision.confidence:.3f}")
        print(f"   Processing time: {decision.processing_time:.3f}s")
        print(f"   Reasoning steps: {len(decision.reasoning_chain)}")
        
        for i, step in enumerate(decision.reasoning_chain):
            print(f"   {i+1}. {step.step_type}: {step.content}")
        
        stats = simple_reasoning.get_stats()
        print(f"\nStats: {stats}")
        
        print("\n🎉 Simple Reasoning Engine test successful!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()