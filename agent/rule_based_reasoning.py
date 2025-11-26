"""
BRAIN: Rule-Based Reasoning Engine - Modular AI Game Agent
=====================================================

Intelligent decision-making layer using rule-based logic and heuristics.
Processes perception and text understanding to make informed game decisions.

This is a practical alternative to the LLM-based reasoning system that works
without requiring large language models.

Integrated with comprehensive debugging system for error tracing.
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import re

# Import our debugging system
from debug_system import get_debugger, monitor_performance, info, debug, warning, error

# Import other layer data structures
from perception_engine import GamePerception, DetectedObject, DetectedText
from text_analyzer import TextAnalysis, TextIntent


@dataclass
class GameContext:
    """Complete game context from all layers"""
    timestamp: float
    perception: GamePerception
    text_analysis: TextAnalysis
    current_state: str              # 'battle', 'menu', 'dialogue', 'exploration', etc.
    confidence: float               # Overall context confidence
    key_entities: List[str]         # Most important entities identified
    available_actions: List[str]    # Actions available in current context
    spatial_info: Dict[str, Any]    # Screen regions and object positions


@dataclass
class ReasoningStep:
    """Individual step in reasoning process"""
    step_type: str          # 'analysis', 'planning', 'decision', 'validation'
    content: str            # The reasoning content
    confidence: float       # Confidence in this step
    evidence: List[str]     # Supporting evidence
    alternatives: List[str] # Alternative options considered


@dataclass
class GameDecision:
    """Final decision made by reasoning engine"""
    timestamp: float
    decision_type: str              # 'action', 'wait', 'explore', 'dialogue_response'
    primary_action: str             # Main action to take
    action_parameters: Dict[str, Any]  # Parameters for the action
    reasoning_chain: List[ReasoningStep]  # Full reasoning process
    confidence: float               # Decision confidence
    fallback_actions: List[str]     # Backup actions if primary fails
    expected_outcome: str           # What we expect to happen
    processing_time: float


class RuleBasedReasoningEngine:
    """
    Rule-based reasoning system for game decision making
    
    Capabilities:
    - Context-aware decision making using rules
    - Multi-step reasoning chains
    - Game strategy planning
    - Action selection and validation
    - Fallback planning
    - Pattern recognition and response
    """
    
    def __init__(self):
        self.debugger = get_debugger()
        self.layer_name = "reasoning"
        
        info(self.layer_name, "BRAIN: Initializing Rule-Based Reasoning Engine...")
        
        # Game knowledge and rules
        self.game_rules = self.load_game_rules()
        self.decision_trees = self.load_decision_trees()
        self.priority_rules = self.load_priority_rules()
        
        # Performance tracking
        self.stats = {
            'decisions_made': 0,
            'avg_reasoning_time': 0.0,
            'successful_actions': 0,
            'failed_actions': 0,
            'confidence_scores': [],
            'rule_usage': {}
        }
        
        info(self.layer_name, "SUCCESS: Rule-Based Reasoning Engine initialized successfully")
    
    def load_game_rules(self) -> Dict[str, Any]:
        """Load game-specific rules and strategies"""
        debug(self.layer_name, "Loading game rules and strategies")
        
        return {
            'battle_rules': {
                'type_effectiveness': {
                    'water': {'beats': ['fire', 'ground', 'rock'], 'weak_to': ['grass', 'electric']},
                    'fire': {'beats': ['grass', 'ice', 'bug'], 'weak_to': ['water', 'ground', 'rock']},
                    'grass': {'beats': ['water', 'ground', 'rock'], 'weak_to': ['fire', 'ice', 'flying', 'bug']},
                    'electric': {'beats': ['water', 'flying'], 'weak_to': ['ground']},
                    # Simplified type chart for demo
                },
                'hp_thresholds': {
                    'critical': 0.25,    # < 25% HP is critical
                    'low': 0.5,          # < 50% HP is low
                    'good': 0.75         # > 75% HP is good
                },
                'battle_priorities': [
                    'heal_if_critical',
                    'use_type_advantage',
                    'switch_if_disadvantaged',
                    'attack_with_best_move',
                    'defend_if_unsure'
                ]
            },
            
            'exploration_rules': {
                'movement_priorities': [
                    'talk_to_npcs',
                    'collect_items',
                    'explore_new_areas',
                    'avoid_unnecessary_battles',
                    'return_to_pokemon_center_when_weak'
                ],
                'interaction_rules': [
                    'examine_suspicious_objects',
                    'talk_to_all_npcs_once',
                    'check_for_hidden_items',
                    'read_all_signs'
                ]
            },
            
            'dialogue_rules': {
                'response_strategies': [
                    'choose_polite_responses',
                    'gather_information_first',
                    'accept_helpful_offers',
                    'decline_risky_propositions'
                ],
                'information_priority': [
                    'quest_information',
                    'item_locations',
                    'battle_tips',
                    'story_progression'
                ]
            },
            
            'menu_rules': {
                'navigation_efficiency': [
                    'use_shortcuts_when_available',
                    'organize_items_logically',
                    'check_pokemon_status_regularly',
                    'save_progress_frequently'
                ]
            }
        }
    
    def load_decision_trees(self) -> Dict[str, Dict]:
        """Load decision trees for different game contexts"""
        debug(self.layer_name, "Loading decision trees")
        
        return {
            'battle_decision_tree': {
                'root': {
                    'condition': 'in_battle',
                    'true_path': {
                        'condition': 'hp_critical',
                        'true_path': {'action': 'use_healing_item', 'priority': 10},
                        'false_path': {
                            'condition': 'type_disadvantage',
                            'true_path': {'action': 'switch_pokemon', 'priority': 8},
                            'false_path': {
                                'condition': 'type_advantage',
                                'true_path': {'action': 'attack_with_advantage', 'priority': 9},
                                'false_path': {'action': 'attack_normally', 'priority': 6}
                            }
                        }
                    },
                    'false_path': {'action': 'wait', 'priority': 1}
                }
            },
            
            'exploration_decision_tree': {
                'root': {
                    'condition': 'exploring',
                    'true_path': {
                        'condition': 'npc_nearby',
                        'true_path': {'action': 'talk_to_npc', 'priority': 8},
                        'false_path': {
                            'condition': 'item_visible',
                            'true_path': {'action': 'collect_item', 'priority': 7},
                            'false_path': {
                                'condition': 'new_area_accessible',
                                'true_path': {'action': 'explore_new_area', 'priority': 6},
                                'false_path': {'action': 'move_randomly', 'priority': 3}
                            }
                        }
                    },
                    'false_path': {'action': 'wait', 'priority': 1}
                }
            },
            
            'dialogue_decision_tree': {
                'root': {
                    'condition': 'in_dialogue',
                    'true_path': {
                        'condition': 'yes_no_choice',
                        'true_path': {
                            'condition': 'beneficial_choice',
                            'true_path': {'action': 'respond_yes', 'priority': 8},
                            'false_path': {'action': 'respond_no', 'priority': 7}
                        },
                        'false_path': {'action': 'continue_dialogue', 'priority': 6}
                    },
                    'false_path': {'action': 'wait', 'priority': 1}
                }
            }
        }
    
    def load_priority_rules(self) -> Dict[str, List]:
        """Load priority rules for action selection"""
        debug(self.layer_name, "Loading priority rules")
        
        return {
            'action_priorities': {
                'emergency': ['heal_critical', 'escape_danger', 'save_game'],
                'battle': ['heal_if_needed', 'use_advantage', 'attack', 'defend'],
                'exploration': ['talk', 'collect', 'examine', 'move'],
                'dialogue': ['gather_info', 'make_choice', 'continue'],
                'menu': ['heal_pokemon', 'organize_items', 'save']
            },
            
            'safety_rules': [
                'never_enter_battle_with_fainted_pokemon',
                'heal_before_major_battles',
                'save_before_important_decisions',
                'dont_waste_rare_items'
            ]
        }
    
    @monitor_performance("reasoning", "decision_making")
    def make_decision(self, 
                     perception: GamePerception, 
                     text_analysis: TextAnalysis,
                     current_state: Optional[str] = None) -> GameDecision:
        """
        Make intelligent game decision using rule-based reasoning
        
        Args:
            perception: Visual perception data
            text_analysis: Text understanding results
            current_state: Optional explicit state override
            
        Returns:
            GameDecision with full reasoning chain
        """
        start_time = time.time()
        
        debug(self.layer_name, "Starting rule-based decision-making process")
        
        try:
            # 1. Create unified game context
            game_context = self._create_game_context(perception, text_analysis, current_state)
            
            # 2. Generate reasoning chain using rules
            reasoning_chain = self._generate_rule_based_reasoning(game_context)
            
            # 3. Apply decision tree logic
            decision = self._apply_decision_tree(game_context, reasoning_chain)
            
            # 4. Validate and finalize decision
            validated_decision = self._validate_decision(decision, game_context)
            
            processing_time = time.time() - start_time
            validated_decision.processing_time = processing_time
            
            # Update statistics
            self._update_stats(validated_decision)
            
            info(self.layer_name, f"Decision made: {validated_decision.primary_action}", 
                 details={'confidence': validated_decision.confidence, 
                         'processing_time': processing_time,
                         'reasoning_steps': len(validated_decision.reasoning_chain)})
            
            return validated_decision
            
        except Exception as e:
            error(self.layer_name, "Decision making failed", exception=e)
            return self._create_fallback_decision()
    
    def _create_game_context(self, 
                           perception: GamePerception, 
                           text_analysis: TextAnalysis,
                           current_state: Optional[str]) -> GameContext:
        """Create unified game context from all layers"""
        debug(self.layer_name, "Creating unified game context")
        
        # Determine current state if not provided
        if current_state is None:
            current_state = self._infer_game_state(perception, text_analysis)
        
        # Extract key entities
        key_entities = self._extract_key_entities(perception, text_analysis)
        
        # Determine available actions
        available_actions = self._determine_available_actions(perception, text_analysis, current_state)
        
        # Create spatial information
        spatial_info = self._create_spatial_info(perception)
        
        # Calculate overall confidence
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
        
        debug(self.layer_name, f"Game context created: state={current_state}, entities={len(key_entities)}")
        return context
    
    def _infer_game_state(self, perception: GamePerception, text_analysis: TextAnalysis) -> str:
        """Infer current game state from available data"""
        debug(self.layer_name, "Inferring game state")
        
        # Check text analysis for context clues
        if text_analysis.intents:
            intent_contexts = [intent.context_category for intent in text_analysis.intents]
            if 'battle' in intent_contexts:
                return 'battle'
            elif 'dialogue' in intent_contexts:
                return 'dialogue'
            elif 'menu' in intent_contexts:
                return 'menu'
        
        # Check for battle-specific UI elements
        if text_analysis.ui_elements:
            ui_lower = [elem.lower() for elem in text_analysis.ui_elements]
            if any(elem in ui_lower for elem in ['fight', 'pokemon', 'bag', 'run']):
                return 'battle'
            elif any(elem in ui_lower for elem in ['yes', 'no']):
                return 'dialogue'
        
        # Check game commands
        if text_analysis.game_commands:
            if any(cmd in text_analysis.game_commands for cmd in ['fight', 'attack']):
                return 'battle'
        
        # Check for battle text patterns
        for text in text_analysis.dialogue_content:
            text_lower = text.lower()
            if any(word in text_lower for word in ['used', 'attack', 'effective', 'fainted', 'hp']):
                return 'battle'
        
        # Default to exploration if no specific context found
        return 'exploration'
    
    def _extract_key_entities(self, perception: GamePerception, text_analysis: TextAnalysis) -> List[str]:
        """Extract most important entities from all data"""
        entities = []
        
        # From text analysis intents
        for intent in text_analysis.intents:
            if intent.entities:
                for entity_type, entity_list in intent.entities.items():
                    entities.extend(entity_list)
        
        # From perception objects
        for obj in perception.detected_objects:
            if hasattr(obj, 'object_type') and obj.object_type:
                entities.append(obj.object_type)
        
        # From UI elements and commands
        entities.extend(text_analysis.ui_elements)
        entities.extend(text_analysis.game_commands)
        
        # Remove duplicates and return top entities
        unique_entities = list(set(entities))
        return unique_entities[:10]  # Limit to most important
    
    def _determine_available_actions(self, 
                                   perception: GamePerception, 
                                   text_analysis: TextAnalysis,
                                   current_state: str) -> List[str]:
        """Determine what actions are available in current context"""
        actions = ['wait', 'observe']  # Always available base actions
        
        # State-specific actions
        if current_state == 'battle':
            if any(elem.lower() == 'fight' for elem in text_analysis.ui_elements):
                actions.append('select_fight')
            if any(elem.lower() == 'pokemon' for elem in text_analysis.ui_elements):
                actions.append('select_pokemon')
            if any(elem.lower() == 'bag' for elem in text_analysis.ui_elements):
                actions.append('select_bag')
            if any(elem.lower() == 'run' for elem in text_analysis.ui_elements):
                actions.append('select_run')
        
        elif current_state == 'dialogue':
            if any(resp.lower() in ['yes', 'no'] for resp in text_analysis.ui_elements):
                actions.extend(['respond_yes', 'respond_no'])
            actions.append('continue_dialogue')
        
        elif current_state == 'menu':
            # Add menu-specific actions based on detected UI elements
            for element in text_analysis.ui_elements:
                actions.append(f'select_{element.lower()}')
        
        elif current_state == 'exploration':
            actions.extend(['move_up', 'move_down', 'move_left', 'move_right', 
                           'interact', 'open_menu', 'move_randomly', 'explore_new_area', 
                           'explore_area'])
        
        return actions
    
    def _create_spatial_info(self, perception: GamePerception) -> Dict[str, Any]:
        """Create spatial information from perception data"""
        spatial = {
            'screen_regions': {'top': [], 'middle': [], 'bottom': []},
            'object_positions': [],
            'text_positions': []
        }
        
        # Categorize by screen region (assuming standard screen height)
        for obj in perception.detected_objects:
            y_center = getattr(obj, 'center', (0, 0))[1]
            if y_center < 150:
                spatial['screen_regions']['top'].append(obj)
            elif y_center < 350:
                spatial['screen_regions']['middle'].append(obj)
            else:
                spatial['screen_regions']['bottom'].append(obj)
            
            spatial['object_positions'].append({
                'object': getattr(obj, 'object_type', 'unknown'),
                'position': getattr(obj, 'center', (0, 0)),
                'confidence': getattr(obj, 'confidence', 0.0)
            })
        
        # Add text positions
        for text in perception.detected_text:
            spatial['text_positions'].append({
                'text': text.text,
                'position': text.center,
                'confidence': text.confidence
            })
        
        return spatial
    
    def _generate_rule_based_reasoning(self, game_context: GameContext) -> List[ReasoningStep]:
        """Generate reasoning chain using rule-based logic"""
        debug(self.layer_name, "Generating rule-based reasoning chain")
        
        reasoning_steps = []
        
        # Step 1: Situation Analysis
        analysis_step = self._analyze_situation(game_context)
        reasoning_steps.append(analysis_step)
        
        # Step 2: Rule Application
        rule_step = self._apply_relevant_rules(game_context)
        reasoning_steps.append(rule_step)
        
        # Step 3: Option Evaluation
        evaluation_step = self._evaluate_options(game_context)
        reasoning_steps.append(evaluation_step)
        
        # Step 4: Decision Justification
        justification_step = self._justify_decision(game_context, reasoning_steps)
        reasoning_steps.append(justification_step)
        
        debug(self.layer_name, f"Generated {len(reasoning_steps)} reasoning steps")
        return reasoning_steps
    
    def _analyze_situation(self, game_context: GameContext) -> ReasoningStep:
        """Analyze the current situation"""
        evidence = []
        
        # Analyze current state
        evidence.append(f"Current state: {game_context.current_state}")
        evidence.append(f"Available actions: {len(game_context.available_actions)}")
        evidence.append(f"Key entities: {', '.join(game_context.key_entities[:5])}")
        
        # Analyze text content
        if game_context.text_analysis.intents:
            intent_types = [intent.intent_type for intent in game_context.text_analysis.intents]
            evidence.append(f"Intent types: {', '.join(set(intent_types))}")
        
        # Analyze HP status if available
        if game_context.text_analysis.stats_info.get('hp_values'):
            hp_info = game_context.text_analysis.stats_info['hp_values'][0]
            if 'max' in hp_info:
                hp_ratio = hp_info['current'] / hp_info['max']
                hp_status = 'critical' if hp_ratio < 0.25 else 'low' if hp_ratio < 0.5 else 'good'
                evidence.append(f"HP status: {hp_status} ({hp_info['current']}/{hp_info['max']})")
        
        content = f"Analyzing {game_context.current_state} situation with {len(evidence)} key factors."
        
        return ReasoningStep(
            step_type='analysis',
            content=content,
            confidence=0.9,
            evidence=evidence,
            alternatives=[]
        )
    
    def _apply_relevant_rules(self, game_context: GameContext) -> ReasoningStep:
        """Apply relevant rules based on current context"""
        applicable_rules = []
        
        state = game_context.current_state
        if state in self.game_rules:
            state_rules = self.game_rules[f"{state}_rules"]
            
            if state == 'battle':
                # Check HP rules
                if game_context.text_analysis.stats_info.get('hp_values'):
                    hp_info = game_context.text_analysis.stats_info['hp_values'][0]
                    if 'max' in hp_info:
                        hp_ratio = hp_info['current'] / hp_info['max']
                        if hp_ratio < state_rules['hp_thresholds']['critical']:
                            applicable_rules.append('heal_if_critical')
                        elif hp_ratio < state_rules['hp_thresholds']['low']:
                            applicable_rules.append('consider_healing')
                
                # Check for type advantage opportunities
                pokemon_entities = []
                for intent in game_context.text_analysis.intents:
                    if intent.entities.get('pokemon'):
                        pokemon_entities.extend(intent.entities['pokemon'])
                
                if pokemon_entities:
                    applicable_rules.append('consider_type_advantage')
            
            elif state == 'exploration':
                # Check for NPCs or interactive elements
                if any('npc' in entity.lower() for entity in game_context.key_entities):
                    applicable_rules.append('talk_to_npcs')
                
                if any('item' in entity.lower() for entity in game_context.key_entities):
                    applicable_rules.append('collect_items')
            
            elif state == 'dialogue':
                # Check for choice options
                if any(elem.lower() in ['yes', 'no'] for elem in game_context.text_analysis.ui_elements):
                    applicable_rules.append('evaluate_yes_no_choice')
        
        content = f"Applied {len(applicable_rules)} relevant rules for {state} context."
        
        return ReasoningStep(
            step_type='planning',
            content=content,
            confidence=0.8,
            evidence=applicable_rules,
            alternatives=[]
        )
    
    def _evaluate_options(self, game_context: GameContext) -> ReasoningStep:
        """Evaluate available options"""
        options = game_context.available_actions
        evaluations = []
        
        for action in options[:5]:  # Evaluate top 5 options
            score = self._score_action(action, game_context)
            evaluations.append(f"{action}: {score:.2f}")
        
        # Find best option
        best_action = max(options, key=lambda a: self._score_action(a, game_context))
        
        content = f"Evaluated {len(options)} options, best: {best_action}"
        
        return ReasoningStep(
            step_type='decision',
            content=content,
            confidence=0.85,
            evidence=evaluations,
            alternatives=options[1:4]  # Alternative options
        )
    
    def _score_action(self, action: str, game_context: GameContext) -> float:
        """Score an action based on current context"""
        score = 0.5  # Base score
        
        state = game_context.current_state
        
        # State-specific scoring
        if state == 'battle':
            if 'fight' in action.lower() and 'fight' in game_context.text_analysis.ui_elements:
                score += 0.4
            elif 'bag' in action.lower() and self._needs_healing(game_context):
                score += 0.6  # Higher score for healing when needed
            elif 'run' in action.lower() and self._should_escape(game_context):
                score += 0.5
        
        elif state == 'dialogue':
            if 'yes' in action.lower() and self._is_beneficial_choice(game_context):
                score += 0.4
            elif 'continue' in action.lower():
                score += 0.3
        
        elif state == 'exploration':
            if 'interact' in action.lower() and game_context.key_entities:
                score += 0.3
            elif 'move' in action.lower():
                score += 0.2
        
        # Safety bonuses
        if action in ['wait', 'observe']:
            score += 0.1  # Safe fallback always gets some points
        
        return min(1.0, score)
    
    def _needs_healing(self, game_context: GameContext) -> bool:
        """Check if healing is needed"""
        if game_context.text_analysis.stats_info.get('hp_values'):
            hp_info = game_context.text_analysis.stats_info['hp_values'][0]
            if 'max' in hp_info:
                hp_ratio = hp_info['current'] / hp_info['max']
                return hp_ratio < 0.5
        return False
    
    def _should_escape(self, game_context: GameContext) -> bool:
        """Check if escaping from battle is advisable"""
        # Simple heuristic: escape if HP is very low
        if game_context.text_analysis.stats_info.get('hp_values'):
            hp_info = game_context.text_analysis.stats_info['hp_values'][0]
            if 'max' in hp_info:
                hp_ratio = hp_info['current'] / hp_info['max']
                return hp_ratio < 0.25
        return False
    
    def _is_beneficial_choice(self, game_context: GameContext) -> bool:
        """Determine if 'yes' would be beneficial in dialogue"""
        # Simple heuristic based on dialogue content
        dialogue_text = ' '.join(game_context.text_analysis.dialogue_content).lower()
        
        beneficial_keywords = ['help', 'heal', 'item', 'gift', 'reward', 'teach']
        risky_keywords = ['danger', 'battle', 'fight', 'lose', 'risk']
        
        beneficial_count = sum(1 for keyword in beneficial_keywords if keyword in dialogue_text)
        risky_count = sum(1 for keyword in risky_keywords if keyword in dialogue_text)
        
        return beneficial_count > risky_count
    
    def _justify_decision(self, game_context: GameContext, reasoning_steps: List[ReasoningStep]) -> ReasoningStep:
        """Create final justification for the decision"""
        
        # Extract key points from previous steps
        key_points = []
        for step in reasoning_steps:
            if step.evidence:
                key_points.extend(step.evidence[:2])  # Take top 2 pieces of evidence
        
        content = f"Decision justified by {len(key_points)} key factors including game state analysis and rule application."
        
        return ReasoningStep(
            step_type='validation',
            content=content,
            confidence=0.8,
            evidence=key_points[:5],  # Top 5 justifications
            alternatives=[]
        )
    
    def _apply_decision_tree(self, game_context: GameContext, reasoning_chain: List[ReasoningStep]) -> GameDecision:
        """Apply decision tree logic to make final decision"""
        debug(self.layer_name, "Applying decision tree logic")
        
        state = game_context.current_state
        tree_name = f"{state}_decision_tree"
        
        # Get decision tree for current state
        if tree_name in self.decision_trees:
            decision_tree = self.decision_trees[tree_name]
            decision_result = self._traverse_decision_tree(decision_tree['root'], game_context)
        else:
            # Fallback decision
            decision_result = {'action': 'wait', 'priority': 1}
        
        # Create decision object
        decision = GameDecision(
            timestamp=time.time(),
            decision_type=self._classify_decision_type(decision_result['action']),
            primary_action=decision_result['action'],
            action_parameters=self._get_action_parameters(decision_result['action'], game_context),
            reasoning_chain=reasoning_chain,
            confidence=self._calculate_confidence(decision_result, game_context),
            fallback_actions=self._get_fallback_actions(decision_result['action'], game_context),
            expected_outcome=self._predict_outcome(decision_result['action'], game_context),
            processing_time=0.0  # Will be set by caller
        )
        
        debug(self.layer_name, f"Decision tree result: {decision_result['action']} (priority: {decision_result['priority']})")
        return decision
    
    def _traverse_decision_tree(self, node: Dict, game_context: GameContext) -> Dict:
        """Traverse decision tree to find best action"""
        
        if 'action' in node:
            # Leaf node - return the action
            return node
        
        if 'condition' in node:
            # Evaluate condition
            condition_met = self._evaluate_condition(node['condition'], game_context)
            
            # Choose path based on condition
            if condition_met and 'true_path' in node:
                return self._traverse_decision_tree(node['true_path'], game_context)
            elif not condition_met and 'false_path' in node:
                return self._traverse_decision_tree(node['false_path'], game_context)
        
        # Fallback
        return {'action': 'wait', 'priority': 1}
    
    def _evaluate_condition(self, condition: str, game_context: GameContext) -> bool:
        """Evaluate a decision tree condition"""
        
        if condition == 'in_battle':
            return game_context.current_state == 'battle'
        elif condition == 'exploring':
            return game_context.current_state == 'exploration'
        elif condition == 'in_dialogue':
            return game_context.current_state == 'dialogue'
        elif condition == 'hp_critical':
            return self._needs_healing(game_context) and self._get_hp_ratio(game_context) < 0.25
        elif condition == 'type_disadvantage':
            return False  # Simplified - would need type analysis
        elif condition == 'type_advantage':
            return False  # Simplified - would need type analysis
        elif condition == 'npc_nearby':
            return any('npc' in entity.lower() for entity in game_context.key_entities)
        elif condition == 'item_visible':
            return any('item' in entity.lower() for entity in game_context.key_entities)
        elif condition == 'new_area_accessible':
            return False  # Would need map analysis
        elif condition == 'yes_no_choice':
            return any(elem.lower() in ['yes', 'no'] for elem in game_context.text_analysis.ui_elements)
        elif condition == 'beneficial_choice':
            return self._is_beneficial_choice(game_context)
        
        return False  # Default to false for unknown conditions
    
    def _get_hp_ratio(self, game_context: GameContext) -> float:
        """Get current HP ratio"""
        if game_context.text_analysis.stats_info.get('hp_values'):
            hp_info = game_context.text_analysis.stats_info['hp_values'][0]
            if 'max' in hp_info and hp_info['max'] > 0:
                return hp_info['current'] / hp_info['max']
        return 1.0  # Assume full HP if unknown
    
    def _classify_decision_type(self, action: str) -> str:
        """Classify decision type"""
        if 'wait' in action.lower() or 'observe' in action.lower():
            return 'wait'
        elif 'respond' in action.lower():
            return 'dialogue_response'
        elif 'move' in action.lower() or 'explore' in action.lower():
            return 'explore'
        else:
            return 'action'
    
    def _get_action_parameters(self, action: str, game_context: GameContext) -> Dict[str, Any]:
        """Get parameters for the action"""
        params = {'context': game_context.current_state}
        
        if game_context.current_state == 'battle':
            params['battle_context'] = True
            if game_context.text_analysis.stats_info:
                params['stats'] = game_context.text_analysis.stats_info
        
        return params
    
    def _calculate_confidence(self, decision_result: Dict, game_context: GameContext) -> float:
        """Calculate confidence in the decision"""
        base_confidence = 0.7
        
        # Boost confidence based on priority
        priority_boost = min(0.2, decision_result.get('priority', 1) * 0.02)
        
        # Boost confidence based on context quality
        context_boost = game_context.confidence * 0.2
        
        # Boost confidence if we have clear UI elements
        ui_boost = 0.1 if game_context.text_analysis.ui_elements else 0.0
        
        total_confidence = min(1.0, base_confidence + priority_boost + context_boost + ui_boost)
        return total_confidence
    
    def _get_fallback_actions(self, primary_action: str, game_context: GameContext) -> List[str]:
        """Get fallback actions"""
        fallbacks = ['wait', 'observe']
        
        # Add context-appropriate fallbacks
        available = game_context.available_actions
        other_actions = [a for a in available if a != primary_action]
        
        fallbacks.extend(other_actions[:3])
        return fallbacks
    
    def _predict_outcome(self, action: str, game_context: GameContext) -> str:
        """Predict expected outcome"""
        action_lower = action.lower()
        
        if 'fight' in action_lower or 'attack' in action_lower:
            return "Execute battle action"
        elif 'heal' in action_lower or 'bag' in action_lower:
            return "Use healing item or access bag"
        elif 'run' in action_lower:
            return "Attempt to escape from battle"
        elif 'respond_yes' in action_lower:
            return "Give positive response to dialogue"
        elif 'respond_no' in action_lower:
            return "Give negative response to dialogue"
        elif 'move' in action_lower:
            return "Move character in specified direction"
        elif 'wait' in action_lower:
            return "Wait and observe for new information"
        else:
            return f"Execute {action} action"
    
    def _validate_decision(self, decision: GameDecision, game_context: GameContext) -> GameDecision:
        """Validate and potentially modify decision"""
        debug(self.layer_name, "Validating decision")
        
        # Ensure action is available
        if decision.primary_action not in game_context.available_actions:
            warning(self.layer_name, f"Action {decision.primary_action} not available, using fallback")
            
            for fallback in decision.fallback_actions:
                if fallback in game_context.available_actions:
                    decision.primary_action = fallback
                    decision.confidence *= 0.8
                    break
        
        # Ensure minimum confidence
        if decision.confidence < 0.3:
            decision.confidence = 0.3
            decision.primary_action = 'wait'
        
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
                    content='Using safe fallback due to error',
                    confidence=0.4,
                    evidence=[],
                    alternatives=[]
                )
            ],
            confidence=0.4,
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
    
    def create_decision_summary(self, decision: GameDecision) -> str:
        """Create human-readable summary of decision"""
        summary = [
            f"Decision: {decision.primary_action}",
            f"Type: {decision.decision_type}", 
            f"Confidence: {decision.confidence:.2f}",
            f"Processing: {decision.processing_time:.3f}s"
        ]
        
        if decision.reasoning_chain:
            reasoning_types = [step.step_type for step in decision.reasoning_chain]
            summary.append(f"Reasoning: {', '.join(set(reasoning_types))}")
        
        return " | ".join(summary)


# Testing and example usage
if __name__ == "__main__":
    # Initialize debugging
    from debug_system import init_debugging
    init_debugging(log_level="DEBUG")
    
    print("BRAIN: Testing Rule-Based Reasoning Engine...")
    
    try:
        # Initialize reasoning engine
        reasoning_engine = RuleBasedReasoningEngine()
        
        # Create mock data (same as before)
        from perception_engine import GamePerception, DetectedObject, DetectedText
        from text_analyzer import TextAnalysis, TextIntent
        
        # Mock perception data
        mock_perception = GamePerception(
            timestamp=time.time(),
            detected_objects=[
                DetectedObject(
                    object_type="pokemon",
                    confidence=0.9,
                    bbox=(100, 100, 200, 200),
                    center=(150, 150)
                )
            ],
            detected_text=[
                DetectedText(
                    text="FIGHT",
                    confidence=0.95,
                    bbox=(50, 400, 120, 430),
                    center=(85, 415),
                    is_menu_item=True
                )
            ],
            game_context="battle",
            dominant_colors=["red", "blue"],
            screen_regions={},
            confidence=0.85,
            processing_time=0.1
        )
        
        # Mock text analysis data
        mock_text_analysis = TextAnalysis(
            timestamp=time.time(),
            original_texts=[],
            processed_texts=["fight"],
            combined_context="BOTTOM: fight",
            intents=[
                TextIntent(
                    intent_type="menu_option",
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
            stats_info={'hp_values': [{'current': 75, 'max': 100}]},
            confidence=0.8,
            processing_time=0.1
        )
        
        # Make decision
        decision = reasoning_engine.make_decision(mock_perception, mock_text_analysis)
        
        # Display results
        print(f"\nRESULTS: Decision Results:")
        print(f"   Primary action: {decision.primary_action}")
        print(f"   Decision type: {decision.decision_type}")
        print(f"   Confidence: {decision.confidence:.3f}")
        print(f"   Processing time: {decision.processing_time:.3f}s")
        print(f"   Reasoning steps: {len(decision.reasoning_chain)}")
        
        # Show reasoning chain
        if decision.reasoning_chain:
            print(f"\nREASONING: Reasoning Chain:")
            for i, step in enumerate(decision.reasoning_chain):
                print(f"   {i+1}. {step.step_type.upper()}: {step.content}")
                if step.evidence:
                    print(f"      Evidence: {', '.join(step.evidence[:3])}")
        
        # Show decision summary
        summary = reasoning_engine.create_decision_summary(decision)
        print(f"\nSUMMARY: {summary}")
        
        # Show performance stats
        stats = reasoning_engine.get_stats()
        print(f"\nPERF: Performance: {stats}")
        
        print("\nSUCCESS: Rule-Based Reasoning Engine test completed!")
        
    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()