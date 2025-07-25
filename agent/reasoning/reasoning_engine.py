"""
BRAIN: Reasoning Engine - Modular AI Game Agent
==========================================

Intelligent decision-making layer using Mistral 7B LLM.
Processes perception and text understanding to make informed game decisions.

Integrated with comprehensive debugging system for error tracing.
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import re

# LLM integration
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Import our debugging system
from ..debug_system import (
    get_debugger,
    monitor_performance,
    info,
    debug,
    warning,
    error,
)

# Import other layer data structures
from ..perception import GamePerception, DetectedObject, DetectedText
from ..nlp import TextAnalysis, TextIntent


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


class ReasoningEngine:
    """
    Advanced reasoning system using Mistral 7B LLM
    
    Capabilities:
    - Context-aware decision making
    - Multi-step reasoning chains
    - Game strategy planning
    - Action selection and validation
    - Learning from outcomes
    - Fallback planning
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 max_context_length: int = 1024,
                 temperature: float = 0.7,
                 use_gpu: bool = True):
        
        self.debugger = get_debugger()
        self.layer_name = "reasoning"
        
        info(self.layer_name, "BRAIN: Initializing Reasoning Engine...")
        
        # LLM configuration
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.temperature = temperature
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize LLM
        self.tokenizer = None
        self.model = None
        self.initialize_llm()
        
        # Game knowledge and prompts
        self.game_knowledge = self.load_game_knowledge()
        self.reasoning_prompts = self.load_reasoning_prompts()
        
        # Performance tracking
        self.stats = {
            'decisions_made': 0,
            'avg_reasoning_time': 0.0,
            'successful_actions': 0,
            'failed_actions': 0,
            'confidence_scores': []
        }
        
        info(self.layer_name, "SUCCESS: Reasoning Engine initialized successfully")
    
    @monitor_performance("reasoning", "llm_initialization")
    def initialize_llm(self):
        """Initialize Mistral 7B model with error handling"""
        try:
            debug(self.layer_name, f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            debug(self.layer_name, f"Loading model: {self.model_name}")
            device_map = "auto" if self.use_gpu else "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                trust_remote_code=True
            )
            
            info(self.layer_name, f"SUCCESS: LLM loaded: {self.model_name} on {'GPU' if self.use_gpu else 'CPU'}")
            
        except Exception as e:
            error(self.layer_name, "Failed to initialize LLM", exception=e)
            raise
    
    def load_game_knowledge(self) -> Dict[str, Any]:
        """Load game-specific knowledge base"""
        debug(self.layer_name, "Loading game knowledge base")
        
        return {
            'pokemon_game_rules': {
                'battle_mechanics': [
                    "Type effectiveness matters (water beats fire, etc.)",
                    "HP reaching 0 means Pokemon faints",
                    "Status effects like paralysis affect battle",
                    "Critical hits do extra damage",
                    "Speed determines move order"
                ],
                'exploration_rules': [
                    "Talk to NPCs for information and items",
                    "Check objects for hidden items",
                    "Wild Pokemon appear in grass",
                    "Gyms require specific badges to enter",
                    "Pokemon Centers heal your team"
                ],
                'menu_navigation': [
                    "Fight > select move in battle",
                    "Pokemon > switch active Pokemon", 
                    "Bag > use items",
                    "Run > escape from battle"
                ]
            },
            
            'common_strategies': {
                'battle_strategy': [
                    "Use type advantages when possible",
                    "Switch Pokemon if current one is weak to opponent",
                    "Use healing items when HP is low",
                    "Save powerful moves for tough opponents"
                ],
                'exploration_strategy': [
                    "Talk to everyone for information",
                    "Explore thoroughly before moving to new areas", 
                    "Keep Pokemon healed at Centers",
                    "Train Pokemon to appropriate levels"
                ]
            },
            
            'action_priorities': {
                'battle': ['type_advantage', 'hp_management', 'status_effects', 'move_power'],
                'exploration': ['dialogue', 'item_collection', 'pokemon_encounters', 'navigation'],
                'menu': ['efficiency', 'context_awareness', 'goal_alignment'],
                'dialogue': ['information_gathering', 'story_progression', 'choice_optimization']
            }
        }
    
    def load_reasoning_prompts(self) -> Dict[str, str]:
        """Load reasoning prompt templates"""
        debug(self.layer_name, "Loading reasoning prompt templates")
        
        return {
            'game_decision_template': """You are an expert Pokemon game player making decisions based on screen analysis.

CURRENT GAME STATE:
{game_context}

PERCEPTION DATA:
Objects detected: {objects}
Text detected: {text_content}
UI elements: {ui_elements}

TEXT ANALYSIS:
Intents identified: {intents}
Game commands available: {commands}
Context category: {context_category}

REASONING TASK:
Analyze the current situation and decide on the best action to take. Consider:
1. What is currently happening in the game?
2. What are the available options?
3. What would an expert player do in this situation?
4. What are the potential risks and benefits of each action?

Provide your reasoning in this format:
ANALYSIS: [Describe what you see and understand]
PLANNING: [Consider the available options and strategies]  
DECISION: [Choose the best action with clear reasoning]
VALIDATION: [Explain why this is the optimal choice]

Action to take: [specific action name]
Parameters: [any parameters needed]
Confidence: [0.0-1.0]
""",

            'battle_decision_template': """You are in a Pokemon battle. Make the optimal battle decision.

BATTLE STATE:
Current Pokemon: {current_pokemon}
Opponent Pokemon: {opponent_pokemon}
HP Status: {hp_status}
Available moves: {available_moves}
Items available: {items}

BATTLE ANALYSIS:
{battle_analysis}

Choose the best action:
1. FIGHT - Select which move to use
2. POKEMON - Switch to different Pokemon
3. BAG - Use an item
4. RUN - Attempt to escape

Reasoning: [Explain your tactical decision]
Action: [FIGHT/POKEMON/BAG/RUN]
Target: [specific move/pokemon/item if applicable]
""",

            'dialogue_response_template': """You are responding to game dialogue. Choose the best response.

DIALOGUE CONTEXT:
Current speaker: {speaker}
Dialogue content: {dialogue}
Available responses: {responses}

CONTEXT ANALYSIS:
Story situation: {story_context}
Character relationship: {character_info}
Potential outcomes: {outcomes}

Choose the response that best advances the story or achieves your goals:
Response: [exact text of chosen response]
Reasoning: [why this response is optimal]
"""
        }
    
    @monitor_performance("reasoning", "decision_making")
    def make_decision(self, 
                     perception: GamePerception, 
                     text_analysis: TextAnalysis,
                     current_state: Optional[str] = None) -> GameDecision:
        """
        Make intelligent game decision based on all available information
        
        Args:
            perception: Visual perception data
            text_analysis: Text understanding results
            current_state: Optional explicit state override
            
        Returns:
            GameDecision with full reasoning chain
        """
        start_time = time.time()
        
        debug(self.layer_name, "Starting decision-making process")
        
        try:
            # 1. Create unified game context
            game_context = self._create_game_context(perception, text_analysis, current_state)
            
            # 2. Generate reasoning chain
            reasoning_chain = self._generate_reasoning_chain(game_context)
            
            # 3. Make final decision
            decision = self._finalize_decision(game_context, reasoning_chain)
            
            # 4. Validate decision
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
        
        # Check for UI elements
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
        
        # Default to exploration
        return 'exploration'
    
    def _extract_key_entities(self, perception: GamePerception, text_analysis: TextAnalysis) -> List[str]:
        """Extract most important entities from all data"""
        entities = []
        
        # From text analysis
        for intent in text_analysis.intents:
            if intent.entities:
                for entity_type, entity_list in intent.entities.items():
                    entities.extend(entity_list)
        
        # From perception (object names)
        for obj in perception.detected_objects:
            if hasattr(obj, 'object_class') and obj.object_class:
                entities.append(obj.object_class)
        
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
            if 'fight' in text_analysis.ui_elements:
                actions.append('select_fight')
            if 'pokemon' in text_analysis.ui_elements:
                actions.append('select_pokemon')
            if 'bag' in text_analysis.ui_elements:
                actions.append('select_bag')
            if 'run' in text_analysis.ui_elements:
                actions.append('select_run')
        
        elif current_state == 'dialogue':
            if any(resp in text_analysis.ui_elements for resp in ['yes', 'no']):
                actions.extend(['respond_yes', 'respond_no'])
            actions.append('continue_dialogue')
        
        elif current_state == 'menu':
            # Add menu-specific actions based on detected UI elements
            for element in text_analysis.ui_elements:
                actions.append(f'select_{element.lower()}')
        
        elif current_state == 'exploration':
            actions.extend(['move_up', 'move_down', 'move_left', 'move_right', 
                           'interact', 'open_menu'])
        
        return actions
    
    def _create_spatial_info(self, perception: GamePerception) -> Dict[str, Any]:
        """Create spatial information from perception data"""
        spatial = {
            'screen_regions': {
                'top': [],
                'middle': [],
                'bottom': []
            },
            'object_positions': [],
            'text_positions': []
        }
        
        # Categorize objects by screen region
        for obj in perception.detected_objects:
            y_center = obj.center[1] if hasattr(obj, 'center') else 0
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
    
    def _generate_reasoning_chain(self, game_context: GameContext) -> List[ReasoningStep]:
        """Generate chain of reasoning steps using LLM"""
        debug(self.layer_name, "Generating reasoning chain")
        
        try:
            # Prepare context for LLM
            context_data = self._prepare_llm_context(game_context)
            
            # Generate reasoning using LLM
            reasoning_text = self._query_llm_for_reasoning(context_data)
            
            # Parse reasoning into structured steps
            reasoning_steps = self._parse_reasoning_steps(reasoning_text)
            
            debug(self.layer_name, f"Generated {len(reasoning_steps)} reasoning steps")
            return reasoning_steps
            
        except Exception as e:
            warning(self.layer_name, "LLM reasoning failed, using fallback", details={'error': str(e)})
            return self._create_fallback_reasoning(game_context)
    
    def _prepare_llm_context(self, game_context: GameContext) -> Dict[str, Any]:
        """Prepare context data for LLM prompt"""
        return {
            'game_context': game_context.current_state,
            'objects': [f"{obj.object_class}({obj.confidence:.2f})" 
                       for obj in game_context.perception.detected_objects[:5]],
            'text_content': [text.text for text in game_context.perception.detected_texts[:5]],
            'ui_elements': game_context.text_analysis.ui_elements,
            'intents': [f"{intent.intent_type}({intent.confidence:.2f})" 
                       for intent in game_context.text_analysis.intents[:3]],
            'commands': game_context.text_analysis.game_commands,
            'context_category': game_context.current_state,
            'available_actions': game_context.available_actions
        }
    
    def _query_llm_for_reasoning(self, context_data: Dict[str, Any]) -> str:
        """Query LLM for reasoning based on context"""
        debug(self.layer_name, "Querying LLM for reasoning")
        
        # Select appropriate prompt template
        if context_data['context_category'] == 'battle':
            template = self.reasoning_prompts['battle_decision_template']
        elif context_data['context_category'] == 'dialogue':
            template = self.reasoning_prompts['dialogue_response_template']
        else:
            template = self.reasoning_prompts['game_decision_template']
        
        # Format prompt with context
        prompt = template.format(**context_data)
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                               max_length=self.max_context_length-512)
        
        if self.use_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part (remove prompt)
        generated_text = response[len(prompt):].strip()
        
        debug(self.layer_name, f"LLM generated {len(generated_text)} characters of reasoning")
        return generated_text
    
    def _parse_reasoning_steps(self, reasoning_text: str) -> List[ReasoningStep]:
        """Parse LLM reasoning text into structured steps"""
        steps = []
        
        # Parse different sections of reasoning
        sections = {
            'ANALYSIS': 'analysis',
            'PLANNING': 'planning', 
            'DECISION': 'decision',
            'VALIDATION': 'validation'
        }
        
        current_section = None
        current_content = []
        
        for line in reasoning_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new section
            section_found = False
            for section_key, section_type in sections.items():
                if line.upper().startswith(section_key + ':'):
                    # Save previous section if exists
                    if current_section and current_content:
                        steps.append(ReasoningStep(
                            step_type=current_section,
                            content=' '.join(current_content),
                            confidence=0.8,  # Default confidence
                            evidence=[],
                            alternatives=[]
                        ))
                    
                    # Start new section
                    current_section = section_type
                    current_content = [line[len(section_key)+1:].strip()]
                    section_found = True
                    break
            
            if not section_found and current_section:
                current_content.append(line)
        
        # Add final section
        if current_section and current_content:
            steps.append(ReasoningStep(
                step_type=current_section,
                content=' '.join(current_content),
                confidence=0.8,
                evidence=[],
                alternatives=[]
            ))
        
        return steps
    
    def _create_fallback_reasoning(self, game_context: GameContext) -> List[ReasoningStep]:
        """Create basic reasoning when LLM fails"""
        return [
            ReasoningStep(
                step_type='analysis',
                content=f"Current game state: {game_context.current_state}",
                confidence=0.6,
                evidence=[f"UI elements: {game_context.text_analysis.ui_elements}"],
                alternatives=[]
            ),
            ReasoningStep(
                step_type='decision',
                content="Taking safe default action based on available options",
                confidence=0.5,
                evidence=[f"Available actions: {game_context.available_actions}"],
                alternatives=[]
            )
        ]
    
    def _finalize_decision(self, game_context: GameContext, reasoning_chain: List[ReasoningStep]) -> GameDecision:
        """Make final decision based on reasoning chain"""
        debug(self.layer_name, "Finalizing decision")
        
        # Extract action from reasoning chain
        primary_action = self._extract_action_from_reasoning(reasoning_chain, game_context)
        
        # Determine decision type
        decision_type = self._classify_decision_type(primary_action, game_context)
        
        # Set action parameters
        action_parameters = self._determine_action_parameters(primary_action, game_context)
        
        # Generate fallback actions
        fallback_actions = self._generate_fallback_actions(primary_action, game_context)
        
        # Calculate confidence
        confidence = self._calculate_decision_confidence(reasoning_chain, game_context)
        
        # Predict expected outcome
        expected_outcome = self._predict_outcome(primary_action, game_context)
        
        decision = GameDecision(
            timestamp=time.time(),
            decision_type=decision_type,
            primary_action=primary_action,
            action_parameters=action_parameters,
            reasoning_chain=reasoning_chain,
            confidence=confidence,
            fallback_actions=fallback_actions,
            expected_outcome=expected_outcome,
            processing_time=0.0  # Will be set by caller
        )
        
        debug(self.layer_name, f"Decision finalized: {primary_action} (confidence: {confidence:.2f})")
        return decision
    
    def _extract_action_from_reasoning(self, reasoning_chain: List[ReasoningStep], game_context: GameContext) -> str:
        """Extract primary action from reasoning chain"""
        
        # Look for explicit action statements in decision steps
        for step in reasoning_chain:
            if step.step_type == 'decision':
                # Look for action patterns in the content
                content_lower = step.content.lower()
                
                # Check available actions
                for action in game_context.available_actions:
                    if action.lower() in content_lower:
                        return action
                
                # Check common action keywords
                action_keywords = ['fight', 'attack', 'run', 'use', 'select', 'move', 'talk', 'wait']
                for keyword in action_keywords:
                    if keyword in content_lower:
                        # Try to find corresponding available action
                        for action in game_context.available_actions:
                            if keyword in action.lower():
                                return action
                        return keyword
        
        # Fallback to first available action
        if game_context.available_actions:
            return game_context.available_actions[0]
        
        return 'wait'
    
    def _classify_decision_type(self, action: str, game_context: GameContext) -> str:
        """Classify the type of decision being made"""
        action_lower = action.lower()
        
        if any(word in action_lower for word in ['wait', 'observe', 'pause']):
            return 'wait'
        elif any(word in action_lower for word in ['move', 'walk', 'go']):
            return 'action'
        elif any(word in action_lower for word in ['respond', 'reply', 'answer']):
            return 'dialogue_response'
        else:
            return 'action'
    
    def _determine_action_parameters(self, action: str, game_context: GameContext) -> Dict[str, Any]:
        """Determine parameters needed for the action"""
        parameters = {}
        
        # Add context-specific parameters
        if game_context.current_state == 'battle':
            parameters['battle_context'] = True
            if game_context.text_analysis.stats_info:
                parameters['stats'] = game_context.text_analysis.stats_info
        
        elif game_context.current_state == 'dialogue':
            parameters['dialogue_context'] = True
            if game_context.text_analysis.dialogue_content:
                parameters['dialogue_options'] = game_context.text_analysis.dialogue_content
        
        # Add spatial parameters if relevant
        if 'move' in action.lower():
            parameters['spatial_info'] = game_context.spatial_info
        
        return parameters
    
    def _generate_fallback_actions(self, primary_action: str, game_context: GameContext) -> List[str]:
        """Generate backup actions if primary action fails"""
        fallbacks = ['wait', 'observe']
        
        # Add state-specific fallbacks
        available = game_context.available_actions
        
        # Remove primary action from available options
        other_actions = [a for a in available if a != primary_action]
        
        # Add top alternatives
        fallbacks.extend(other_actions[:3])
        
        return fallbacks
    
    def _calculate_decision_confidence(self, reasoning_chain: List[ReasoningStep], game_context: GameContext) -> float:
        """Calculate confidence in the decision"""
        if not reasoning_chain:
            return 0.3
        
        # Average reasoning step confidence
        step_confidence = sum(step.confidence for step in reasoning_chain) / len(reasoning_chain)
        
        # Boost confidence based on context quality
        context_boost = game_context.confidence * 0.3
        
        # Boost confidence based on available information
        info_boost = 0.0
        if game_context.text_analysis.ui_elements:
            info_boost += 0.1
        if game_context.text_analysis.game_commands:
            info_boost += 0.1
        if game_context.perception.detected_objects:
            info_boost += 0.1
        
        total_confidence = min(1.0, step_confidence + context_boost + info_boost)
        return total_confidence
    
    def _predict_outcome(self, action: str, game_context: GameContext) -> str:
        """Predict expected outcome of the action"""
        action_lower = action.lower()
        
        if 'fight' in action_lower or 'attack' in action_lower:
            return "Battle action will be executed"
        elif 'run' in action_lower:
            return "Attempt to escape from battle"
        elif 'move' in action_lower:
            return "Character will move in specified direction"
        elif 'wait' in action_lower:
            return "No immediate change, observe for new information"
        elif 'respond' in action_lower:
            return "Dialogue will progress based on response"
        else:
            return f"Execute {action} command"
    
    def _validate_decision(self, decision: GameDecision, game_context: GameContext) -> GameDecision:
        """Validate and potentially modify decision"""
        debug(self.layer_name, "Validating decision")
        
        # Check if action is actually available
        if decision.primary_action not in game_context.available_actions:
            warning(self.layer_name, f"Primary action {decision.primary_action} not available, using fallback")
            
            # Use first available fallback
            for fallback in decision.fallback_actions:
                if fallback in game_context.available_actions:
                    decision.primary_action = fallback
                    decision.confidence *= 0.8  # Reduce confidence for fallback
                    break
        
        # Ensure minimum confidence threshold
        if decision.confidence < 0.2:
            decision.confidence = 0.2
            decision.primary_action = 'wait'  # Conservative fallback
        
        return decision
    
    def _create_fallback_decision(self) -> GameDecision:
        """Create safe fallback decision when everything fails"""
        return GameDecision(
            timestamp=time.time(),
            decision_type='wait',
            primary_action='wait',
            action_parameters={},
            reasoning_chain=[
                ReasoningStep(
                    step_type='fallback',
                    content='Using safe fallback action due to processing error',
                    confidence=0.3,
                    evidence=[],
                    alternatives=[]
                )
            ],
            confidence=0.3,
            fallback_actions=['observe', 'wait'],
            expected_outcome='No immediate action, wait for clearer information',
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
            self.stats['confidence_scores'].pop(0)  # Keep only recent scores
    
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
            f"Processing time: {decision.processing_time:.3f}s"
        ]
        
        if decision.reasoning_chain:
            reasoning_types = [step.step_type for step in decision.reasoning_chain]
            summary.append(f"Reasoning steps: {', '.join(set(reasoning_types))}")
        
        if decision.expected_outcome:
            summary.append(f"Expected outcome: {decision.expected_outcome}")
        
        return " | ".join(summary)


# Testing and example usage
if __name__ == "__main__":
    # Initialize debugging
    from ..debug_system import init_debugging
    init_debugging(log_level="DEBUG")
    
    print("BRAIN: Testing Reasoning Engine...")
    
    try:
        # Initialize reasoning engine
        reasoning_engine = ReasoningEngine()
        
        # Create mock perception and text analysis data
        from ..perception import GamePerception, DetectedObject, DetectedText
        from ..nlp import TextAnalysis, TextIntent
        
        # Mock perception data
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
            scene_description="Battle screen with menu options",
            confidence=0.85
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
            stats_info={},
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
                print(f"   {i+1}. {step.step_type.upper()}: {step.content[:100]}...")
        
        # Show decision summary
        summary = reasoning_engine.create_decision_summary(decision)
        print(f"\nSUMMARY: {summary}")
        
        # Show performance stats
        stats = reasoning_engine.get_stats()
        print(f"\nPERF: Performance: {stats}")
        
        print("\nSUCCESS: Reasoning Engine test completed!")
        
    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()