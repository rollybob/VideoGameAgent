"""
TEXT: Text Analyzer - Modular AI Game Agent
=======================================

Advanced text understanding layer using spaCy NLP and Sentence-Transformers.
Processes detected text to extract meaning, intent, and context for intelligent
decision making.

Integrated with comprehensive debugging system for error tracing.
"""

import re
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from pathlib import Path

# NLP and ML libraries
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

# Import our debugging system
from ..debug_system import (
    get_debugger,
    monitor_performance,
    info,
    debug,
    warning,
    error,
)

# Import perception data structures
from ..perception import DetectedText


@dataclass
class TextIntent:
    """Represents the understood intent from text"""
    intent_type: str        # 'command', 'dialogue', 'information', 'question', etc.
    confidence: float       # Confidence in intent classification
    entities: Dict[str, List[str]]  # Named entities found
    keywords: List[str]     # Important keywords
    sentiment: str          # 'positive', 'negative', 'neutral'
    action_required: bool   # Does this text require an action?
    context_category: str   # 'battle', 'menu', 'dialogue', 'shop', etc.


@dataclass
class SemanticMatch:
    """Represents semantic similarity match"""
    text: str
    similarity_score: float
    matched_pattern: str
    context: str


@dataclass 
class TextAnalysis:
    """Complete analysis of detected text"""
    timestamp: float
    original_texts: List[DetectedText]
    processed_texts: List[str]      # Cleaned and normalized
    combined_context: str           # All text combined with context
    intents: List[TextIntent]
    semantic_matches: List[SemanticMatch]
    game_commands: List[str]        # Identified game commands
    dialogue_content: List[str]     # Extracted dialogue
    ui_elements: List[str]          # Menu items, buttons, etc.
    stats_info: Dict[str, Any]      # Extracted numeric information
    confidence: float               # Overall analysis confidence
    processing_time: float


class TextAnalyzer:
    """
    Advanced text understanding system using spaCy + Sentence-Transformers
    
    Capabilities:
    - Natural language processing with spaCy
    - Semantic similarity matching
    - Intent recognition for game contexts
    - Entity extraction (Pokemon names, moves, items)
    - Command identification and parsing
    - Context-aware text interpretation
    - Game-specific text pattern recognition
    """
    
    def __init__(self, 
                 spacy_model: str = "en_core_web_sm",
                 sentence_model: str = "all-MiniLM-L6-v2"):
        
        self.debugger = get_debugger()
        self.layer_name = "text_understanding"
        
        info(self.layer_name, "INIT: Initializing Text Analyzer...")
        
        # Initialize NLP models
        self.nlp = None
        self.sentence_model = None
        self.initialize_models(spacy_model, sentence_model)
        
        # Game-specific knowledge bases
        self.game_patterns = self.load_game_patterns()
        self.pokemon_entities = self.load_pokemon_knowledge()
        self.command_patterns = self.load_command_patterns()
        
        # Performance tracking
        self.stats = {
            'texts_processed': 0,
            'intents_identified': 0,
            'entities_extracted': 0,
            'avg_processing_time': 0.0
        }
        
        info(self.layer_name, "SUCCESS: Text Analyzer initialized successfully")
    
    @monitor_performance("text_understanding", "model_initialization")
    def initialize_models(self, spacy_model: str, sentence_model: str):
        """Initialize NLP models with error handling"""
        try:
            # Load spaCy model
            debug(self.layer_name, f"Loading spaCy model: {spacy_model}")
            self.nlp = spacy.load(spacy_model)
            info(self.layer_name, f"SUCCESS: spaCy model loaded: {spacy_model}")
            
            # Load sentence transformer
            debug(self.layer_name, f"Loading sentence transformer: {sentence_model}")
            self.sentence_model = SentenceTransformer(sentence_model)
            info(self.layer_name, f"SUCCESS: Sentence transformer loaded: {sentence_model}")
            
        except Exception as e:
            error(self.layer_name, "Failed to initialize NLP models", exception=e)
            raise
    
    def load_game_patterns(self) -> Dict[str, List[str]]:
        """Load game-specific text patterns"""
        debug(self.layer_name, "Loading game-specific text patterns")
        
        return {
            'battle_patterns': [
                r'\b\w+ used \w+\b',
                r'It\'s (super effective|not very effective|critically hit)',
                r'\b\w+ fainted\b',
                r'HP: \d+/\d+',
                r'Level \d+',
                r'(Attack|Defense|Speed|Special) (rose|fell)'
            ],
            
            'dialogue_patterns': [
                r'\b(said|says|asked|replied|shouted)\b',
                r'Professor \w+',
                r'Welcome to',
                r'Would you like to',
                r'Do you want to'
            ],
            
            'menu_patterns': [
                r'\b(Fight|Pokemon|Bag|Run)\b',
                r'\b(Items|Save|Exit|Options)\b',
                r'Select (a|an) \w+',
                r'Choose \w+'
            ],
            
            'status_patterns': [
                r'HP: \d+',
                r'Level \d+',
                r'EXP: \d+',
                r'\$\d+',
                r'\d+/\d+'
            ]
        }
    
    def load_pokemon_knowledge(self) -> Dict[str, Set[str]]:
        """Load Pokemon-specific entities and knowledge"""
        debug(self.layer_name, "Loading Pokemon knowledge base")
        
        return {
            'pokemon_names': {
                'pikachu', 'charizard', 'blastoise', 'venusaur', 'mewtwo',
                'mew', 'alakazam', 'gengar', 'dragonite', 'snorlax',
                'lucario', 'garchomp', 'rayquaza', 'dialga', 'palkia'
                # This would be expanded with full Pokemon database
            },
            
            'move_names': {
                'thunderbolt', 'flamethrower', 'hydro pump', 'solar beam',
                'psychic', 'earthquake', 'ice beam', 'shadow ball',
                'dragon claw', 'hyper beam', 'tackle', 'scratch'
                # This would be expanded with full move database
            },
            
            'item_names': {
                'potion', 'super potion', 'hyper potion', 'full heal',
                'pokeball', 'great ball', 'ultra ball', 'master ball',
                'rare candy', 'revive', 'max revive', 'ether'
                # This would be expanded with full item database
            },
            
            'location_names': {
                'pallet town', 'viridian city', 'pewter city', 'cerulean city',
                'vermilion city', 'lavender town', 'celadon city', 'fuchsia city',
                'saffron city', 'cinnabar island', 'indigo plateau'
                # This would be expanded with full location database
            }
        }
    
    def load_command_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load command patterns for different game contexts"""
        debug(self.layer_name, "Loading command patterns")
        
        return {
            'battle_commands': {
                'attack_patterns': [r'use \w+', r'attack with \w+', r'\w+ attack'],
                'defensive_patterns': [r'defend', r'guard', r'protect'],
                'item_patterns': [r'use (potion|heal|item)', r'drink \w+'],
                'escape_patterns': [r'run away', r'flee', r'escape']
            },
            
            'navigation_commands': {
                'movement_patterns': [r'go (up|down|left|right)', r'move to \w+', r'walk \w+'],
                'interaction_patterns': [r'talk to \w+', r'examine \w+', r'use \w+'],
                'menu_patterns': [r'open (menu|bag|pokemon)', r'check \w+']
            },
            
            'dialogue_commands': {
                'response_patterns': [r'yes', r'no', r'okay', r'sure', r'maybe'],
                'question_patterns': [r'what is \w+', r'how do \w+', r'where is \w+'],
                'continuation_patterns': [r'next', r'continue', r'more']
            }
        }
    
    @monitor_performance("text_understanding", "text_analysis")
    def analyze_texts(self, detected_texts: List[DetectedText]) -> TextAnalysis:
        """
        Comprehensive text analysis - main entry point
        
        Args:
            detected_texts: List of DetectedText from perception layer
            
        Returns:
            TextAnalysis with complete understanding results
        """
        start_time = time.time()
        
        debug(self.layer_name, f"Starting analysis of {len(detected_texts)} text elements")
        
        if not detected_texts:
            return self._create_empty_analysis()
        
        try:
            # 1. Text preprocessing and cleaning
            processed_texts = self._preprocess_texts(detected_texts)
            
            # 2. Combine texts with spatial context
            combined_context = self._create_combined_context(detected_texts, processed_texts)
            
            # 3. Intent recognition
            intents = self._analyze_intents(processed_texts)
            
            # 4. Semantic similarity matching
            semantic_matches = self._find_semantic_matches(processed_texts)
            
            # 5. Extract specific content types
            game_commands = self._extract_game_commands(processed_texts)
            dialogue_content = self._extract_dialogue_content(detected_texts, processed_texts)
            ui_elements = self._extract_ui_elements(detected_texts, processed_texts)
            stats_info = self._extract_stats_information(processed_texts)
            
            # 6. Calculate overall confidence
            confidence = self._calculate_analysis_confidence(intents, semantic_matches)
            
            processing_time = time.time() - start_time
            
            # Create analysis result
            analysis = TextAnalysis(
                timestamp=start_time,
                original_texts=detected_texts,
                processed_texts=processed_texts,
                combined_context=combined_context,
                intents=intents,
                semantic_matches=semantic_matches,
                game_commands=game_commands,
                dialogue_content=dialogue_content,
                ui_elements=ui_elements,
                stats_info=stats_info,
                confidence=confidence,
                processing_time=processing_time
            )
            
            # Update statistics
            self._update_stats(processing_time, intents)
            
            info(self.layer_name, f"Analysis completed: {len(intents)} intents, {len(game_commands)} commands", 
                 details={'processing_time': processing_time, 'confidence': confidence})
            
            return analysis
            
        except Exception as e:
            error(self.layer_name, "Text analysis failed", exception=e)
            return self._create_empty_analysis()
    
    def _preprocess_texts(self, detected_texts: List[DetectedText]) -> List[str]:
        """Clean and preprocess detected texts"""
        debug(self.layer_name, "Preprocessing detected texts")
        
        processed = []
        
        for text_obj in detected_texts:
            # Basic cleaning
            text = text_obj.text.strip()
            
            # Remove extra spaces and normalize
            text = re.sub(r'\s+', ' ', text)
            
            # Handle special game characters
            text = text.replace('(M)', 'male').replace('(F)', 'female')
            
            # Convert to lowercase for processing (preserve original case in metadata)
            processed_text = text.lower()
            
            if len(processed_text) >= 2:  # Filter very short texts
                processed.append(processed_text)
        
        debug(self.layer_name, f"Preprocessed {len(processed)} texts from {len(detected_texts)} detected")
        return processed
    
    def _create_combined_context(self, detected_texts: List[DetectedText], processed_texts: List[str]) -> str:
        """Create combined context from all texts with spatial information"""
        debug(self.layer_name, "Creating combined text context")
        
        # Group texts by screen regions
        top_texts = []
        middle_texts = []
        bottom_texts = []
        
        for i, text_obj in enumerate(detected_texts):
            if i < len(processed_texts):
                cx, cy = text_obj.center
                # Assume screen height context (this could be parameterized)
                if cy < 150:  # Top region
                    top_texts.append(processed_texts[i])
                elif cy < 350:  # Middle region
                    middle_texts.append(processed_texts[i])
                else:  # Bottom region
                    bottom_texts.append(processed_texts[i])
        
        # Combine with spatial context
        context_parts = []
        if top_texts:
            context_parts.append(f"TOP: {' | '.join(top_texts)}")
        if middle_texts:
            context_parts.append(f"MIDDLE: {' | '.join(middle_texts)}")
        if bottom_texts:
            context_parts.append(f"BOTTOM: {' | '.join(bottom_texts)}")
        
        combined = " || ".join(context_parts)
        debug(self.layer_name, f"Combined context: {combined[:100]}...")
        
        return combined
    
    def _analyze_intents(self, processed_texts: List[str]) -> List[TextIntent]:
        """Analyze texts to identify intents and purposes"""
        debug(self.layer_name, "Analyzing text intents")
        
        intents = []
        
        for text in processed_texts:
            try:
                # Use spaCy for NLP analysis
                doc = self.nlp(text)
                
                # Extract entities
                entities = self._extract_entities(doc)
                
                # Extract keywords (important tokens)
                keywords = [token.lemma_ for token in doc 
                           if not token.is_stop and not token.is_punct and len(token.text) > 2]
                
                # Classify intent type
                intent_type = self._classify_intent_type(text, doc)
                
                # Determine sentiment
                sentiment = self._analyze_sentiment(doc)
                
                # Check if action is required
                action_required = self._requires_action(text, intent_type)
                
                # Determine context category
                context_category = self._determine_context_category(text)
                
                # Calculate confidence based on various factors
                confidence = self._calculate_intent_confidence(doc, entities, keywords)
                
                intent = TextIntent(
                    intent_type=intent_type,
                    confidence=confidence,
                    entities=entities,
                    keywords=keywords,
                    sentiment=sentiment,
                    action_required=action_required,
                    context_category=context_category
                )
                
                intents.append(intent)
                
            except Exception as e:
                warning(self.layer_name, f"Failed to analyze intent for text: {text[:50]}...", 
                       details={'error': str(e)})
        
        debug(self.layer_name, f"Identified {len(intents)} intents")
        return intents
    
    def _extract_entities(self, doc) -> Dict[str, List[str]]:
        """Extract named entities using spaCy + game knowledge"""
        entities = {
            'pokemon': [],
            'moves': [],
            'items': [],
            'locations': [],
            'numbers': [],
            'names': []
        }
        
        # spaCy named entity recognition
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG']:
                entities['names'].append(ent.text.lower())
            elif ent.label_ in ['CARDINAL', 'MONEY']:
                entities['numbers'].append(ent.text)
            elif ent.label_ == 'GPE':  # Geopolitical entity (locations)
                entities['locations'].append(ent.text.lower())
        
        # Game-specific entity matching
        text_lower = doc.text.lower()
        
        for pokemon in self.pokemon_entities['pokemon_names']:
            if pokemon in text_lower:
                entities['pokemon'].append(pokemon)
        
        for move in self.pokemon_entities['move_names']:
            if move in text_lower:
                entities['moves'].append(move)
        
        for item in self.pokemon_entities['item_names']:
            if item in text_lower:
                entities['items'].append(item)
        
        return entities
    
    def _classify_intent_type(self, text: str, doc) -> str:
        """Classify the type of intent from text"""
        
        # Check for questions
        if any(token.text.lower() in ['what', 'how', 'where', 'when', 'why', 'who'] for token in doc):
            return 'question'
        
        # Check for commands (imperatives)
        if doc[0].pos_ == 'VERB' and doc[0].dep_ == 'ROOT':
            return 'command'
        
        # Check for battle-related text
        battle_keywords = ['used', 'attack', 'hit', 'effective', 'fainted', 'hp', 'damage']
        if any(keyword in text.lower() for keyword in battle_keywords):
            return 'battle_information'
        
        # Check for dialogue
        dialogue_keywords = ['said', 'says', 'asked', 'replied', 'welcome', 'hello']
        if any(keyword in text.lower() for keyword in dialogue_keywords):
            return 'dialogue'
        
        # Check for menu/UI
        menu_keywords = ['fight', 'pokemon', 'bag', 'run', 'items', 'save', 'exit']
        if any(keyword in text.lower() for keyword in menu_keywords):
            return 'menu_option'
        
        # Check for status information
        if re.search(r'\d+', text):  # Contains numbers
            return 'status_information'
        
        return 'information'
    
    def _analyze_sentiment(self, doc) -> str:
        """Simple sentiment analysis"""
        # This is a basic implementation - could be enhanced with sentiment models
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'super', 'effective'}
        negative_words = {'bad', 'terrible', 'awful', 'failed', 'not very', 'weak', 'fainted'}
        
        text_lower = doc.text.lower()
        
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        else:
            return 'neutral'
    
    def _requires_action(self, text: str, intent_type: str) -> bool:
        """Determine if text requires an action response"""
        action_intents = ['command', 'question', 'menu_option']
        action_keywords = ['choose', 'select', 'pick', 'use', 'go', 'attack', 'run']
        
        return (intent_type in action_intents or 
                any(keyword in text.lower() for keyword in action_keywords))
    
    def _determine_context_category(self, text: str) -> str:
        """Determine the game context category"""
        text_lower = text.lower()
        
        # Battle context
        if any(pattern_list for pattern_list in self.game_patterns['battle_patterns'] 
               if re.search(pattern_list, text_lower)):
            return 'battle'
        
        # Menu context
        if any(re.search(pattern, text_lower) for pattern in self.game_patterns['menu_patterns']):
            return 'menu'
        
        # Dialogue context
        if any(re.search(pattern, text_lower) for pattern in self.game_patterns['dialogue_patterns']):
            return 'dialogue'
        
        # Status context
        if any(re.search(pattern, text_lower) for pattern in self.game_patterns['status_patterns']):
            return 'status'
        
        return 'general'
    
    def _calculate_intent_confidence(self, doc, entities: Dict[str, List[str]], keywords: List[str]) -> float:
        """Calculate confidence score for intent classification"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on entity matches
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        confidence += min(0.3, total_entities * 0.1)
        
        # Boost confidence based on keywords
        confidence += min(0.2, len(keywords) * 0.05)
        
        # Boost confidence based on text length and structure
        if len(doc) > 3:  # Multi-word text
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _find_semantic_matches(self, processed_texts: List[str]) -> List[SemanticMatch]:
        """Find semantic similarity matches with known patterns"""
        debug(self.layer_name, "Finding semantic matches")
        
        if self.sentence_model is None:
            warning(self.layer_name, "Sentence model not available for semantic matching")
            return []
        
        matches = []
        
        # Known game patterns for semantic matching
        known_patterns = {
            'battle_start': "a wild pokemon appeared",
            'battle_win': "enemy pokemon fainted",
            'battle_lose': "your pokemon fainted", 
            'level_up': "pokemon leveled up",
            'item_use': "used an item",
            'pokemon_center': "welcome to pokemon center",
            'shop_greeting': "welcome to the shop",
            'gym_challenge': "gym leader battle"
        }
        
        try:
            # Encode all texts
            text_embeddings = self.sentence_model.encode(processed_texts)
            pattern_embeddings = self.sentence_model.encode(list(known_patterns.values()))
            
            # Calculate similarities
            for i, text in enumerate(processed_texts):
                for j, (pattern_name, pattern_text) in enumerate(known_patterns.items()):
                    # Calculate cosine similarity
                    similarity = np.dot(text_embeddings[i], pattern_embeddings[j]) / (
                        np.linalg.norm(text_embeddings[i]) * np.linalg.norm(pattern_embeddings[j])
                    )
                    
                    if similarity > 0.7:  # High similarity threshold
                        match = SemanticMatch(
                            text=text,
                            similarity_score=float(similarity),
                            matched_pattern=pattern_name,
                            context=pattern_text
                        )
                        matches.append(match)
            
            debug(self.layer_name, f"Found {len(matches)} semantic matches")
            
        except Exception as e:
            warning(self.layer_name, "Semantic matching failed", details={'error': str(e)})
        
        return matches
    
    def _extract_game_commands(self, processed_texts: List[str]) -> List[str]:
        """Extract identifiable game commands"""
        debug(self.layer_name, "Extracting game commands")
        
        commands = []
        
        for text in processed_texts:
            # Direct command matching
            command_words = ['fight', 'pokemon', 'bag', 'run', 'attack', 'use', 'go', 'talk', 'examine']
            
            for word in command_words:
                if word in text.lower():
                    commands.append(word)
        
        return list(set(commands))  # Remove duplicates
    
    def _extract_dialogue_content(self, detected_texts: List[DetectedText], processed_texts: List[str]) -> List[str]:
        """Extract dialogue content from texts"""
        dialogue = []
        
        for i, text_obj in enumerate(detected_texts):
            if text_obj.is_dialogue and i < len(processed_texts):
                dialogue.append(processed_texts[i])
        
        return dialogue
    
    def _extract_ui_elements(self, detected_texts: List[DetectedText], processed_texts: List[str]) -> List[str]:
        """Extract UI elements (menu items, buttons)"""
        ui_elements = []
        
        for i, text_obj in enumerate(detected_texts):
            if text_obj.is_menu_item and i < len(processed_texts):
                ui_elements.append(processed_texts[i])
        
        return ui_elements
    
    def _extract_stats_information(self, processed_texts: List[str]) -> Dict[str, Any]:
        """Extract numeric/statistical information"""
        debug(self.layer_name, "Extracting statistical information")
        
        stats = {
            'hp_values': [],
            'levels': [],
            'experience': [],
            'money': [],
            'other_numbers': []
        }
        
        for text in processed_texts:
            # HP patterns
            hp_matches = re.findall(r'hp:?\s*(\d+)(?:/(\d+))?', text, re.IGNORECASE)
            for match in hp_matches:
                if match[1]:  # HP: current/max format
                    stats['hp_values'].append({'current': int(match[0]), 'max': int(match[1])})
                else:  # HP: value format
                    stats['hp_values'].append({'current': int(match[0])})
            
            # Level patterns
            level_matches = re.findall(r'(?:level|lv\.?)\s*(\d+)', text, re.IGNORECASE)
            stats['levels'].extend([int(match) for match in level_matches])
            
            # Experience patterns
            exp_matches = re.findall(r'exp:?\s*(\d+)', text, re.IGNORECASE)
            stats['experience'].extend([int(match) for match in exp_matches])
            
            # Money patterns
            money_matches = re.findall(r'\$(\d+)', text)
            stats['money'].extend([int(match) for match in money_matches])
            
            # Other numbers
            other_numbers = re.findall(r'\b(\d+)\b', text)
            stats['other_numbers'].extend([int(match) for match in other_numbers])
        
        return stats
    
    def _calculate_analysis_confidence(self, intents: List[TextIntent], semantic_matches: List[SemanticMatch]) -> float:
        """Calculate overall analysis confidence"""
        if not intents:
            return 0.0
        
        # Average intent confidence
        intent_confidence = sum(intent.confidence for intent in intents) / len(intents)
        
        # Boost confidence based on semantic matches
        semantic_boost = min(0.2, len(semantic_matches) * 0.1)
        
        return min(1.0, intent_confidence + semantic_boost)
    
    def _create_empty_analysis(self) -> TextAnalysis:
        """Create empty analysis for error cases"""
        return TextAnalysis(
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
            confidence=0.0,
            processing_time=0.0
        )
    
    def _update_stats(self, processing_time: float, intents: List[TextIntent]):
        """Update performance statistics"""
        self.stats['texts_processed'] += 1
        self.stats['intents_identified'] += len(intents)
        
        # Update average processing time
        current_avg = self.stats['avg_processing_time']
        count = self.stats['texts_processed']
        self.stats['avg_processing_time'] = (current_avg * (count - 1) + processing_time) / count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.stats.copy()
    
    def create_text_summary(self, analysis: TextAnalysis) -> str:
        """Create human-readable summary of text analysis"""
        summary = []
        
        if analysis.game_commands:
            summary.append(f"Commands identified: {', '.join(analysis.game_commands)}")
        
        if analysis.dialogue_content:
            summary.append(f"Dialogue: {len(analysis.dialogue_content)} messages")
        
        if analysis.ui_elements:
            summary.append(f"UI elements: {', '.join(analysis.ui_elements)}")
        
        high_confidence_intents = [intent for intent in analysis.intents if intent.confidence > 0.7]
        if high_confidence_intents:
            intent_types = [intent.intent_type for intent in high_confidence_intents]
            summary.append(f"Primary intents: {', '.join(set(intent_types))}")
        
        if analysis.semantic_matches:
            match_patterns = [match.matched_pattern for match in analysis.semantic_matches]
            summary.append(f"Semantic matches: {', '.join(set(match_patterns))}")
        
        return " | ".join(summary) if summary else "No significant text content identified"


# Testing and example usage
if __name__ == "__main__":
    # Initialize debugging
    from ..debug_system import init_debugging
    init_debugging(log_level="DEBUG")
    
    # Test text analyzer
    analyzer = TextAnalyzer()
    
    # Create sample detected texts (simulating perception layer output)
    sample_texts = [
        DetectedText(
            text="PIKACHU used THUNDERBOLT!",
            confidence=0.95,
            bbox=(50, 300, 250, 320),
            center=(150, 310),
            is_dialogue=True
        ),
        DetectedText(
            text="It's super effective!",
            confidence=0.92,
            bbox=(50, 325, 200, 345),
            center=(125, 335),
            is_dialogue=True
        ),
        DetectedText(
            text="HP: 45/100",
            confidence=0.89,
            bbox=(300, 50, 400, 70),
            center=(350, 60),
            is_stat=True
        ),
        DetectedText(
            text="FIGHT",
            confidence=0.96,
            bbox=(50, 400, 120, 430),
            center=(85, 415),
            is_menu_item=True
        )
    ]
    
    print("TEST: Testing Text Analyzer with sample data...")
    
    # Analyze the sample texts
    analysis = analyzer.analyze_texts(sample_texts)
    
    # Display results
    print(f"\nRESULTS: Analysis Results:")
    print(f"   Processing time: {analysis.processing_time:.3f}s")
    print(f"   Overall confidence: {analysis.confidence:.3f}")
    print(f"   Intents found: {len(analysis.intents)}")
    print(f"   Game commands: {analysis.game_commands}")
    print(f"   Dialogue content: {analysis.dialogue_content}")
    print(f"   UI elements: {analysis.ui_elements}")
    print(f"   Stats info: {analysis.stats_info}")
    
    # Show text summary
    summary = analyzer.create_text_summary(analysis)
    print(f"\nSUMMARY: {summary}")
    
    # Show performance stats
    stats = analyzer.get_stats()
    print(f"\nPERF: Performance: {stats}")
    
    print("\nSUCCESS: Text Analyzer test completed!")