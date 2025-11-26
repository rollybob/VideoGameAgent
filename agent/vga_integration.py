"""
VGA Integration Module - Universal AI Integration
===============================================

Integrates the Universal AI Manager into the existing VGA.py system,
providing a seamless bridge between the legacy agent and the new
modular AI architecture.

This allows users to switch between different AI modes while maintaining
backward compatibility with existing functionality.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

# Import the Universal AI Manager
from universal_ai_manager import UniversalAIManager, UniversalGameContext, ActionResult

# Import debugging system
from debug_system import get_debugger, monitor_performance, info, debug, warning, error


class VGAUniversalAIBridge:
    """
    Bridge between VGA.py legacy system and Universal AI Manager
    
    Features:
    - Seamless integration with existing VGA.py structure
    - Backward compatibility with legacy decision-making
    - Platform-agnostic action translation
    - Performance monitoring and debugging
    - Graceful fallback handling
    """
    
    def __init__(self, 
                 enable_universal_ai: bool = True,
                 platform_hint: str = "gameboy",
                 debug_level: str = "INFO"):
        
        self.debugger = get_debugger()
        self.layer_name = "vga_integration"
        
        info(self.layer_name, "Initializing VGA Universal AI Bridge...")
        
        # Configuration
        self.enable_universal_ai = enable_universal_ai
        self.platform_hint = platform_hint
        self.debug_level = debug_level
        
        # Initialize Universal AI Manager
        self.universal_ai = None
        if self.enable_universal_ai:
            try:
                self.universal_ai = UniversalAIManager(
                    enable_perception=True,
                    enable_text_analysis=True,
                    enable_reasoning=True,
                    debug_level=debug_level
                )
                info(self.layer_name, "Universal AI Manager integrated")
            except Exception as e:
                warning(self.layer_name, "Failed to initialize Universal AI, using legacy mode", 
                       details={'error': str(e)})
                self.enable_universal_ai = False
        
        # Legacy compatibility mappings
        self.legacy_mappings = self.create_legacy_mappings()
        
        # Performance tracking
        self.stats = {
            'total_decisions': 0,
            'universal_ai_decisions': 0,
            'legacy_decisions': 0,
            'avg_processing_time': 0.0,
            'successful_translations': 0,
            'failed_translations': 0
        }
        
        info(self.layer_name, f"VGA Bridge initialized (Universal AI: {'ON' if self.enable_universal_ai else 'OFF'})")
    
    def create_legacy_mappings(self) -> Dict[str, str]:
        """Create mappings between Universal AI actions and legacy VGA actions"""
        return {
            # Movement mappings
            'navigate_up': 'up',
            'navigate_down': 'down', 
            'navigate_left': 'left',
            'navigate_right': 'right',
            'move_up': 'up',
            'move_down': 'down',
            'move_left': 'left', 
            'move_right': 'right',
            
            # Interaction mappings
            'primary_action': 'A',
            'secondary_action': 'B',
            'confirm_action': 'A',
            'cancel_action': 'B',
            'menu_action': 'start',
            'interact': 'A',
            'examine': 'A',
            'talk': 'A',
            
            # Battle mappings
            'select_fight': 'A',
            'select_pokemon': 'A',
            'select_bag': 'A', 
            'select_run': 'A',
            'battle_action': 'A',
            'use_move': 'A',
            
            # Strategic mappings
            'wait_observe': 'wait',
            'wait': 'wait',
            'observe': 'wait',
            'explore': 'up',  # Default exploration direction
            'explore_area': 'up',
            
            # Response mappings  
            'respond_yes': 'A',
            'respond_no': 'B',
            'continue_dialogue': 'A',
            
            # Default fallback
            'unknown': 'wait'
        }
    
    @monitor_performance("vga_integration", "decision_making")
    def make_universal_decision(self, 
                              game_frame: np.ndarray,
                              current_state: str = "unknown",
                              screen_analysis: Dict[str, Any] = None) -> Tuple[str, str, float]:
        """
        Make decision using Universal AI system
        
        Args:
            game_frame: Current game frame
            current_state: Current game state from legacy detection
            screen_analysis: Legacy screen analysis data
            
        Returns:
            Tuple of (action, reasoning, confidence)
        """
        start_time = time.time()
        
        try:
            if not self.enable_universal_ai or self.universal_ai is None:
                return self._fallback_to_legacy_decision(current_state, screen_analysis)
            
            debug(self.layer_name, f"Making Universal AI decision for state: {current_state}")
            
            # Step 1: Analyze frame with Universal AI
            context = self.universal_ai.analyze_frame(game_frame, self.platform_hint)
            
            # Step 2: Translate Universal AI decision to legacy action
            universal_action = context.decision.primary_action
            legacy_action = self._translate_to_legacy_action(universal_action, context)
            
            # Step 3: Create reasoning
            reasoning = self._create_reasoning(context, legacy_action, current_state)
            
            # Step 4: Update statistics
            processing_time = time.time() - start_time
            self._update_stats(processing_time, True)
            
            debug(self.layer_name, f"Universal AI decision: {legacy_action} (confidence: {context.confidence:.2f})")
            
            return legacy_action, reasoning, context.confidence
            
        except Exception as e:
            error(self.layer_name, "Universal AI decision failed, falling back to legacy", exception=e)
            return self._fallback_to_legacy_decision(current_state, screen_analysis)
    
    def _translate_to_legacy_action(self, universal_action: str, context: UniversalGameContext) -> str:
        """Translate Universal AI action to legacy VGA action"""
        
        # Direct mapping if available
        if universal_action in self.legacy_mappings:
            legacy_action = self.legacy_mappings[universal_action]
            self.stats['successful_translations'] += 1
            return legacy_action
        
        # Intelligent pattern matching
        action_lower = universal_action.lower()
        
        # Movement patterns
        if any(word in action_lower for word in ['up', 'north']):
            return 'up'
        elif any(word in action_lower for word in ['down', 'south']):
            return 'down'
        elif any(word in action_lower for word in ['left', 'west']):
            return 'left'
        elif any(word in action_lower for word in ['right', 'east']):
            return 'right'
        
        # Exploration patterns - convert to actual movement
        elif any(word in action_lower for word in ['move_randomly', 'random', 'explore']):
            import random
            return random.choice(['up', 'down', 'left', 'right'])
        elif any(word in action_lower for word in ['explore_new_area', 'explore_area']):
            return 'up'  # Default exploration direction
        
        # Interaction patterns
        elif any(word in action_lower for word in ['attack', 'fight', 'battle']):
            return 'A'
        elif any(word in action_lower for word in ['select', 'choose', 'confirm']):
            return 'A'
        elif any(word in action_lower for word in ['cancel', 'back', 'exit']):
            return 'B'
        elif any(word in action_lower for word in ['menu', 'start']):
            return 'start'
        
        # Context-aware translation
        elif context.text_analysis.ui_elements:
            ui_elements = [elem.lower() for elem in context.text_analysis.ui_elements]
            if 'fight' in ui_elements or 'attack' in ui_elements:
                return 'A'  # Select fight option
            elif 'yes' in ui_elements:
                return 'A'  # Confirm yes
            elif 'no' in ui_elements:
                return 'B'  # Select no
        
        # Default fallback
        self.stats['failed_translations'] += 1
        warning(self.layer_name, f"Could not translate action '{universal_action}', using fallback")
        return 'wait'
    
    def _create_reasoning(self, context: UniversalGameContext, legacy_action: str, current_state: str) -> str:
        """Create human-readable reasoning for the decision"""
        
        reasoning_parts = []
        
        # Add AI decision reasoning
        if context.decision.reasoning_chain:
            primary_reasoning = context.decision.reasoning_chain[0].content
            reasoning_parts.append(f"AI Analysis: {primary_reasoning[:100]}")
        
        # Add context information
        if context.perception.detected_objects:
            reasoning_parts.append(f"Detected {len(context.perception.detected_objects)} objects")
        
        if context.text_analysis.intents:
            intent_types = [intent.intent_type for intent in context.text_analysis.intents[:2]]
            reasoning_parts.append(f"Intents: {', '.join(intent_types)}")
        
        # Add confidence and state info
        reasoning_parts.append(f"State: {current_state} -> Action: {legacy_action}")
        reasoning_parts.append(f"Confidence: {context.confidence:.2f}")
        
        return " | ".join(reasoning_parts)
    
    def _fallback_to_legacy_decision(self, current_state: str, screen_analysis: Dict[str, Any]) -> Tuple[str, str, float]:
        """Fallback to legacy decision-making logic"""
        
        debug(self.layer_name, f"Using legacy decision for state: {current_state}")
        
        # Simple state-based decision making (legacy compatibility)
        if current_state in ["Overworld", "Water/Flying", "Indoor/Cave"]:
            action = "up"  # Default exploration
            reasoning = f"Legacy exploration in {current_state}"
            confidence = 0.6
            
        elif current_state in ["Battle", "Wild Battle", "Trainer Battle"]:
            action = "A"  # Default battle action
            reasoning = f"Legacy battle action in {current_state}"
            confidence = 0.7
            
        elif current_state == "Dialogue/Menu":
            action = "A"  # Default menu selection
            reasoning = f"Legacy menu interaction in {current_state}"
            confidence = 0.8
            
        elif current_state in ["Pokemon Center", "Shop", "Gym"]:
            action = "A"  # Default interaction
            reasoning = f"Legacy location interaction in {current_state}"
            confidence = 0.7
            
        else:
            action = "wait"  # Safe fallback
            reasoning = f"Legacy fallback for unknown state: {current_state}"
            confidence = 0.3
        
        self._update_stats(0.001, False)  # Fast legacy decision
        return action, reasoning, confidence
    
    def _update_stats(self, processing_time: float, used_universal_ai: bool):
        """Update performance statistics"""
        self.stats['total_decisions'] += 1
        
        if used_universal_ai:
            self.stats['universal_ai_decisions'] += 1
        else:
            self.stats['legacy_decisions'] += 1
        
        # Update average processing time
        current_avg = self.stats['avg_processing_time']
        count = self.stats['total_decisions']
        self.stats['avg_processing_time'] = (current_avg * (count - 1) + processing_time) / count
    
    def set_platform_hint(self, platform: str):
        """Update platform hint for better action translation"""
        self.platform_hint = platform
        debug(self.layer_name, f"Platform hint updated to: {platform}")
    
    def toggle_universal_ai(self, enabled: bool) -> bool:
        """Enable or disable Universal AI mode"""
        if enabled and self.universal_ai is None:
            try:
                self.universal_ai = UniversalAIManager()
                self.enable_universal_ai = True
                info(self.layer_name, "Universal AI mode enabled")
                return True
            except Exception as e:
                warning(self.layer_name, "Failed to enable Universal AI", details={'error': str(e)})
                return False
        else:
            self.enable_universal_ai = enabled
            status = "enabled" if enabled else "disabled" 
            info(self.layer_name, f"Universal AI mode {status}")
            return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status and statistics"""
        status = {
            'universal_ai_enabled': self.enable_universal_ai,
            'universal_ai_available': self.universal_ai is not None,
            'platform_hint': self.platform_hint,
            'statistics': self.stats.copy()
        }
        
        # Add Universal AI stats if available
        if self.universal_ai:
            status['universal_ai_stats'] = self.universal_ai.get_stats()
        
        # Calculate success rates
        total_decisions = self.stats['total_decisions']
        if total_decisions > 0:
            status['statistics']['universal_ai_usage_rate'] = \
                self.stats['universal_ai_decisions'] / total_decisions
            status['statistics']['translation_success_rate'] = \
                self.stats['successful_translations'] / max(1, self.stats['successful_translations'] + self.stats['failed_translations'])
        
        return status
    
    def create_status_summary(self) -> str:
        """Create human-readable status summary"""
        status = self.get_status()
        
        summary_parts = [
            f"Universal AI: {'ON' if status['universal_ai_enabled'] else 'OFF'}",
            f"Platform: {status['platform_hint']}",
            f"Decisions: {status['statistics']['total_decisions']}",
            f"AI Usage: {status['statistics'].get('universal_ai_usage_rate', 0):.1%}",
            f"Avg Time: {status['statistics']['avg_processing_time']:.3f}s"
        ]
        
        return " | ".join(summary_parts)


def create_vga_ai_bridge(enable_ai: bool = True, platform: str = "gameboy") -> VGAUniversalAIBridge:
    """
    Factory function to create VGA AI Bridge
    
    Args:
        enable_ai: Whether to enable Universal AI
        platform: Platform hint for action translation
        
    Returns:
        Configured VGAUniversalAIBridge instance
    """
    return VGAUniversalAIBridge(
        enable_universal_ai=enable_ai,
        platform_hint=platform,
        debug_level="INFO"
    )


# Testing and example usage
if __name__ == "__main__":
    # Initialize debugging
    from debug_system import init_debugging
    init_debugging(log_level="DEBUG")
    
    print("Testing VGA Universal AI Bridge...")
    
    try:
        # Create bridge
        bridge = create_vga_ai_bridge(enable_ai=True, platform="gameboy")
        
        # Create mock game frame
        mock_frame = np.zeros((320, 480, 3), dtype=np.uint8)
        mock_frame[150:200, 200:250] = [0, 255, 0]  # Green square
        
        # Test decision making
        print(f"\nTesting Universal AI Decision Making...")
        action, reasoning, confidence = bridge.make_universal_decision(
            game_frame=mock_frame,
            current_state="Overworld",
            screen_analysis={}
        )
        
        print(f"Decision Results:")
        print(f"   Action: {action}")
        print(f"   Reasoning: {reasoning}")
        print(f"   Confidence: {confidence:.3f}")
        
        # Test legacy fallback
        print(f"\nTesting Legacy Fallback...")
        bridge.toggle_universal_ai(False)
        action2, reasoning2, confidence2 = bridge.make_universal_decision(
            game_frame=mock_frame,
            current_state="Battle",
            screen_analysis={}
        )
        
        print(f"Legacy Results:")
        print(f"   Action: {action2}")
        print(f"   Reasoning: {reasoning2}")
        print(f"   Confidence: {confidence2:.3f}")
        
        # Test platform switching
        print(f"\nTesting Platform Switching...")
        bridge.toggle_universal_ai(True)
        platforms = ["gameboy", "pc", "playstation"]
        for platform in platforms:
            bridge.set_platform_hint(platform)
            action3, reasoning3, confidence3 = bridge.make_universal_decision(
                game_frame=mock_frame,
                current_state="Overworld", 
                screen_analysis={}
            )
            print(f"   {platform}: {action3} (confidence: {confidence3:.2f})")
        
        # Show status summary
        status_summary = bridge.create_status_summary()
        print(f"\nStatus Summary: {status_summary}")
        
        # Show detailed status
        status = bridge.get_status()
        print(f"\nDetailed Status:")
        print(f"   Universal AI Available: {status['universal_ai_available']}")
        print(f"   Total Decisions: {status['statistics']['total_decisions']}")
        print(f"   AI Usage Rate: {status['statistics'].get('universal_ai_usage_rate', 0):.1%}")
        print(f"   Translation Success: {status['statistics'].get('translation_success_rate', 0):.1%}")
        
        print("\nVGA Universal AI Bridge test completed!")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()