#!/usr/bin/env python3
"""
🌟 Universal AI Integration Demo
==============================

This script demonstrates how the Universal AI system integrates with VGA.py
and how it works across different gaming platforms.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add agent directory to path
sys.path.append('agent')

def demo_universal_ai_integration():
    """Demonstrate Universal AI integration with VGA.py"""
    
    print("Universal AI Integration Demo")
    print("=" * 50)
    
    try:
        # Step 1: Import and create Universal AI Bridge
        print("\nStep 1: Creating Universal AI Bridge...")
        from vga_integration import create_vga_ai_bridge
        
        bridge = create_vga_ai_bridge(
            enable_ai=True,
            platform="gameboy"  # Can be changed to "pc", "playstation", etc.
        )
        print("Universal AI Bridge created successfully")
        
        # Step 2: Demonstrate decision making
        print("\nStep 2: Universal AI Decision Making...")
        
        # Create mock game frames for different scenarios
        scenarios = [
            ("Battle Scene", create_battle_frame()),
            ("Overworld Exploration", create_overworld_frame()),
            ("Menu Interaction", create_menu_frame())
        ]
        
        for scenario_name, game_frame in scenarios:
            print(f"\n   Scenario: {scenario_name}")
            
            # Make Universal AI decision
            action, reasoning, confidence = bridge.make_universal_decision(
                game_frame=game_frame,
                current_state=scenario_name.split()[0].lower(),
                screen_analysis={}
            )
            
            print(f"      Action: {action}")
            print(f"      Reasoning: {reasoning[:60]}...")
            print(f"      Confidence: {confidence:.3f}")
        
        # Step 3: Platform versatility demonstration
        print("\nStep 3: Platform Versatility...")
        
        platforms = ["gameboy", "pc", "playstation", "nintendo"]
        sample_frame = create_overworld_frame()
        
        for platform in platforms:
            bridge.set_platform_hint(platform)
            action, reasoning, confidence = bridge.make_universal_decision(
                game_frame=sample_frame,
                current_state="exploration",
                screen_analysis={}
            )
            print(f"   {platform.ljust(12)}: {action.ljust(10)} (confidence: {confidence:.2f})")
        
        # Step 4: Show status and statistics
        print("\nStep 4: System Status...")
        status = bridge.get_status()
        print(f"   Universal AI Available: {status['universal_ai_available']}")
        print(f"   Platform: {status['platform_hint']}")
        print(f"   Total Decisions: {status['statistics']['total_decisions']}")
        print(f"   AI Usage Rate: {status['statistics'].get('universal_ai_usage_rate', 0):.1%}")
        
        # Step 5: Show how it integrates with VGA.py
        print("\nStep 5: VGA.py Integration Example...")
        print("""
   In VGA.py agent loop (line ~2717):
   
   # Check if Universal AI should handle this (highest priority)
   if (self.agent.use_universal_ai and 
       self.agent.universal_ai_bridge is not None):
       
       # Use Universal AI for decision making
       universal_action, universal_reasoning, universal_confidence = \\
           self.agent.universal_ai_bridge.make_universal_decision(
               game_frame=ml_frame,
               current_state=state,
               screen_analysis=screen_analysis
           )
       
       if universal_action in ["up", "down", "left", "right", "A", "B", "start", "select", "wait"]:
           if universal_action != "wait":
               self.ai_press(universal_action, f"Universal AI ({universal_confidence:.2f})")
           action_taken = universal_action
           reasoning = f"Universal AI: {universal_reasoning}"
        """)
        
        print("\nDemo completed successfully!")
        print("\nHow to use in VGA.py:")
        print("   1. Click 'Init Universal' button")
        print("   2. Click 'Universal ON' to enable")
        print("   3. Universal AI will make decisions with highest priority")
        print("   4. Falls back to Neural Agent or legacy logic if needed")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def create_battle_frame():
    """Create mock battle frame"""
    frame = np.zeros((320, 480, 3), dtype=np.uint8)
    
    # Battle background (darker)
    frame[:, :, :] = [40, 40, 60]
    
    # Pokemon sprites (red vs blue)
    frame[100:150, 100:150, 0] = 200  # Red player Pokemon
    frame[50:100, 350:400, 2] = 200   # Blue enemy Pokemon
    
    # UI elements (white/gray)
    frame[250:320, :, :] = [180, 180, 180]  # Battle menu area
    
    return frame


def create_overworld_frame():
    """Create mock overworld frame"""
    frame = np.zeros((320, 480, 3), dtype=np.uint8)
    
    # Grass (green)
    frame[:, :, 1] = 120
    
    # Path (brown)
    frame[140:180, :, :] = [139, 69, 19]
    
    # Trees (dark green)
    frame[50:120, 100:150, 1] = 80
    frame[50:120, 330:380, 1] = 80
    
    # Player character (red)
    frame[150:170, 230:250, 0] = 255
    
    return frame


def create_menu_frame():
    """Create mock menu frame"""
    frame = np.zeros((320, 480, 3), dtype=np.uint8)
    
    # Menu background (dark blue)
    frame[:, :, :] = [30, 30, 80]
    
    # Menu items (white/gray rectangles)
    for i, y in enumerate([80, 120, 160, 200]):
        frame[y:y+30, 50:400, :] = [150, 150, 150]
        if i == 1:  # Highlight second item
            frame[y:y+30, 50:400, :] = [100, 150, 255]
    
    return frame


if __name__ == "__main__":
    demo_universal_ai_integration()