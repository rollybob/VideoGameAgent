#!/usr/bin/env python3
"""
🎮 LIVE PERCEPTION DEMO
======================

Real-time demo showing:
1. How synthetic GameBoy screens are created
2. Object detection in action
3. Performance metrics

Press ESC to exit
"""

import cv2
import numpy as np
import time
import random
from typing import Dict, List, Tuple

class LivePerceptionDemo:
    """Live demo of perception layer in action"""
    
    def __init__(self):
        self.window_name = "Perception Layer Demo"
        self.screen_width = 240
        self.screen_height = 160
        self.demo_scale = 3  # Scale up for visibility
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def create_synthetic_screen(self, screen_type: str) -> np.ndarray:
        """Create a synthetic GameBoy screen of specified type"""
        screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        if screen_type == 'overworld':
            return self._create_overworld_screen(screen)
        elif screen_type == 'battle':
            return self._create_battle_screen(screen)
        elif screen_type == 'menu':
            return self._create_menu_screen(screen)
        elif screen_type == 'pokemon_center':
            return self._create_pokemon_center_screen(screen)
        else:
            return screen
            
    def _create_overworld_screen(self, screen):
        """Create overworld-style screen with random elements"""
        # Background (grass green with variation)
        grass_color = [34 + random.randint(-10, 10), 139 + random.randint(-20, 20), 34 + random.randint(-10, 10)]
        screen[:, :] = grass_color
        
        # Player character (yellow square that moves slightly)
        player_x = 120 + random.randint(-10, 10)
        player_y = 80 + random.randint(-10, 10)
        cv2.rectangle(screen, (player_x-5, player_y-5), (player_x+5, player_y+5), (255, 255, 0), -1)
        
        # Random trees/obstacles
        num_trees = random.randint(3, 7)
        for _ in range(num_trees):
            tree_x = random.randint(20, 220)
            tree_y = random.randint(20, 140)
            tree_size = random.randint(6, 12)
            cv2.circle(screen, (tree_x, tree_y), tree_size, (0, 100, 0), -1)
            
        # HP display with random values
        hp_current = random.randint(60, 100)
        cv2.rectangle(screen, (5, 5), (80, 25), (0, 0, 0), -1)
        cv2.putText(screen, f"HP {hp_current}/100", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return screen
        
    def _create_battle_screen(self, screen):
        """Create battle-style screen with animated elements"""
        # Background (darker with slight variation)
        bg_color = [50 + random.randint(-10, 10), 50 + random.randint(-10, 10), 80 + random.randint(-10, 10)]
        screen[:, :] = bg_color
        
        # Player Pokemon area (with slight movement for "animation")
        player_offset = random.randint(-2, 2)
        cv2.rectangle(screen, (10 + player_offset, 90), (100 + player_offset, 140), (100, 150, 100), -1)
        cv2.putText(screen, "PIKACHU", (15 + player_offset, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Enemy Pokemon area
        enemy_offset = random.randint(-1, 1)
        cv2.rectangle(screen, (140 + enemy_offset, 20), (230 + enemy_offset, 70), (150, 100, 100), -1)
        cv2.putText(screen, "RATTATA", (145 + enemy_offset, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Animated HP bars
        player_hp = random.randint(40, 90)
        enemy_hp = random.randint(20, 80)
        
        # Player HP bar
        cv2.rectangle(screen, (10, 120), (10 + player_hp, 130), (0, 255, 0) if player_hp > 30 else (255, 0, 0), -1)
        cv2.rectangle(screen, (10, 120), (90, 130), (255, 255, 255), 1)  # Border
        
        # Enemy HP bar
        cv2.rectangle(screen, (150, 50), (150 + enemy_hp, 60), (0, 255, 0) if enemy_hp > 30 else (255, 0, 0), -1)
        cv2.rectangle(screen, (150, 50), (220, 60), (255, 255, 255), 1)  # Border
        
        # Battle menu
        cv2.rectangle(screen, (5, 140), (235, 155), (200, 200, 200), -1)
        cv2.rectangle(screen, (5, 140), (235, 155), (0, 0, 0), 1)  # Border
        cv2.putText(screen, "FIGHT  BAG  PKMN  RUN", (8, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        return screen
        
    def _create_menu_screen(self, screen):
        """Create menu-style screen"""
        # Background
        screen[:, :] = [200, 200, 220]
        
        # Menu box with slight shadow effect
        cv2.rectangle(screen, (22, 22), (222, 142), (100, 100, 100), -1)  # Shadow
        cv2.rectangle(screen, (20, 20), (220, 140), (255, 255, 255), -1)   # Main box
        cv2.rectangle(screen, (20, 20), (220, 140), (0, 0, 0), 2)          # Border
        
        # Menu items with selection indicator
        menu_items = ["POKEDEX", "POKEMON", "ITEMS", "SAVE", "OPTION", "EXIT"]
        selected_item = random.randint(0, len(menu_items) - 1)
        
        for i, item in enumerate(menu_items):
            y_pos = 40 + i * 15
            
            # Highlight selected item
            if i == selected_item:
                cv2.rectangle(screen, (25, y_pos - 8), (215, y_pos + 2), (100, 150, 255), -1)
                text_color = (255, 255, 255)
            else:
                text_color = (0, 0, 0)
                
            cv2.putText(screen, item, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            
        return screen
        
    def _create_pokemon_center_screen(self, screen):
        """Create Pokemon Center-style screen"""
        # Background (pinkish)
        screen[:, :] = [255, 200, 200]
        
        # Counter
        cv2.rectangle(screen, (50, 100), (190, 130), (150, 75, 0), -1)
        cv2.rectangle(screen, (50, 100), (190, 130), (0, 0, 0), 1)  # Border
        
        # Nurse Joy (animated with slight movement)
        nurse_x = 120 + random.randint(-2, 2)
        cv2.circle(screen, (nurse_x, 90), 15, (255, 220, 177), -1)  # Face
        cv2.circle(screen, (nurse_x, 90), 15, (0, 0, 0), 1)         # Face border
        
        # Nurse hat
        cv2.rectangle(screen, (nurse_x - 10, 80), (nurse_x + 10, 85), (255, 255, 255), -1)
        cv2.circle(screen, (nurse_x, 80), 3, (255, 0, 0), -1)  # Red cross
        
        cv2.putText(screen, "NURSE", (nurse_x - 15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        # Dialogue box with animated text
        cv2.rectangle(screen, (10, 130), (230, 155), (255, 255, 255), -1)
        cv2.rectangle(screen, (10, 130), (230, 155), (0, 0, 0), 1)
        
        messages = [
            "Welcome to Pokemon Center!",
            "Would you like to heal?",
            "Your Pokemon are healed!",
            "Come back anytime!"
        ]
        message = random.choice(messages)
        cv2.putText(screen, message, (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        return screen
        
    def detect_objects(self, screen: np.ndarray, screen_type: str) -> List[Dict]:
        """Simulate object detection on the screen"""
        objects = []
        
        if screen_type == 'overworld':
            # Detect HP display
            objects.append({
                'type': 'hp_display',
                'bbox': [5, 5, 80, 25],
                'confidence': 0.95
            })
            # Detect player character (search for yellow pixels)
            yellow_pixels = np.where((screen[:, :, 0] < 100) & (screen[:, :, 1] > 200) & (screen[:, :, 2] > 200))
            if len(yellow_pixels[0]) > 0:
                y_min, y_max = np.min(yellow_pixels[0]), np.max(yellow_pixels[0])
                x_min, x_max = np.min(yellow_pixels[1]), np.max(yellow_pixels[1])
                objects.append({
                    'type': 'player_character',
                    'bbox': [x_min, y_min, x_max, y_max],
                    'confidence': 0.92
                })
                
        elif screen_type == 'battle':
            objects.extend([
                {'type': 'player_pokemon', 'bbox': [10, 90, 100, 140], 'confidence': 0.88},
                {'type': 'enemy_pokemon', 'bbox': [140, 20, 230, 70], 'confidence': 0.91},
                {'type': 'player_hp_bar', 'bbox': [10, 120, 90, 130], 'confidence': 0.96},
                {'type': 'enemy_hp_bar', 'bbox': [150, 50, 220, 60], 'confidence': 0.94},
                {'type': 'battle_menu', 'bbox': [5, 140, 235, 155], 'confidence': 0.99}
            ])
            
        elif screen_type == 'menu':
            objects.append({
                'type': 'menu_box',
                'bbox': [20, 20, 220, 140],
                'confidence': 0.97
            })
            
        elif screen_type == 'pokemon_center':
            objects.extend([
                {'type': 'counter', 'bbox': [50, 100, 190, 130], 'confidence': 0.93},
                {'type': 'nurse_character', 'bbox': [105, 75, 135, 105], 'confidence': 0.87},
                {'type': 'dialogue_box', 'bbox': [10, 130, 230, 155], 'confidence': 0.98}
            ])
            
        return objects
        
    def draw_detections(self, screen: np.ndarray, objects: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on detected objects"""
        display_screen = screen.copy()
        
        for obj in objects:
            bbox = obj['bbox']
            obj_type = obj['type']
            confidence = obj['confidence']
            
            # Choose color based on object type
            if 'hp' in obj_type:
                color = (0, 255, 0)  # Green for HP
            elif 'pokemon' in obj_type or 'character' in obj_type:
                color = (255, 0, 0)  # Blue for characters
            elif 'menu' in obj_type or 'box' in obj_type:
                color = (0, 255, 255)  # Yellow for UI elements
            else:
                color = (255, 255, 255)  # White for others
            
            # Draw bounding box
            cv2.rectangle(display_screen, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
            
            # Draw label
            label = f"{obj_type} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            
            # Background for label
            cv2.rectangle(display_screen, 
                         (bbox[0], bbox[1] - label_size[1] - 5),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)
            
            # Label text
            cv2.putText(display_screen, label, (bbox[0], bbox[1] - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            
        return display_screen
        
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
            
    def draw_info_panel(self, display_screen: np.ndarray, screen_type: str, num_objects: int) -> np.ndarray:
        """Draw information panel on the screen"""
        # Info background
        info_height = 60
        info_panel = np.zeros((info_height, display_screen.shape[1], 3), dtype=np.uint8)
        info_panel[:, :] = [40, 40, 40]  # Dark gray
        
        # Draw info text
        y_offset = 15
        cv2.putText(info_panel, f"Screen Type: {screen_type.upper()}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 15
        cv2.putText(info_panel, f"Objects Detected: {num_objects}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        y_offset += 15
        cv2.putText(info_panel, f"FPS: {self.current_fps}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Processing status
        cv2.putText(info_panel, "LIVE PERCEPTION DEMO", (display_screen.shape[1] - 180, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        cv2.putText(info_panel, "Press ESC to exit", (display_screen.shape[1] - 150, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Combine with main screen
        combined = np.vstack([display_screen, info_panel])
        return combined
        
    def run_demo(self):
        """Run the live perception demo"""
        print("Starting Live Perception Demo...")
        print("- Shows how synthetic GameBoy screens are created")
        print("- Demonstrates real-time object detection")
        print("- Press ESC to exit")
        
        screen_types = ['overworld', 'battle', 'menu', 'pokemon_center']
        current_type_index = 0
        frame_count = 0
        type_switch_interval = 120  # Switch screen type every 120 frames (~4 seconds at 30fps)
        
        while True:
            # Switch screen type periodically
            if frame_count % type_switch_interval == 0:
                current_type_index = (current_type_index + 1) % len(screen_types)
                
            current_screen_type = screen_types[current_type_index]
            
            # Create synthetic screen
            screen = self.create_synthetic_screen(current_screen_type)
            
            # Detect objects
            detected_objects = self.detect_objects(screen, current_screen_type)
            
            # Draw detections
            display_screen = self.draw_detections(screen, detected_objects)
            
            # Scale up for visibility
            display_screen = cv2.resize(display_screen, 
                                      (self.screen_width * self.demo_scale, 
                                       self.screen_height * self.demo_scale),
                                      interpolation=cv2.INTER_NEAREST)
            
            # Add info panel
            display_screen = self.draw_info_panel(display_screen, current_screen_type, len(detected_objects))
            
            # Update FPS
            self.update_fps()
            
            # Display
            cv2.imshow(self.window_name, display_screen)
            
            # Check for exit
            key = cv2.waitKey(33) & 0xFF  # ~30 FPS
            if key == 27:  # ESC key
                break
                
            frame_count += 1
            
        cv2.destroyAllWindows()
        print("Demo ended.")

def main():
    """Run the live perception demo"""
    demo = LivePerceptionDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()