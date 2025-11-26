#!/usr/bin/env python3
"""
🔍 SYNTHETIC DATA VISUALIZER
============================

Shows exactly how synthetic GameBoy screens are created and what the
perception layer detects. Saves images so you can see the process.
"""

import cv2
import numpy as np
import json
import random
from pathlib import Path
from typing import Dict, List

class SyntheticDataVisualizer:
    """Visualize how synthetic training data is created"""
    
    def __init__(self):
        self.output_dir = Path("visualization_output")
        self.output_dir.mkdir(exist_ok=True)
        self.screen_width = 240
        self.screen_height = 160
        
    def create_and_visualize_all_types(self):
        """Create and visualize all screen types"""
        screen_types = ['overworld', 'battle', 'menu', 'pokemon_center']
        
        print("Creating synthetic GameBoy screens...")
        print("=" * 50)
        
        for i, screen_type in enumerate(screen_types):
            print(f"\n{i+1}. Creating {screen_type.upper()} screen...")
            
            # Create the synthetic screen
            screen = self.create_synthetic_screen(screen_type)
            
            # Detect objects (simulate what the AI sees)
            detected_objects = self.detect_objects(screen, screen_type)
            
            # Draw detection boxes
            screen_with_detections = self.draw_detections(screen.copy(), detected_objects)
            
            # Save original screen
            original_path = self.output_dir / f"{screen_type}_original.png"
            cv2.imwrite(str(original_path), screen)
            
            # Save screen with detections
            detection_path = self.output_dir / f"{screen_type}_with_detections.png"
            cv2.imwrite(str(detection_path), screen_with_detections)
            
            # Save detection metadata
            metadata = {
                'screen_type': screen_type,
                'objects_detected': detected_objects,
                'explanation': self.get_screen_explanation(screen_type)
            }
            
            metadata_path = self.output_dir / f"{screen_type}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"   • Detected {len(detected_objects)} objects")
            print(f"   • Saved: {original_path.name}")
            print(f"   • Saved: {detection_path.name}")
            print(f"   • Saved: {metadata_path.name}")
            
        print(f"\n✅ All synthetic screens created in: {self.output_dir}")
        
    def create_synthetic_screen(self, screen_type: str) -> np.ndarray:
        """Create a synthetic GameBoy screen"""
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
        """Create realistic overworld screen"""
        print("     - Drawing grass background")
        screen[:, :] = [34, 139, 34]  # Grass green
        
        print("     - Adding player character (yellow square)")
        player_x, player_y = 120, 80
        cv2.rectangle(screen, (player_x-5, player_y-5), (player_x+5, player_y+5), (255, 255, 0), -1)
        
        print("     - Placing random trees")
        for i in range(5):
            tree_x = random.randint(20, 220)
            tree_y = random.randint(20, 140)
            cv2.circle(screen, (tree_x, tree_y), 8, (0, 100, 0), -1)
            
        print("     - Drawing HP display")
        cv2.rectangle(screen, (5, 5), (80, 25), (0, 0, 0), -1)
        cv2.putText(screen, "HP 85/100", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return screen
        
    def _create_battle_screen(self, screen):
        """Create realistic battle screen"""
        print("     - Drawing battle background")
        screen[:, :] = [50, 50, 80]  # Dark battle background
        
        print("     - Adding player Pokemon (PIKACHU)")
        cv2.rectangle(screen, (10, 90), (100, 140), (100, 150, 100), -1)
        cv2.putText(screen, "PIKACHU", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        print("     - Adding enemy Pokemon (RATTATA)")
        cv2.rectangle(screen, (140, 20), (230, 70), (150, 100, 100), -1)
        cv2.putText(screen, "RATTATA", (145, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        print("     - Drawing HP bars")
        cv2.rectangle(screen, (10, 120), (90, 130), (255, 0, 0), -1)  # Player HP (red)
        cv2.rectangle(screen, (150, 50), (220, 60), (255, 0, 0), -1)   # Enemy HP (red)
        
        print("     - Creating battle menu")
        cv2.rectangle(screen, (5, 140), (235, 155), (200, 200, 200), -1)
        cv2.putText(screen, "FIGHT  BAG  PKMN  RUN", (8, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        return screen
        
    def _create_menu_screen(self, screen):
        """Create realistic menu screen"""
        print("     - Drawing menu background")
        screen[:, :] = [200, 200, 220]
        
        print("     - Creating menu box")
        cv2.rectangle(screen, (20, 20), (220, 140), (255, 255, 255), -1)
        cv2.rectangle(screen, (20, 20), (220, 140), (0, 0, 0), 2)
        
        print("     - Adding menu items")
        menu_items = ["POKEDEX", "POKEMON", "ITEMS", "SAVE", "OPTION", "EXIT"]
        for i, item in enumerate(menu_items):
            y_pos = 40 + i * 15
            cv2.putText(screen, item, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
        return screen
        
    def _create_pokemon_center_screen(self, screen):
        """Create realistic Pokemon Center screen"""
        print("     - Drawing Pokemon Center background")
        screen[:, :] = [255, 200, 200]  # Pink background
        
        print("     - Adding reception counter")
        cv2.rectangle(screen, (50, 100), (190, 130), (150, 75, 0), -1)
        
        print("     - Drawing Nurse Joy")
        cv2.circle(screen, (120, 90), 15, (255, 220, 177), -1)  # Face
        cv2.putText(screen, "NURSE", (100, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        print("     - Creating dialogue box")
        cv2.rectangle(screen, (10, 130), (230, 155), (255, 255, 255), -1)
        cv2.rectangle(screen, (10, 130), (230, 155), (0, 0, 0), 1)
        cv2.putText(screen, "Welcome to Pokemon Center!", (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        return screen
        
    def detect_objects(self, screen: np.ndarray, screen_type: str) -> List[Dict]:
        """Simulate object detection (what the AI would see)"""
        objects = []
        
        if screen_type == 'overworld':
            objects = [
                {'type': 'hp_display', 'bbox': [5, 5, 80, 25], 'confidence': 0.95},
                {'type': 'player_character', 'bbox': [115, 75, 125, 85], 'confidence': 0.92}
            ]
            
        elif screen_type == 'battle':
            objects = [
                {'type': 'player_pokemon', 'bbox': [10, 90, 100, 140], 'confidence': 0.88},
                {'type': 'enemy_pokemon', 'bbox': [140, 20, 230, 70], 'confidence': 0.91},
                {'type': 'player_hp_bar', 'bbox': [10, 120, 90, 130], 'confidence': 0.96},
                {'type': 'enemy_hp_bar', 'bbox': [150, 50, 220, 60], 'confidence': 0.94},
                {'type': 'battle_menu', 'bbox': [5, 140, 235, 155], 'confidence': 0.99}
            ]
            
        elif screen_type == 'menu':
            objects = [
                {'type': 'menu_box', 'bbox': [20, 20, 220, 140], 'confidence': 0.97}
            ]
            
        elif screen_type == 'pokemon_center':
            objects = [
                {'type': 'counter', 'bbox': [50, 100, 190, 130], 'confidence': 0.93},
                {'type': 'nurse_character', 'bbox': [105, 75, 135, 105], 'confidence': 0.87},
                {'type': 'dialogue_box', 'bbox': [10, 130, 230, 155], 'confidence': 0.98}
            ]
            
        return objects
        
    def draw_detections(self, screen: np.ndarray, objects: List[Dict]) -> np.ndarray:
        """Draw bounding boxes showing what the AI detected"""
        for obj in objects:
            bbox = obj['bbox']
            obj_type = obj['type']
            confidence = obj['confidence']
            
            # Color coding for different object types
            if 'hp' in obj_type:
                color = (0, 255, 0)  # Green for HP bars
            elif 'pokemon' in obj_type or 'character' in obj_type:
                color = (255, 0, 0)  # Blue for characters
            elif 'menu' in obj_type or 'box' in obj_type:
                color = (0, 255, 255)  # Yellow for UI elements
            else:
                color = (255, 255, 255)  # White for others
            
            # Draw bounding box
            cv2.rectangle(screen, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{obj_type} ({confidence:.2f})"
            cv2.putText(screen, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
        return screen
        
    def get_screen_explanation(self, screen_type: str) -> str:
        """Get explanation of how the screen was created"""
        explanations = {
            'overworld': 
                "This overworld screen was created by:\n"
                "1. Filling background with grass green color [34, 139, 34]\n"
                "2. Drawing yellow player character at position (120, 80)\n"
                "3. Adding 5 random dark green trees as circles\n"
                "4. Creating HP display box with white text 'HP 85/100'\n"
                "5. AI detects: HP display and player character",
                
            'battle': 
                "This battle screen was created by:\n"
                "1. Dark background [50, 50, 80] for battle atmosphere\n"
                "2. Player Pokemon area with 'PIKACHU' text\n"
                "3. Enemy Pokemon area with 'RATTATA' text\n"
                "4. Red HP bars for both Pokemon\n"
                "5. Battle menu with 'FIGHT BAG PKMN RUN' options\n"
                "6. AI detects: 5 objects including Pokemon and UI elements",
                
            'menu': 
                "This menu screen was created by:\n"
                "1. Light gray background [200, 200, 220]\n"
                "2. White menu box with black border\n"
                "3. 6 menu items: POKEDEX, POKEMON, ITEMS, SAVE, OPTION, EXIT\n"
                "4. AI detects: Menu box structure",
                
            'pokemon_center': 
                "This Pokemon Center screen was created by:\n"
                "1. Pink background [255, 200, 200] for Pokemon Center feel\n"
                "2. Brown reception counter\n"
                "3. Nurse Joy character as flesh-colored circle\n"
                "4. Dialogue box with welcome message\n"
                "5. AI detects: Counter, nurse, and dialogue box"
        }
        
        return explanations.get(screen_type, "No explanation available")
        
    def create_comparison_sheet(self):
        """Create a comparison sheet showing all screen types"""
        print("\n📊 Creating comparison sheet...")
        
        # Load all created images
        screen_types = ['overworld', 'battle', 'menu', 'pokemon_center']
        
        # Create a large comparison image
        comparison_width = self.screen_width * 4  # 4 screens side by side
        comparison_height = self.screen_height * 2  # Original + detected versions
        comparison_sheet = np.zeros((comparison_height, comparison_width, 3), dtype=np.uint8)
        
        for i, screen_type in enumerate(screen_types):
            # Load original and detection images
            original_path = self.output_dir / f"{screen_type}_original.png"
            detection_path = self.output_dir / f"{screen_type}_with_detections.png"
            
            if original_path.exists() and detection_path.exists():
                original_img = cv2.imread(str(original_path))
                detection_img = cv2.imread(str(detection_path))
                
                # Place in comparison sheet
                x_offset = i * self.screen_width
                
                # Original on top
                comparison_sheet[0:self.screen_height, x_offset:x_offset+self.screen_width] = original_img
                
                # Detection on bottom
                comparison_sheet[self.screen_height:, x_offset:x_offset+self.screen_width] = detection_img
                
                # Add labels
                cv2.putText(comparison_sheet, screen_type.upper(), 
                           (x_offset + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(comparison_sheet, "DETECTED", 
                           (x_offset + 5, self.screen_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Save comparison sheet
        comparison_path = self.output_dir / "comparison_sheet.png"
        cv2.imwrite(str(comparison_path), comparison_sheet)
        print(f"   • Saved: {comparison_path.name}")
        
    def generate_training_report(self):
        """Generate a detailed training report"""
        report = {
            "synthetic_data_creation": {
                "method": "Programmatic OpenCV drawing",
                "no_emulator_needed": True,
                "screen_types_created": ["overworld", "battle", "menu", "pokemon_center"],
                "screen_dimensions": f"{self.screen_width}x{self.screen_height}",
                "color_depth": "24-bit RGB"
            },
            "object_detection_simulation": {
                "objects_per_screen": {
                    "overworld": 2,
                    "battle": 5,
                    "menu": 1,
                    "pokemon_center": 3
                },
                "detection_types": [
                    "hp_display", "player_character", "player_pokemon", 
                    "enemy_pokemon", "hp_bars", "menu_elements", 
                    "dialogue_boxes", "characters"
                ]
            },
            "advantages": [
                "No real game needed",
                "Infinite training data generation",
                "Perfect ground truth labels",
                "Controllable variations",
                "Fast generation (milliseconds per screen)",
                "Consistent labeling"
            ],
            "next_steps": {
                "scale_up": "Generate 1000+ samples",
                "add_variations": "Different colors, positions, text",
                "real_data_mixing": "Combine with actual game screenshots",
                "advanced_detection": "Train YOLOv8 on this data"
            }
        }
        
        report_path = self.output_dir / "training_data_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📋 Training report saved: {report_path.name}")

def main():
    """Main visualization function"""
    visualizer = SyntheticDataVisualizer()
    
    print("SYNTHETIC TRAINING DATA VISUALIZER")
    print("="*50)
    print("This shows exactly how the perception layer creates")
    print("training data WITHOUT needing a real emulator!")
    print()
    
    # Create all screen types
    visualizer.create_and_visualize_all_types()
    
    # Create comparison sheet
    visualizer.create_comparison_sheet()
    
    # Generate report
    visualizer.generate_training_report()
    
    print("\n🎉 VISUALIZATION COMPLETE!")
    print(f"📁 Check all files in: {visualizer.output_dir}")
    print("\nKey insights:")
    print("• No emulator needed - everything is drawn with OpenCV")
    print("• Perfect labels since we know exactly what we drew")
    print("• Can generate thousands of variations instantly")
    print("• Each 'screen' is just programmed rectangles and text")

if __name__ == "__main__":
    main()