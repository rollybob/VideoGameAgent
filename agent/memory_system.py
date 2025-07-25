import json
import os
import time
import hashlib
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import pytesseract
from PIL import Image

class GameMemory:
    def __init__(self, maps_folder: str = "maps"):
        self.maps_folder = maps_folder
        self.current_game_file = None
        self.memory_data = {}
        self.current_area_hash = None
        self.grid_size = (64, 64)  # Adjustable based on game
        
        # Ensure maps folder exists
        if not os.path.exists(self.maps_folder):
            os.makedirs(self.maps_folder)
        
        # Memory tracking
        self.visited_tiles = set()
        self.area_signatures = {}  # area_hash -> area_name
        self.last_save_time = time.time()
        self.save_interval = 30  # Save every 30 seconds
        
    def extract_game_title(self, frame) -> str:
        """Extract game title using OCR and heuristics"""
        # For now, skip OCR entirely to avoid random names
        # Just use a consistent generic name
        return "GBA_Game"
    
    def initialize_game_memory(self, frame):
        """Initialize memory for current game"""
        game_title = self.extract_game_title(frame)
        self.current_game_file = os.path.join(self.maps_folder, f"{game_title}.json")
        
        # Load existing memory or create new
        if os.path.exists(self.current_game_file):
            try:
                with open(self.current_game_file, 'r') as f:
                    self.memory_data = json.load(f)
                    
                # Check if memory is corrupted or has too many areas
                if len(self.memory_data.get("areas", {})) > 50:
                    print(f"Memory file has {len(self.memory_data['areas'])} areas - resetting due to corruption")
                    self.create_new_memory(game_title)
                else:
                    # Clean up any corrupted tile data
                    self.cleanup_corrupted_tiles()
                    print(f"Loaded existing memory for: {game_title}")
            except Exception as e:
                print(f"Error loading memory: {e}")
                self.create_new_memory(game_title)
        else:
            self.create_new_memory(game_title)
            print(f"Created new memory for: {game_title}")
    
    def create_new_memory(self, game_title: str):
        """Create new memory structure for a game"""
        self.memory_data = {
            "game_info": {
                "title": game_title,
                "created": datetime.now().isoformat(),
                "last_played": datetime.now().isoformat(),
                "total_playtime": 0,
                "version": "1.0"
            },
            "world_map": {
                "grid_size": self.grid_size,
                "visited_tiles": [],
                "obstacles": [],
                "landmarks": {
                    "pokemon_centers": [],
                    "shops": [],
                    "gyms": [],
                    "npcs": [],
                    "items": [],
                    "doors": [],
                    "warp_points": []
                }
            },
            "areas": {},  # area_hash -> area_data
            "encounters": {
                "wild_pokemon": [],
                "trainer_battles": [],
                "npc_interactions": []
            },
            "progress": {
                "badges": 0,
                "pokemon_seen": 0,
                "pokemon_caught": 0,
                "areas_discovered": [],
                "key_events": []
            },
            "statistics": {
                "steps_taken": 0,
                "battles_won": 0,
                "battles_lost": 0,
                "items_found": 0,
                "money_earned": 0
            }
        }
    
    def get_area_signature(self, frame) -> str:
        """Create unique signature for current area/screen"""
        if frame is None:
            return "unknown"
        
        # Resize to very small size for extremely stable hashing
        resized = cv2.resize(frame, (8, 8))
        
        # Convert to grayscale and apply maximum blur
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Heavy quantization to reduce sensitivity
        quantized = (blurred // 64) * 64
        
        # Create hash from image data
        image_hash = hashlib.md5(quantized.tobytes()).hexdigest()[:6]
        return image_hash
    
    def update_current_position(self, frame, game_state: str, position: Tuple[int, int]):
        """Update memory with current position and context"""
        if not self.memory_data:
            self.initialize_game_memory(frame)
        
        area_hash = self.get_area_signature(frame)
        
        # Check if we're in a new area (only add if actually new)
        if area_hash != self.current_area_hash:
            self.current_area_hash = area_hash
            self.add_area_to_memory(area_hash, frame, game_state)
        
        # Use area hash as position instead of estimated coordinates
        # This prevents duplicate counting and is more reliable
        current_time = time.time()
        
        # Only record if we haven't been in this exact area recently (last 10 seconds)
        recent_tiles = self.memory_data["world_map"]["visited_tiles"][-20:]
        time_threshold = current_time - 10.0  # 10 second threshold
        
        # Check if we've been in this area hash recently
        recently_visited = any(
            len(t) >= 4 and isinstance(t[3], (int, float)) and t[3] > time_threshold and 
            (t[0] == area_hash if isinstance(t[0], str) else False)
            for t in recent_tiles
        )
        
        # Don't count tiles during battle states since there's no actual movement
        battle_states = ["Battle", "Wild Battle", "Trainer Battle"]
        
        if not recently_visited and game_state not in battle_states:
            # Store area hash as position for more reliable tracking
            tile_data = [area_hash, game_state, current_time, position]
            self.memory_data["world_map"]["visited_tiles"].append(tile_data)
            
            # Limit total visited tiles to prevent memory bloat
            if len(self.memory_data["world_map"]["visited_tiles"]) > 500:  # Reduced limit
                # Remove oldest 100 entries
                self.memory_data["world_map"]["visited_tiles"] = self.memory_data["world_map"]["visited_tiles"][100:]
        
        # Auto-save periodically (reduced to 10 seconds for more frequent saves)
        if time.time() - self.last_save_time > 10.0:
            self.save_memory()
            print(f"Auto-saved map data: {len(self.memory_data.get('areas', {}))} areas")
    
    def add_area_to_memory(self, area_hash: str, frame, game_state: str):
        """Add new area information to memory"""
        if area_hash not in self.memory_data["areas"]:
            area_name = self.generate_area_name(frame, game_state)
            
            self.memory_data["areas"][area_hash] = {
                "name": area_name,
                "first_visited": datetime.now().isoformat(),
                "game_state": game_state,
                "visit_count": 1,
                "landmarks_detected": [],
                "screenshot_hash": area_hash
            }
            
            # Add to discovered areas
            if area_name not in self.memory_data["progress"]["areas_discovered"]:
                self.memory_data["progress"]["areas_discovered"].append(area_name)
        else:
            # Increment visit count
            self.memory_data["areas"][area_hash]["visit_count"] += 1
    
    def generate_area_name(self, frame, game_state: str) -> str:
        """Generate descriptive name for area based on visual features"""
        # Use game state as primary indicator
        if game_state == "Pokemon Center":
            return f"Pokemon_Center_{len(self.memory_data['areas']) + 1}"
        elif game_state == "Shop":
            return f"Shop_{len(self.memory_data['areas']) + 1}"
        elif game_state == "Gym":
            return f"Gym_{len(self.memory_data['areas']) + 1}"
        elif game_state == "Wild Battle":
            return f"Wild_Area_{len(self.memory_data['areas']) + 1}"
        elif game_state == "Trainer Battle":
            return f"Trainer_Area_{len(self.memory_data['areas']) + 1}"
        elif game_state == "Overworld":
            return f"Overworld_{len(self.memory_data['areas']) + 1}"
        elif game_state == "Water/Flying":
            return f"Water_Area_{len(self.memory_data['areas']) + 1}"
        elif game_state == "Indoor/Cave":
            return f"Indoor_Area_{len(self.memory_data['areas']) + 1}"
        else:
            return f"Unknown_Area_{len(self.memory_data['areas']) + 1}"
    
    def analyze_dominant_colors(self, frame) -> List[str]:
        """Analyze dominant colors in frame for area classification"""
        if frame is None:
            return []
        
        # Resize for faster processing
        resized = cv2.resize(frame, (64, 64))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        colors = []
        
        # Check for green (grass, trees)
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        if np.sum(green_mask > 0) > 500:
            colors.append("green")
        
        # Check for blue (water, sky)
        blue_mask = cv2.inRange(hsv, np.array([100, 40, 40]), np.array([130, 255, 255]))
        if np.sum(blue_mask > 0) > 500:
            colors.append("blue")
        
        # Check for brown (buildings, paths)
        brown_mask = cv2.inRange(hsv, np.array([10, 50, 20]), np.array([25, 255, 200]))
        if np.sum(brown_mask > 0) > 500:
            colors.append("brown")
        
        return colors
    
    def record_encounter(self, encounter_type: str, data: Dict[str, Any]):
        """Record various types of encounters"""
        encounter_data = {
            "timestamp": datetime.now().isoformat(),
            "area_hash": self.current_area_hash,
            "type": encounter_type,
            "data": data
        }
        
        if encounter_type == "wild_pokemon":
            self.memory_data["encounters"]["wild_pokemon"].append(encounter_data)
            # Safely increment pokemon_seen counter
            if "statistics" in self.memory_data:
                self.memory_data["statistics"]["pokemon_seen"] = self.memory_data["statistics"].get("pokemon_seen", 0) + 1
        elif encounter_type == "trainer_battle":
            self.memory_data["encounters"]["trainer_battles"].append(encounter_data)
        elif encounter_type == "npc_interaction":
            self.memory_data["encounters"]["npc_interactions"].append(encounter_data)
    
    def record_landmark(self, landmark_type: str, position: Tuple[int, int], name: str = ""):
        """Record landmarks like Pokemon Centers, Shops, etc."""
        landmark_data = [position[0], position[1], name, datetime.now().isoformat()]
        
        if landmark_type in self.memory_data["world_map"]["landmarks"]:
            # Avoid duplicates
            existing = self.memory_data["world_map"]["landmarks"][landmark_type]
            if not any(l[0] == position[0] and l[1] == position[1] for l in existing):
                existing.append(landmark_data)
    
    def get_visited_areas(self) -> List[str]:
        """Get list of discovered area names"""
        return self.memory_data.get("progress", {}).get("areas_discovered", [])
    
    def get_area_visit_count(self, area_hash: str) -> int:
        """Get how many times an area has been visited"""
        return self.memory_data.get("areas", {}).get(area_hash, {}).get("visit_count", 0)
    
    def save_memory(self):
        """Save memory data to file"""
        if not self.memory_data or not self.current_game_file:
            return
        
        try:
            # Update last played time
            self.memory_data["game_info"]["last_played"] = datetime.now().isoformat()
            
            # Save to file with backup
            backup_file = self.current_game_file + ".backup"
            if os.path.exists(self.current_game_file):
                os.rename(self.current_game_file, backup_file)
            
            with open(self.current_game_file, 'w') as f:
                json.dump(self.memory_data, f, indent=2)
            
            # Remove backup if save successful
            if os.path.exists(backup_file):
                os.remove(backup_file)
            
            self.last_save_time = time.time()
            print(f"Memory saved to {self.current_game_file}")
            
        except Exception as e:
            print(f"Error saving memory: {e}")
            # Restore backup if save failed
            if os.path.exists(backup_file):
                os.rename(backup_file, self.current_game_file)
    
    def cleanup_corrupted_tiles(self):
        """Clean up corrupted tile data that might cause comparison errors"""
        if "world_map" in self.memory_data and "visited_tiles" in self.memory_data["world_map"]:
            original_count = len(self.memory_data["world_map"]["visited_tiles"])
            
            # Filter out corrupted entries
            clean_tiles = []
            for tile in self.memory_data["world_map"]["visited_tiles"]:
                try:
                    # Check if tile data is in expected format
                    if (isinstance(tile, list) and len(tile) >= 3 and 
                        isinstance(tile[2], (int, float))):
                        clean_tiles.append(tile)
                except (TypeError, IndexError):
                    continue  # Skip corrupted entries
            
            self.memory_data["world_map"]["visited_tiles"] = clean_tiles
            
            if original_count != len(clean_tiles):
                print(f"Cleaned up {original_count - len(clean_tiles)} corrupted tile entries")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get summary statistics about current memory"""
        if not self.memory_data:
            return {}
        
        return {
            "game_title": self.memory_data.get("game_info", {}).get("title", "Unknown"),
            "areas_discovered": len(self.memory_data.get("areas", {})),
            "tiles_visited": len(self.memory_data.get("world_map", {}).get("visited_tiles", [])),
            "pokemon_seen": self.memory_data.get("statistics", {}).get("pokemon_seen", 0),
            "steps_taken": self.memory_data.get("statistics", {}).get("steps_taken", 0),
            "current_area": self.memory_data.get("areas", {}).get(self.current_area_hash, {}).get("name", "Unknown")
        }