import heapq
import numpy as np
from typing import List, Tuple, Optional, Set
import cv2

class Node:
    def __init__(self, position: Tuple[int, int], g_cost: float = 0, h_cost: float = 0, parent=None):
        self.position = position
        self.g_cost = g_cost  # Cost from start
        self.h_cost = h_cost  # Heuristic cost to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.position == other.position
    
    def __hash__(self):
        return hash(self.position)

class GamePathfinder:
    def __init__(self, grid_size: Tuple[int, int] = (32, 32)):
        self.grid_width, self.grid_height = grid_size
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        self.visited_map = np.zeros((self.grid_height, self.grid_width), dtype=int)
        
        # Movement directions (4-directional for Pokemon-like games)
        self.directions = [
            (0, -1),  # up
            (0, 1),   # down
            (-1, 0),  # left
            (1, 0)    # right
        ]
        
        # Direction names for movement commands
        self.direction_names = {
            (0, -1): "up",
            (0, 1): "down", 
            (-1, 0): "left",
            (1, 0): "right"
        }
    
    def update_obstacle_map(self, frame):
        """Update obstacle map based on current game frame"""
        if frame is None:
            return
        
        # Resize frame to grid size for processing
        h, w, _ = frame.shape
        resized = cv2.resize(frame, (self.grid_width, self.grid_height))
        
        # Convert to HSV for better obstacle detection
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Detect obstacles (walls, trees, buildings, water)
        obstacles = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        # Enhanced dark areas detection (walls, trees, buildings)
        dark_mask = gray < 50  # Stricter threshold
        obstacles |= dark_mask
        
        # Water detection (blue areas player can't walk on)
        water_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        obstacles |= (water_mask > 0)
        
        # Very bright areas (might be walls or unreachable)
        bright_mask = gray > 220  # Lower threshold to catch more walls
        obstacles |= bright_mask
        
        # Edge detection for walls
        edges = cv2.Canny(gray, 50, 150)
        edge_obstacles = edges > 100
        obstacles |= edge_obstacles
        
        # Add border obstacles (screen edges) - make them thicker
        border_thickness = 2
        obstacles[:border_thickness, :] = True   # Top
        obstacles[-border_thickness:, :] = True  # Bottom
        obstacles[:, :border_thickness] = True   # Left
        obstacles[:, -border_thickness:] = True  # Right
        
        # Morphological operations to fill small gaps and reduce noise
        kernel = np.ones((2, 2), np.uint8)
        obstacles = cv2.morphologyEx(obstacles.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        
        self.obstacle_map = obstacles
    
    def get_player_position(self, frame) -> Optional[Tuple[int, int]]:
        """Estimate player position on the grid"""
        if frame is None:
            return None
        
        # For Pokemon games, player is usually in the center-bottom area
        h, w, _ = frame.shape
        
        # Look for player sprite in center area
        center_x = self.grid_width // 2
        center_y = int(self.grid_height * 0.6)  # Slightly below center
        
        # Simple heuristic: player is likely near center of screen
        return (center_x, center_y)
    
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid and not an obstacle"""
        x, y = pos
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return False
        return not self.obstacle_map[y, x]
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[str]:
        """A* pathfinding algorithm"""
        if not self.is_valid_position(start) or not self.is_valid_position(goal):
            return []
        
        # Priority queue for open set
        open_set = []
        heapq.heappush(open_set, Node(start, 0, self.heuristic(start, goal)))
        
        # Sets for tracking visited nodes
        closed_set: Set[Tuple[int, int]] = set()
        open_positions: Set[Tuple[int, int]] = {start}
        
        # Dictionary for quick lookup of nodes in open set
        open_nodes = {start: Node(start, 0, self.heuristic(start, goal))}
        
        while open_set:
            current = heapq.heappop(open_set)
            current_pos = current.position
            
            # Remove from open tracking
            open_positions.discard(current_pos)
            
            # Goal reached
            if current_pos == goal:
                path = []
                node = current
                while node.parent is not None:
                    # Calculate direction from parent to current
                    dx = node.position[0] - node.parent.position[0]
                    dy = node.position[1] - node.parent.position[1]
                    direction = self.direction_names.get((dx, dy), "up")
                    path.append(direction)
                    node = node.parent
                return list(reversed(path))
            
            closed_set.add(current_pos)
            
            # Check all neighbors
            for dx, dy in self.directions:
                neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                if not self.is_valid_position(neighbor_pos) or neighbor_pos in closed_set:
                    continue
                
                # Calculate costs
                g_cost = current.g_cost + 1
                h_cost = self.heuristic(neighbor_pos, goal)
                
                # Check if this path to neighbor is better
                if neighbor_pos in open_positions:
                    existing_node = open_nodes[neighbor_pos]
                    if g_cost < existing_node.g_cost:
                        existing_node.g_cost = g_cost
                        existing_node.f_cost = g_cost + h_cost
                        existing_node.parent = current
                        heapq.heapify(open_set)  # Re-heapify since we changed costs
                else:
                    # Add new node to open set
                    neighbor_node = Node(neighbor_pos, g_cost, h_cost, current)
                    heapq.heappush(open_set, neighbor_node)
                    open_positions.add(neighbor_pos)
                    open_nodes[neighbor_pos] = neighbor_node
        
        # No path found
        return []
    
    def find_nearest_goal(self, start: Tuple[int, int], goal_type: str) -> Optional[Tuple[int, int]]:
        """Find nearest goal of specified type using simple search"""
        if goal_type == "unexplored":
            return self.find_nearest_unexplored(start)
        elif goal_type == "center":
            return (self.grid_width // 2, self.grid_height // 2)
        elif goal_type == "edge":
            return self.find_nearest_edge(start)
        return None
    
    def find_nearest_unexplored(self, start: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find nearest unexplored area"""
        min_distance = float('inf')
        best_pos = None
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.is_valid_position((x, y)) and self.visited_map[y, x] == 0:
                    distance = self.heuristic(start, (x, y))
                    if distance < min_distance:
                        min_distance = distance
                        best_pos = (x, y)
        
        return best_pos
    
    def find_nearest_edge(self, start: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find nearest accessible edge position"""
        edges = []
        
        # Top and bottom edges
        for x in range(1, self.grid_width - 1):
            if self.is_valid_position((x, 1)):
                edges.append((x, 1))
            if self.is_valid_position((x, self.grid_height - 2)):
                edges.append((x, self.grid_height - 2))
        
        # Left and right edges
        for y in range(1, self.grid_height - 1):
            if self.is_valid_position((1, y)):
                edges.append((1, y))
            if self.is_valid_position((self.grid_width - 2, y)):
                edges.append((self.grid_width - 2, y))
        
        if not edges:
            return None
        
        # Find closest edge
        min_distance = float('inf')
        best_edge = None
        
        for edge in edges:
            distance = self.heuristic(start, edge)
            if distance < min_distance:
                min_distance = distance
                best_edge = edge
        
        return best_edge
    
    def mark_visited(self, position: Tuple[int, int]):
        """Mark a position as visited"""
        x, y = position
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            self.visited_map[y, x] += 1
    
    def get_exploration_score(self, position: Tuple[int, int]) -> float:
        """Get exploration score for a position (higher = less visited)"""
        x, y = position
        if not self.is_valid_position(position):
            return 0.0
        
        visit_count = self.visited_map[y, x]
        # Return inverse of visit count + 1 to avoid division by zero
        return 1.0 / (visit_count + 1)
    
    def get_smart_exploration_direction(self, current_pos: Tuple[int, int]) -> Optional[str]:
        """Get the best direction for exploration"""
        best_direction = None
        best_score = -1
        
        for dx, dy in self.directions:
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            if self.is_valid_position(new_pos):
                score = self.get_exploration_score(new_pos)
                if score > best_score:
                    best_score = score
                    best_direction = self.direction_names[(dx, dy)]
        
        return best_direction