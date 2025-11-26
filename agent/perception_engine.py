"""
VISION: Perception Engine - Modular AI Game Agent
=============================================

The "eyes" of our intelligent agent - combines YOLOv8 object detection with 
advanced OCR to create rich, structured understanding of game screens.

Replaces the limited CNN state classification with comprehensive visual analysis.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging

# YOLOv8 for object detection
from ultralytics import YOLO

# OCR engines
import easyocr
import pytesseract
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """Represents a detected game object"""
    object_type: str        # 'health_bar', 'button', 'pokemon', 'text_box', etc.
    confidence: float       # Detection confidence (0-1)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int] # Center point (x, y)
    attributes: Dict[str, Any] = None  # Additional properties
    
    @property
    def area(self) -> int:
        """Calculate bounding box area"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class DetectedText:
    """Represents detected text on screen"""
    text: str               # The actual text content
    confidence: float       # OCR confidence (0-1)
    bbox: Tuple[int, int, int, int]  # Bounding box
    center: Tuple[int, int] # Center point
    font_size: Optional[int] = None
    is_dialogue: bool = False
    is_menu_item: bool = False
    is_stat: bool = False


@dataclass
class GamePerception:
    """Complete perception data from a single frame"""
    timestamp: float
    detected_objects: List[DetectedObject]
    detected_text: List[DetectedText]
    game_context: str       # 'battle', 'overworld', 'menu', etc.
    dominant_colors: List[str]
    screen_regions: Dict[str, Any]
    confidence: float       # Overall perception confidence
    processing_time: float


class PerceptionEngine:
    """
    Advanced perception system using YOLOv8 + OCR
    
    Capabilities:
    - Object detection (health bars, buttons, characters, items)
    - Text recognition (dialogue, menus, stats)
    - Scene understanding and context classification
    - Game-agnostic detection with specialized adapters
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        
        # Initialize object detector (YOLOv8)
        self.yolo_model = None
        self.initialize_object_detector()
        
        # Initialize OCR engines
        self.easyocr_reader = None
        self.initialize_ocr_engines()
        
        # Game-specific configurations
        self.game_context_rules = self.load_context_rules()
        
        # Performance tracking
        self.stats = {
            'frames_processed': 0,
            'avg_processing_time': 0,
            'total_objects_detected': 0,
            'total_text_detected': 0
        }
    
    def initialize_object_detector(self):
        """Initialize YOLOv8 model for object detection"""
        try:
            if self.model_path and Path(self.model_path).exists():
                logger.info(f"Loading custom YOLO model: {self.model_path}")
                self.yolo_model = YOLO(self.model_path)
            else:
                logger.info("Loading default YOLOv8n model")
                self.yolo_model = YOLO('yolov8n.pt')  # Nano version for speed
                
            logger.info("SUCCESS: YOLOv8 object detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO: {e}")
            self.yolo_model = None
    
    def initialize_ocr_engines(self):
        """Initialize OCR engines with optimal settings"""
        try:
            # EasyOCR - better for diverse fonts and languages
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)  # CPU for compatibility
            logger.info("SUCCESS: EasyOCR initialized")
            
            # Pytesseract config for game text (fallback)
            self.tesseract_config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?-: '
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            self.easyocr_reader = None
    
    def load_context_rules(self) -> Dict[str, Any]:
        """Load game context classification rules"""
        return {
            'battle_indicators': [
                'health_bar', 'pokemon', 'move_button', 'battle_ui'
            ],
            'menu_indicators': [
                'menu_cursor', 'list_items', 'navigation_buttons'
            ],
            'dialogue_indicators': [
                'text_box', 'dialogue_frame', 'character_portrait'
            ],
            'overworld_indicators': [
                'minimap', 'player_character', 'buildings', 'terrain'
            ]
        }
    
    def analyze_frame(self, frame: np.ndarray) -> GamePerception:
        """
        Comprehensive frame analysis - main entry point
        
        Args:
            frame: BGR image array from screen capture
            
        Returns:
            GamePerception object with all detected information
        """
        start_time = time.time()
        
        if frame is None or frame.size == 0:
            return self._create_empty_perception()
        
        try:
            # 1. Object Detection with YOLOv8
            detected_objects = self._detect_objects(frame)
            
            # 2. Text Recognition with OCR
            detected_text = self._detect_text(frame)
            
            # 3. Scene Analysis
            game_context = self._classify_game_context(detected_objects, detected_text)
            dominant_colors = self._analyze_colors(frame)
            screen_regions = self._analyze_screen_regions(frame, detected_objects, detected_text)
            
            # 4. Calculate overall confidence
            confidence = self._calculate_overall_confidence(detected_objects, detected_text)
            
            processing_time = time.time() - start_time
            
            # Create perception result
            perception = GamePerception(
                timestamp=start_time,
                detected_objects=detected_objects,
                detected_text=detected_text,
                game_context=game_context,
                dominant_colors=dominant_colors,
                screen_regions=screen_regions,
                confidence=confidence,
                processing_time=processing_time
            )
            
            # Update stats
            self._update_stats(processing_time, detected_objects, detected_text)
            
            return perception
            
        except Exception as e:
            logger.error(f"Error in frame analysis: {e}")
            return self._create_empty_perception()
    
    def _detect_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detect game objects using YOLOv8"""
        if self.yolo_model is None:
            return []
        
        try:
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            detected_objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box data
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Get class name
                        class_name = self.yolo_model.names[class_id]
                        
                        # Map to game-specific object types
                        object_type = self._map_to_game_object(class_name, x1, y1, x2, y2, frame)
                        
                        detected_object = DetectedObject(
                            object_type=object_type,
                            confidence=confidence,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            center=(int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            attributes={'yolo_class': class_name}
                        )
                        
                        detected_objects.append(detected_object)
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return []
    
    def _detect_text(self, frame: np.ndarray) -> List[DetectedText]:
        """Detect and recognize text using OCR"""
        if self.easyocr_reader is None:
            return []
        
        try:
            # Use EasyOCR as primary
            results = self.easyocr_reader.readtext(frame)
            detected_text = []
            
            for result in results:
                bbox, text, confidence = result
                
                # Clean and validate text
                text = text.strip()
                if len(text) < 2 or confidence < 0.3:
                    continue
                
                # Convert bbox format
                x1, y1 = int(min(p[0] for p in bbox)), int(min(p[1] for p in bbox))
                x2, y2 = int(max(p[0] for p in bbox)), int(max(p[1] for p in bbox))
                
                # Analyze text properties
                text_properties = self._analyze_text_properties(text, x1, y1, x2, y2, frame)
                
                detected_text_obj = DetectedText(
                    text=text,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    is_dialogue=text_properties.get('is_dialogue', False),
                    is_menu_item=text_properties.get('is_menu_item', False),
                    is_stat=text_properties.get('is_stat', False)
                )
                
                detected_text.append(detected_text_obj)
            
            return detected_text
            
        except Exception as e:
            logger.error(f"Text detection error: {e}")
            return []
    
    def _map_to_game_object(self, yolo_class: str, x1: float, y1: float, x2: float, y2: float, frame: np.ndarray) -> str:
        """Map YOLO classes to game-specific object types"""
        
        # Direct mappings
        game_object_map = {
            'person': 'character',
            'sports ball': 'pokeball', 
            'bottle': 'potion',
            'cell phone': 'device',
            'remote': 'game_controller'
        }
        
        # Position-based heuristics for UI elements
        frame_h, frame_w = frame.shape[:2]
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 1
        
        # Top area - likely UI elements
        if y1 < frame_h * 0.2:
            if aspect_ratio > 3:  # Long horizontal rectangle
                return 'health_bar'
            elif width < frame_w * 0.1:  # Small square
                return 'status_icon'
        
        # Bottom area - likely buttons/text
        elif y1 > frame_h * 0.7:
            if aspect_ratio < 2 and width > frame_w * 0.15:
                return 'button'
        
        # Use YOLO class or map it
        return game_object_map.get(yolo_class, yolo_class)
    
    def _analyze_text_properties(self, text: str, x1: int, y1: int, x2: int, y2: int, frame: np.ndarray) -> Dict[str, bool]:
        """Analyze text to determine its purpose and properties"""
        properties = {
            'is_dialogue': False,
            'is_menu_item': False,
            'is_stat': False
        }
        
        text_lower = text.lower()
        frame_h, frame_w = frame.shape[:2]
        
        # Dialogue detection
        dialogue_keywords = ['said', 'says', 'used', 'appeared', 'fainted', 'critical hit', 'effective']
        if any(keyword in text_lower for keyword in dialogue_keywords):
            properties['is_dialogue'] = True
        
        # Long text in bottom half is often dialogue
        if len(text) > 20 and y1 > frame_h * 0.6:
            properties['is_dialogue'] = True
        
        # Menu item detection
        menu_keywords = ['fight', 'pokemon', 'bag', 'run', 'items', 'save', 'exit']
        if any(keyword in text_lower for keyword in menu_keywords):
            properties['is_menu_item'] = True
        
        # Stat detection (numbers, HP, level indicators)
        if any(char.isdigit() for char in text) and len(text) < 10:
            stat_keywords = ['hp', 'level', 'lv', 'exp', 'atk', 'def', 'spd']
            if any(keyword in text_lower for keyword in stat_keywords):
                properties['is_stat'] = True
        
        return properties
    
    def _classify_game_context(self, objects: List[DetectedObject], text: List[DetectedText]) -> str:
        """Classify the overall game context/scene type"""
        
        # Count indicators for each context
        context_scores = {
            'battle': 0,
            'menu': 0,
            'dialogue': 0,
            'overworld': 0,
            'unknown': 0
        }
        
        # Score based on detected objects
        for obj in objects:
            if obj.object_type in ['health_bar', 'pokemon', 'character']:
                context_scores['battle'] += 2
            elif obj.object_type in ['button', 'menu_cursor']:
                context_scores['menu'] += 1
            elif obj.object_type == 'text_box':
                context_scores['dialogue'] += 2
        
        # Score based on detected text
        for text_obj in text:
            if text_obj.is_dialogue:
                context_scores['dialogue'] += 2
            elif text_obj.is_menu_item:
                context_scores['menu'] += 1
            elif text_obj.is_stat:
                context_scores['battle'] += 1
        
        # Return highest scoring context
        best_context = max(context_scores.items(), key=lambda x: x[1])
        return best_context[0] if best_context[1] > 0 else 'unknown'
    
    def _analyze_colors(self, frame: np.ndarray) -> List[str]:
        """Analyze dominant colors in the frame"""
        # Simple color analysis - can be expanded
        avg_color = np.mean(frame, axis=(0, 1))
        b, g, r = avg_color
        
        # Categorize dominant color
        if r > g and r > b:
            return ['red']
        elif g > r and g > b:
            return ['green']
        elif b > r and b > g:
            return ['blue']
        else:
            return ['neutral']
    
    def _analyze_screen_regions(self, frame: np.ndarray, objects: List[DetectedObject], text: List[DetectedText]) -> Dict[str, Any]:
        """Analyze different regions of the screen"""
        h, w = frame.shape[:2]
        
        regions = {
            'top': {'objects': [], 'text': [], 'area': (0, 0, w, h//3)},
            'middle': {'objects': [], 'text': [], 'area': (0, h//3, w, 2*h//3)},
            'bottom': {'objects': [], 'text': [], 'area': (0, 2*h//3, w, h)}
        }
        
        # Distribute objects and text by regions
        for obj in objects:
            cx, cy = obj.center
            if cy < h//3:
                regions['top']['objects'].append(obj)
            elif cy < 2*h//3:
                regions['middle']['objects'].append(obj)
            else:
                regions['bottom']['objects'].append(obj)
        
        for text_obj in text:
            cx, cy = text_obj.center
            if cy < h//3:
                regions['top']['text'].append(text_obj)
            elif cy < 2*h//3:
                regions['middle']['text'].append(text_obj)
            else:
                regions['bottom']['text'].append(text_obj)
        
        return regions
    
    def _calculate_overall_confidence(self, objects: List[DetectedObject], text: List[DetectedText]) -> float:
        """Calculate overall perception confidence"""
        if not objects and not text:
            return 0.0
        
        total_confidence = 0
        count = 0
        
        for obj in objects:
            total_confidence += obj.confidence
            count += 1
            
        for text_obj in text:
            total_confidence += text_obj.confidence
            count += 1
        
        return total_confidence / count if count > 0 else 0.0
    
    def _create_empty_perception(self) -> GamePerception:
        """Create empty perception for error cases"""
        return GamePerception(
            timestamp=time.time(),
            detected_objects=[],
            detected_text=[],
            game_context='unknown',
            dominant_colors=[],
            screen_regions={},
            confidence=0.0,
            processing_time=0.0
        )
    
    def _update_stats(self, processing_time: float, objects: List[DetectedObject], text: List[DetectedText]):
        """Update performance statistics"""
        self.stats['frames_processed'] += 1
        self.stats['total_objects_detected'] += len(objects)
        self.stats['total_text_detected'] += len(text)
        
        # Update average processing time
        current_avg = self.stats['avg_processing_time']
        frame_count = self.stats['frames_processed']
        self.stats['avg_processing_time'] = (current_avg * (frame_count - 1) + processing_time) / frame_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.stats.copy()
    
    def create_debug_visualization(self, frame: np.ndarray, perception: GamePerception) -> np.ndarray:
        """Create annotated frame for debugging/visualization"""
        debug_frame = frame.copy()
        
        # Draw object detections
        for obj in perception.detected_objects:
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{obj.object_type} {obj.confidence:.2f}"
            cv2.putText(debug_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw text detections
        for text_obj in perception.detected_text:
            x1, y1, x2, y2 = text_obj.bbox
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Truncate long text for display
            display_text = text_obj.text[:20] + "..." if len(text_obj.text) > 20 else text_obj.text
            cv2.putText(debug_frame, display_text, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Add context and stats
        info_text = f"Context: {perception.game_context} | Confidence: {perception.confidence:.2f}"
        cv2.putText(debug_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_frame


# Testing and example usage
if __name__ == "__main__":
    # Initialize perception engine
    engine = PerceptionEngine()
    
    print("VISION: Perception Engine initialized!")
    print("INFO: Capabilities:")
    print("  - YOLOv8 Object Detection")
    print("  - EasyOCR Text Recognition")
    print("  - Game Context Classification") 
    print("  - Performance Monitoring")
    print("\nREADY: Ready for integration with main agent!")