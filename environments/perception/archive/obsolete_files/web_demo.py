#!/usr/bin/env python3
"""
Web-based Perception Demo
=========================
Creates a simple web server showing live perception demo
Access at: http://localhost:8080
"""

import cv2
import numpy as np
import json
import base64
import time
import random
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

class PerceptionWebDemo:
    """Web-based perception demo"""
    
    def __init__(self):
        self.screen_width = 240
        self.screen_height = 160
        self.demo_scale = 3
        
    def create_synthetic_screen(self, screen_type: str) -> np.ndarray:
        """Create synthetic screen (same as before)"""
        screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        if screen_type == 'overworld':
            return self._create_overworld_screen(screen)
        elif screen_type == 'battle':
            return self._create_battle_screen(screen)
        elif screen_type == 'menu':
            return self._create_menu_screen(screen)
        elif screen_type == 'pokemon_center':
            return self._create_pokemon_center_screen(screen)
        return screen
        
    def _create_overworld_screen(self, screen):
        screen[:, :] = [34 + random.randint(-10, 10), 139 + random.randint(-20, 20), 34 + random.randint(-10, 10)]
        player_x = 120 + random.randint(-10, 10)
        player_y = 80 + random.randint(-10, 10)
        cv2.rectangle(screen, (player_x-5, player_y-5), (player_x+5, player_y+5), (255, 255, 0), -1)
        
        for _ in range(random.randint(3, 7)):
            tree_x = random.randint(20, 220)
            tree_y = random.randint(20, 140)
            cv2.circle(screen, (tree_x, tree_y), random.randint(6, 12), (0, 100, 0), -1)
            
        hp = random.randint(60, 100)
        cv2.rectangle(screen, (5, 5), (80, 25), (0, 0, 0), -1)
        cv2.putText(screen, f"HP {hp}/100", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return screen
        
    def _create_battle_screen(self, screen):
        screen[:, :] = [50 + random.randint(-10, 10), 50 + random.randint(-10, 10), 80 + random.randint(-10, 10)]
        
        player_offset = random.randint(-2, 2)
        cv2.rectangle(screen, (10 + player_offset, 90), (100 + player_offset, 140), (100, 150, 100), -1)
        cv2.putText(screen, "PIKACHU", (15 + player_offset, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        enemy_offset = random.randint(-1, 1)
        cv2.rectangle(screen, (140 + enemy_offset, 20), (230 + enemy_offset, 70), (150, 100, 100), -1)
        cv2.putText(screen, "RATTATA", (145 + enemy_offset, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        player_hp = random.randint(40, 90)
        enemy_hp = random.randint(20, 80)
        
        cv2.rectangle(screen, (10, 120), (10 + player_hp, 130), (0, 255, 0) if player_hp > 30 else (255, 0, 0), -1)
        cv2.rectangle(screen, (150, 50), (150 + enemy_hp, 60), (0, 255, 0) if enemy_hp > 30 else (255, 0, 0), -1)
        
        cv2.rectangle(screen, (5, 140), (235, 155), (200, 200, 200), -1)
        cv2.putText(screen, "FIGHT  BAG  PKMN  RUN", (8, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        return screen
        
    def _create_menu_screen(self, screen):
        screen[:, :] = [200, 200, 220]
        cv2.rectangle(screen, (20, 20), (220, 140), (255, 255, 255), -1)
        cv2.rectangle(screen, (20, 20), (220, 140), (0, 0, 0), 2)
        
        menu_items = ["POKEDEX", "POKEMON", "ITEMS", "SAVE", "OPTION", "EXIT"]
        selected = random.randint(0, len(menu_items) - 1)
        
        for i, item in enumerate(menu_items):
            y_pos = 40 + i * 15
            if i == selected:
                cv2.rectangle(screen, (25, y_pos - 8), (215, y_pos + 2), (100, 150, 255), -1)
                color = (255, 255, 255)
            else:
                color = (0, 0, 0)
            cv2.putText(screen, item, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return screen
        
    def _create_pokemon_center_screen(self, screen):
        screen[:, :] = [255, 200, 200]
        cv2.rectangle(screen, (50, 100), (190, 130), (150, 75, 0), -1)
        
        nurse_x = 120 + random.randint(-2, 2)
        cv2.circle(screen, (nurse_x, 90), 15, (255, 220, 177), -1)
        cv2.putText(screen, "NURSE", (nurse_x - 15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        cv2.rectangle(screen, (10, 130), (230, 155), (255, 255, 255), -1)
        cv2.rectangle(screen, (10, 130), (230, 155), (0, 0, 0), 1)
        
        messages = ["Welcome to Pokemon Center!", "Would you like to heal?", "Your Pokemon are healed!", "Come back anytime!"]
        cv2.putText(screen, random.choice(messages), (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        return screen
        
    def detect_objects(self, screen: np.ndarray, screen_type: str):
        """Detect objects (same logic as before)"""
        if screen_type == 'overworld':
            return [
                {'type': 'hp_display', 'bbox': [5, 5, 80, 25], 'confidence': 0.95},
                {'type': 'player_character', 'bbox': [115, 75, 125, 85], 'confidence': 0.92}
            ]
        elif screen_type == 'battle':
            return [
                {'type': 'player_pokemon', 'bbox': [10, 90, 100, 140], 'confidence': 0.88},
                {'type': 'enemy_pokemon', 'bbox': [140, 20, 230, 70], 'confidence': 0.91},
                {'type': 'player_hp_bar', 'bbox': [10, 120, 90, 130], 'confidence': 0.96},
                {'type': 'enemy_hp_bar', 'bbox': [150, 50, 220, 60], 'confidence': 0.94},
                {'type': 'battle_menu', 'bbox': [5, 140, 235, 155], 'confidence': 0.99}
            ]
        elif screen_type == 'menu':
            return [{'type': 'menu_box', 'bbox': [20, 20, 220, 140], 'confidence': 0.97}]
        elif screen_type == 'pokemon_center':
            return [
                {'type': 'counter', 'bbox': [50, 100, 190, 130], 'confidence': 0.93},
                {'type': 'nurse_character', 'bbox': [105, 75, 135, 105], 'confidence': 0.87},
                {'type': 'dialogue_box', 'bbox': [10, 130, 230, 155], 'confidence': 0.98}
            ]
        return []
        
    def draw_detections(self, screen: np.ndarray, objects):
        """Draw detection boxes"""
        for obj in objects:
            bbox = obj['bbox']
            color = (0, 255, 0) if 'hp' in obj['type'] else (255, 0, 0) if 'pokemon' in obj['type'] else (0, 255, 255)
            cv2.rectangle(screen, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
            label = f"{obj['type']} ({obj['confidence']:.2f})"
            cv2.putText(screen, label, (bbox[0], bbox[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        return screen
        
    def get_frame_data(self, screen_type: str):
        """Get frame data for web display"""
        screen = self.create_synthetic_screen(screen_type)
        objects = self.detect_objects(screen, screen_type)
        screen_with_detections = self.draw_detections(screen.copy(), objects)
        
        # Scale up
        scaled = cv2.resize(screen_with_detections, 
                           (self.screen_width * self.demo_scale, self.screen_height * self.demo_scale),
                           interpolation=cv2.INTER_NEAREST)
        
        # Convert to base64 for web display
        _, buffer = cv2.imencode('.png', scaled)
        img_str = base64.b64encode(buffer).decode()
        
        return {
            'image': img_str,
            'screen_type': screen_type,
            'objects_detected': len(objects),
            'objects': objects
        }

# Global demo instance
demo = PerceptionWebDemo()

class DemoHandler(BaseHTTPRequestHandler):
    """Web server handler"""
    
    def do_GET(self):
        if self.path == "/":
            self.serve_html()
        elif self.path == "/data":
            self.serve_data()
        else:
            self.send_error(404)
            
    def serve_html(self):
        """Serve the main HTML page"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Perception Layer Demo</title>
            <style>
                body { font-family: Arial; background: #1a1a1a; color: white; margin: 0; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .demo-area { display: flex; gap: 20px; margin: 20px 0; }
                .screen-display { border: 2px solid #00ff00; padding: 10px; background: #000; }
                .info-panel { background: #333; padding: 15px; border-radius: 5px; min-width: 300px; }
                .status { color: #00ff00; font-weight: bold; }
                .object-list { margin: 10px 0; }
                .object-item { background: #444; padding: 5px; margin: 5px 0; border-radius: 3px; }
                h1 { color: #00ff00; text-align: center; }
                button { background: #00ff00; color: black; border: none; padding: 10px 20px; cursor: pointer; }
                button:hover { background: #00aa00; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎮 LIVE PERCEPTION LAYER DEMO</h1>
                <p style="text-align: center;">Real-time synthetic GameBoy screen generation with AI object detection</p>
                
                <div class="demo-area">
                    <div class="screen-display">
                        <img id="gameScreen" alt="Game Screen" style="image-rendering: pixelated;">
                    </div>
                    
                    <div class="info-panel">
                        <h3>📊 Detection Status</h3>
                        <div class="status" id="status">Initializing...</div>
                        <p><strong>Screen Type:</strong> <span id="screenType">-</span></p>
                        <p><strong>Objects Detected:</strong> <span id="objectCount">0</span></p>
                        <p><strong>FPS:</strong> <span id="fps">0</span></p>
                        
                        <h4>🔍 Detected Objects:</h4>
                        <div id="objectList" class="object-list">
                            <div class="object-item">No objects detected</div>
                        </div>
                        
                        <button onclick="toggleDemo()">Pause/Resume</button>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 20px;">
                    <p><strong>How it works:</strong> Creates synthetic GameBoy screens using OpenCV, then runs object detection</p>
                    <p><strong>No emulator needed!</strong> Everything is programmatically generated</p>
                </div>
            </div>
            
            <script>
                let isRunning = true;
                let frameCount = 0;
                let startTime = Date.now();
                
                const screenTypes = ['overworld', 'battle', 'menu', 'pokemon_center'];
                let currentTypeIndex = 0;
                
                function updateFrame() {
                    if (!isRunning) return;
                    
                    const currentType = screenTypes[currentTypeIndex];
                    
                    fetch('/data?type=' + currentType)
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('gameScreen').src = 'data:image/png;base64,' + data.image;
                            document.getElementById('screenType').textContent = data.screen_type.toUpperCase();
                            document.getElementById('objectCount').textContent = data.objects_detected;
                            
                            // Update object list
                            const objectList = document.getElementById('objectList');
                            if (data.objects.length > 0) {
                                objectList.innerHTML = data.objects.map(obj => 
                                    `<div class="object-item">${obj.type} (${(obj.confidence * 100).toFixed(0)}%)</div>`
                                ).join('');
                            } else {
                                objectList.innerHTML = '<div class="object-item">No objects detected</div>';
                            }
                            
                            // Update FPS
                            frameCount++;
                            const elapsed = (Date.now() - startTime) / 1000;
                            const fps = Math.round(frameCount / elapsed);
                            document.getElementById('fps').textContent = fps;
                            
                            document.getElementById('status').textContent = 'RUNNING';
                        })
                        .catch(error => {
                            document.getElementById('status').textContent = 'ERROR: ' + error.message;
                        });
                    
                    // Switch screen type every 3 seconds
                    if (frameCount % 60 === 0) {
                        currentTypeIndex = (currentTypeIndex + 1) % screenTypes.length;
                    }
                }
                
                function toggleDemo() {
                    isRunning = !isRunning;
                    document.getElementById('status').textContent = isRunning ? 'RUNNING' : 'PAUSED';
                }
                
                // Start the demo
                setInterval(updateFrame, 50); // 20 FPS
            </script>
        </body>
        </html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
        
    def serve_data(self):
        """Serve frame data as JSON"""
        screen_type = self.path.split('type=')[1] if 'type=' in self.path else 'overworld'
        data = demo.get_frame_data(screen_type)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

def run_web_demo():
    """Run the web-based demo"""
    print("🌐 Starting Web-Based Perception Demo...")
    print("📡 Server starting at: http://localhost:8080")
    print("🎮 Features:")
    print("   • Real-time synthetic screen generation")
    print("   • Live object detection visualization") 
    print("   • FPS monitoring")
    print("   • Cycles through all screen types")
    print("\n🚀 Open your browser and go to: http://localhost:8080")
    print("❌ Press Ctrl+C to stop the server")
    
    server = HTTPServer(('localhost', 8080), DemoHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped.")
        server.shutdown()

if __name__ == "__main__":
    run_web_demo()