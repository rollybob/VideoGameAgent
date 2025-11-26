#!/usr/bin/env python3
"""
🔄 Shared Class Manager - Centralized Object Class Management
============================================================

Single source of truth for object classes across all perception tools.
Automatically syncs between comp_model_trainer and backseat_data_collector.
"""

import json
import threading
import time
from pathlib import Path
from typing import List, Dict, Set, Callable, Optional
from datetime import datetime
import hashlib

class SharedClassManager:
    """Centralized management of object classes with auto-sync"""
    
    def __init__(self, config_file: str = "shared_classes.json"):
        self.config_file = Path(config_file)
        self.lock = threading.Lock()
        self.subscribers = []  # Tools that want to be notified of changes
        self.last_hash = ""
        
        # Default classes (fallback)
        self.default_classes = [
            "building", "dialogue_box", "hp_bar", "menu_box", 
            "npc_character", "player_character", "pokemon_sprite", 
            "text_area", "tree", "pokeball_item", "grass", "water", 
            "exp_bar", "level_indicator", "missed_object", "other"
        ]
        
        # Load or create initial class list
        self.classes = self._load_classes()
        self._save_classes()
        
        # Start file watcher thread
        self.monitoring = True
        self.watcher_thread = threading.Thread(target=self._watch_file_changes, daemon=True)
        self.watcher_thread.start()
    
    def _load_classes(self) -> List[str]:
        """Load classes from config file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    classes = data.get('classes', self.default_classes)
                    self.last_hash = self._compute_hash(classes)
                    print(f"[ClassManager] Loaded {len(classes)} classes from {self.config_file}")
                    return classes
        except Exception as e:
            print(f"[ClassManager] Error loading classes: {e}")
        
        # Use defaults
        print(f"[ClassManager] Using default {len(self.default_classes)} classes")
        return self.default_classes.copy()
    
    def _save_classes(self):
        """Save classes to config file"""
        try:
            with self.lock:
                data = {
                    'classes': self.classes,
                    'last_updated': datetime.now().isoformat(),
                    'version': self._compute_hash(self.classes),
                    'metadata': {
                        'total_classes': len(self.classes),
                        'source': 'SharedClassManager'
                    }
                }
                
                with open(self.config_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.last_hash = data['version']
                print(f"[ClassManager] Saved {len(self.classes)} classes")
                
        except Exception as e:
            print(f"[ClassManager] Error saving classes: {e}")
    
    def _compute_hash(self, classes: List[str]) -> str:
        """Compute hash of class list for change detection"""
        class_str = json.dumps(sorted(classes))
        return hashlib.md5(class_str.encode()).hexdigest()[:8]
    
    def _watch_file_changes(self):
        """Watch for external file changes and notify subscribers"""
        while self.monitoring:
            try:
                if self.config_file.exists():
                    with open(self.config_file, 'r') as f:
                        data = json.load(f)
                        file_hash = data.get('version', '')
                        
                        if file_hash != self.last_hash:
                            # File was changed externally
                            new_classes = data.get('classes', [])
                            if new_classes:
                                with self.lock:
                                    self.classes = new_classes
                                    self.last_hash = file_hash
                                
                                print(f"[ClassManager] External change detected - {len(new_classes)} classes")
                                self._notify_subscribers()
                
            except Exception as e:
                pass  # Ignore file read errors during monitoring
            
            time.sleep(2)  # Check every 2 seconds
    
    def subscribe(self, callback: Callable[[List[str]], None], name: str = "Unknown"):
        """Subscribe to class list changes"""
        with self.lock:
            self.subscribers.append({'callback': callback, 'name': name})
            print(f"[ClassManager] {name} subscribed to class updates")
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from class list changes"""
        with self.lock:
            self.subscribers = [s for s in self.subscribers if s['callback'] != callback]
    
    def _notify_subscribers(self):
        """Notify all subscribers of class changes"""
        with self.lock:
            for subscriber in self.subscribers:
                try:
                    subscriber['callback'](self.classes.copy())
                    print(f"[ClassManager] Notified {subscriber['name']}")
                except Exception as e:
                    print(f"[ClassManager] Error notifying {subscriber['name']}: {e}")
    
    def get_classes(self) -> List[str]:
        """Get current class list"""
        with self.lock:
            return self.classes.copy()
    
    def add_class(self, class_name: str, source: str = "Unknown") -> bool:
        """Add a new class"""
        with self.lock:
            if class_name not in self.classes:
                self.classes.append(class_name)
                self._save_classes()
                print(f"[ClassManager] Added class '{class_name}' from {source}")
                self._notify_subscribers()
                return True
            return False
    
    def remove_class(self, class_name: str, source: str = "Unknown") -> bool:
        """Remove a class"""
        with self.lock:
            if class_name in self.classes:
                self.classes.remove(class_name)
                self._save_classes()
                print(f"[ClassManager] Removed class '{class_name}' from {source}")
                self._notify_subscribers()
                return True
            return False
    
    def update_classes(self, new_classes: List[str], source: str = "Unknown") -> bool:
        """Update entire class list"""
        with self.lock:
            if set(new_classes) != set(self.classes):
                self.classes = new_classes.copy()
                self._save_classes()
                print(f"[ClassManager] Updated to {len(new_classes)} classes from {source}")
                self._notify_subscribers()
                return True
            return False
    
    def get_stats(self) -> Dict:
        """Get manager statistics"""
        with self.lock:
            return {
                'total_classes': len(self.classes),
                'subscribers': len(self.subscribers),
                'config_file': str(self.config_file),
                'last_hash': self.last_hash,
                'subscriber_names': [s['name'] for s in self.subscribers]
            }
    
    def shutdown(self):
        """Shutdown the manager"""
        self.monitoring = False
        print("[ClassManager] Shutdown complete")

# Global instance
_shared_manager = None

def get_shared_class_manager() -> SharedClassManager:
    """Get the global shared class manager"""
    global _shared_manager
    if _shared_manager is None:
        _shared_manager = SharedClassManager()
    return _shared_manager

def cleanup_shared_manager():
    """Cleanup global manager (call on app exit)"""
    global _shared_manager
    if _shared_manager:
        _shared_manager.shutdown()
        _shared_manager = None

# Convenience functions for direct use
def get_classes() -> List[str]:
    """Get current class list"""
    return get_shared_class_manager().get_classes()

def add_class(class_name: str, source: str = "Direct") -> bool:
    """Add a class"""
    return get_shared_class_manager().add_class(class_name, source)

def remove_class(class_name: str, source: str = "Direct") -> bool:
    """Remove a class"""
    return get_shared_class_manager().remove_class(class_name, source)

def subscribe_to_changes(callback: Callable[[List[str]], None], name: str = "Unknown"):
    """Subscribe to class changes"""
    get_shared_class_manager().subscribe(callback, name)

# Testing
if __name__ == "__main__":
    print("Testing Shared Class Manager...")
    
    manager = SharedClassManager("test_classes.json")
    
    # Test callback
    def test_callback(classes):
        print(f"   Callback received: {len(classes)} classes")
    
    manager.subscribe(test_callback, "Test")
    
    print(f"Initial classes: {len(manager.get_classes())}")
    
    # Test add/remove
    manager.add_class("test_class", "Test")
    manager.remove_class("test_class", "Test")
    
    print(f"Final classes: {len(manager.get_classes())}")
    print(f"Stats: {manager.get_stats()}")
    
    manager.shutdown()
    print("Test complete!")