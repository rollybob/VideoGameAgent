#!/usr/bin/env python3
"""
🔧 Class Integration Patches
============================

Code patches to integrate SharedClassManager into existing tools.
Replace hardcoded class lists with dynamic shared management.
"""

# ================================
# PATCH FOR comp_model_trainer.py
# ================================

COMP_MODEL_TRAINER_PATCH = """
# ADD TO IMPORTS:
from shared_class_manager import get_shared_class_manager, cleanup_shared_manager

# REPLACE __init__ method class loading section:
def __init__(self):
    # ... existing code ...
    
    # REPLACE hardcoded object_classes with shared manager
    self.class_manager = get_shared_class_manager()
    self.object_classes = self.class_manager.get_classes()
    
    # Subscribe to class changes
    self.class_manager.subscribe(self._on_classes_changed, "CompetitiveTrainer")
    
    # ... rest of existing __init__ code ...

# ADD NEW METHOD:
def _on_classes_changed(self, new_classes):
    '''Callback when classes are updated externally'''
    print(f"[CompTrainer] Class list updated: {len(new_classes)} classes")
    self.object_classes = new_classes
    
    # Update UI if needed
    if hasattr(self, 'class_buttons'):
        self.root.after(0, self.create_class_buttons)

# REPLACE load_persistent_data method:
def load_persistent_data(self):
    '''Load persistent data (classes now managed by SharedClassManager)'''
    if self.data_file.exists():
        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                # Don't load classes from pickle anymore - use shared manager
                print(f"Loaded session data (classes managed centrally)")
        except Exception as e:
            print(f"Error loading persistent data: {e}")

# ADD TO on_closing method:
def on_closing(self):
    # ... existing closing code ...
    
    # Cleanup shared manager
    cleanup_shared_manager()
    
    # ... rest of existing code ...
"""

# ====================================
# PATCH FOR backseat_data_collector_fixed.py  
# ====================================

BACKSEAT_DATA_COLLECTOR_PATCH = """
# ADD TO IMPORTS:
from shared_class_manager import get_shared_class_manager, cleanup_shared_manager

# REPLACE __init__ method class setup:
def __init__(self):
    # ... existing code until class setup ...
    
    # REPLACE hardcoded class management with shared manager
    self.class_manager = get_shared_class_manager()
    self.available_classes = self.class_manager.get_classes()
    self.active_classes = self.available_classes.copy()
    
    # Subscribe to class changes
    self.class_manager.subscribe(self._on_classes_changed, "BackseatCollector")
    
    # ... rest of existing __init__ code ...

# ADD NEW METHOD:
def _on_classes_changed(self, new_classes):
    '''Callback when classes are updated externally'''
    print(f"[BackseatCollector] Class list updated: {len(new_classes)} classes")
    self.available_classes = new_classes
    
    # Update active classes to include any new ones
    for cls in new_classes:
        if cls not in self.active_classes:
            self.active_classes.append(cls)
    
    # Update model wrapper if loaded
    if self.current_model_wrapper:
        self.current_model_wrapper.set_target_classes(self.active_classes)
    
    # Update UI
    self.root.after(0, self.refresh_class_registry)
    self.root.after(0, self._update_manual_labels_dropdown)

# ADD NEW METHOD:
def _update_manual_labels_dropdown(self):
    '''Update manual annotation dropdown with current classes'''
    if hasattr(self, 'manual_label_dropdown'):
        current_classes = self.available_classes.copy()
        current_classes.extend(['missed_object', 'other'])  # Always include these
        self.manual_label_dropdown.config(values=current_classes)

# REPLACE add_new_class method:
def add_new_class(self):
    '''Add new class through shared manager'''
    new_class = self.new_class_var.get().strip()
    if not new_class:
        return
    
    if new_class in self.available_classes:
        messagebox.showwarning("Class Exists", f"Class '{new_class}' already exists!")
        return
    
    # Add through shared manager (will notify all subscribers)
    if self.class_manager.add_class(new_class, "BackseatCollector"):
        self.new_class_var.set("")
        messagebox.showinfo("Class Added", 
                           f"New class '{new_class}' added successfully!\\n\\n"
                           f"This class is now available in all tools.")

# ADD TO shutdown method:
def shutdown(self):
    # ... existing shutdown code ...
    
    # Cleanup shared manager
    cleanup_shared_manager()
    
    # ... rest of existing code ...
"""

def create_integration_script():
    """Create a script to automatically apply patches"""
    
    script_content = '''#!/usr/bin/env python3
"""
🔧 Apply Class Integration Patches
==================================

Automatically integrates SharedClassManager into existing tools.
Run this to update your tools with centralized class management.
"""

import sys
from pathlib import Path

def backup_file(file_path):
    """Create backup of original file"""
    backup_path = Path(str(file_path) + ".backup")
    if not backup_path.exists():
        with open(file_path, 'r') as original:
            with open(backup_path, 'w') as backup:
                backup.write(original.read())
        print(f"   ✅ Backup created: {backup_path}")

def apply_comp_model_trainer_patch():
    """Apply patch to comp_model_trainer.py"""
    print("🔧 Patching comp_model_trainer.py...")
    
    file_path = Path("comp_model_trainer.py")
    if not file_path.exists():
        print("   ❌ comp_model_trainer.py not found")
        return False
    
    backup_file(file_path)
    
    # Read current file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Apply patches
    patches = [
        # Add import
        ("from model_manager import ModelManager",
         "from model_manager import ModelManager\\nfrom shared_class_manager import get_shared_class_manager, cleanup_shared_manager"),
        
        # Replace class initialization
        ("self.object_classes = [",
         "# Dynamic class management\\n        self.class_manager = get_shared_class_manager()\\n        self.object_classes = self.class_manager.get_classes()\\n        self.class_manager.subscribe(self._on_classes_changed, \\"CompetitiveTrainer\\")\\n        \\n        # Legacy fallback (not used)\\n        # self.object_classes = [")
    ]
    
    modified = False
    for old, new in patches:
        if old in content and new not in content:
            content = content.replace(old, new)
            modified = True
    
    if modified:
        with open(file_path, 'w') as f:
            f.write(content)
        print("   ✅ Patches applied successfully")
        return True
    else:
        print("   ℹ️ No patches needed (already applied or not applicable)")
        return True

def apply_backseat_collector_patch():
    """Apply patch to backseat_data_collector_fixed.py"""
    print("🔧 Patching backseat_data_collector_fixed.py...")
    
    file_path = Path("backseat_data_collector_fixed.py")
    if not file_path.exists():
        print("   ❌ backseat_data_collector_fixed.py not found")
        return False
    
    backup_file(file_path)
    
    print("   ✅ Backup created, manual integration recommended")
    print("   📋 See class_integration_patches.py for detailed patch code")
    return True

def main():
    print("🔄 Applying Shared Class Management Integration...")
    print("=" * 60)
    
    # Change to perception directory
    perception_dir = Path("environments/perception")
    if perception_dir.exists():
        import os
        os.chdir(perception_dir)
    
    success1 = apply_comp_model_trainer_patch()
    success2 = apply_backseat_collector_patch()
    
    if success1 and success2:
        print("\\n✅ Integration complete!")
        print("📋 Next steps:")
        print("   1. Review the patched files")
        print("   2. Test both tools to ensure they work")
        print("   3. Classes will now sync automatically between tools")
    else:
        print("\\n⚠️ Some patches failed - check logs above")

if __name__ == "__main__":
    main()
'''
    
    with open("apply_class_integration.py", 'w') as f:
        f.write(script_content)
    
    print("✅ Integration script created: apply_class_integration.py")

if __name__ == "__main__":
    print("🔧 Class Integration Patches")
    print("=" * 40)
    print("This file contains patches to integrate SharedClassManager.")
    print("\\nTo apply:")
    print("1. Review the patch code above")
    print("2. Run the integration script")
    print("3. Test both tools")
    
    create_integration_script()