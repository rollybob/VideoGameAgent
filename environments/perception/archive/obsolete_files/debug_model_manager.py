#!/usr/bin/env python3
"""Debug script to test ModelManager regex patterns"""

import re
from pathlib import Path

# Test directory names we have
test_names = [
    "pokemon_object_detector",
    "pokemon_object_detector2", 
    "pokemon_object_detector7",
    "pokemon_object_detector2_competitive_20250728_210432",
    "pokemon_object_detector_competitive_20250728_193422",
    "pokemon_object_detector_competitive_20250728_201327"
]

base_model_name = "pokemon_object_detector"

patterns = [
    (rf"^{base_model_name}(\d*)$", "Original pattern"),
    (rf"^{base_model_name}(\d*)_competitive_\d+$", "Numbered competitive"),
    (rf"^{base_model_name}_competitive_\d+$", "Unnumbered competitive"),
]

print("Testing regex patterns against actual model directory names:")
print("=" * 70)

for name in test_names:
    print(f"\nTesting: {name}")
    for pattern_str, description in patterns:
        pattern = re.compile(pattern_str)
        match = pattern.match(name)
        status = "MATCH" if match else "NO MATCH"
        print(f"  {description:20} | {status}")

print("\n" + "="*70)
print("Checking actual models directory:")

models_dir = Path("models")
if models_dir.exists():
    for item in models_dir.iterdir():
        if item.is_dir():
            weights_path = item / "weights" / "best.pt" 
            has_weights = "HAS_WEIGHTS" if weights_path.exists() else "NO_WEIGHTS"
            print(f"{has_weights} {item.name}")

print("\nNow testing the actual ModelManager:")
from model_manager import ModelManager

manager = ModelManager()
models = manager.get_all_model_versions()
print(f"\nModelManager found {len(models)} models:")
for name, path, time in models:
    print(f"  - {name} ({time})")