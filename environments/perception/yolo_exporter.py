
# yolo_exporter.py
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2

# Canonical, sorted class order synchronized with shared class manager
CANONICAL_CLASSES = [
    "building",
    "dialogue_box",
    "exp_bar",
    "grass", 
    "hp_bar",
    "level_indicator",
    "menu_box",
    "missed_object",
    "npc_character",
    "other",
    "player_character",
    "pokeball_item",
    "pokemon_sprite",
    "text_area",
    "tree",
    "water",
]

# Normalize free-text labels to the canonical set above
_ALIASES = {
    # Experience bar variations
    "exp._bar": "exp_bar",
    "exp bar": "exp_bar", 
    "exp": "exp_bar",
    "experience_bar": "exp_bar",
    
    # HP bar variations
    "hp": "hp_bar",
    "hpbar": "hp_bar", 
    "hp bar": "hp_bar",
    "health_bar": "hp_bar",
    
    # Pokeball/item variations
    "pokeball/item": "pokeball_item",
    "pokeball": "pokeball_item",
    "item": "pokeball_item",
    "pokeball_item": "pokeball_item",  # Identity mapping for safety
    
    # Text/dialogue variations
    "textbox": "text_area", 
    "text_box": "text_area",
    "dialogue": "dialogue_box",
    "dialog_box": "dialogue_box",
    "dialog": "dialogue_box",
    
    # Level indicator variations
    "level": "level_indicator",
    "lvl": "level_indicator",
    "level_info": "level_indicator",
    
    # Menu variations
    "menu": "menu_box",
    "menu_window": "menu_box",
    
    # Character variations
    "player": "player_character",
    "protagonist": "player_character",
    "npc": "npc_character",
    "trainer": "npc_character",
    "rival": "npc_character",
    
    # Pokemon variations
    "pokemon": "pokemon_sprite",
    "sprite": "pokemon_sprite",
    
    # Environment variations
    "building": "building",  # Identity mapping
    "structure": "building",
    "house": "building",
    "trees": "tree",
    "forest": "tree",
    "vegetation": "grass",
    "plants": "grass",
    
    # Catch-all variations
    "unknown": "other",
    "misc": "other",
    "miscellaneous": "other",
    "missed": "missed_object",
    "missing": "missed_object",
}

def _canonize(label: str) -> str:
    s = label.strip().lower()
    s = s.replace("/", "_").replace(" ", "_")
    s = s.replace("__", "_")
    return _ALIASES.get(s, s)

def write_classes_txt(labels_out: Path, classes: List[str] = None) -> None:
    classes = classes or CANONICAL_CLASSES
    labels_out.mkdir(parents=True, exist_ok=True)
    (labels_out / "classes.txt").write_text("\n".join(classes), encoding="utf-8")

def _clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def _to_yolo_line(cls_id: int, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> str:
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"{cls_id} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"

def save_yolo_frame(
    image_rgb: np.ndarray,
    detections: List[Dict],
    labels_out: Path,
    images_out: Path,
    classes: List[str] = None,
    stem: str = None,
) -> Tuple[Path, Path]:
    """Save one frame + its detections to YOLO format.
    detections: list of dicts with keys 'bbox'=[x1,y1,x2,y2], 'type'=label string.
    """
    classes = classes or CANONICAL_CLASSES
    name_to_id = {c: i for i, c in enumerate(classes)}

    # pick filename stem
    import time
    if not stem:
        stem = f"frame_{int(time.time()*1000)}"

    labels_out = Path(labels_out)
    images_out = Path(images_out)
    labels_out.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    # ensure classes.txt exists
    write_classes_txt(labels_out, classes)

    # image save (ensure BGR for cv2)
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image_rgb.copy()
    img_path = images_out / f"{stem}.png"
    cv2.imwrite(str(img_path), img_bgr)

    h, w = image_rgb.shape[:2]
    lines = []
    for det in detections:
        label = _canonize(det.get("type", ""))
        if label not in name_to_id:
            # skip unknown labels
            continue
        bbox = det.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        clipped = _clip_box(x1, y1, x2, y2, w, h)
        if not clipped:
            continue
        x1, y1, x2, y2 = clipped
        lines.append(_to_yolo_line(name_to_id[label], x1, y1, x2, y2, w, h))

    label_path = labels_out / f"{stem}.txt"
    label_path.write_text("\n".join(lines), encoding="utf-8")

    return img_path, label_path
