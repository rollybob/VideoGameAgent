# verify_dataset.py
import os, glob

IMAGES = r"F:\GameAgentUSB\datasets\gba\images\train"
LABELS = r"F:\GameAgentUSB\datasets\gba\labels\train"

imgs = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(IMAGES, "*.*"))}
labs = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(LABELS, "*.txt"))}

missing_labels = sorted(imgs - labs)
missing_images = sorted(labs - imgs)

print(f"Images: {len(imgs)}  Labels: {len(labs)}")
print(f"Missing labels: {len(missing_labels)}")
for x in missing_labels[:15]: print("  no label for:", x)
print(f"Missing images: {len(missing_images)}")
for x in missing_images[:15]: print("  no image for:", x)