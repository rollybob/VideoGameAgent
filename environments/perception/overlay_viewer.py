import cv2, glob, os, random

IMAGES = r"F:\GameAgentUSB\datasets\gba\images\train"
LABELS = r"F:\GameAgentUSB\datasets\gba\labels\train"
CLASSES = ["building","dialogue_box","exp_bar","grass","hp_bar",
           "menu_box","npc_character","player_character",
           "pokeball_item","pokemon_sprite","tree","water"]

paths = random.sample(glob.glob(os.path.join(IMAGES, "*.*")), k=min(20, len(glob.glob(os.path.join(IMAGES, "*.*")))))
os.makedirs("overlay_checks", exist_ok=True)

for p in paths:
    img = cv2.imread(p)
    h, w = img.shape[:2]
    lab = os.path.join(LABELS, os.path.splitext(os.path.basename(p))[0]+".txt")
    if os.path.exists(lab):
        with open(lab) as f:
            for line in f:
                cid, cx, cy, bw, bh = line.strip().split()
                cid = int(cid); cx=float(cx)*w; cy=float(cy)*h; bw=float(bw)*w; bh=float(bh)*h
                x1=int(cx-bw/2); y1=int(cy-bh/2); x2=int(cx+bw/2); y2=int(cy+bh/2)
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
                name = CLASSES[cid] if 0<=cid<len(CLASSES) else str(cid)
                cv2.putText(img,name,(x1,max(10,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1,cv2.LINE_AA)
    out = os.path.join("overlay_checks", os.path.basename(p))
    cv2.imwrite(out, cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
print("Wrote overlays to ./overlay_checks")