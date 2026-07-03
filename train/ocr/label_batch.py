"""Resumable labeling helper. For a game, find clean_manifest crops not yet in the labels
store, emit the next N as one readable contact sheet (<2000px) + print ids & tesseract seed.
Claude views the sheet, then appends {game,id,label} lines to realcap_labels.jsonl.
  python label_batch.py <game> [N]            -> writes /tmp/batch_<game>.png, prints ids
"""
import cv2, numpy as np, json, os, sys
LF="train/ocr/data/realcap_labels.jsonl"
game=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 16
d=f"train/ocr/data/realcap_{game}"
done=set()
if os.path.exists(LF):
    for line in open(LF):
        e=json.loads(line)
        if e["game"]==game: done.add(e["id"])
clean=json.load(open(f"{d}/clean_manifest.json"))
todo=[e for e in clean if e["id"] not in done]
batch=todo[:N]
rows=[]
for e in batch:
    im=cv2.imread(f"{d}/uniq_crops/{e['id']}.png")
    if im is None: continue
    h,w=im.shape[:2]; sc=max(1,int(38/max(1,h)))
    nw,nh=w*sc,h*sc
    MAXW=900
    if nw>MAXW:                       # scale wide lines DOWN to fit (show full text)
        f=MAXW/float(nw); nw,nh=MAXW,max(12,int(nh*f))
    im=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA if nw<w*sc else cv2.INTER_NEAREST)
    c=np.full((max(nh,22),215+nw,3),255,np.uint8)
    cv2.putText(c,e["id"],(2,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(120,0,0),1)
    c[0:nh,215:215+nw]=im
    rows.append(cv2.copyMakeBorder(c,0,3,0,0,cv2.BORDER_CONSTANT,value=(170,170,170)))
if rows:
    W=max(r.shape[1] for r in rows)
    rows=[cv2.copyMakeBorder(r,0,0,0,W-r.shape[1],cv2.BORDER_CONSTANT,value=(255,255,255)) for r in rows]
    cv2.imwrite(f"/tmp/batch_{game}.png", np.vstack(rows))
print(f"{game}: {len(done)} done, {len(todo)} clean remaining. this batch={len(batch)} -> /tmp/batch_{game}.png")
for e in batch: print(f"  {e['id']:>16}  tess=[{e.get('tesseract','')}]")
