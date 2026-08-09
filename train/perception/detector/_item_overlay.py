import os,sys,numpy as np,torch
sys.argv=['x']
import train_detector as T
from PIL import Image
net=T.Net().to(T.DEV); net.load_state_dict(torch.load(os.path.join(T.HERE,"detector.pt"))); net.eval()
d=np.load(os.path.join(T.DATA,"item_held.npz")); F=d["frames"]   # HELD-OUT key states
tiles=[]
for k in range(0,min(10*60,len(F)),60):
    fr=F[k].copy()
    with torch.no_grad(): pr=net(T.to_in(F[k:k+1]).to(T.DEV)).cpu().numpy()[0]
    for ch,c in ((0,[0,255,0]),(2,[0,128,255])):   # link green, ITEM blue
        px,py=T.decode_peak(pr[ch]); px,py=int(round(px)),int(round(py))
        for dd in range(-3,4):
            if 0<=py<160 and 0<=px+dd<240: fr[py,px+dd]=c
            if 0<=px<240 and 0<=py+dd<160: fr[py+dd,px]=c
    tiles.append(np.kron(fr,np.ones((2,2,1),np.uint8)))
Image.fromarray(np.vstack(tiles)).save(os.path.join(T.HERE,"_item_overlay.png"))
print("WROTE _item_overlay.png (held-out key states; green=Link blue=ITEM)")
