"""Phase 2/3 (3-class): heatmap detector for {Link, enemy, item(collectible)}.
Input (3,160,240) -> 3 heatmap channels at stride 2 (80x120). RAM-taught, pixels-only.
TRAIN: rooms{0,1,3} (enemy/Link) + item_train (keydrop items).
HELD-OUT: rooms{2,4} (enemy/Link) + item_held (unseen key states).
Bars: >=100 FPS; held-out enemy>=80% & item>=80% @4px (Link is camera-centered/confounded)."""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
DEV="cuda"; STRIDE=2; HH,HW=80,120; SIGMA=2.0; NCH=3   # 0=link 1=enemy 2=item
TRAIN_FILES=["room0.npz","room1.npz","room3.npz","item_train.npz"]
HELD_FILES =["room2.npz","room4.npz","item_held.npz"]
CLASSES=["link","enemy","item"]

def load(files):
    F_,L_,E_,I_=[],[],[],[]
    for f in files:
        p=os.path.join(DATA,f)
        if not os.path.exists(p): print(f"  (skip missing {f})",flush=True); continue
        d=np.load(p); F_.append(d["frames"]); L_.append(d["link"]); E_.append(d["enem"]); I_.append(d["item"])
    return np.concatenate(F_),np.concatenate(L_),np.concatenate(E_),np.concatenate(I_)

def heatmap(pts):
    hm=np.zeros((HH,HW),np.float32); yy,xx=np.mgrid[0:HH,0:HW]
    for sx,sy in pts:
        hx,hy=sx/STRIDE,sy/STRIDE
        hm=np.maximum(hm,np.exp(-((xx-hx)**2+(yy-hy)**2)/(2*SIGMA**2)))
    return hm

def make_targets(link,enem,item):
    n=len(link); T=np.zeros((n,NCH,HH,HW),np.float32)
    for i in range(n):
        T[i,0]=heatmap([tuple(link[i])])
        es=[tuple(e) for e in enem[i] if e[0]>=0];  T[i,1]=heatmap(es) if es else 0
        it=[tuple(e) for e in item[i] if e[0]>=0];  T[i,2]=heatmap(it) if it else 0
    return T

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i,o,st): return nn.Sequential(nn.Conv2d(i,o,3,st,1),nn.BatchNorm2d(o),nn.ReLU(True))
        s.e1=blk(3,16,1); s.e2=blk(16,32,2); s.e3=blk(32,64,2); s.e4=blk(64,64,2)
        s.u1=nn.Sequential(nn.ConvTranspose2d(64,32,4,2,1),nn.BatchNorm2d(32),nn.ReLU(True))
        s.u2=nn.Sequential(nn.ConvTranspose2d(32,32,4,2,1),nn.BatchNorm2d(32),nn.ReLU(True))
        s.head=nn.Conv2d(32,NCH,1)
    def forward(s,x): return s.head(s.u2(s.u1(s.e4(s.e3(s.e2(s.e1(x)))))))

def to_in(frames): return torch.from_numpy(frames).float().permute(0,3,1,2).div_(255.)

def decode_peak(hm):
    idx=int(hm.argmax()); hy,hx=divmod(idx,HW)
    y0,y1=max(0,hy-2),min(HH,hy+3); x0,x1=max(0,hx-2),min(HW,hx+3)
    win=np.clip(hm[y0:y1,x0:x1],0,None); s=win.sum()
    if s>1e-6:
        ys,xs=np.mgrid[y0:y1,x0:x1]; hy=(win*ys).sum()/s; hx=(win*xs).sum()/s
    return hx*STRIDE, hy*STRIDE

def main():
    Ftr,Ltr,Etr,Itr=load(TRAIN_FILES); Fhe,Lhe,Ehe,Ihe=load(HELD_FILES)
    print(f"train {len(Ftr)} frames, held-out {len(Fhe)}",flush=True)
    Xtr=to_in(Ftr); Ttr=torch.from_numpy(make_targets(Ltr,Etr,Itr))
    net=Net().to(DEV); opt=torch.optim.Adam(net.parameters(),1e-3)
    npar=sum(p.numel() for p in net.parameters()); print(f"params={npar/1e6:.2f}M",flush=True)
    bs=32; n=len(Xtr)
    for ep in range(30):
        net.train(); perm=torch.randperm(n); tot=0
        for i in range(0,n,bs):
            idx=perm[i:i+bs]; x=Xtr[idx].to(DEV); t=Ttr[idx].to(DEV)
            p=net(x); w=1+10*t; loss=(w*(p-t)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()*len(idx)
        if ep%5==0 or ep==29: print(f"ep{ep} loss={tot/n:.4f}",flush=True)
    def loc(F_,L_,E_,I_):
        net.eval(); N=len(F_); lk4=lk8=0; e4=e8=en=0; i4=i8=it=0
        with torch.no_grad():
            for k in range(0,N,64):
                pr=net(to_in(F_[k:k+64]).to(DEV)).cpu().numpy()
                for j in range(len(pr)):
                    px,py=decode_peak(pr[j,0]); gx,gy=L_[k+j]
                    dm=max(abs(px-gx),abs(py-gy)); lk4+=dm<=4; lk8+=dm<=8
                    for arr,ch,c4,c8,ct in ((E_,1,'e4','e8','en'),(I_,2,'i4','i8','it')):
                        es=[e for e in arr[k+j] if e[0]>=0]
                        if len(es)==1:
                            ex,ey=decode_peak(pr[j,ch]); d=max(abs(ex-es[0][0]),abs(ey-es[0][1]))
                            if ch==1: en+=1; e4+=d<=4; e8+=d<=8
                            else:     it+=1; i4+=d<=4; i8+=d<=8
        f=lambda h,N_:round(100*h/N_,1) if N_ else None
        return {"link@4":f(lk4,N),"link@8":f(lk8,N),"enemy@4":f(e4,en),"enemy@8":f(e8,en),
                "enemy_n":en,"item@4":f(i4,it),"item@8":f(i8,it),"item_n":it}
    held=loc(Fhe,Lhe,Ehe,Ihe); train_=loc(Ftr,Ltr,Etr,Itr)
    net.eval(); x1=to_in(Fhe[:1]).to(DEV)
    with torch.no_grad():
        for _ in range(10): net(x1)
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(300): net(x1)
        torch.cuda.synchronize(); fps=300/(time.time()-t0)
    res={"params_M":round(npar/1e6,3),"fps_bs1":round(fps,1),"heldout":held,"train":train_}
    print("RESULT",json.dumps(res),flush=True)
    json.dump(res,open(os.path.join(HERE,"_detector_result.json"),"w"),indent=2)
    torch.save(net.state_dict(),os.path.join(HERE,"detector.pt"))
    # overlay on held-out: link green, enemy red, item blue
    from PIL import Image
    net.eval(); tiles=[]; COL=[[0,255,0],[255,0,0],[0,128,255]]
    idxs=list(range(0,len(Fhe),max(1,len(Fhe)//10)))[:10]
    with torch.no_grad():
        for k in idxs:
            fr=Fhe[k].copy(); pr=net(to_in(Fhe[k:k+1]).to(DEV)).cpu().numpy()[0]
            for ch in range(NCH):
                px,py=decode_peak(pr[ch]); px,py=int(round(px)),int(round(py))
                for dd in range(-3,4):
                    if 0<=py<160 and 0<=px+dd<240: fr[py,px+dd]=COL[ch]
                    if 0<=px<240 and 0<=py+dd<160: fr[py+dd,px]=COL[ch]
            tiles.append(np.kron(fr,np.ones((2,2,1),np.uint8)))
    if tiles: Image.fromarray(np.vstack(tiles)).save(os.path.join(HERE,"_pred_overlay.png"))
    print("DONE",flush=True)

if __name__=="__main__": main()
