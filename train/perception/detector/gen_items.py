"""Item (collectible) detection data from keydrop states (key on the floor).
Link random-walks (in-room, biased AWAY from the key so he doesn't collect it); the key's
SCREEN position varies as the camera follows Link -> position diversity. Label rule:
on-screen sprite slot HP>0 = enemy, HP==0 = ITEM (key/rupee/heart). Split the keydrop
states 70/30 -> item_train.npz / item_held.npz (held = unseen key states = generalization).
Same schema as gen_data (frames/link/enem/item)."""
import os, sys, glob, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); RL=os.path.abspath(os.path.join(HERE,"..","..","rl"))
sys.path.insert(0,RL)
from alttp_ppo_env import AlttpPpoEnv
from harvest_key_frames import u16
CAM_X,CAM_Y=0x02B82,0x02B86
DIRS=[1,2,3,4,5,6,7,8]

def onscreen(iw,cx,cy):
    # ITEM = SLOT 3 ONLY (v5 relabel, 2026-08-20): the keydrop banks' key lives in
    # slot 3 (harvest convention). The old any-hp==0-slot rule labeled room4's
    # STATUES as items -> the net fired on them at key-level conf (composer round-1
    # router theft). Statues/junk slots now stay unlabeled = background negatives.
    en=[]; it=[]
    for i in range(16):
        ex,ey=u16(iw,0x03846+4*i),u16(iw,0x03848+4*i)
        if (ex,ey)==(0,0): continue
        sx,sy=ex-cx,ey-cy
        if 0<=sx<240 and 0<=sy<160:
            if int(iw[0x03250+i])>0: en.append((sx,sy))
            elif i==3: it.append((sx,sy))
    return en,it

def pad(lst):
    a=np.full((8,2),-1,np.int16)
    for j,(sx,sy) in enumerate(lst[:8]): a[j]=(sx,sy)
    return a

def gen_state(st, n_frames=45, seed=200):
    from mgba._pylib import ffi
    def new():
        e=AlttpPpoEnv(state_path=st,horizon=10**7,death_terminates=True); e.reset(seed=seed)
        return e, ffi.cast("uint8_t *",e.core._native.memory.iwram)
    env,iw=new(); rng=random.Random(seed)
    F=[];L=[];E=[];I=[]; d=rng.choice(DIRS); prev=None; since=0; guard=0
    while len(F)<n_frames and guard<n_frames*30:
        guard+=1
        cx,cy=u16(iw,CAM_X),u16(iw,CAM_Y)
        lsx,lsy=u16(iw,0x038F4)-cx,u16(iw,0x038F0)-cy
        en,it=onscreen(iw,cx,cy)
        if it:                                   # steer away from the key so we don't collect it
            kx,ky=it[0]
            if abs(lsx-kx)+abs(lsy-ky)<45:
                d=(3 if kx>lsx else 4) if abs(lsx-kx)>=abs(lsy-ky) else (1 if ky>lsy else 2)
        if   lsx<35:  d=4
        elif lsx>205: d=3
        elif lsy<35:  d=2
        elif lsy>135: d=1
        elif rng.random()<0.15: d=rng.choice(DIRS)
        env.step(np.array([d,0,0,0,0],dtype=np.int64)); since+=1   # no attack
        if env.oracle.read_health(env.core)<=0 or since>=300:
            env.close(); env,iw=new(); prev=None; since=0; continue
        if since%2: continue
        cx,cy=u16(iw,CAM_X),u16(iw,CAM_Y)
        en,it=onscreen(iw,cx,cy)
        if not it:                               # key collected/gone -> respawn the state
            env.close(); env,iw=new(); prev=None; continue
        if prev is not None and abs(cx-prev[0])+abs(cy-prev[1])>6:
            prev=(cx,cy); continue
        prev=(cx,cy)
        lsx,lsy=u16(iw,0x038F4)-cx,u16(iw,0x038F0)-cy
        if not (0<=lsx<240 and 0<=lsy<160): continue
        F.append(np.array(env._frames[-1],np.uint8)); L.append((lsx,lsy)); E.append(pad(en)); I.append(pad(it))
    env.close()
    return F,L,E,I

if __name__=="__main__":
    os.makedirs(os.path.join(HERE,"data"),exist_ok=True)
    banks=[os.path.join(RL,"..","perception",b,"states") for b in ("keybank_holdout2","keyprobe_v2","keybank_holdout")]
    states=[]
    for b in banks: states+=sorted(glob.glob(os.path.join(b,"keydrop-*.state")))
    random.Random(0).shuffle(states)
    ntr=int(len(states)*0.7)
    for name,sub in (("item_train",states[:ntr]),("item_held",states[ntr:])):
        F=[];L=[];E=[];I=[]
        for si,st in enumerate(sub):
            f,l,e,i=gen_state(st,seed=200+si); F+=f;L+=l;E+=e;I+=i
        np.savez_compressed(os.path.join(HERE,"data",name+".npz"),
            frames=np.stack(F),link=np.array(L,np.int16),enem=np.stack(E),item=np.stack(I))
        ipf=np.mean([(np.asarray(ii)[:,0]>=0).sum() for ii in I]) if I else 0
        print(f"{name}: {len(F)} frames from {len(sub)} states, items/frame={ipf:.2f}",flush=True)
