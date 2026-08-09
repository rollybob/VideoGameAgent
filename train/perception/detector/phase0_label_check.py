"""Phase 0 (v2): overlay RAM labels using the FOUND camera origin (0x02B82/0x02B86).
screen = world - camera. Link green, enemy-slots red. Confirms the label pipeline."""
import os, sys, warnings, random
warnings.filterwarnings("ignore")
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); RL=os.path.abspath(os.path.join(HERE,"..","..","rl"))
sys.path.insert(0,RL)
from alttp_ppo_env import AlttpPpoEnv
from harvest_key_frames import u16
from chain_harvest import ChainHunter, DIR_KEYS
from PIL import Image
CAM_X,CAM_Y=0x02B82,0x02B86
def eslot(iw,i): return u16(iw,0x03846+4*i), u16(iw,0x03848+4*i)
def cross(img,x,y,c,r=3):
    h,w,_=img.shape
    for d in range(-r,r+1):
        if 0<=y<h and 0<=x+d<w: img[y,x+d]=c
        if 0<=x<w and 0<=y+d<h: img[y+d,x]=c
def sample(state,n=3,seed=6):
    from mgba._pylib import ffi
    env=AlttpPpoEnv(state_path=state,horizon=400); env.reset(seed=seed)
    iw=ffi.cast("uint8_t *",env.core._native.memory.iwram)
    rng=random.Random(seed); _=rng.random(); h=ChainHunter(iw,env.oracle,env.core,rng,validate=True)
    shots=[]; drops=[]; step=0; grab=set(np.linspace(30,240,n).astype(int))
    while step<=max(grab):
        env.step(np.array(h.act(drops),dtype=np.int64)); step+=1
        if step in grab:
            f=np.array(env._frames[-1],np.uint8).copy()
            cx,cy=u16(iw,CAM_X),u16(iw,CAM_Y)
            lsx,lsy=u16(iw,0x038F4)-cx,u16(iw,0x038F0)-cy
            cross(f,lsx,lsy,[0,255,0])
            for i in range(16):
                ex,ey=eslot(iw,i)
                if (ex,ey)==(0,0): continue
                sx,sy=ex-cx,ey-cy
                if 0<=sx<240 and 0<=sy<160: cross(f,sx,sy,[255,0,0])
            shots.append(np.kron(f,np.ones((2,2,1),np.uint8)))
    env.close(); return shots
alls=[]
for rm in ("02","03","04"): alls+=sample(os.path.join(RL,"states",f"alttp_human-{rm}.state"))
sep=np.full((3,alls[0].shape[1],3),255,np.uint8); grid=[]
for s in alls: grid+=[s,sep]
Image.fromarray(np.vstack(grid)).save(os.path.join(HERE,"_label_check.png"))
print("WROTE _label_check.png (rooms 2,3,4; green=Link red=enemy-slots, camera-corrected)")
