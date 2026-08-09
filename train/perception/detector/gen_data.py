"""Phase 1 (v2): RAM-labeled detection data. screen = world - camera(0x02B82/86).
Improvements: (1) center-biased walk keeps Link IN the room (avoids door transitions);
(2) drop transition frames (room-cell change, or camera jump >6px/step); (3) drop
death/gameover (death_terminates + health>0); (4) recreate the env every ~300 frames --
the mgba binding SEGFAULTS on long runs, so keep each instance short-lived.
Labels: Link screen (near-constant, camera-centered) + enemy sprite slots (the REAL,
high-variance target). One room per process (--room), driven by run_detector.sh + retry."""
import os, sys, warnings, random, argparse
warnings.filterwarnings("ignore")
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); RL=os.path.abspath(os.path.join(HERE,"..","..","rl"))
sys.path.insert(0,RL)
from alttp_ppo_env import AlttpPpoEnv
from harvest_key_frames import u16
CAM_X,CAM_Y=0x02B82,0x02B86
DIRS=[1,2,3,4,5,6,7,8]   # env dir idx: 1=U 2=D 3=L 4=R (+diagonals)

def gen_room(rm, n_frames=900, seed=100):
    from mgba._pylib import ffi
    state=os.path.join(RL,"states",f"alttp_human-{rm:02d}.state")
    def new_env():
        e=AlttpPpoEnv(state_path=state,horizon=10**7,death_terminates=True)
        e.reset(seed=seed); iw=ffi.cast("uint8_t *",e.core._native.memory.iwram)
        return e,iw
    env,iw=new_env()
    start_cell=(u16(iw,0x038F4)>>9,u16(iw,0x038F0)>>9)
    rng=random.Random(seed*13+rm)
    frames=[]; links=[]; enem=[]; item=[]
    d=rng.choice(DIRS); prev_cam=None; since=0
    while len(frames)<n_frames:
        cx,cy=u16(iw,CAM_X),u16(iw,CAM_Y)
        lsx,lsy=u16(iw,0x038F4)-cx,u16(iw,0x038F0)-cy
        if   lsx<40:  d=4                       # center-bias: steer away from edges
        elif lsx>200: d=3
        elif lsy<40:  d=2
        elif lsy>130: d=1
        elif rng.random()<0.12: d=rng.choice(DIRS)
        atk=1 if rng.random()<0.10 else 0
        env.step(np.array([d,0,atk,0,0],dtype=np.int64)); since+=1
        term=env.oracle.read_health(env.core)<=0
        if term or since>=300:                  # death OR periodic recreate (segfault dodge)
            env.close(); env,iw=new_env(); prev_cam=None; since=0; continue
        if since%2: continue                    # subsample
        h=env.oracle.read_health(env.core)
        if not (0<h<=200): continue             # skip dead/garbage
        cx,cy=u16(iw,CAM_X),u16(iw,CAM_Y)
        if (u16(iw,0x038F4)>>9,u16(iw,0x038F0)>>9)!=start_cell:   # left the room -> transition
            env.close(); env,iw=new_env(); prev_cam=None; since=0; continue
        if prev_cam is not None and abs(cx-prev_cam[0])+abs(cy-prev_cam[1])>6:
            prev_cam=(cx,cy); continue           # camera jump = mid-scroll transition -> drop
        prev_cam=(cx,cy)
        lsx,lsy=u16(iw,0x038F4)-cx,u16(iw,0x038F0)-cy
        if not (0<=lsx<240 and 0<=lsy<160): continue
        en=[]; it=[]                             # on-screen sprites split by HP
        for i in range(16):
            ex,ey=u16(iw,0x03846+4*i),u16(iw,0x03848+4*i)
            if (ex,ey)==(0,0): continue
            sx,sy=ex-cx,ey-cy
            if 0<=sx<240 and 0<=sy<160:
                (en if int(iw[0x03250+i])>0 else it).append((sx,sy))   # HP>0 enemy / HP==0 item
        frames.append(np.array(env._frames[-1],np.uint8))
        links.append((lsx,lsy))
        def pad(lst):
            a=np.full((8,2),-1,np.int16)
            for j,(sx,sy) in enumerate(lst[:8]): a[j]=(sx,sy)
            return a
        enem.append(pad(en)); item.append(pad(it))
    env.close()
    return (np.stack(frames).astype(np.uint8),np.array(links,np.int16),
            np.stack(enem).astype(np.int16),np.stack(item).astype(np.int16))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--room",type=int,required=True); a=ap.parse_args()
    os.makedirs(os.path.join(HERE,"data"),exist_ok=True)
    F,L,E,I=gen_room(a.room)
    np.savez_compressed(os.path.join(HERE,"data",f"room{a.room}.npz"),frames=F,link=L,enem=E,item=I)
    ne=np.sum(E[:,:,0]>=0,axis=1); ni=np.sum(I[:,:,0]>=0,axis=1)
    print(f"room{a.room}: {len(F)} frames | enemies/f={ne.mean():.2f} items/f={ni.mean():.2f} "
          f"| link_x {int(L[:,0].min())}-{int(L[:,0].max())} y {int(L[:,1].min())}-{int(L[:,1].max())}",flush=True)
