import os, sys, warnings
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); RL=os.path.abspath(os.path.join(HERE,"..","..","rl"))
sys.path.insert(0,RL)
from alttp_ppo_env import AlttpPpoEnv
from harvest_key_frames import u16
from mgba._pylib import ffi
# gather link_world + full IWRAM per room
rooms=[]
for i in range(7):
    env=AlttpPpoEnv(state_path=os.path.join(RL,"states",f"alttp_human-0{i}.state"),horizon=16)
    env.reset(seed=0); iw=ffi.cast("uint8_t *",env.core._native.memory.iwram)
    lx,ly=u16(iw,0x038F4),u16(iw,0x038F0)
    snap=bytes(iw[a] for a in range(0x8000))   # 32KB IWRAM snapshot
    rooms.append((lx,ly,snap)); env.close()
def val(snap,a): return snap[a]|(snap[a+1]<<8)
# origin_x addr: link_world_x - RAM[addr] in [0,240) for ALL rooms; likewise y in [0,160)
xc=[a for a in range(0,0x7FFE,2) if all(0<=r[0]-val(r[2],a)<240 for r in rooms)]
yc=[a for a in range(0,0x7FFE,2) if all(0<=r[1]-val(r[2],a)<160 for r in rooms)]
print(f"origin_X candidates ({len(xc)}): "+", ".join(f"0x{a:05X}" for a in xc[:20]))
print(f"origin_Y candidates ({len(yc)}): "+", ".join(f"0x{a:05X}" for a in yc[:20]))
# show implied Link screen pos per room for the first few joint (x,y adjacent) candidates
for ax in xc[:8]:
    for ay in (ax-4,ax-2,ax+2,ax+4):
        if ay in yc:
            scr=[(r[0]-val(r[2],ax), r[1]-val(r[2],ay)) for r in rooms]
            print(f"  Xaddr 0x{ax:05X} Yaddr 0x{ay:05X} -> link_screen per room: {scr}")
