import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); RL=os.path.abspath(os.path.join(HERE,"..","..","rl"))
sys.path.insert(0,RL)
from alttp_ppo_env import AlttpPpoEnv
from harvest_key_frames import u16
from PIL import Image
from mgba._pylib import ffi
tiles=[]
for rm in ("02","04"):
    env=AlttpPpoEnv(state_path=os.path.join(RL,"states",f"alttp_human-{rm}.state"),horizon=16)
    env.reset(seed=0); iw=ffi.cast("uint8_t *",env.core._native.memory.iwram)
    lwx,lwy=u16(iw,0x038F4),u16(iw,0x038F0); ex,ey=u16(iw,0x03852),u16(iw,0x03854)
    print(f"room{rm}: link_world=({lwx},{lwy})  enemy_world=({ex},{ey})  rel_enemy=({ex-lwx},{ey-lwy})")
    f=np.array(env._frames[-1],np.uint8)
    # draw a 10px grid so I can read screen pixel coords off the image
    g=f.copy()
    for x in range(0,240,20): g[:,x]=[80,80,80]
    for y in range(0,160,20): g[y,:]=[80,80,80]
    tiles.append(np.kron(g,np.ones((3,3,1),np.uint8))); env.close()
Image.fromarray(np.hstack(tiles)).save(os.path.join(HERE,"_calib.png"))
print("WROTE _calib.png (room2 | room4), 3x, 20px grid (=60px at 3x)")
