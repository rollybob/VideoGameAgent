"""Contact sheet of the 7 human room states' first frames so Tim can map our
numbering to the real game (dungeon key-room vs outside). Labels narrated separately."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from alttp_ppo_env import AlttpPpoEnv
try:
    from PIL import Image
except ImportError:
    import torchvision  # fallback
    Image = None

tiles = []
info = []
for i in range(7):
    env = AlttpPpoEnv(state_path=os.path.join(HERE,"states",f"alttp_human-0{i}.state"), horizon=16)
    env.reset(seed=0)
    x,y = env.oracle.read_pos(env.core); cell=(x>>9,y>>9)
    hp = env.oracle.read_health(env.core)
    f = env._frames[-1].copy().astype(np.uint8)        # (160,240,3) RGB
    # 4px white border + index stripe so tiles are separable at a glance
    f = np.pad(f, ((6,6),(6,6),(0,0)), constant_values=255)
    f[:6, :, :] = [255,0,0] if i in (2,4) else [40,40,40]   # red border = key room
    tiles.append(f); info.append((i,cell,hp))
    env.close()
while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[r:r+4]) for r in range(0, len(tiles), 4)]
sheet = np.vstack(rows)
out = os.path.join(HERE, "_states_contact.png")
Image.fromarray(sheet).save(out)
print("WROTE", out, sheet.shape)
print("order left->right, top row rooms 0-3, bottom row 4-6; red top-border = key room {2,4}")
for i,cell,hp in info: print(f"  room{i}: cell{cell} hp{hp}")
