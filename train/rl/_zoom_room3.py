import os,sys,warnings; warnings.filterwarnings("ignore")
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from alttp_ppo_env import AlttpPpoEnv
from PIL import Image
# room3 first frame at 4x, clean, plus frames after 40/80 idle steps to show whether
# the "enemies" MOVE (real enemy) or stay put (statue).
env=AlttpPpoEnv(state_path=os.path.join(HERE,"states","alttp_human-03.state"),horizon=200)
env.reset(seed=0)
shots=[env._frames[-1].copy()]
import numpy as np
for _ in range(2):
    for _ in range(40):
        env.step(np.array([0,0,0,0,0]))   # idle (no input)
    shots.append(env._frames[-1].copy())
imgs=[np.array(s,np.uint8) for s in shots]
# stack the 3 timepoints vertically, 4x zoom
big=[np.kron(im,np.ones((4,4,1),np.uint8)) for im in imgs]
sep=np.full((6,big[0].shape[1],3),255,np.uint8)
out=np.vstack([big[0],sep,big[1],sep,big[2]])
Image.fromarray(out).save(os.path.join(HERE,"_room3_zoom.png"))
print("WROTE _room3_zoom.png", out.shape, "(3 timepoints: t=0,40,80 idle)")
