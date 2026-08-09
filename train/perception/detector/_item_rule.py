import os,sys,json,warnings
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); RL=os.path.abspath(os.path.join(HERE,"..","..","rl"))
sys.path.insert(0,RL)
from alttp_ppo_env import AlttpPpoEnv
from harvest_key_frames import u16
from mgba._pylib import ffi
def dump(tag, state):
    env=AlttpPpoEnv(state_path=state,horizon=16); env.reset(seed=0)
    iw=ffi.cast("uint8_t *",env.core._native.memory.iwram)
    print(f"--- {tag} ({os.path.basename(state)}) ---")
    for i in range(16):
        ex,ey=u16(iw,0x03846+4*i),u16(iw,0x03848+4*i)
        if (ex,ey)==(0,0): continue
        hp=int(iw[0x03250+i])
        print(f"  slot{i:2d} pos=({ex},{ey}) hp@0x0325{i:X}={hp}")
    env.close()
KB=os.path.join(RL,"..","perception","keybank_holdout","states")
dump("KEY on floor", os.path.join(KB,"keydrop-00.state"))
dump("ENEMY alive room2", os.path.join(RL,"states","alttp_human-02.state"))
dump("ENEMY alive room4", os.path.join(RL,"states","alttp_human-04.state"))
