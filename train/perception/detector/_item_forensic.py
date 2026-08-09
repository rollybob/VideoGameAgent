import os,sys,json,glob,warnings
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); RL=os.path.abspath(os.path.join(HERE,"..","..","rl"))
sys.path.insert(0,RL)
from alttp_ppo_env import AlttpPpoEnv
from harvest_key_frames import u16
from mgba._pylib import ffi
BANK=os.path.join(RL,"..","perception","keybank_holdout","states")
slot_hits={}; broad_hits={}
states=sorted(glob.glob(os.path.join(BANK,"keydrop-*.state")))[:12]
for st in states:
    j=json.load(open(st.replace(".state",".json"))); dx,dy=j["drop"]
    env=AlttpPpoEnv(state_path=st,horizon=16); env.reset(seed=0)
    iw=ffi.cast("uint8_t *",env.core._native.memory.iwram)
    # (a) enemy sprite table slots 0x03846+4i
    ms=[i for i in range(16) if abs(u16(iw,0x03846+4*i)-dx)<=8 and abs(u16(iw,0x03848+4*i)-dy)<=8]
    for i in ms: slot_hits[i]=slot_hits.get(i,0)+1
    # (b) broad IWRAM scan: adjacent (x,y) u16 pair near the drop
    ba=[a for a in range(0x3000,0x4000,2) if abs(u16(iw,a)-dx)<=8 and abs(u16(iw,a+2)-dy)<=8]
    for a in ba: broad_hits[a]=broad_hits.get(a,0)+1
    print(f"{os.path.basename(st)} drop=({dx},{dy}) enemy-table slots={ms} broad={[hex(a) for a in ba[:6]]}")
    env.close()
n=len(states)
print(f"\n=== consistency over {n} states ===")
print("enemy-table slot hit counts:", {k:v for k,v in sorted(slot_hits.items())})
print("broad addr hit counts (top):", sorted(broad_hits.items(),key=lambda x:-x[1])[:8])
