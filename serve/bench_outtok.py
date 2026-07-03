"""Confirm the real latency lever: OUTPUT tokens. Sweep max_new_tokens + a terse-reason
prompt, at default pixels. Also measure raw decode tok/s."""
import glob, time, re
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
BUTTONS=["up","down","left","right","A","B","start","select","L","R","wait"]
SYSTEM=("You are VGA, an agent that plays video games from the screen alone. You are given one "
 "GBA screenshot and choose the single best controller input now. Read on-screen text/menus and "
 "follow them. Advance dialogue/confirm with A; cancel B; move D-pad; menu start; wait during "
 "animations. Pick ONE input. Pixels only.")
def instr(terse):
    tail=('Respond with ONLY compact JSON: {"button":"<one of: '+" ".join(BUTTONS)+'>","repeats":<1-3>}'
          if terse else
          'Respond with ONLY a compact JSON object: {"reason":"<one short sentence>","button":"<one of: '
          +" ".join(BUTTONS)+'>","repeats":<1,2,3>}')
    return "Step 0. Current screen. Objective: start the game and reach the overworld.\n"+tail
frames=sorted(glob.glob('/work/captures/pokemon_real/*.png'))
picks=[frames[0],frames[len(frames)//3],frames[2*len(frames)//3],frames[-1]]
print("loading...",flush=True)
model=AutoModelForImageTextToText.from_pretrained('/models',torch_dtype=torch.bfloat16,device_map='cuda').eval()
proc=AutoProcessor.from_pretrained('/models')
def run(terse,mnt):
    lats=[];ntoks=[]
    for fp in picks:
        img=Image.open(fp).convert('RGB')
        msgs=[{"role":"system","content":[{"type":"text","text":SYSTEM}]},
              {"role":"user","content":[{"type":"image","image":img},{"type":"text","text":instr(terse)}]}]
        chat=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=proc(text=[chat],images=[img],return_tensors="pt").to(model.device)
        for r in range(2):
            t0=time.time()
            with torch.no_grad(): gen=model.generate(**inp,max_new_tokens=mnt,do_sample=False)
            torch.cuda.synchronize();dt=time.time()-t0
        out=[o[len(i):] for i,o in zip(inp.input_ids,gen)]
        nt=len(out[0]); txt=proc.batch_decode(out,skip_special_tokens=True)[0].replace("\n"," ")
        lats.append(dt);ntoks.append(nt)
        print(f"    terse={terse} mnt={mnt:3d} {fp.split('/')[-1]:16s} out_tok={nt:3d} lat={dt:5.2f}s  {txt[:90]}",flush=True)
    print(f"  -> terse={terse} mnt={mnt}: mean_lat={sum(lats)/len(lats):.2f}s mean_out_tok={sum(ntoks)/len(ntoks):.0f}",flush=True)
for terse,mnt in [(False,128),(False,48),(True,24),(True,12)]:
    run(terse,mnt)
