"""Task A benchmark: sweep max_pixels, measure latency + action quality on real GBA frames.
Runs the SAME generate path as serve_vlm.py. Picks frames likely to contain text."""
import glob, time, json, re
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

BUTTONS = ["up","down","left","right","A","B","start","select","L","R","wait"]
SYSTEM = ("You are VGA, an agent that plays video games from the screen alone, like a human. "
 "You are given one screenshot of a Game Boy Advance game and must choose the single best "
 "controller input to make progress right now. Read on-screen text, menus, and tutorials and "
 "follow them. Advance dialogue and confirm menus with A; cancel with B; move with the D-pad; "
 "open the menu with start. If an animation or transition is playing and nothing needs doing, "
 'choose "wait". Pick exactly ONE input; you will see the result on the next screenshot. You '
 "see pixels only - there is no hidden game state. Base every decision on what is visible.")
def instr():
    return ('Step 0. This is the current screen.\nObjective: start the game and reach the '
     'overworld.\nRespond with ONLY a compact JSON object and nothing else, of the form: '
     '{"reason": "<one short sentence>", "button": "<one of: '+" ".join(BUTTONS)+'>", "repeats": <1, 2, or 3>}')

frames = sorted(glob.glob('/work/captures/pokemon_real/*.png'))
# spread across the capture so we hit different scenes (title/menu/overworld)
picks = [frames[0], frames[len(frames)//3], frames[2*len(frames)//3], frames[-1]]

print("loading model...", flush=True)
model = AutoModelForImageTextToText.from_pretrained('/models', torch_dtype=torch.bfloat16, device_map='cuda').eval()

CONFIGS = [("default", None), ("64x28^2(~native)", 64*28*28), ("32x28^2", 32*28*28)]
for name, mp in CONFIGS:
    kw = {'max_pixels': mp, 'min_pixels': 16*28*28} if mp else {}
    proc = AutoProcessor.from_pretrained('/models', **kw)
    print(f"\n===== max_pixels config: {name} =====", flush=True)
    # warmup once
    for warm in range(1):
        pass
    for fp in picks:
        img = Image.open(fp).convert('RGB')
        msgs=[{"role":"system","content":[{"type":"text","text":SYSTEM}]},
              {"role":"user","content":[{"type":"image","image":img},{"type":"text","text":instr()}]}]
        chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[chat], images=[img], return_tensors="pt").to(model.device)
        grid = inp['image_grid_thw'][0].tolist(); vtok=(grid[0]*grid[1]*grid[2])//4
        # two runs: discard first (warmup), time second
        for r in range(2):
            t0=time.time()
            with torch.no_grad():
                gen=model.generate(**inp, max_new_tokens=128, do_sample=False)
            torch.cuda.synchronize(); dt=time.time()-t0
        out=[o[len(i):] for i,o in zip(inp.input_ids,gen)]
        txt=proc.batch_decode(out, skip_special_tokens=True)[0]
        m=re.search(r'\{.*\}', txt, re.DOTALL)
        act=m.group(0) if m else txt[:100]
        print(f"  {fp.split('/')[-1]:20s} vtok={vtok:4d} lat={dt:5.2f}s  {act}", flush=True)
