"""drive_agent.py -- 3-TIER integrated agent (v2, SMARTER): live-detector + System-1 reflex +
System-2 VLM, driving ALttP headless and capturing an annotated video.

v1 failed (stuck in Link's house 30 min). Root causes + fixes here:
  1. S1 is a DUNGEON COMBAT reflex -> it wanders and sabotages navigation. FIX: detector-gated
     arbitration -- S1 drives ONLY when enemies are on screen; otherwise the VLM navigates.
  2. No sense of being STUCK -> repeated pushing a wall forever. FIX: read Link's true WORLD
     position from RAM (0x038F4/0x038F0) as a movement ORACLE (control signal, not perception);
     if a direction is issued but Link isn't moving, ESCAPE-sweep directions until he moves
     (un-sticks from walls, finds the open path / doorway).
  3. VLM got no feedback -> kept re-affirming a blocked action. FIX: send last_action/last_changed/
     looping to /act so it re-plans; phase-aware goal (leave the house, then explore).

VLM runs async (~9s/call) so gameplay never freezes. Segmented (rebuild core every SEG decisions)
to dodge the mgba long-run segfault. Real-time throttle. PIL overlay -> JPEG frames -> host ffmpeg
(run_drive.sh). Run in thor-rl:cu130 with the VLM up on 127.0.0.1:8077.
"""
import os, sys, io, json, time, base64, threading, subprocess, traceback, warnings
warnings.filterwarnings("ignore")
import numpy as np
ROOT = "/work"
DET_DIR = os.path.join(ROOT, "train/perception/detector")
sys.path.insert(0, DET_DIR); sys.path.insert(0, os.path.join(ROOT, "train/rl"))

MINUTES = float(os.environ.get("MINUTES", "30"))
FRAME_SKIP = 4
FPS = int(os.environ.get("FPS", "15"))
DECISIONS = int(MINUTES * 60 * FPS)
WALL_CAP_S = MINUTES * 60 * 2.0 + 600
SEG = 240
STUCK_K = int(os.environ.get("STUCK_K", "15"))            # decisions of no world-movement before the anti-freeze forces action
ESC_HOLD = 6                                              # decisions per escape button
ESC_SWEEP = ["a", "down", "right", "a", "up", "left"]     # anti-freeze rotation: advance dialogs (a) + try each direction
MOVE_TH = 4                                                # world-units over ~6 decisions counts as "moving"
MENU_HOLD = 6                                              # decisions per button while mashing through menus
MENU_CYCLE = ["start", "a", "start", "a", "down", "a", "right", "a", "up", "a"]   # advances title/file-select/dialog
VLM = os.environ.get("VLM", "http://127.0.0.1:8077")
OUT = os.environ.get("OUT", os.path.join(ROOT, "sessions/agent_run/run.mp4"))
STATE = os.environ.get("STATE", os.path.join(ROOT, "train/rl/states/alttp_start_normal.state"))
SKIP_MENU = os.environ.get("SKIP_MENU", "0") == "1"       # start already in gameplay -> skip the title/menu masher
ROM = os.path.join(ROOT, "Emulator/mGBA/roms/Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
S1_CKPT = os.environ.get("S1", os.path.join(ROOT, "train/rl/runs/alttp_s1v2_scratch04/ppo_alttp_600000_steps.zip"))
DET_PT = os.path.join(DET_DIR, "detector.pt")
GOAL = ("You are playing The Legend of Zelda: A Link to the Past on GBA. Objectives in order: "
        "(1) get through the title screen / any menus by pressing Start or A; (2) you begin INSIDE "
        "a house -- LEAVE it: walk to the doorway (usually DOWN) to get outside; (3) then EXPLORE "
        "new areas and look for hidden or secret passages behind walls, bushes, or gaps. CRUCIAL: "
        "if last_changed is false you are BLOCKED/stuck -- do NOT repeat that button, choose a "
        "DIFFERENT direction or exit. Prefer reaching NEW areas over re-checking the same spot. "
        "AVOID 'wait' -- only wait if the screen is clearly black/loading. If you can see the game "
        "world OR any text box, do NOT wait: MOVE (pick a direction) or press A to advance text. "
        "Reply with the single best next button.")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def notify(msg):
    try:
        subprocess.run([os.path.expanduser("~/.local/bin/notify"), msg], timeout=20)
    except Exception:
        print("NOTIFY:", msg, flush=True)


# ---- perception -------------------------------------------------------------
import torch
from train_detector import Net, to_in
from eval_detector import peak_single, peaks_multi
DEV = "cuda"
net = Net().to(DEV); net.load_state_dict(torch.load(DET_PT, map_location=DEV)); net.eval()


def detect(frame):
    with torch.no_grad():
        hm = net(to_in(frame[None]).to(DEV)).cpu().numpy()[0]
    return peak_single(hm[0]), peaks_multi(hm[1], thr=0.7), peaks_multi(hm[2], thr=0.6)


# ---- System-1 ---------------------------------------------------------------
from stable_baselines3 import PPO
s1 = PPO.load(S1_CKPT, device=DEV)
print(f"loaded S1 {os.path.basename(S1_CKPT)}", flush=True)

# ---- raw mgba core ----------------------------------------------------------
import mgba.core, mgba.image, mgba.gba as gba, mgba.log
from mgba._pylib import ffi
from agent_memory import RoomMemory
mgba.log.silence()
K = gba.GBA
BIT = {n: 1 << getattr(K, "KEY_" + n) for n in ("A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "R", "L")}
BTN = {"up": BIT["UP"], "down": BIT["DOWN"], "left": BIT["LEFT"], "right": BIT["RIGHT"], "a": BIT["A"], "b": BIT["B"],
       "start": BIT["START"], "select": BIT["SELECT"], "l": BIT["L"], "r": BIT["R"], "wait": 0}
BTN_TO_DIR = {"up": 1, "down": 2, "left": 3, "right": 4}
DIR_MASK = {0: 0, 1: BIT["UP"], 2: BIT["DOWN"], 3: BIT["LEFT"], 4: BIT["RIGHT"]}
DIR_NAME = {0: "-", 1: "up", 2: "down", 3: "left", 4: "right"}
DIR_ROT = {1: 4, 4: 2, 2: 3, 3: 1}                          # clockwise sweep to escape a block
DIRBITS = {0: 0, 1: BIT["UP"], 2: BIT["DOWN"], 3: BIT["LEFT"], 4: BIT["RIGHT"],
           5: BIT["UP"] | BIT["RIGHT"], 6: BIT["UP"] | BIT["LEFT"], 7: BIT["DOWN"] | BIT["RIGHT"], 8: BIT["DOWN"] | BIT["LEFT"]}


def new_core(state_bytes=None):
    core = mgba.core.load_path(ROM); w, h = core.desired_video_dimensions()
    img = mgba.image.Image(w, h); core.set_video_buffer(img); core.reset()
    with open(STATE, "rb") as f:
        core.load_raw_state(f.read())
    if state_bytes is not None:
        core.load_raw_state(state_bytes)
    return core, img, w, h


def grab(img, w, h):
    return np.frombuffer(bytes(ffi.buffer(img.buffer, w * h * 4)), np.uint8).reshape(h, w, 4)[..., :3].copy()


def link_world(core):
    iw = ffi.cast("uint8_t *", core._native.memory.iwram)   # movement ORACLE (control, not perception)
    return (iw[0x038F4] | (iw[0x038F5] << 8), iw[0x038F0] | (iw[0x038F1] << 8))


def room_cell(core):
    x, y = link_world(core)
    return (x >> 9, y >> 9)


def s1_action_to_mask(a):
    a = np.asarray(a).ravel(); m = DIRBITS.get(int(a[0]), 0)
    for i, nm in enumerate(("A", "B", "L", "R"), start=1):
        if len(a) > i and int(a[i]): m |= BIT[nm]
    return m


# ---- System-2 (VLM /act) async, with feedback -------------------------------
import urllib.request
shared = {"frame": None, "det": ((-1, -1), [], []), "step": 0, "last_btn": "wait", "last_changed": False, "stuck": False}
directive = {"button": "wait", "subgoal": "(booting VLM...)", "reason": "", "n": 0}
lock = threading.Lock(); stop_flag = threading.Event()
hist = []


def post_act(frame, det, step, last_btn, last_changed, stuck, brief="", receipts=None):
    (lx, ly), en, it = det
    ctx = f"Perception: Link@({lx},{ly}); enemies={[tuple(map(int,e)) for e in en[:3]]}."
    b = io.BytesIO()
    from PIL import Image
    Image.fromarray(frame).save(b, format="PNG")
    # LOGIC LOOP (2026-08-09): the room-attempt brief rides /act's task_phase slot
    # (authoritative host-measured state, the framing that beats the echo bug) and
    # transition receipts ride the skills slot. The VLM now REASONS over what has
    # been tried in this room instead of re-deciding refuted moves statelessly.
    payload = {"image_b64": base64.b64encode(b.getvalue()).decode(), "goal": GOAL + " " + ctx,
               "step": step, "history": hist[-6:], "last_action": last_btn,
               "last_changed": bool(last_changed), "looping": bool(stuck),
               "task_phase": brief, "skills": receipts or []}
    req = urllib.request.Request(VLM + "/act", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


def vlm_worker():
    while not stop_flag.is_set():
        with lock:
            fr = None if shared["frame"] is None else shared["frame"].copy()
            det, stp, lb, lc, st = shared["det"], shared["step"], shared["last_btn"], shared["last_changed"], shared["stuck"]
            brief, rcpt = shared.get("brief", ""), shared.get("receipts", [])
        if fr is None:
            time.sleep(0.2); continue
        try:
            _t = time.time()
            j = post_act(fr, det, stp, lb, lc, st, brief=brief, receipts=rcpt)
            btn = str(j.get("button", "wait")).lower()
            with lock:
                directive.update(button=btn if btn in BTN else "wait",
                                 subgoal=str(j.get("subgoal", ""))[:60], reason=str(j.get("reason", ""))[:80],
                                 n=directive["n"] + 1)
            hist.append({"button": btn, "reason": str(j.get("subgoal", ""))[:40], "changed": bool(lc)})
            print(f"[vlm] #{directive['n']} {time.time()-_t:.1f}s -> {btn} moved={lc} stuck={st} {str(j.get('subgoal',''))[:32]!r}", flush=True)
        except Exception as e:
            print(f"[vlm] ERROR {str(e)[:90]}", flush=True)
            time.sleep(1.0)
        time.sleep(0.2)


# ---- overlay + JPEG frames --------------------------------------------------
from PIL import Image, ImageDraw
Z = 3
FRAMES = os.path.join(os.path.dirname(OUT), "frames")
os.makedirs(FRAMES, exist_ok=True)
for _f in os.listdir(FRAMES):
    if _f.endswith(".jpg"):
        os.remove(os.path.join(FRAMES, _f))


def write_overlay(frame, link, en, it, drv, info, idx):
    im = Image.fromarray(frame).resize((240 * Z, 160 * Z), Image.NEAREST).convert("RGB")
    dr = ImageDraw.Draw(im)

    def bx(x, y, c, r=6):
        if x is None or x < 0:
            return
        dr.rectangle([x * Z - r, y * Z - r, x * Z + r, y * Z + r], outline=c, width=2)
    bx(link[0], link[1], (0, 255, 0))
    for e in en:
        bx(e[0], e[1], (255, 40, 40), 5)
    for i in it:
        bx(i[0], i[1], (40, 140, 255), 5)
    col = {"S1": (255, 120, 120), "S2nav": (120, 200, 255), "S2": (120, 200, 255), "ESC": (255, 210, 60), "S2wait": (150, 150, 150)}.get(drv, (200, 200, 200))
    canvas = Image.new("RGB", (240 * Z, 160 * Z + 46), (0, 0, 0)); canvas.paste(im, (0, 0))
    dr2 = ImageDraw.Draw(canvas)
    dr2.text((6, 160 * Z + 5), f"DRIVER:{drv}  nav:{info['nav']}  VLM#{info['n']}:{info['btn']}  moved:{info['moved']}", fill=col)
    dr2.text((6, 160 * Z + 25), (f"goal:{info['subgoal']}")[:80], fill=(225, 225, 225))
    canvas.save(os.path.join(FRAMES, f"f{idx:06d}.jpg"), quality=80)


# ---- main loop --------------------------------------------------------------
def main():
    t0 = time.time()
    core, img, w, h = new_core()
    stack = None
    threading.Thread(target=vlm_worker, daemon=True).start()
    print(f"driving {DECISIONS} decisions (~{MINUTES}min game) -> {OUT}", flush=True)
    nav_dir = 0; stuck_run = 0; last_vlm_n = -1; last_btn = "wait"; moved_since_vlm = False; escaping = False; entered_game = SKIP_MENU; enemy_run = 0
    wpos = []; nxt = time.time(); drv_counts = {}
    mem = RoomMemory(); prev_room = None; sweep_jig = 0     # LOGIC LOOP state
    PERP = {"up": ("left", "right"), "down": ("right", "left"),
            "left": ("down", "up"), "right": ("up", "down")}
    for step in range(DECISIONS):
        if time.time() - t0 > WALL_CAP_S:
            print("wall-clock cap hit", flush=True); break
        frame = grab(img, w, h)
        if stack is None:
            stack = [frame] * 4
        link, en, it = detect(frame)
        lw = link_world(core); wpos.append(lw)
        if len(wpos) > 8:
            wpos.pop(0)
        moving = len(wpos) >= 6 and (abs(lw[0] - wpos[-6][0]) + abs(lw[1] - wpos[-6][1])) > MOVE_TH
        if moving:
            moved_since_vlm = True; entered_game = True
        with lock:
            cur_n = directive["n"]; vbtn = directive["button"]; vsub = directive["subgoal"]

        room = room_cell(core) if entered_game else None     # movement-oracle room id (control, not perception)
        if entered_game and prev_room is None:
            mem.enter(room, step); prev_room = room
        if entered_game and room != prev_room:               # TRANSITION receipt: what we were doing WORKED
            mem.room_changed(prev_room, room, step, last_btn)
            print(f"[mem] ROOM CHANGE {prev_room}->{room} at step {step} via {last_btn!r} "
                  f"(rooms={mem.stats()['rooms_visited']})", flush=True)
            prev_room = room
        esc_lvl = mem.escalation(room) if entered_game else 0
        sweep_edge = None

        enemy_run = enemy_run + 1 if len(en) >= 2 else 0     # hysteresis + >=2 -> filter OOD false-positive enemies
        if not entered_game:                                 # MENU/intro -> structured cycle-tap to advance
            b = MENU_CYCLE[(step // MENU_HOLD) % len(MENU_CYCLE)]
            mask = BTN[b] if step % 2 == 0 else 0            # TAP (press/release edges -- menus need edges, not holds)
            drv = "MENU"; last_btn = b; nav_dir = 0; stuck_run = 0; escaping = False
        elif enemy_run >= 4:                                 # SUSTAINED (>=4 decisions) real enemies -> S1 combat
            obs = np.concatenate(stack, axis=-1)
            act, _ = s1.predict(obs, deterministic=False)
            mask = s1_action_to_mask(act); drv = "S1"; nav_dir = 0; stuck_run = 0; escaping = False
        elif esc_lvl >= 2:                                   # LOGIC-LOOP LEVEL 2: reasoning failed -> SYSTEMATIC exit sweep.
            # Probe the least-tried screen edge in bounded bursts: hold its direction with a
            # wall-slip jiggle (perpendicular diagonal every 3rd decision, alternating sides)
            # + an A-tap every 6th (clears NPC dialogs like the (4,5) guide trap). Every burst's
            # outcome feeds back into mem's edge stats, so blocked edges rotate out.
            sweep_edge, sd = mem.sweep_edge(room, step)
            ph = step % 6
            if ph == 5:
                mask = BTN["a"] if step % 2 == 0 else 0
            elif ph in (2, 4):
                sweep_jig ^= 1
                mask = DIR_MASK[BTN_TO_DIR[sd]] | DIR_MASK[BTN_TO_DIR[PERP[sd][sweep_jig]]]
            else:
                mask = DIR_MASK[BTN_TO_DIR[sd]]
            drv = "SWEEP"; last_btn = f"sweep:{sweep_edge}"; nav_dir = BTN_TO_DIR[sd]; escaping = False; stuck_run = 0
        else:                                                # gameplay: VLM navigates; MASTER anti-freeze if Link is stuck
            if cur_n != last_vlm_n:
                last_vlm_n = cur_n; last_btn = vbtn; moved_since_vlm = False
            if moving:                                       # progress = Link's WORLD position changed
                stuck_run = 0; escaping = False
            else:
                stuck_run += 1
                if stuck_run >= STUCK_K:
                    escaping = True                          # stuck too long (wall / wait / dialogue) -> force action
            if escaping:                                     # ANTI-FREEZE: sweep A + directions until Link moves again
                esc = ESC_SWEEP[(step // ESC_HOLD) % len(ESC_SWEEP)]
                mask = (BTN[esc] if step % 2 == 0 else 0) if esc in ("a", "b", "start") else DIR_MASK[BTN_TO_DIR[esc]]
                drv = "ESC"; last_btn = esc; nav_dir = 0
            elif vbtn in BTN_TO_DIR:                          # VLM direction -> walk
                nav_dir = BTN_TO_DIR[vbtn]; mask = DIR_MASK[nav_dir]; drv = "S2nav"
            elif vbtn in ("a", "b", "start", "select"):      # VLM action -> tap
                mask = BTN[vbtn] if step % 2 == 0 else 0; drv = "S2"; nav_dir = 0
            else:                                            # VLM wait (brief -- stuck_run climbs, escape takes over)
                mask = 0; drv = "S2wait"; nav_dir = 0

        if entered_game and drv != "MENU":                   # record the decision's outcome in the room log
            mem.note(room, last_btn if drv != "SWEEP" else DIR_NAME[nav_dir], moving, step,
                     edge=sweep_edge)
        with lock:
            shared.update(frame=frame, det=(link, en, it), step=step,
                          last_btn=last_btn, last_changed=moved_since_vlm, stuck=(stuck_run > 0 or escaping),
                          brief=(mem.brief(room) if entered_game else ""),
                          receipts=(mem.receipts() if entered_game else []))
        for _ in range(FRAME_SKIP):
            core.set_keys(raw=mask); core.run_frame()
        stack = stack[1:] + [grab(img, w, h)]
        drv_counts[drv] = drv_counts.get(drv, 0) + 1
        write_overlay(frame, link, en, it, drv,
                      {"nav": DIR_NAME[nav_dir], "n": cur_n, "btn": last_btn if nav_dir or vbtn in BTN else vbtn,
                       "moved": moving,
                       "subgoal": (f"[rooms:{mem.stats()['rooms_visited']} esc:{esc_lvl}] " if entered_game else "") + vsub}, step)
        if step and step % SEG == 0:
            core, img, w, h = new_core(core.save_raw_state())
            if step % (SEG * 4) == 0:
                print(f"  step {step}/{DECISIONS} t={int(time.time()-t0)}s room={room_cell(core)} drv={drv} moving={moving} vlm#{cur_n} goal={vsub!r}", flush=True)
        nxt += 1.0 / FPS
        _sl = nxt - time.time()
        if _sl > 0:
            time.sleep(_sl)
        else:
            nxt = time.time()
    stop_flag.set()
    nf = len([f for f in os.listdir(FRAMES) if f.endswith(".jpg")])
    ms = mem.stats()
    print(f"FRAMES_DONE {nf} frames, {int(time.time()-t0)}s wall, vlm_calls={directive['n']}, final_room={room_cell(core)}, drivers={drv_counts} -> {FRAMES}", flush=True)
    print(f"MEM_STATS rooms_visited={ms['rooms_visited']} transitions={ms['transitions']} order={ms['order']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc(); print(tb, flush=True)
        notify(f"VGA 3-tier run ERROR (aborted): {tb.strip().splitlines()[-1][:120]}")
        sys.exit(1)
