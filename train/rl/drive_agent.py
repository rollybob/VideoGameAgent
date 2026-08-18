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
import os, sys, io, re, json, time, base64, threading, subprocess, traceback, warnings
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
ESC_SWEEP = ["a", "down", "b", "right", "a", "up", "b", "left"]  # anti-freeze rotation: advance dialogs (a) + SLASH obstacles like bushes (b) + try each direction
MOVE_TH = 4                                                # world-units over ~6 decisions counts as "moving"
MENU_HOLD = 6                                              # decisions per button while mashing through menus
MENU_CYCLE = ["start", "a", "start", "a", "down", "a", "right", "a", "up", "a"]   # advances title/file-select/dialog
VLM = os.environ.get("VLM", "http://127.0.0.1:8077")
OUT = os.environ.get("OUT", os.path.join(ROOT, "sessions/agent_run/run.mp4"))
STATE = os.environ.get("STATE", os.path.join(ROOT, "train/rl/states/alttp_start_normal.state"))
SKIP_MENU = os.environ.get("SKIP_MENU", "0") == "1"       # start already in gameplay -> skip the title/menu masher
ROM = os.path.join(ROOT, "Emulator/mGBA/roms/Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba")
S1_CKPT = os.environ.get("S1") or os.path.join(ROOT, "train/rl/runs/alttp_s1v2_scratch04/ppo_alttp_600000_steps.zip")   # `or`: empty-string env (wrapper passthrough) falls back too
DET_PT = os.path.join(DET_DIR, "detector.pt")
# General prompt (2026-08-09): the earlier version hard-scripted "you are in a house, leave it,
# press DOWN", and the weak 8B parroted that out of context (said "leave house" in dungeons and
# overworld). Keep the goal + game mechanics general; let the model read the actual screen instead
# of leaning on a scripted step. Obstacle-interaction is stated as a PRINCIPLE, not a per-scene script.
GOAL = ("You are controlling the hero in The Legend of Zelda: A Link to the Past (GBA), a top-down "
        "action-adventure, using only what you see on screen. Your goal is to make progress by "
        "EXPLORING: reach rooms and areas you have not visited yet, and keep heading toward new "
        "territory rather than re-checking places you have already been. You can walk in four "
        "directions, attack with your sword, and use the action button to interact or advance text. "
        "Not every obstacle is a wall -- bushes, pots, and enemies can often be cut, moved, or "
        "defeated to open a path, so if a route looks blocked consider acting on what blocks it "
        "before turning back. If last_changed is false your last button did nothing (you are "
        "blocked) -- do NOT repeat it; choose a different direction or action. Only 'wait' if the "
        "screen is black or loading; if you can see the world or a text box, act. Select and Start "
        "open menus/the map and do NOT move you -- never use them to explore. Reply with the "
        "single best next button.")
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


LINK_THR = 0.5   # peak_single is argmax-ALWAYS (no threshold) -> on rain/OOD frames the "peak" is
                 # noise and drew a green box on rain. Gate on confidence; below it report absent (-1).
                 # Cosmetic: Link CONTROL reads RAM link_world, not this detection.
def detect(frame):
    with torch.no_grad():
        hm = net(to_in(frame[None]).to(DEV)).cpu().numpy()[0]
    link = peak_single(hm[0]) if float(hm[0].max()) >= LINK_THR else (-1.0, -1.0)
    return link, peaks_multi(hm[1], thr=0.7), peaks_multi(hm[2], thr=0.6)


# ---- System-1 ---------------------------------------------------------------
from stable_baselines3 import PPO
s1 = PPO.load(S1_CKPT, device=DEV)
print(f"loaded S1 {os.path.basename(S1_CKPT)}", flush=True)

# ---- System-1b: goal-conditioned WALKER (the "legs", 2026-08-09) -------------
# Turns a target SCREEN position into learned navigation: obs = the same 12ch
# stack + a goal-blob channel (walker_env.goal_channel -- one renderer for train
# and inference), action = Discrete(9) direction. Targets: a detected ITEM
# (walk-to-collect, the rung-3 fix) or the VLM's direction rendered as a screen-
# edge point (intent -> executed walk instead of a blind button hold).
# WALKER env var = checkpoint path; unset/missing -> feature off, prior behavior.
WALKER_CKPT = os.environ.get("WALKER", "")
walker = None
if WALKER_CKPT and os.path.exists(WALKER_CKPT):
    from walker_env import goal_channel
    walker = PPO.load(WALKER_CKPT, device=DEV)
    print(f"loaded WALKER {os.path.basename(WALKER_CKPT)}", flush=True)
else:
    print("walker OFF (set WALKER=<ckpt.zip> to enable)", flush=True)
WALK_EDGE_PT = {1: lambda lx, ly: (lx, 8), 2: lambda lx, ly: (lx, 151),
                3: lambda lx, ly: (8, ly), 4: lambda lx, ly: (231, ly)}


def walker_dir(stack, sx, sy):
    obs = np.concatenate(stack + [goal_channel(sx, sy)], axis=-1)
    a, _ = walker.predict(obs, deterministic=False)
    return DIRBITS.get(int(np.asarray(a).ravel()[0]), 0)

# ---- raw mgba core ----------------------------------------------------------
import mgba.core, mgba.image, mgba.gba as gba, mgba.log
from mgba._pylib import ffi
from agent_memory import RoomMemory
from ram_text import extract_text                         # on-screen message text (2026-08-10 comprehension scaffold)
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


def camera(core):
    """Camera world origin (walker_env's 0x02B82/86 convention). screen = world - camera:
    lets a WORLD-anchored target stay valid while the camera scrolls during the walk."""
    iw = ffi.cast("uint8_t *", core._native.memory.iwram)
    return (iw[0x02B82] | (iw[0x02B83] << 8), iw[0x02B86] | (iw[0x02B87] << 8))


def health(core):
    """Link's HP (WRAM 0x0234D, the oracle.py convention). ==0 means dead/game-over screen:
    a NON-SPATIAL state where rooms/edges/targets are meaningless (v2.1, brainlegs run:
    the spatial escalation ground against the continue menu for minutes)."""
    return int(ffi.cast("uint8_t *", core._native.memory.wram)[0x0234D])


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


def post_act(frame, det, step, last_btn, last_changed, stuck, brief="", receipts=None, dialog="", already_read=False):
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
               "task_phase": brief, "skills": receipts or [],
               "dialog_text": dialog, "already_read": bool(already_read)}
    req = urllib.request.Request(VLM + "/act", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


# ---- LEVEL-3 target pointing (v2.1): S2 names a BOX, the walker executes ----
# Format matters enormously (offline probe 2026-08-09, GT = RAM Link pos): freeform
# "TARGET x,y" = 0/4 hits (corner echo, what sank brainlegs); the model's NATIVE
# grounding format -- bounding box normalized to 0-1000 on a 3x-upscaled frame --
# = 4/4 hits at 17-28px. Ask in the trained format, convert the center host-side.
POINT_SYS = ("You are a visual grounding assistant looking at ONE Game Boy Advance screen from "
             "Zelda: A Link to the Past. You output ONE bounding box with coordinates normalized "
             "to 0-1000, in the exact format asked. Nothing else.")


def post_point(frame, avoid):
    prompt = ("Link is STUCK in this room and must LEAVE it. Locate the ONE best visible thing to "
              "walk to and interact with to get out: an open doorway, a staircase, stairs down, "
              "a chest, a floor switch, a pot, or a gap in the walls. "
              + ("These were already tried and did NOT work, pick something ELSE: "
                 + "; ".join(avoid) + ". " if avoid else "")
              + "Output ONLY: (x1,y1),(x2,y2) | what it is")
    b = io.BytesIO()
    from PIL import Image
    Image.fromarray(frame).resize((720, 480), Image.NEAREST).save(b, format="PNG")
    payload = {"image_b64": base64.b64encode(b.getvalue()).decode(), "prompt": prompt,
               "system": POINT_SYS, "max_new_tokens": 48}
    req = urllib.request.Request(VLM + "/read", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode()).get("text", "")


def vlm_worker():
    last_dlg = ""                                            # last on-screen message sent -> set already_read on repeats
    while not stop_flag.is_set():
        with lock:
            treq = shared.pop("tgt_req", None)
        if treq is not None:                                  # point query preempts the /act cycle
            t_fr, t_cam, t_avoid = treq
            try:
                _t = time.time()
                text = post_point(t_fr, t_avoid)
                n = [int(v) for v in re.findall(r"\d{1,4}", text.split("|", 1)[0])]
                pt = None
                if len(n) >= 4 and all(0 <= v <= 1000 for v in n[:4]):
                    pt = (min(int((n[0] + n[2]) / 2 * 240 / 1000), 239),
                          min(int((n[1] + n[3]) / 2 * 160 / 1000), 159))
                lbl = (text.split("|", 1)[1].strip()[:24] if "|" in text else text.strip()[:24]) or "?"
                with lock:
                    shared["tgt_resp"] = {"pt": pt, "cam": t_cam, "label": lbl}
                print(f"[point] {time.time()-_t:.1f}s -> {pt} {lbl!r} raw={text[:60]!r}", flush=True)
            except Exception as e:
                print(f"[point] ERROR {str(e)[:90]}", flush=True)
                with lock:
                    shared["tgt_resp"] = {"pt": None, "cam": t_cam, "label": "error"}
            continue
        with lock:
            fr = None if shared["frame"] is None else shared["frame"].copy()
            det, stp, lb, lc, st = shared["det"], shared["step"], shared["last_btn"], shared["last_changed"], shared["stuck"]
            brief, rcpt = shared.get("brief", ""), shared.get("receipts", [])
            dlg = shared.get("dialog", "")
        if fr is None:
            time.sleep(0.2); continue
        already = (dlg != "" and dlg == last_dlg)             # same message across calls -> "already read"
        try:
            _t = time.time()
            j = post_act(fr, det, stp, lb, lc, st, brief=brief, receipts=rcpt, dialog=dlg, already_read=already)
            btn = str(j.get("button", "wait")).lower()
            with lock:
                directive.update(button=btn if btn in BTN else "wait",
                                 subgoal=str(j.get("subgoal", ""))[:60], reason=str(j.get("reason", ""))[:80],
                                 n=directive["n"] + 1)
            hist.append({"button": btn, "reason": str(j.get("subgoal", ""))[:40], "changed": bool(lc)})
            print(f"[vlm] #{directive['n']} {time.time()-_t:.1f}s -> {btn} moved={lc} stuck={st} {str(j.get('subgoal',''))[:32]!r}", flush=True)
            last_dlg = dlg
        except Exception as e:
            print(f"[vlm] ERROR {str(e)[:90]}", flush=True)
            time.sleep(1.0)
        time.sleep(0.2)


# ---- overlay -> mp4 (in-container: imageio/ffmpeg baked into thor-rl) --------
# Stream overlay frames straight to OUT (.mp4). No JPEG intermediate and no host-side
# ffmpeg stitch in run_drive.sh -- the container now has imageio+ffmpeg. The canvas is
# 720x526 (even), so yuv420p is valid; macro_block_size=1 keeps that exact size (no pad).
from PIL import Image, ImageDraw
import imageio.v2 as imageio
import atexit
Z = 3
VID = imageio.get_writer(OUT, fps=FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
_NF = [0]                 # frames written (list -> mutable without `global` in write_overlay)
_vid_closed = [False]
def _finalize_video():
    if not _vid_closed[0]:
        _vid_closed[0] = True
        VID.close()
atexit.register(_finalize_video)   # finalize the mp4 even on early exit / exception


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
    if info.get("tgt") is not None:
        tx, ty = info["tgt"]
        if -20 < tx < 260 and -20 < ty < 180:
            bx(min(max(tx, 4), 236), min(max(ty, 4), 156), (255, 0, 255), 7)
    col = {"S1": (255, 120, 120), "S2nav": (120, 200, 255), "S2": (120, 200, 255), "ESC": (255, 210, 60), "S2wait": (150, 150, 150),
           "SWEEP": (255, 160, 40), "TGTwalk": (255, 0, 255), "TGTa": (255, 90, 255), "WALKit": (90, 255, 190), "WALKnav": (140, 230, 255),
           "GAMEOVER": (255, 60, 60)}.get(drv, (200, 200, 200))
    canvas = Image.new("RGB", (240 * Z, 160 * Z + 46), (0, 0, 0)); canvas.paste(im, (0, 0))
    dr2 = ImageDraw.Draw(canvas)
    dr2.text((6, 160 * Z + 5), f"DRIVER:{drv}  nav:{info['nav']}  VLM#{info['n']}:{info['btn']}  moved:{info['moved']}", fill=col)
    dr2.text((6, 160 * Z + 25), (f"goal:{info['subgoal']}")[:80], fill=(225, 225, 225))
    VID.append_data(np.asarray(canvas)); _NF[0] += 1


# ---- main loop --------------------------------------------------------------
def main():
    t0 = time.time()
    core, img, w, h = new_core()
    stack = None
    threading.Thread(target=vlm_worker, daemon=True).start()
    print(f"driving {DECISIONS} decisions (~{MINUTES}min game) -> {OUT}", flush=True)
    nav_dir = 0; stuck_run = 0; last_vlm_n = -1; last_btn = "wait"; moved_since_vlm = False; escaping = False; entered_game = SKIP_MENU; enemy_run = 0
    wpos = []; nxt = time.time(); drv_counts = {}; item_run = 0
    mem = RoomMemory(); prev_room = None; sweep_jig = 0     # LOGIC LOOP state
    tgt = None; tgt_deadline = 0; tgt_apress = 0; tgt_cool = 0; tgt_hist = {}   # LEVEL-3 state
    tgt_asks = {}; tgt_failed = {}                          # per-room ask budget + failed world points (dedup)
    just_gameover = False; alive_run = 0; dead_run = 0      # suppress the bogus transition receipt on respawn
    traj = []                                               # full (x, y, drv) trajectory, dumped at end
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
            if just_gameover:                                # respawn jump, not an earned exit -- no receipt
                mem.enter(room, step)
                print(f"[mem] RESPAWN {prev_room}->{room} at step {step} (no receipt)", flush=True)
            else:
                mem.room_changed(prev_room, room, step, last_btn)
                print(f"[mem] ROOM CHANGE {prev_room}->{room} at step {step} via {last_btn!r} "
                      f"(rooms={mem.stats()['rooms_visited']})", flush=True)
            prev_room = room
            tgt = None; tgt_apress = 0; tgt_cool = 0         # targets are room-local
            with lock:                                       # drop stale point asks/answers from the old room
                shared.pop("tgt_req", None); shared.pop("tgt_resp", None)
        esc_lvl = mem.escalation(room) if entered_game else 0
        sweep_edge = None; note_btn = None

        enemy_run = enemy_run + 1 if len(en) >= 2 else 0     # hysteresis + >=2 -> filter OOD false-positive enemies
        item_run = item_run + 1 if len(it) >= 1 else 0       # sustained detected item -> walk-to-collect candidate
        hp = health(core) if entered_game else 1
        dead_run = dead_run + 1 if (entered_game and hp <= 0) else 0   # hysteresis: savestate BOOT reads hp=0
        if hp > 0 and just_gameover:                         # for a few decisions (brainlegs2 logged phantom
            alive_run += 1                                   # RESPAWNs at t=0); a real game-over holds for 100s
            if alive_run > 150:                              # same-cell respawn: receipt guard expires quietly
                just_gameover = False
        if not entered_game:                                 # MENU/intro -> structured cycle-tap to advance
            b = MENU_CYCLE[(step // MENU_HOLD) % len(MENU_CYCLE)]
            mask = BTN[b] if step % 2 == 0 else 0            # TAP (press/release edges -- menus need edges, not holds)
            drv = "MENU"; last_btn = b; nav_dir = 0; stuck_run = 0; escaping = False
        elif dead_run >= 8:                                  # GAME OVER (v2.1): NON-SPATIAL state -- rooms/edges/
            # targets are meaningless on the continue menu; the spatial escalation ground
            # against it for minutes in the brainlegs run. Menu-tap through (dpad moves the
            # cursor, A confirms) until the continue restores health.
            b = MENU_CYCLE[(step // MENU_HOLD) % len(MENU_CYCLE)]
            mask = BTN[b] if step % 2 == 0 else 0
            drv = "GAMEOVER"; last_btn = b; nav_dir = 0; stuck_run = 0; escaping = False
            just_gameover = True; alive_run = 0; tgt = None
        elif enemy_run >= 4:                                 # SUSTAINED (>=4 decisions) real enemies -> S1 combat
            obs = np.concatenate(stack, axis=-1)
            act, _ = s1.predict(obs, deterministic=False)
            mask = s1_action_to_mask(act); drv = "S1"; nav_dir = 0; stuck_run = 0; escaping = False
        elif walker is not None and 4 <= item_run <= 120:    # WALKER walk-to-collect: sustained detected item, no enemies.
            # 120-decision cap per continuous sighting: a phantom/unreachable item cannot
            # starve the sweep -- after the cap it falls through until the item leaves view.
            mask = walker_dir(stack, it[0][0], it[0][1])
            drv = "WALKit"; last_btn = "walk:item"; nav_dir = 0; stuck_run = 0; escaping = False
        elif esc_lvl >= 2:                                   # LOGIC-LOOP L2/L3: reasoning failed -> host takes over.
            ccx, ccy = camera(core)
            if esc_lvl >= 3 and walker is not None and tgt is None and tgt_cool <= step:
                # LEVEL 3 (v2, post-braintest): blunt edge-probing refuted itself -> S2 names a
                # TARGET via /read ("TARGET x,y | chest"), anchored to WORLD coords with the
                # ask-time camera; the walker executes; A on arrival. Failed targets are recorded
                # and excluded from the next ask. Sweep keeps driving while an ask is in flight.
                with lock:
                    if "tgt_resp" in shared:
                        resp = shared.pop("tgt_resp")
                        if resp.get("pt"):
                            kx, ky = resp["cam"]
                            px, py = kx + resp["pt"][0], ky + resp["pt"][1]
                            if any(abs(px - fx) + abs(py - fy) <= 24 for fx, fy in tgt_failed.get(room, [])):
                                # v2.1: the 8B repeats failed points despite the exclusion text
                                # (brainlegs: same "TARGET 239,15" forever) -> hard dedup + long cooldown
                                print(f"[tgt] room {room} DUP-FAILED point ({px},{py}) -> sweep", flush=True)
                                tgt_cool = step + 300
                            else:
                                tgt = (px, py, resp["label"])
                                tgt_deadline = step + 200; tgt_apress = 0
                                print(f"[tgt] room {room} target {resp['label']!r} world=({px},{py})", flush=True)
                        else:
                            tgt_cool = step + 90             # unparseable answer -> sweep interlude, re-ask later
                    elif "tgt_req" not in shared and tgt_asks.get(room, 0) < 6:
                        tgt_asks[room] = tgt_asks.get(room, 0) + 1   # per-room ask budget: no /read churn
                        shared["tgt_req"] = (frame.copy(), (ccx, ccy), list(tgt_hist.get(room, []))[-4:])
            if tgt is not None:                              # LEVEL-3 drive: walk to the named point, interact
                wx, wy, wlbl = tgt
                if abs(lw[0] - wx) + abs(lw[1] - wy) <= 12:
                    mask = BTN["a"] if step % 2 == 0 else 0
                    tgt_apress += 1
                    drv = "TGTa"; last_btn = "tgt:a"; note_btn = "a"
                    if tgt_apress >= 14:                     # reached + interacted, room unchanged -> target refuted
                        tgt_hist.setdefault(room, []).append(f"{wlbl} (reached, A did not exit)")
                        tgt_failed.setdefault(room, []).append((wx, wy))
                        print(f"[tgt] room {room} REFUTED {wlbl!r} (A no-exit)", flush=True)
                        tgt = None; tgt_cool = step + 90
                elif step >= tgt_deadline:
                    tgt_hist.setdefault(room, []).append(f"{wlbl} (walker could not reach it)")
                    tgt_failed.setdefault(room, []).append((wx, wy))
                    print(f"[tgt] room {room} UNREACHABLE {wlbl!r}", flush=True)
                    tgt = None; tgt_cool = step + 90
                    mask = 0; drv = "TGTwalk"; note_btn = "walk"
                else:
                    mask = walker_dir(stack, wx - ccx, wy - ccy)
                    drv = "TGTwalk"; last_btn = "tgt:" + wlbl[:12]; note_btn = "walk"
                nav_dir = 0; escaping = False; stuck_run = 0
            else:                                            # LEVEL 2 (also L3 while an ask is pending): edge sweep.
                # Probe the least-tried screen edge in bounded bursts: hold its direction with a
                # wall-slip jiggle (perpendicular diagonal every 3rd decision, alternating sides)
                # + an A-tap every 6th (clears NPC dialogs like the (4,5) guide trap). Every burst's
                # outcome feeds back into mem's edge stats, so blocked edges rotate out.
                sweep_edge, sd = mem.sweep_edge(room, step)
                ph = step % 6
                if ph == 5:                                  # interact phase: SLASH with the SWORD (b) to
                    # cut bushes/pots blocking an edge (the (5,5) failure). NEVER A here -- probe confirmed
                    # A is the ITEM button (fires the equipped torch, magic 28->0; B drains 0). Mashing A
                    # across 85%-sweep was the magic-empty spam. The old A-tap was dead code (parity:
                    # ph==5 is always odd) so sword-only loses nothing that ever actually fired.
                    mask = BTN["b"]
                    note_btn = "b"
                else:
                    if ph in (2, 4):
                        sweep_jig ^= 1
                        mask = DIR_MASK[BTN_TO_DIR[sd]] | DIR_MASK[BTN_TO_DIR[PERP[sd][sweep_jig]]]
                    else:
                        mask = DIR_MASK[BTN_TO_DIR[sd]]
                    note_btn = sd
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
                nav_dir = BTN_TO_DIR[vbtn]
                if walker is not None:                        # intent EXECUTED: walker to a screen-edge target
                    lsx, lsy = (link[0], link[1]) if link[0] >= 0 else (120, 80)
                    mask = walker_dir(stack, *WALK_EDGE_PT[nav_dir](lsx, lsy))
                    drv = "WALKnav"
                else:
                    mask = DIR_MASK[nav_dir]; drv = "S2nav"
            elif vbtn in ("a", "b", "start", "select"):      # VLM action -> tap
                mask = BTN[vbtn] if step % 2 == 0 else 0; drv = "S2"; nav_dir = 0
            else:                                            # VLM wait (brief -- stuck_run climbs, escape takes over)
                mask = 0; drv = "S2wait"; nav_dir = 0

        if entered_game and drv not in ("MENU", "GAMEOVER"):  # record the decision's outcome in the room log
            mem.note(room, note_btn if note_btn is not None else last_btn, moving, step,
                     edge=sweep_edge)
        with lock:
            shared.update(frame=frame, det=(link, en, it), step=step,
                          last_btn=last_btn, last_changed=moved_since_vlm,
                          stuck=(esc_lvl >= 1 or stuck_run > 0 or escaping),   # v2 fix: escalation IS stuck (jiggle no longer masks it)
                          brief=(mem.brief(room) if entered_game else ""),
                          receipts=(mem.receipts() if entered_game else []),
                          dialog=extract_text(core))          # on-screen message text -> /act dialog_text (2026-08-10)
        for _ in range(FRAME_SKIP):
            core.set_keys(raw=mask); core.run_frame()
        stack = stack[1:] + [grab(img, w, h)]
        drv_counts[drv] = drv_counts.get(drv, 0) + 1
        traj.append((lw[0], lw[1], drv))
        _cx, _cy = camera(core) if tgt is not None else (0, 0)
        write_overlay(frame, link, en, it, drv,
                      {"nav": DIR_NAME[nav_dir], "n": cur_n, "btn": last_btn if nav_dir or vbtn in BTN else vbtn,
                       "moved": moving, "tgt": (tgt[0] - _cx, tgt[1] - _cy) if tgt is not None else None,
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
    _finalize_video()                  # close the mp4 writer (atexit is the backstop)
    nf = _NF[0]
    ms = mem.stats()
    drv_names = sorted(set(d for _, _, d in traj))
    drv_code = {d: i for i, d in enumerate(drv_names)}
    np.savez_compressed(os.path.join(os.path.dirname(OUT), "traj.npz"),
                        x=np.array([t[0] for t in traj], np.int32),
                        y=np.array([t[1] for t in traj], np.int32),
                        drv=np.array([drv_code[t[2]] for t in traj], np.uint8),
                        drv_names=np.array(drv_names))
    print(f"FRAMES_DONE {nf} frames, {int(time.time()-t0)}s wall, vlm_calls={directive['n']}, final_room={room_cell(core)}, drivers={drv_counts} -> {OUT}", flush=True)
    print(f"MEM_STATS rooms_visited={ms['rooms_visited']} transitions={ms['transitions']} order={ms['order']}", flush=True)
    for rm, hs in tgt_hist.items():
        print(f"TGT_HIST room {rm}: {hs}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc(); print(tb, flush=True)
        notify(f"VGA 3-tier run ERROR (aborted): {tb.strip().splitlines()[-1][:120]}")
        sys.exit(1)
