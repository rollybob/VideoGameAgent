"""Fresh-drop item data (detector v5, 2026-08-20 composer arc).

ChainHunter (RAM-scripted, 12/12 kill reliability in rooms 2/4) kills the slot-3
key enemy; after a 15-frame PERSISTENCE GATE (slot3 hp==0, position stable, so the
death-poof window can never be labeled -- kill_drop_forensic 2026-08-08), Link
random-walks AWAY from the key while frames are captured. Labels:
  item = slot 3 ONLY. gen_items labeled EVERY on-screen hp==0 slot as an item,
         which taught the net that room4's statues are items at key-level
         confidence (the composer round-1 router theft). Here statues and other
         junk slots stay UNLABELED = correct background negatives, present in
         the very frames that used to teach the confusion.
  link = RAM position (always reliable).
  enem = on-screen hp>0 slots, RECORDED but the trainer MASKS the enemy channel
         for these files (v3 lesson: partially-labeled enemies in new rooms
         poison the enemy head; room2 is multi-enemy and slot coverage there is
         unvalidated).
Splits BY DROP EPISODE: room4 -> drops_room4.npz (train); room2 -> the first
ROOM2_HELD episodes to drops_room2_held.npz (the >=80%@4 bar), rest to
drops_room2_train.npz. Held = unseen drop EPISODES, not an unseen room -- the
unseen-room test remains the composer re-fire + a future third key room.

  docker run --rm --runtime nvidia --user 1000:1000 -v ~/projects/VGA:/vga \\
      -w /vga/train/perception/detector thor-rl:cu130 python3 -u gen_drops.py
Knobs: SMOKE=1 (1 episode, room2, verbose).
"""
import os
import sys
import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.abspath(os.path.join(HERE, "..", "..", "rl"))
sys.path.insert(0, RL)
from alttp_ppo_env import AlttpPpoEnv                       # noqa: E402
from harvest_key_frames import u16, ENEMY_X, ENEMY_Y       # noqa: E402
from chain_harvest import ChainHunter                       # noqa: E402

CAM_X, CAM_Y = 0x02B82, 0x02B86
SLOT3_HP = 0x03253
DIRS = [1, 2, 3, 4, 5, 6, 7, 8]
SMOKE = os.environ.get("SMOKE", "0") == "1"
GATE_DEC = 4            # persistence gate: 4 decisions = 16 emu frames > 12-frame poof
CAP_PER_EP = 40         # captured (labeled) frames per successful drop episode
EPS = {"room4": 10, "room2": 14}
ROOM2_HELD = 5          # first N room2 episodes -> held (by episode, no frame leakage)


def gold_blob(frame, wx, wy, r=20, excl=None, min_px=6):
    """Exact key label by PIXELS: strict-gold blob centroid inside a small window
    centered on the RAM-derived guess (wx,wy). Returns (x,y) or None.

    Why: (a) the room2 floor key RENDERS ~26-29px ABOVE its pickup/world anchor
    (render offset -- same trap as the 2026-08-08 pot forensic); (b) the object
    system CULLS the sprite when Link is far, so anchor-only labels produce
    phantom keys on empty floor (caught 2026-08-20 by patch inspection before
    any training). RAM anchors the REGION; pixels give the point; absence of a
    blob VETOES the frame. Strict-gold = key palette (255,214,66)/(231,181,49)/
    (255,255,74); excludes torches (windowed), Link (excl radius), HUD (window
    never reaches y<16 in practice)."""
    x0, x1 = max(0, wx - r), min(240, wx + r)
    # y floor 24: the HUD band (y<16) contains a literal gold KEY ICON at ~(112,8)
    # that the palette mask matches -- caught labeling it 2026-08-20 when the
    # camera scrolled the real key above the interior. Room content never renders
    # under the HUD, so clamping costs nothing.
    y0, y1 = max(24, wy - r), min(160, wy + r)
    if x1 <= x0 or y1 <= y0:
        return None
    win = frame[y0:y1, x0:x1].astype(int)
    m = (win[..., 0] >= 200) & (win[..., 1] >= 140) & (win[..., 2] <= 110) \
        & (win[..., 0] - win[..., 2] >= 90)
    ys, xs = np.where(m)
    # excl = list of sprite positions whose own gold pixels must not be taken for
    # the key: Link (shield/hair) AND live enemies (the room2 knight's trim sat a
    # constant (+10,+2) off his slot pos and impersonated the key -- probe
    # 2026-08-20). 14px covers a 16x16 sprite from its anchor.
    for e in (excl or []):
        er = e[2] if len(e) > 2 else 14
        keep = (np.abs(xs + x0 - e[0]) > er) | (np.abs(ys + y0 - e[1]) > er)
        xs, ys = xs[keep], ys[keep]
    if len(xs) < min_px:
        return None
    return (float(xs.mean()) + x0, float(ys.mean()) + y0)


def slots_onscreen(iw, cx, cy):
    """On-screen hp>0 slots (enemy candidates). Item labeling is slot-3-only and
    handled by the caller; hp==0 slots are deliberately IGNORED here."""
    en = []
    for i in range(16):
        ex, ey = u16(iw, 0x03846 + 4 * i), u16(iw, 0x03848 + 4 * i)
        if (ex, ey) == (0, 0):
            continue
        sx, sy = ex - cx, ey - cy
        if 0 <= sx < 240 and 0 <= sy < 160 and int(iw[0x03250 + i]) > 0:
            en.append((sx, sy))
    return en


def pad(lst):
    a = np.full((8, 2), -1, np.int16)
    for j, (sx, sy) in enumerate(lst[:8]):
        a[j] = (sx, sy)
    return a


def drop_episode(state, seed):
    """One kill -> gate -> away-walk capture. Returns (frames, link, enem, item)
    lists (possibly short if the key despawns / gets collected)."""
    from mgba._pylib import ffi
    env = AlttpPpoEnv(state_path=state, horizon=6000)
    env.reset(seed=seed)
    iw = ffi.cast("uint8_t *", env.core._native.memory.iwram)
    rng = random.Random(seed * 11 + 3)
    hunter = ChainHunter(iw, env.oracle, env.core, rng, validate=True)
    F, L, E, I = [], [], [], []
    drops = []
    key_pos = None
    gate = 0
    d = rng.choice(DIRS)
    prev_cam = None
    step = 0
    preroll = rng.randint(0, 15)     # chain_harvest recipe: diversifies the engage
    while len(F) < CAP_PER_EP and step < 1200:
        step += 1
        if key_pos is None and step >= 600:
            break                    # fail fast: hunter deadlocked (enemy in an
                                     # unreachable pocket) -- retry with a new seed
        if key_pos is None and step <= preroll:
            act = np.array([rng.choice(DIRS), 0, 0, 0, 0], dtype=np.int64)
        elif key_pos is None:
            act = np.array(hunter.act(drops), dtype=np.int64)
        else:
            # away-biased walk (gen_items pattern): vary the camera, never collect
            cx, cy = u16(iw, CAM_X), u16(iw, CAM_Y)
            lsx, lsy = u16(iw, 0x038F4) - cx, u16(iw, 0x038F0) - cy
            ksx, ksy = key_pos[0] - cx, key_pos[1] - cy
            if abs(lsx - ksx) + abs(lsy - ksy) < 45:
                d = (3 if ksx > lsx else 4) if abs(lsx - ksx) >= abs(lsy - ksy) \
                    else (1 if ksy > lsy else 2)
            if lsx < 35:
                d = 4
            elif lsx > 205:
                d = 3
            elif lsy < 35:
                d = 2
            elif lsy > 135:
                d = 1
            elif rng.random() < 0.15:
                d = rng.choice(DIRS)
            act = np.array([d, 0, 0, 0, 0], dtype=np.int64)
        ex_pre, ey_pre = u16(iw, ENEMY_X), u16(iw, ENEMY_Y)
        obs, r, term, trunc, info = env.step(act)
        if SMOKE and (step <= 3 or step % 100 == 0):
            print(f"    [gd] step{step} mode={hunter.mode} slot3=({ex_pre},{ey_pre})"
                  f" hp={int(iw[SLOT3_HP])} link={env.oracle.read_pos(env.core)}"
                  f" health={env.oracle.read_health(env.core)} key_pos={key_pos}", flush=True)
        if SMOKE and step == 1:
            cx0, cy0 = u16(iw, CAM_X), u16(iw, CAM_Y)
            for i in range(16):
                sx_, sy_ = u16(iw, 0x03846 + 4 * i), u16(iw, 0x03848 + 4 * i)
                if (sx_, sy_) != (0, 0) and int(iw[0x03250 + i]) == 0:
                    on = 0 <= sx_ - cx0 < 240 and 0 <= sy_ - cy0 < 160
                    print(f"    [gd] hp0-slot{i}: world({sx_},{sy_}) "
                          f"screen({sx_ - cx0},{sy_ - cy0}) onscreen={on}", flush=True)
        if term or trunc or env.oracle.read_health(env.core) <= 0:
            if SMOKE:
                print(f"    [gd] BREAK step{step} term={term} trunc={trunc} "
                      f"health={env.oracle.read_health(env.core)}", flush=True)
            break
        collected = False
        for _t, ch, _d, val in info["events"]:
            if (ch == "enemy_dmg" and val == 0 and key_pos is None
                    and (ex_pre, ey_pre) != (0, 0) and int(iw[SLOT3_HP]) == 0):
                # slot3 hp==0 gate: in multi-enemy room2 a DIFFERENT slot's death
                # must not arm the capture on a still-alive slot3 position
                drops.append([ex_pre, ey_pre, step])
                hunter.on_kill((ex_pre, ey_pre))
                key_pos = (ex_pre, ey_pre)
                gate = 0
            elif ch == "key":
                # Pre-kill key events = the room's FLOOR key collected en route
                # (room2 has one at a fixed spot -- discovered 2026-08-20, it had
                # been polluting every room2 chain metric). Collecting it is
                # GOOD: it leaves the room clean for the drop capture. Only a
                # post-kill pickup (the drop itself) ends the episode.
                collected = key_pos is not None     # walked onto it despite the bias -- end episode
        if collected:
            if SMOKE:
                print(f"    [gd] COLLECTED-BREAK step{step} events={info['events']}", flush=True)
            break
        if key_pos is None:
            continue
        # persistence gate + liveness: slot3 must stay hp==0 at the drop position
        kx, ky = u16(iw, ENEMY_X), u16(iw, ENEMY_Y)
        if (kx, ky) == (0, 0):
            if SMOKE:
                print(f"    [gd] DESPAWN-BREAK step{step}", flush=True)
            break                    # despawned (no real drop this kill) -- keep nothing
        if abs(kx - key_pos[0]) + abs(ky - key_pos[1]) > 4 or int(iw[SLOT3_HP]) != 0:
            gate = 0
            key_pos = (kx, ky)
            continue
        gate += 1
        if gate < GATE_DEC or step % 2:
            continue
        cx, cy = u16(iw, CAM_X), u16(iw, CAM_Y)
        if prev_cam is not None and abs(cx - prev_cam[0]) + abs(cy - prev_cam[1]) > 6:
            prev_cam = (cx, cy)
            continue
        prev_cam = (cx, cy)
        lsx, lsy = u16(iw, 0x038F4) - cx, u16(iw, 0x038F0) - cy
        if not (0 <= lsx < 240 and 0 <= lsy < 160):
            continue
        ksx, ksy = kx - cx, ky - cy
        fr = np.array(env._frames[-1], np.uint8)
        en = slots_onscreen(iw, cx, cy)
        blob = gold_blob(fr, ksx, ksy, r=14, excl=[(lsx, lsy)] + en) \
            if (0 <= ksx < 240 and 0 <= ksy < 160) else None
        if blob is None:
            continue                 # key not rendered / occluded -> frame vetoed
        F.append(fr)
        L.append((lsx, lsy))
        E.append(pad(en))
        I.append(pad([(int(round(blob[0])), int(round(blob[1])))]))
    if SMOKE:
        print(f"    [gd] EXIT step{step} F={len(F)} key_pos={key_pos}", flush=True)
    env.close()
    return F, L, E, I, key_pos is not None


def floorkey_anchor(state, seed=5000):
    """Room2's key is a PRE-PLACED floor key (object system, invisible to the
    sprite-slot table -- discovered 2026-08-20; it had silently supplied every
    historical room2 'chain' success). Anchor its WORLD position from the key
    EVENT itself: drive the deterministic hunter approach until 'key' fires;
    Link is standing on it at that instant. RAM-honest, no manual eyes."""
    from mgba._pylib import ffi
    env = AlttpPpoEnv(state_path=state, horizon=2000)
    env.reset(seed=seed)
    iw = ffi.cast("uint8_t *", env.core._native.memory.iwram)
    rng = random.Random(seed * 11 + 3)
    hunter = ChainHunter(iw, env.oracle, env.core, rng, validate=True)
    anchor = None
    for _ in range(400):
        act = np.array(hunter.act([]), dtype=np.int64)
        _o, _r, term, trunc, info = env.step(act)
        if any(ch == "key" for _t, ch, _d, _v in info["events"]):
            anchor = env.oracle.read_pos(env.core)
            break
        if term or trunc:
            break
    env.close()
    return anchor


def floorkey_approach(state, seed, anchor):
    """Room2 HELD-eval reals: capture during the hunter's APPROACH -- the only
    regime where the floor key provably renders (cull radius ~50; the orbit
    variant starved). Discrimination: min_px=14 (the knight's trim alone is a
    small blob and cannot qualify; the key body is ~27px) + tight enemy
    exclusion r=10 (so a key NEXT to the knight survives the exclusion).
    Ends at the key event (~step 54). Yields a handful of frames per seed."""
    from mgba._pylib import ffi
    env = AlttpPpoEnv(state_path=state, horizon=2000)
    env.reset(seed=seed)
    iw = ffi.cast("uint8_t *", env.core._native.memory.iwram)
    rng = random.Random(seed * 11 + 3)
    hunter = ChainHunter(iw, env.oracle, env.core, rng, validate=True)
    F, L, E, I = [], [], [], []
    kx, ky = anchor[0], anchor[1] - 26
    preroll = rng.randint(0, 10)
    for step in range(1, 200):
        if step <= preroll:
            act = np.array([rng.choice(DIRS), 0, 0, 0, 0], dtype=np.int64)
        else:
            act = np.array(hunter.act([]), dtype=np.int64)
        _o, _r, term, trunc, info = env.step(act)
        if term or trunc or env.oracle.read_health(env.core) <= 0:
            break
        if any(ch == "key" for _t, ch, _d, _v in info["events"]):
            break
        if step % 2:
            continue
        cx, cy = u16(iw, CAM_X), u16(iw, CAM_Y)
        lsx, lsy = u16(iw, 0x038F4) - cx, u16(iw, 0x038F0) - cy
        if not (0 <= lsx < 240 and 0 <= lsy < 160):
            continue
        ksx, ksy = kx - cx, ky - cy
        if not (0 <= ksx < 240 and 0 <= ksy < 160):
            continue
        fr = np.array(env._frames[-1], np.uint8)
        en = slots_onscreen(iw, cx, cy)
        blob = gold_blob(fr, ksx, ksy, r=16, min_px=14,
                         excl=[(lsx, lsy, 14)] + [(ex, ey, 10) for ex, ey in en])
        if blob is None:
            continue
        F.append(fr)
        L.append((lsx, lsy))
        E.append(pad(en))
        I.append(pad([(int(round(blob[0])), int(round(blob[1])))]))
    env.close()
    return F, L, E, I


def floorkey_episode(state, seed, anchor):
    """Away-biased walk capturing the resting floor key at world `anchor`.
    Labels: item = anchor-camera while on-screen; enemy channel masked at train
    (the knight roams and other slots are unvalidated here)."""
    from mgba._pylib import ffi
    env = AlttpPpoEnv(state_path=state, horizon=4000)
    env.reset(seed=seed)
    iw = ffi.cast("uint8_t *", env.core._native.memory.iwram)
    rng = random.Random(seed * 13 + 7)
    F, L, E, I = [], [], [], []
    d = rng.choice(DIRS)
    prev_cam = None
    kx, ky = anchor
    # The key SPRITE renders ~26px above the pickup anchor, and the object system
    # CULLS it beyond Link-distance ~50 (probe 2026-08-20: blob absent at dist
    # 59-117, key visibly rendered at dist <=~30-50). So: orbit TIGHT around the
    # RENDER point -- close enough to keep the sprite alive, never on its tile.
    krx, kry = kx, ky - 26
    for step in range(1, 700):
        cx, cy = u16(iw, CAM_X), u16(iw, CAM_Y)
        lsx, lsy = u16(iw, 0x038F4) - cx, u16(iw, 0x038F0) - cy
        ksx, ksy = krx - cx, kry - cy
        man = abs(lsx - ksx) + abs(lsy - ksy)
        if man < 22:                                # never step on the key
            d = (3 if ksx > lsx else 4) if abs(lsx - ksx) >= abs(lsy - ksy) \
                else (1 if ksy > lsy else 2)
        elif man > 42:                              # stay inside the cull radius
            d = (4 if ksx > lsx else 3) if abs(lsx - ksx) >= abs(lsy - ksy) \
                else (2 if ksy > lsy else 1)
        if lsx < 35:
            d = 4
        elif lsx > 205:
            d = 3
        elif lsy < 35:
            d = 2
        elif lsy > 135:
            d = 1
        elif rng.random() < 0.15:
            d = rng.choice(DIRS)
        _o, _r, term, trunc, info = env.step(np.array([d, 0, 0, 0, 0], dtype=np.int64))
        if term or trunc or env.oracle.read_health(env.core) <= 0:
            break
        if any(ch == "key" for _t, ch, _d, _v in info["events"]):
            break                                   # stepped on it anyway -- key gone
        if len(F) >= CAP_PER_EP or step % 2:
            continue
        cx, cy = u16(iw, CAM_X), u16(iw, CAM_Y)
        if prev_cam is not None and abs(cx - prev_cam[0]) + abs(cy - prev_cam[1]) > 6:
            prev_cam = (cx, cy)
            continue
        prev_cam = (cx, cy)
        lsx, lsy = u16(iw, 0x038F4) - cx, u16(iw, 0x038F0) - cy
        if not (0 <= lsx < 240 and 0 <= lsy < 160):
            continue
        ksx, ksy = krx - cx, kry - cy
        fr = np.array(env._frames[-1], np.uint8)
        en = slots_onscreen(iw, cx, cy)
        blob = gold_blob(fr, ksx, ksy, r=16, excl=[(lsx, lsy)] + en) \
            if (0 <= ksx < 240 and 0 <= ksy < 160) else None
        if blob is None:
            continue                 # culled/occluded -> frame vetoed (no phantoms)
        F.append(fr)
        L.append((lsx, lsy))
        E.append(pad(en))
        I.append(pad([(int(round(blob[0])), int(round(blob[1])))]))
    env.close()
    return F, L, E, I


def main():
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    banks = {}
    # room4: true fresh DROPS via the hunter (its home room; drops proven by the
    # keydrop bank's existence). room2: floor-key captures (drops unproven there;
    # the hunter cannot even reach the roaming knight from this state).
    n4 = 1 if SMOKE else EPS["room4"]
    state4 = os.path.join(RL, "states", "alttp_human-04.state")
    eps_data = []
    got = 0
    ep = 0
    while got < n4 and ep < n4 * 6:
        F, L, E, I, killed = drop_episode(state4, seed=4400 + ep)
        ep += 1
        if len(F) >= (5 if SMOKE else 15):
            eps_data.append((F, L, E, I))
            got += 1
        print(f"room4 ep{ep - 1}: kill={killed} frames={len(F)} "
              f"(kept {got}/{n4})", flush=True)
    banks["room4"] = eps_data
    state2 = os.path.join(RL, "states", "alttp_human-02.state")
    anchor = floorkey_anchor(state2)
    print(f"room2 floor-key anchor: {anchor}", flush=True)
    if anchor is not None:
        # HELD-eval reals only; room2 TRAIN context comes from paste augmentation
        # (gen_paste.py) -- the orbit-capture variant starved on the cull radius.
        n2 = 2 if SMOKE else 12
        eps_data = []
        for ep in range(n2):
            F, L, E, I = floorkey_approach(state2, seed=5000 + ep, anchor=anchor)
            if F:
                eps_data.append((F, L, E, I))
            print(f"room2 approach ep{ep}: frames={len(F)}", flush=True)
        banks["room2"] = eps_data
    outs = {}
    if "room4" in banks:
        outs["drops_room4"] = banks["room4"]
    if "room2" in banks:
        outs["approach2_held"] = banks["room2"]
    for name, eps_data in outs.items():
        if not eps_data:
            print(f"{name}: EMPTY (skipped)", flush=True)
            continue
        F, L, E, I = [], [], [], []
        for f, l, e, i in eps_data:
            F += f; L += l; E += e; I += i
        np.savez_compressed(os.path.join(HERE, "data", name + ".npz"),
                            frames=np.stack(F), link=np.array(L, np.int16),
                            enem=np.stack(E), item=np.stack(I))
        ipf = float(np.mean([(np.asarray(ii)[:, 0] >= 0).sum() for ii in I]))
        print(f"{name}: {len(F)} frames from {len(eps_data)} eps, "
              f"items/frame={ipf:.2f}", flush=True)


if __name__ == "__main__":
    main()
