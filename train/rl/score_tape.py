"""Score a solo_recorder human tape through the AlttpOracle -- the human CEILING:
per-channel reward breakdown + the room LADDER (progress rungs), the reference the
project has been missing (we only had the random floor).

Replays the tape deterministically (prefail.state + input masks, exactly as
solo_ram_capture) and runs the oracle each frame, using the SAME channel_reward()
as the training env, so the total sits on the same scale as random/policy evals.

enemy_dmg is -04-slot-specific (0x03253 = sprite slot 3), so it is REPORTED but
only rewarded with --enemy-ok (pass it only for a tape that stays in -04).

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga -w /vga/train/rl \\
      thor-rl:cu130 python3 score_tape.py <tape-dir>
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mgba.core
import mgba.image
import mgba.log

mgba.log.silence()
from alttp_ppo_env import ROM, channel_reward
from oracle import AlttpOracle

CHANNEL_ORDER = ["key", "key_used", "enemy_dmg", "rupees", "heal", "magic",
                 "heart_container", "new_room", "explore", "damage", "death"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="solo_recorder tape dir (prefail.state + inputs.json)")
    ap.add_argument("--rom", default=os.environ.get("FOUR_SWORDS_ROM", ROM))
    ap.add_argument("--enemy-ok", action="store_true",
                    help="reward enemy_dmg (only valid if the tape stays in -04)")
    ap.add_argument("--state", default=None, help="override start state")
    a = ap.parse_args()

    with open(os.path.join(a.dump, "inputs.json")) as f:
        masks = json.load(f)["masks"]
    state_path = a.state or os.path.join(a.dump, "prefail.state")

    core = mgba.core.load_path(a.rom)
    img = mgba.image.Image(*core.desired_video_dimensions())
    core.set_video_buffer(img)
    core.reset()
    with open(state_path, "rb") as f:
        assert core.load_raw_state(f.read()), "load_raw_state failed"

    orc = AlttpOracle()
    orc.step(0, core)  # prime baseline + seed the start room/cell (discard events)
    ev = collections.Counter()
    rew = collections.defaultdict(float)
    total = 0.0
    ladder = [orc.room_of(*orc.read_pos(core))]
    for k, mask in enumerate(masks):
        core.set_keys(raw=int(mask))
        core.run_frame()
        for _t, ch, delta, _v in orc.step(k + 1, core):
            ev[ch] += 1
            r = channel_reward(ch, delta, a.enemy_ok)
            rew[ch] += r
            total += r
        room = orc.room_of(*orc.read_pos(core))
        if room != ladder[-1]:
            ladder.append(room)

    n = len(masks)
    print("TAPE %s : %d frames (%.0fs), start %s"
          % (os.path.basename(os.path.normpath(a.dump)), n, n / 60.0,
             os.path.basename(state_path)))
    print("TOTAL reward %+.1f | rooms-visited %d, cells %d | keys %d, dmg %d, deaths %d"
          % (total, len(orc.rooms_seen), len(orc.cells_seen),
             ev.get("key", 0), ev.get("damage", 0), ev.get("death", 0)))
    print("per-channel (events -> reward):")
    for ch in CHANNEL_ORDER:
        if ev.get(ch, 0):
            note = "   [NOT rewarded here: -04-slot-specific]" if (
                ch == "enemy_dmg" and not a.enemy_ok) else ""
            print("  %-15s %5d -> %+9.1f%s" % (ch, ev[ch], rew[ch], note))
    print("room ladder (%d unique, %d rungs): %s"
          % (len(orc.rooms_seen), len(ladder), " -> ".join("(%d,%d)" % r for r in ladder)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
