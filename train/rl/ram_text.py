"""ram_text.py -- read the game's on-screen dialogue/hints straight from RAM.

FOUND 2026-08-09 (night, autonomous): "The Legend of Zelda: A Link to the Past & Four
Swords" (GBA) stores the active message text as PLAIN ASCII in EWRAM starting ~0x05485.
No OCR and no bigger VLM needed. Verified LIVE (buffer updates per message):
  - intro backstory ("A mysterious wizard known as Agahnim ...")
  - Zelda's telepathy ("... I am a prisoner in the castle dungeon. My name is Zelda ...")
  - the uncle ("... I don't want you leaving the house until then ...")
  - the KEY navigation hint: "Somewhere outside the castle you should find a hidden
    entrance to the palace garden."

WHY THIS MATTERS: the 3-tier agent IGNORED on-screen hints (Tim, 2026-08-09: "it read
nothing"). Root cause was that the 8B VLM can't reliably read GBA text (perception-bench
finding). This is the fix for the COMPREHENSION gap -- feed extract_text() into the VLM's
context (task_phase / a knowledge store) so the agent finally acts on the game's own
directions (e.g. head for the castle garden's hidden entrance).

REFINEMENT TODO: 0x05480-0x05820 spans the whole message scrollback, so extract_text can
bleed in stale text or menu labels ("Continue Game Save Sleep"). To isolate the *current*
box exactly, find the active text-position pointer or gate on a text-box-active flag.
Good enough as v1: the current message is the last coherent sentence(s) in the buffer.
"""
import numpy as np
from mgba._pylib import ffi

TEXT_LO, TEXT_HI = 0x05480, 0x05820   # EWRAM dialogue buffer (ALttP-FS GBA)
MSG_SEP = 0x18                        # separates messages; ACTIVE msg = [TEXT_LO, first 0x18)


def extract_text(core):
    """Return the CURRENT (most-recent) dialogue message as clean ASCII, or "".

    The ALttP-FS message engine decodes text into EWRAM at 0x05480 with inline
    control bytes (<0x20: line/word breaks; 0x1F<arg> = button icon) and a 0x18
    byte SEPARATING messages. The buffer is NOT cleared when a box closes, and a
    new message overwrites only its own length -- so the bytes AFTER the first
    0x18 are STALE scrollback from older messages (verified 2026-08-10 by hex
    dump of the human-08/-14/ingame states). The active message is therefore
    exactly 0x05480 up to the first 0x18. Control bytes render as spaces and
    whitespace is collapsed. Returns "" when the buffer holds no real text
    (empty, or <3 letters) so a caller can treat "" as "no message".

    STILL OPEN: this returns the most-recent message even when the box is CLOSED
    (the buffer stays stale). A true "box on screen NOW" gate needs a box-active
    flag (unfound) or a pixel box detector -- built alongside the labeler.
    """
    mem = ffi.cast("uint8_t *", core._native.memory.wram)
    chars = []
    for off in range(TEXT_LO, TEXT_HI):
        b = mem[off]
        if b == MSG_SEP:
            break
        chars.append(chr(b) if 0x20 <= b <= 0x7E else " ")
    s = " ".join("".join(chars).split())
    return s if sum(c.isalpha() for c in s) >= 3 else ""


LINE_BREAKS = frozenset({0x0C, 0x0E, 0x0F, 0x17})   # intra-box line + page breaks (verified 2026-08-10)


def extract_lines(core):
    """The active message split into VISUAL lines. The engine marks line/page breaks with
    control bytes 0x0C/0E/0F/17 (0x18 ends the message; 0x1F<arg> is a button icon). Each
    returned line pairs with one visual line of a single-page box -- the per-line LABEL
    source for the pixel reader's line crops (teacher for the CRNN fine-tune).

    Caveat: multi-page messages return ALL pages' lines, but a single frame shows only the
    current page -- so pair by matching line COUNT (use only boxes where the number of RAM
    lines equals the number of segmented visual lines).
    """
    mem = ffi.cast("uint8_t *", core._native.memory.wram)
    lines, cur = [], []
    for off in range(TEXT_LO, TEXT_HI):
        b = mem[off]
        if b == MSG_SEP:
            break
        if b in LINE_BREAKS:
            s = "".join(cur).strip()
            if any(c.isalpha() for c in s):
                lines.append(s)
            cur = []
        elif 0x20 <= b <= 0x7E:
            cur.append(chr(b))
    s = "".join(cur).strip()
    if any(c.isalpha() for c in s):
        lines.append(s)
    return lines
