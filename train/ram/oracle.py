#!/usr/bin/env python3
"""RAM oracle: turn raw emulator memory into structured game STATE and a scalar
REWARD/progress signal. TRAINING+EVAL ONLY - runtime perception stays pixels-only
(NORTH_STAR Sec 3). This is the "teacher" the reason layer has been missing: the VLM
plays from pixels and has no idea whether it is making progress; the oracle reads the
truth from RAM and can score it.

Two jobs:
  read_state(emu) -> dict   : the structured fields we trust (player x/y, cursor, ...)
  reward(prev, cur) -> float: how much better did that step make things?

`reward` here is deliberately simple, honest shaping - exploration (new tiles seen) and
navigation (cursor moved). It is NOT a learned or task-complete reward; it is enough to
(a) tell "the agent is progressing" from "the agent is spinning in a menu", and (b) label
rollouts so a filtered behavioral-cloning / self-imitation pass can keep the good steps.
Richer per-game reward (HP, battles won, story flags) slots in as we map more addresses.

The oracle also exposes stuck detection: a long run of zero reward at the SAME state is
the RAM-truth version of the menu-loop the pixel heuristics guess at - and a stuck->escape
transition is exactly the kind of lesson a reflection pass should write into the SkillStore
(vga/reason/skills.py), closing the loop from this teacher back to the policy.
"""

from __future__ import annotations

import os
from typing import Optional

from addresses import FFTA, POKEMON_AI_RED


class Oracle:
    """Base per-game oracle. Subclasses supply the address map and any reward shaping.
    State is a flat dict of trusted fields; reward compares two consecutive states."""
    name = "generic"
    addrs: dict = {}

    def read_state(self, emu) -> dict:
        """Read every registered address into a flat dict {field: value}."""
        out = {}
        for field, (addr, width) in self.addrs.items():
            out[field] = {"u8": emu.u8, "u16": emu.u16, "u32": emu.u32}[width](addr)
        return out

    def boot_to_play(self, emu) -> None:
        """Advance past the intro/title to interactive play. Default: just boot-settle;
        games with a language/title/load flow override this."""
        emu.run(300)

    def reward(self, prev: Optional[dict], cur: dict) -> float:
        """Scalar progress for the step prev->cur. Override per game. Default: 0."""
        return 0.0

    def progress_summary(self, cur: dict) -> str:
        """One-line human-readable state, for logs/notify."""
        return " ".join(f"{k}={v}" for k, v in cur.items())

    def checkpoints(self, cur: dict) -> list:
        """Ordered milestone ladder for a task, as INSTANTANEOUS, stateless predicates over
        ONE step's state dict: [(name, bool), ...] in progression order. The eval metric is
        the FURTHEST rung ever satisfied in an episode (train/ram/analyze_rollout.py), which
        is dense and diagnostic - it says WHERE in a long action-chain the agent stalled,
        not just pass/fail (see docs/CHECKPOINT_LADDER_PLAN.md). Default: no ladder."""
        return []


class PokemonAIRed(Oracle):
    name = "pokemon_ai_red"
    addrs = POKEMON_AI_RED

    def __init__(self):
        # Exploration memory: reward the FIRST visit to each tile, so wandering into new
        # map is progress and pacing back and forth is not. This is the RAM-truth analog
        # of "am I actually getting somewhere" that the pixel loop-detector approximates.
        self._visited: set[tuple[int, int]] = set()

    def boot_to_play(self, emu) -> None:
        # Recipe from the F21 de-risk (find_coords6.py): boot, mash A to clear the
        # continue/intro dialogs, then settle to the interactive overworld.
        emu.run(300)
        for _ in range(6):
            emu.tap("A", hold=4, then=20)
        emu.run(600)

    def reward(self, prev: Optional[dict], cur: dict) -> float:
        tile = (cur.get("player_x"), cur.get("player_y"))
        r = 0.0
        if None not in tile and tile not in self._visited:
            self._visited.add(tile)
            r += 1.0          # new tile explored
        if prev is not None:
            moved = (prev.get("player_x"), prev.get("player_y")) != tile
            if not moved:
                r -= 0.05     # small penalty for a step that changed nothing on the map
        return r

    def progress_summary(self, cur: dict) -> str:
        return (f"pos=({cur.get('player_x')},{cur.get('player_y')}) "
                f"tiles_explored={len(self._visited)}")


class Ffta(Oracle):
    name = "ffta"
    addrs = FFTA

    # Which checkpoint ladder this oracle scores. "herb" = the pub->mission->travel task;
    # "naming" = the name-entry micro-ladder (RESCUE_PLAN P2b). Set by rollout(task=...).
    task = "herb"

    NAMING_SIG_ON = 0x888888F9   # naming_sig while the keyboard is up
    DIALOG_SIG_ON = 0x99999999   # dialog_sig while a dialog/confirm box is up

    def __init__(self):
        self._cursor_history: list[int] = []

    def boot_to_play(self, emu) -> None:
        # FFTA needs language -> title -> Saved Game -> Load -> slot to reach the world map
        # (see addresses.py). We don't script that whole flow here; callers that want FFTA
        # world-map state should load the ffta_worldmap.state savestate before rolling out.
        emu.run(300)

    def reward(self, prev: Optional[dict], cur: dict) -> float:
        # World map is node-graph navigation: reward reaching a cursor/location value we
        # have not been at before (exploration), nothing for sitting still.
        c = cur.get("worldmap_cursor")
        if c is None:
            return 0.0
        if c not in self._cursor_history:
            self._cursor_history.append(c)
            return 1.0
        return 0.0

    def progress_summary(self, cur: dict) -> str:
        return (f"worldmap_cursor={cur.get('worldmap_cursor')} "
                f"visited={len(self._cursor_history)} funds={cur.get('clan_funds')} "
                f"mode_overlay={cur.get('mode_overlay')}")

    # Baseline party gil at ffta_pub.state; any drop = an info fee was paid = a mission was
    # accepted. Stateless because every eval episode starts from that fixed savestate.
    PUB_FUNDS_BASELINE = 5000

    def checkpoints(self, cur: dict) -> list:
        """The Herb Picking ladder. Rungs 0-2 verified 2026-07-02 by savestate diffing
        (train/ram/find_mission.py); rungs 3-5 added 2026-07-03 from the multi-group EWRAM
        diff (train/ram/find_wm_state.py: scene@0x0200027f 6=town 7=wm 14=battle,
        clan_pos@0x02001f69 18=Cyril 20=Giza; constant across all 27 dumps in
        sessions/wm-ram-dumps-0703/).

            pub_open         : at the pub (definitionally true at load - the rung-0 anchor).
            mission_list     : reached the Missions list (mode_overlay==208, not Rumors).
            mission_accepted : paid the info fee -> mission accepted (clan_funds < baseline).
            worldmap_regained: accepted AND back on the world map (scene==7). Gated on
                accepted so exiting the pub WITHOUT the mission scores nothing.
            at_giza          : accepted AND clan node == Giza, Herb Picking's battle site.
            battle_entered   : accepted AND on the battle map (scene==14).
        Still deferred: battle-turn rungs (need battle-grid RAM, see addresses.py TODO)."""
        if self.task == "naming":
            return self._checkpoints_naming(cur)
        funds = cur.get("clan_funds")
        mode = cur.get("mode_overlay")
        scene = cur.get("scene")
        pos = cur.get("clan_pos")
        accepted = funds is not None and funds < self.PUB_FUNDS_BASELINE
        return [
            ("pub_open", True),
            ("mission_list", mode == 208),
            ("mission_accepted", accepted),
            ("worldmap_regained", bool(accepted and scene == 7)),
            ("at_giza", bool(accepted and pos == 20)),
            ("battle_entered", bool(accepted and scene == 14)),
        ]

    def _checkpoints_naming(self, cur: dict) -> list:
        """Name-entry micro-ladder (RESCUE_PLAN P2b), from train/ram/ffta_naming.state.
        Signals verified 2026-07-03 (see addresses.py naming_sig/dialog_sig):

            naming_open    : keyboard up (anchor rung, true at load).
            confirm_reached: the "Confirm action. OK? Yes/No" box opened (Start pressed).
                Diagnostic rung for the known wall: reaching confirm but committing No.
            name_committed : keyboard gone. B cannot escape the naming screen, so the
                only way the sig drops is Yes on the confirm box = name committed.
        """
        nsig = cur.get("naming_sig")
        dsig = cur.get("dialog_sig")
        naming_on = nsig == self.NAMING_SIG_ON
        return [
            ("naming_open", True),
            ("confirm_reached", bool(naming_on and dsig == self.DIALOG_SIG_ON)),
            ("name_committed", bool(nsig == 0)),
        ]


# Registry by ROM filename substring. Keep in sync with addresses.py.
_REGISTRY = [
    ("pokemon ai red", PokemonAIRed),
    ("ffta", Ffta),
    ("final fantasy tactics", Ffta),
]


def for_rom(rom_path: str) -> Oracle:
    """Pick the oracle for a ROM by filename. Raises if we have no address map for it."""
    base = os.path.basename(rom_path).lower()
    for key, cls in _REGISTRY:
        if key in base:
            return cls()
    raise ValueError(f"No RAM oracle registered for ROM '{base}'. Add addresses + a "
                     f"registry entry in train/ram/oracle.py.")
