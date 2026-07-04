#!/usr/bin/env python3
"""Per-game RAM address registry for the F21 oracle (TRAINING+EVAL ONLY --
runtime perception stays pixels-only). Addresses found by scripted-movement
diffing (find_coords*.py), NOT from known vanilla offsets, so they hold for
ROM hacks whose SaveBlock has drifted.

Each entry: absolute GBA bus address + width. Read via emu.Emu.u8/u16/u32.
"""

# "Pokemon AI Red" (FRLG hack). Verified 2026-06-30: player coords track
# movement tile-by-tile; object-event copy sits at map+7 (FRLG MAP_OFFSET),
# an independent confirmation.
POKEMON_AI_RED = {
    "player_x":     (0x02025554, "u16"),   # SaveBlock map tile X
    "player_y":     (0x02025556, "u16"),   # SaveBlock map tile Y
    "obj_player_x": (0x02036e48, "u16"),   # gObjectEvents[0] X == player_x + 7
    "obj_player_y": (0x02036e4a, "u16"),   # gObjectEvents[0] Y == player_y + 7
    # TODO: map_id, party HP/PP, money, battle flag, menu cursor (same method)
}

# Final Fantasy Tactics Advance (E). Verified 2026-06-30 the harness generalizes:
# loaded the commercial ROM headless, drove language->title->Saved Game->Load->
# slot1 to the WORLD MAP, and movement-diffing surfaced a position struct at
# 0x02002c__. NUANCE: the world map is NODE-GRAPH navigation (travel along paths
# between locations), NOT free tile x/y -- every direction from the start node
# snaps 0x02002c10 (97->65) to the single reachable neighbor. So the clean
# tile-coordinate model (Pokemon) does not apply here; this is a location/cursor
# value. Real grid x/y live on a BATTLE map (needs story advance: pub -> accept a
# quest, per Tim) -- do that when we want FFTA battle-grid coords.
# Clean world-map savestate: ffta_worldmap.state
FFTA = {
    "worldmap_cursor": (0x02002c10, "u16"),   # changes on world-map navigation (node-graph)
    # Found 2026-07-02 by known-change savestate diffing from ffta_pub.state
    # (train/ram/find_mission.py, accept-vs-decline control):
    "clan_funds":      (0x02001f64, "u16"),   # party gil; 5000 at the pub baseline, drops on
                                              # mission acceptance (300-gil info fee). Verified
                                              # via live memory API AND the decline control.
    "mode_overlay":    (0x02003cb7, "u8"),    # UI-overlay flag: 208=missions list, 255=wm-menu,
                                              # 204=area list, 3=help, 0=field/pub/dialog.
    # Found 2026-07-03 by multi-group EWRAM diffing (train/ram/find_wm_state.py; 27 dumps in
    # sessions/wm-ram-dumps-0703/) -- constant within scene groups, distinct across them:
    "scene":           (0x0200027f, "u8"),    # 6=town/pub interior, 7=world map, 14=battle map
                                              # (0x02000282 mirrors it; fallback if 027f drifts).
    "clan_pos":        (0x02001f69, "u8"),    # clan's world-map node: 18=Cyril, 20=Giza Plains,
                                              # 26=Sprohm. Herb Picking's battle is at Giza (20)
                                              # per the bat1/bat2 dumps.
    # Naming-screen signals, found 2026-07-03 by multi-group diffing around
    # train/ram/ffta_naming.state (docs/RESCUE_PLAN.md P2b). Both are u32 reads of
    # 4-byte UI tile-buffer signatures -- a single byte was too collision-prone:
    "naming_sig":      (0x02003d30, "u32"),   # 0x888888f9 while the naming keyboard is up
                                              # (incl. under the confirm box); 0 once the
                                              # name is committed. B cannot escape naming,
                                              # so sig ON->OFF == name committed.
    "dialog_sig":      (0x02004eb0, "u32"),   # 0x99999999 while a dialog/confirm box is up.
                                              # NOT unique to the confirm box (post-commit
                                              # dialogue uses the same buffer) -- only
                                              # meaningful combined with naming_sig ON.
    # TODO (battle map): grid cursor x/y, unit HP/positions -- battle placement-phase dumps
    # exist (sessions/wm-ram-dumps-0703/bat2); travel is scriptable, no Tim-driven state needed.
}
