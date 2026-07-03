#!/usr/bin/env bash
# F18 Phase 2.2 - REAL background + multi-font harvest. Boot a DIVERSE set of GBA ROMs
# (distinct engines => distinct fonts) headless, drive each with capture_frames.py, and
# bank cropped GBA frames under captures/harvest/<slug>/. A later step runs EAST over
# these to extract text-line crops (real backgrounds for the synthetic compositor +
# a hard cross-font eval set). Reuses the run_capture.sh recipe (Xvfb + twm REQUIRED).
# Run as a thor-job (boots ~22 games; tens of minutes). No GPU.
set -uo pipefail
cd "$HOME/projects/VGA"
export DISPLAY=:99
VENV="$HOME/projects/VGA/.venv/bin/python"
ROMS="$HOME/projects/VGA/Emulator/mGBA/roms"
N="${N:-90}"; STRIDE="${STRIDE:-6}"

pgrep -x Xvfb >/dev/null || ( Xvfb :99 -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 & )
sleep 2
pgrep -x twm >/dev/null || ( twm >/tmp/twm.log 2>&1 & )
sleep 1

# slug|romfile -- curated for FONT diversity across engines (one per distinct font family).
GAMES=(
  "pokefrlg|Pokemon AI Red.gba"
  "pokeemerald|Pokemon All Things It Devours V1.1.gba"
  "pokemystery|Pokemon Mystery Dungeon - Red Rescue Team (U)(RDG).gba"
  "fe7|Fire Emblem (USA, Australia).gba"
  "fe6|Fire Emblem - Sealed Sword.gba"
  "fe8|Fire Emblem - the Sacred Stones # GBA.GBA"
  "ffta|Final Fantasy Tactics Advance (E)(Surplus).gba"
  "advancewars|Advance Wars (USA) (Rev 1).gba"
  "advancewars2|Advance Wars 2 - Black Hole Rising (USA).gba"
  "zeldaminish|The Legend of Zelda - The Minish Cap (U)(DCS).gba"
  "zeldalttp|Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba"
  "doom|Doom.gba"
  "dkc|Donkey Kong Country # GBA.GBA"
  "dkc2|Donkey Kong Country 2 # GBA.GBA"
  "mariopinball|Mario Pinball Land (U).gba"
  "metalslug|Metal Slug Advance (U)(Independent).gba"
  "sonicpinball|Sonic Pinball Party (U) (M6).gba"
  "namco|Namco Museum 50th Anniversary (E)(sUppLeX).gba"
  "yugioh|Yu-Gi-Oh! - Destiny Board Traveler (U).gba"
  "dragonball|Dragon Ball - Advanced Adventure (U)(Ongaku).gba"
  "spyro|Legend of Spyro, The - A New Beginning (U).gba"
  "baldursgate|Baldur's Gate - Dark Alliance # GBA.GBA"
)

ok=0; fail=0
for entry in "${GAMES[@]}"; do
  slug="${entry%%|*}"; rom="$ROMS/${entry#*|}"
  if [ ! -f "$rom" ]; then echo "[harvest] MISSING $slug ($rom)"; fail=$((fail+1)); continue; fi
  echo "[harvest] $(date -Is) $slug"
  pkill -x mgba-qt 2>/dev/null; sleep 1
  nohup /usr/games/mgba-qt "$rom" >"/tmp/mgba_$slug.log" 2>&1 &
  found=0
  for i in $(seq 1 40); do xdotool search --name mGBA >/dev/null 2>&1 && { found=1; break; }; sleep 0.3; done
  if [ "$found" = 0 ]; then echo "[harvest] $slug window never appeared; skip"; fail=$((fail+1)); continue; fi
  sleep 6   # boot through logos
  if "$VENV" train/ocr/capture_frames.py --name "$slug" --out "captures/harvest/$slug" --n "$N" --stride "$STRIDE"; then
    ok=$((ok+1))
  else
    echo "[harvest] $slug capture FAILED"; fail=$((fail+1))
  fi
done
pkill -x mgba-qt 2>/dev/null
echo "[harvest] $(date -Is) DONE ok=$ok fail=$fail -> captures/harvest/*/"
