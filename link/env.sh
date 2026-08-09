# Source this to run the link engine against the on-box mGBA Python bindings.
#   source ~/projects/VGA/link/env.sh ; "$LINK_PY" link_engine.py
# The bindings live in the mGBA build tree; the venv (cffi + cached_property) is the
# scouting sandbox venv so we don't re-provision. Migrate into a repo-local venv later.

export MGBA_BUILD="$HOME/src/mgba/build-py"
export LD_LIBRARY_PATH="$MGBA_BUILD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$MGBA_BUILD/python/lib.linux-aarch64-cpython-312${PYTHONPATH:+:$PYTHONPATH}"
export LINK_PY="$HOME/mgba-mp-probe/.venv/bin/python3"
export FOUR_SWORDS_ROM="$HOME/projects/VGA/Emulator/mGBA/roms/Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba"
