#!/bin/bash
# v3 savestate-coherence test wrapper: save phase then resume phase, separate
# processes. Exit codes from the phases are unreliable (teardown segfault at
# interpreter exit), so success is judged by artifact files, not rc.
cd ~/projects/VGA/link
source env.sh
rm -f sessions/linktest_v3/verdict.txt sessions/linktest_v3/control_rate.txt
"$LINK_PY" link_test_v3_savestate.py save
echo "--- save phase process exited rc=$? (segfault-at-exit is known/harmless)"
if [ ! -f sessions/linktest_v3/control_rate.txt ]; then
    echo "save phase did not complete (no control_rate.txt)"; exit 1
fi
"$LINK_PY" link_test_v3_savestate.py resume
echo "--- resume phase process exited rc=$? (segfault-at-exit is known/harmless)"
if [ "$(cat sessions/linktest_v3/verdict.txt 2>/dev/null)" = "PASS" ]; then
    echo "SAVESTATE COHERENCE: PASS"; exit 0
fi
echo "SAVESTATE COHERENCE: FAIL"; exit 1
