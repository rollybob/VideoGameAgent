-- VGA human-demonstration logger for mGBA.
-- Load via the mGBA GUI: Tools > Scripting... opens a (blank) Scripting window; then in that
-- window File > Load script... > this file. Load AFTER the ROM is running.
--
-- SELF-CONTAINED: on load it creates its OWN timestamped run dir
--   ~/projects/VGA/sessions/human-<epoch>/frames
-- (mGBA opens the full Lua stdlib via luaL_openlibs, so os.execute/os.time/io are available)
-- and logs (screenshot, GBA key bitmask) at every INPUT CHANGE into that dir's inputs.csv.
-- No host prep needed -- just load and play. emu:screenshot dumps the clean native 240x160
-- framebuffer (no window chrome). Releases (keymask 0) are skipped, so there is no idle-wait
-- spam -- every logged sample is a deliberate human input on the frame it was made.
--
-- Key bit order = authoritative GBAKey enum (verified vs mGBA source):
--   0=A 1=B 2=Select 3=Start 4=Right 5=Left 6=Up 7=Down 8=R 9=L
-- (the bundled socketserver.lua example has Left/Right SWAPPED -- do not trust it).
--
-- RELOAD CAVEAT: mGBA "Load script" does NOT remove a prior script's callbacks -- reloading
-- STACKS another logger. With self-created dirs this no longer corrupts data (each logger
-- writes to its own fresh dir), but you would get duplicate captures. To start ONE clean
-- run, Reset the scripting environment (or restart mGBA) before loading.

local ROOT = "/home/timothy/projects/VGA/sessions"
local RUN_DIR = string.format("%s/human-%d", ROOT, os.time())
os.execute(string.format("mkdir -p '%s/frames'", RUN_DIR))

local csv = assert(io.open(RUN_DIR .. "/inputs.csv", "w"),
	"cannot create " .. RUN_DIR .. "/inputs.csv (mkdir failed?)")
local lastkeys = 0
local n = 0

local function onKeys()
	local keys = emu:getKeys()
	if keys == lastkeys then return end   -- only on change
	lastkeys = keys
	if keys == 0 then return end          -- log presses/chords, not releases
	emu:screenshot(string.format("%s/frames/%06d.png", RUN_DIR, n))
	csv:write(string.format("%d,frames/%06d.png,%d\n", n, n, keys))
	csv:flush()
	n = n + 1
end

callbacks:add("keysRead", onKeys)
console:log("VGA human logger active -> " .. RUN_DIR)
