"""
Stage-1 plumbing test for the deterministic Python link coordinator.
Validates the four load-bearing primitives on TWO real GBA cores at once:
  1. attach a Python GBASIODriver to each core (attach_sio, SIO_MULTI)
  2. read/write the SIO struct directly (siocnt / rcnt / mode)
  3. read/write the IO register array (memory.io[REG >> 1]) -- e.g. SIOMULTI0..3
  4. raise the SIO IRQ from Python (GBARaiseIRQ + GBA_IRQ_SIO)
  5. read both cores' framebuffers independently
If all pass, the pure-Python coordinator is buildable with no rebuild.
"""
import mgba.core, mgba.gba, mgba.image, mgba.log
from mgba._pylib import ffi, lib

mgba.log.silence()

# GBA hardware IO register offsets (C macros, not exposed as cffi constants).
REG_SIOMULTI0 = 0x120
REG_SIOCNT    = 0x128
REG_SIOMLT_SEND = 0x12A
REG_RCNT      = 0x134

ROM = "/home/timothy/projects/VGA/Emulator/mGBA/roms/" \
      "Legend of Zelda, The - A Link To The Past Four Swords (U) [!].gba"

def io_read(core, reg):
    return core._native.memory.io[reg >> 1]
def io_write(core, reg, val):
    core._native.memory.io[reg >> 1] = val & 0xFFFF

class LogDriver(mgba.gba.GBASIODriver):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.writes = []
    # NOTE: the binding's callback dispatcher looks up camelCase "writeRegister"
    def writeRegister(self, address, value):
        self.writes.append((address, value))
        return value

results = {}
cores, drivers, images = [], [], []
for i in range(2):
    c = mgba.core.load_path(ROM)
    w, h = c.desired_video_dimensions()
    img = mgba.image.Image(w, h)
    c.set_video_buffer(img)      # set buffer BEFORE reset (known-good order)
    c.reset()
    d = LogDriver("p%d" % i)
    c.attach_sio(d, lib.SIO_MULTI)
    cores.append(c); drivers.append(d); images.append(img)

print("type(core[0]):", type(cores[0]).__name__)
results["1_attach_sio_two_cores"] = True

# 5-FIRST: render on clean cores (before any destructive SIO pokes)
try:
    for _ in range(320):
        cores[0].run_frame(); cores[1].run_frame()
    b0 = bytes(mgba.image.ffi.buffer(images[0].buffer))
    b1 = bytes(mgba.image.ffi.buffer(images[1].buffer))
    nz0 = sum(1 for x in b0 if x); nz1 = sum(1 for x in b1 if x)
    results["5_two_framebuffers"] = "p0 nz=%d p1 nz=%d" % (nz0, nz1)
except Exception as e:
    results["5_two_framebuffers"] = "ERR %r" % e

# 2. direct SIO struct read/write
try:
    for i, c in enumerate(cores):
        _ = c._native.sio.siocnt, c._native.sio.rcnt, c._native.sio.mode
        c._native.sio.siocnt = 0x0008  # write a probe bit
    ok = all(c._native.sio.siocnt == 0x0008 for c in cores)
    results["2_sio_struct_rw"] = ok
except Exception as e:
    results["2_sio_struct_rw"] = "ERR %r" % e

# 3. IO register array read/write (SIOMULTI0)
try:
    io_write(cores[0], REG_SIOMULTI0, 0x1234)
    io_write(cores[1], REG_SIOMULTI0, 0x5678)
    ok = io_read(cores[0], REG_SIOMULTI0) == 0x1234 and io_read(cores[1], REG_SIOMULTI0) == 0x5678
    results["3_io_array_rw"] = ok
except Exception as e:
    results["3_io_array_rw"] = "ERR %r" % e

# 4. raise SIO IRQ from Python
try:
    lib.GBARaiseIRQ(cores[0]._native, lib.GBA_IRQ_SIO, 0)
    results["4_raise_sio_irq"] = True
except Exception as e:
    results["4_raise_sio_irq"] = "ERR %r" % e

# 6. bitfield helpers behave (sanity: set/get ready on a siocnt word)
try:
    s = 0
    s = lib.GBASIOMultiplayerSetReady(s, 1)
    results["6_bitfield_helpers"] = bool(lib.GBASIOMultiplayerIsReady(s))
except Exception as e:
    results["6_bitfield_helpers"] = "ERR %r" % e

print("\n=== PLUMBING RESULTS ===")
for k in sorted(results):
    print("  %-26s %s" % (k, results[k]))
allpass = all(v is True or (isinstance(v, str) and not v.startswith("ERR")) for v in results.values())
print("\nALL PRIMITIVES OK:", allpass)
