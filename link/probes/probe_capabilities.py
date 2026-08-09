"""
Capability probe: what C surface does the mGBA Python binding already expose?
This decides whether a deterministic single-thread link coordinator can be
built in pure Python (calling C primitives via cffi.lib) with NO rebuild, or
whether we must extend the cdef and rebuild the bindings.
"""
from mgba._pylib import ffi, lib

def show(title, names):
    print("\n== {} ({}) ==".format(title, len(names)))
    for n in sorted(names):
        print("  ", n)

allnames = [n for n in dir(lib)]
print("total lib symbols:", len(allnames))

show("SIO functions",        [n for n in allnames if 'SIO' in n or 'Sio' in n])
show("IRQ functions",        [n for n in allnames if 'IRQ' in n or 'Irq' in n])
show("Timing functions",     [n for n in allnames if 'Timing' in n or 'timing' in n])
show("Lockstep functions",   [n for n in allnames if 'ockstep' in n])
show("Core run/step",        [n for n in allnames if 'Run' in n or 'Step' in n or 'runLoop' in n])
show("GBA multiplayer bitfns", [n for n in allnames if 'Multiplayer' in n])

# Can we see the GBA struct internals (sio, memory.io) and construct a driver?
print("\n== struct field access probe ==")
for typ, field in [("struct GBA", "sio"), ("struct GBA", "memory"),
                   ("struct GBASIO", "siocnt"), ("struct GBASIO", "rcnt"),
                   ("struct GBASIO", "mode"), ("struct GBASIO", "driver")]:
    try:
        t = ffi.typeof(typ)
        flds = [f[0] for f in t.fields] if t.fields else []
        print("  {}: {} -> has '{}': {}".format(typ, "OK", field, field in flds))
    except Exception as e:
        print("  {}: ERR {}".format(typ, e))

# Key enums / constants
print("\n== constants ==")
for c in ["SIO_MULTI", "SIO_NORMAL_8", "IRQ_SIO", "GBA_IRQ_SIO", "REG_SIOCNT",
          "REG_SIOMULTI0", "REG_SIOMLT_SEND", "MAX_GBAS"]:
    try:
        print("  {} = {}".format(c, getattr(lib, c)))
    except Exception as e:
        print("  {}: (absent) {}".format(c, type(e).__name__))
