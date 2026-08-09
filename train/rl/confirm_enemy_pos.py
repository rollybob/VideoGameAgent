"""Confirm IWRAM 0x03852 (X) / 0x03854 (Y) is the live enemy position.

The idle-capture trajectory already shows a sprite homing onto stationary Link
and stopping adjacent, in Link's own coordinate frame. This is the independent
check that the address is AUTHORITATIVE, not a read-only shadow: poke the enemy
far from Link and keep idling.

  - if the write STICKS (reads back the poked value next frame, not the old one)
    and then EVOLVES smoothly from there (the enemy re-homes toward Link), the
    engine reads and writes this byte -- it is the position.
  - if the poke is overwritten instantly, it is a shadow and the master is
    elsewhere (still fine to READ for reward, but worth knowing).

No pixels needed; behaviour in RAM is the evidence.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

X, Y = 0x03852, 0x03854
LX_ADDR, LY_ADDR = 0x038F4, 0x038F0


def main():
    from alttp_ppo_env import AlttpPpoEnv
    from mgba._pylib import ffi

    env = AlttpPpoEnv(state_path=os.path.join(HERE, "states", "alttp_human-04.state"),
                      horizon=10 ** 9, death_terminates=False)
    env.reset()
    core = env.core
    iw = ffi.cast("uint8_t *", core._native.memory.iwram)

    def u16(a):
        return int(iw[a]) | (int(iw[a + 1]) << 8)

    def idle(n):
        for _ in range(n):
            core.set_keys(raw=0)
            core.run_frame()

    lx, ly = u16(LX_ADDR), u16(LY_ADDR)
    idle(40)
    print("Link (%d,%d).  enemy before poke: (%d,%d)  dist=%.0f"
          % (lx, ly, u16(X), u16(Y), ((u16(X) - lx) ** 2 + (u16(Y) - ly) ** 2) ** 0.5))

    # poke the enemy 300 units UP and 120 LEFT, well away from Link
    newx, newy = u16(X) - 120, u16(Y) - 300
    iw[X], iw[X + 1] = newx & 0xFF, (newx >> 8) & 0xFF
    iw[Y], iw[Y + 1] = newy & 0xFF, (newy >> 8) & 0xFF
    print("poked enemy to (%d,%d)" % (newx, newy))

    idle(1)
    stuck = abs(u16(X) - newx) <= 4 and abs(u16(Y) - newy) <= 4
    print("1 frame after poke: (%d,%d)  -> %s"
          % (u16(X), u16(Y), "STUCK (authoritative)" if stuck else "SNAPPED BACK (shadow)"))

    print("re-homing over the next 60 idle frames (dist to Link should shrink):")
    for k in range(6):
        idle(10)
        ex, ey = u16(X), u16(Y)
        print("  +%2d frames: (%d,%d)  dist=%.0f"
              % ((k + 1) * 10, ex, ey, ((ex - lx) ** 2 + (ey - ly) ** 2) ** 0.5))

    env.close()
    return 0 if stuck else 1


if __name__ == "__main__":
    sys.exit(main())
