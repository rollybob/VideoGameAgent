"""Find an address from a human-marked event (link/ram_ring.py dumps).

METHOD (this is the one that has actually worked -- see the ALttP health-oracle
and Sea of Trees notes): a human makes ONE real, visible thing happen and marks
it. The address is then the byte that (a) held a constant value for a while,
(b) changed inside the window, and (c) held the new value after. Almost nothing
else in RAM behaves like that -- animation noise churns continuously, timers
ramp, and pointers jump around.

MODES -- pick the one that matches the event the human performed:

  down       taking damage: HP steps DOWN.  (default; unchanged, and this is the
             path that found ALttP health IWRAM 0x00428)
  up         picking something up: a counter steps UP. Keys, bombs, arrows.
  any        the value changed and settled, direction unconstrained. Room IDs on
             a one-way transition.
  roundtrip  v1 -> v2 -> v1. THE strong pattern for anything without a HUD
             readout, because returning to the exact original value is very
             unlikely by chance. Two shapes, both supported:
               - across two dumps: pick a key up (0 -> 1) in dump A, spend it on
                 the door (1 -> 0) in dump B
               - inside one dump: through a door and back, if both fit the ~4 s
                 window

WHY ROUNDTRIP EXISTS (2026-07-31): the write-test that confirmed the rupee
counter cannot confirm a key counter. It worked for rupees only because the
rupee HUD is an ANIMATED rolling counter that visibly redraws toward whatever
you poke; a key count has no such animation, so "poke it and look" is
inconclusive rather than negative. An up-then-back-DOWN round trip needs no HUD
redraw at all -- the RAM proves itself.

TOLERANCE: the up/any/roundtrip modes require a stable DWELL either side of the
change but allow any number of intermediate values, rather than demanding
exactly one clean step. That is deliberate. The strict single-step filter is
what rejected the real health address earlier the same day, because damage
drains over several frames; a room ID can likewise pass through loading values
mid-transition. `down` keeps the strict detector it was validated with.

The eighths fingerprint is the strong extra filter for HEALTH specifically:
Zelda hearts are quarters stored in eighths, so real HP values are multiples of
8 and a hit costs a whole number of eighths. Candidates matching that are ranked
first. Counters (keys) are scored on a different fingerprint: +/-1 and small.

  docker run --rm --user 1000:1000 -v ~/projects/VGA:/vga thor-torch:cu130 \
      python3 /vga/train/rl/ram_ring_diff.py \
      /vga/link/sessions/ramhits_fs/<dump-dir>

  # key round trip: pickup dump first, spend dump second
  ... ram_ring_diff.py --mode roundtrip <pickup-dump> <spend-dump>

  # self-check the detectors against real RAM with a synthetic event injected
  ... ram_ring_diff.py --selftest link/sessions/ramhits_solo/*-hit
"""
import argparse
import json
import os

import numpy as np

REGIONS = (("IWRAM", 0, 32 * 1024), ("EWRAM", 32 * 1024, 256 * 1024))


def load(dump):
    with open(os.path.join(dump, "meta.json")) as f:
        meta = json.load(f)
    raw = np.fromfile(os.path.join(dump, "ring.bin"), dtype=np.uint8)
    n, snap = meta["n"], meta["snap_size"]
    if raw.size != n * snap:
        # a dump written while the ring was still filling, or a torn last write
        n = raw.size // snap
        raw = raw[:n * snap]
        print("warning: ring.bin short, using %d snapshots" % n)
    return meta, raw.reshape(n, snap)


def single_step_down(col):
    """Return (before, after, index) if col is constant, drops once, constant."""
    change = np.flatnonzero(col[1:] != col[:-1])
    if change.size != 1:
        return None
    i = int(change[0])
    before, after = int(col[0]), int(col[-1])
    if after >= before:
        return None
    # require real dwell either side, so a value merely passing through is out
    if i < 2 or i > col.size - 3:
        return None
    return before, after, i


def runs(col):
    """Run-length encode a column -> [(value, start, length), ...].

    Every tolerant detector below is expressed on runs rather than on raw
    samples, because "held still, changed, held still" is a statement about runs
    and gets muddled when written against individual transitions.
    """
    edge = np.flatnonzero(col[1:] != col[:-1]) + 1
    bounds = np.concatenate(([0], edge, [col.size]))
    return [(int(col[bounds[k]]), int(bounds[k]), int(bounds[k + 1] - bounds[k]))
            for k in range(bounds.size - 1)]


def stable_transition(col, min_dwell, direction):
    """Constant -> (anything) -> constant, different value. None if it does not fit.

    Unlike single_step_down this permits intermediate values: a real counter can
    take a frame or two to settle, and a room ID can pass through a loading
    value. What it does NOT permit is failing to hold still either side, which
    is what rules out the churning animation bytes.
    """
    r = runs(col)
    if len(r) < 2:
        return None
    before, after = r[0][0], r[-1][0]
    if before == after:
        return None
    if r[0][2] < min_dwell or r[-1][2] < min_dwell:
        return None
    if direction == "up" and after <= before:
        return None
    if direction == "down" and after >= before:
        return None
    return before, after, r[0][2], len(r) - 1


def within_round_trip(col, min_dwell):
    """v1 -> v2 -> v1 inside ONE window. (v1, v2, n_changes, v2_dwell) or None.

    Requires a real dwell at v2 as well as at both ends, so a byte that merely
    flickers through another value on its way back does not qualify.

    v2_dwell is returned because it is the main thing separating a real visit
    from noise: if the human walked into the next room and back, or picked a key
    up and spent it, the intermediate value HELD for a good fraction of the
    window. A wobbling animation byte returns home almost immediately.
    """
    r = runs(col)
    if len(r) < 3:
        return None
    v1 = r[0][0]
    if r[-1][0] != v1 or r[0][2] < min_dwell or r[-1][2] < min_dwell:
        return None
    middle = [x for x in r[1:-1] if x[0] != v1 and x[2] >= min_dwell]
    if not middle:
        return None
    best = max(middle, key=lambda x: x[2])
    return v1, best[0], len(r) - 1, best[2]


def prefilter(ring, min_dwell, same_ends):
    """Vectorised cut before the per-address Python loop.

    The ring is ~35 MB; testing every byte in Python is slow enough to matter
    when this runs several times per capture session. Both dwell requirements
    and the ends-equal/ends-differ split are pure numpy, and they remove the
    overwhelming majority of bytes (which either never move or never hold).
    """
    head = np.all(ring[:min_dwell] == ring[0], axis=0)
    tail = np.all(ring[-min_dwell:] == ring[-1], axis=0)
    moved = ring.max(axis=0) != ring.min(axis=0)
    ends_equal = ring[0] == ring[-1]
    ok = head & tail & moved
    ok &= ends_equal if same_ends else ~ends_equal
    return np.flatnonzero(ok)


def score_counter(before, after, nchanges, counter_max):
    """How much this looks like a COLLECTIBLE COUNTER (keys, bombs, arrows).

    Different fingerprint from health entirely: counters move by exactly one and
    live in single digits, where health moves in eighths and sits in the tens.
    Scored, never hard-filtered -- the same rule that saved the Sea of Trees
    address from the eighths filter.
    """
    delta = abs(after - before)
    score = 0
    if delta == 1:
        score += 4                     # a counter increments, it does not ramp
    elif delta <= 4:
        score += 1
    if 0 < max(before, after) <= counter_max:
        score += 2                     # key counts are single digits
    if before == 0 or after == 0:
        score += 2                     # 0 -> 1 (first key) / 1 -> 0 (spent it)
    if nchanges == 1:
        score += 1                     # settled instantly; weak, not required
    return score


def fmt_addr(addr):
    for name, base, size in REGIONS:
        if base <= addr < base + size:
            return "%s 0x%05X" % (name, addr - base)
    return "?? 0x%05X" % addr


def neighbour_max(ring, addr, before):
    """Look for a constant byte just beside `addr` holding >= the pre-hit value.

    Both confirmed health addresses to date sat immediately after a max-HP byte
    (ALttP, and Sea of Trees 0x004A0=max / 0x004A1=cur), so a stable neighbour
    that bounds the value from above is a strong structural tell.
    """
    for off in (-1, 1, -2, 2, -4, 4):
        j = addr + off
        if j < 0 or j >= ring.shape[1]:
            continue
        col = ring[:, j]
        v = int(col[0])
        if v >= before and 0 < v <= 176 and np.all(col == col[0]):
            return j, v
    return None


def score_health(ring, addr, before, after):
    """HEALTH fingerprint (NOT the counter one): eighths, magnitude, max-HP neighbour.

    Kept separate from the counter scoring because the two want opposite things.
    A counter is small, moves by one and touches zero; health sits in the tens
    and steps by whole eighths. Applying the counter fingerprint to a health
    capture buries the real address -- measured 2026-08-01, `--mode any` ranked
    the confirmed 0x0234D 274th on a bee dump.
    """
    drop = before - after
    score = 0
    if drop % 8 == 0:
        score += 3
    elif drop % 4 == 0:
        score += 1
    elif drop % 2 == 0:
        # QUARTER hearts. Tim 2026-08-01: bees do a quarter-heart of damage,
        # which is a drop of 2 in eighths and matches NEITHER %8 nor %4. A
        # capture whose only hit came from a bee would therefore have scored
        # its real health address as unquantised and buried it. Scored below
        # half-hearts because 2 is a common junk delta, but never zero --
        # same rule that saved Sea of Trees from the strict eighths filter.
        # CONFIRMED on real data the same day: a bee hit steps 0x0234D by
        # exactly -2 (12 -> 10 -> 8), so quarter-hearts really are eighths.
        score += 1
    if 0 < before <= 176:
        score += 2
    if drop <= 64:
        score += 2
    nb = neighbour_max(ring, int(addr), before)
    if nb is not None:
        score += 3
    return score, nb


def candidates(dump, tolerant=False, min_dwell=5):
    """Scored down-candidates for one dump: {addr: (score,b,a,drop,i,nb)}.

    tolerant=False is the STRICT single-step detector that found IWRAM 0x00428;
    the validated path is left bit-identical on purpose.

    tolerant=True is the `drain` mode, added 2026-08-01 for a real gap. Damage
    drains over several frames and a longer window catches several hits, so the
    strict detector rejects the true address outright: a bee capture stepping
    12 -> 10 -> 8 is TWO changes, and `--mode down` therefore did not surface
    the confirmed 0x0234D at all. `--mode any` does detect it but scores it as a
    counter and ranked it 274th. Neither combination could find health in a
    multi-hit window, which the 10 s ring now makes the normal case.
    """
    meta, ring = load(dump)
    changed = np.flatnonzero(ring[0] != ring[-1])
    out = {}
    for addr in changed:
        col = ring[:, addr]
        if tolerant:
            got = stable_transition(col, min_dwell, "down")
            if got is None:
                continue
            before, after = got[0], got[1]
            i = got[2]
        else:
            got = single_step_down(col)
            if got is None:
                continue
            before, after, i = got
        score, nb = score_health(ring, addr, before, after)
        out[int(addr)] = (score, before, after, before - after, i, nb)
    return meta, changed.size, out


def intersect(dumps, tolerant=False, min_dwell=5):
    """The real confirmation bar: which address steps down in EVERY dump.

    One dump is a hypothesis -- plenty of unrelated bytes happen to tick down
    once in any given 4 s window. An address that does it across several
    independent hits, in the same area, is the health byte.
    """
    per = []
    for d in dumps:
        meta, nchanged, cands = candidates(d, tolerant, min_dwell)
        ticks = meta.get("ticks") or [0]
        print("  %-26s %5d changed, %4d candidates, ticks %d-%d" % (
            os.path.basename(os.path.normpath(d)), nchanged, len(cands),
            ticks[0], ticks[-1]))
        per.append((d, meta, cands))

    # flag windows that overlap in emulator time -- they are not independent hits
    for a, b in zip(per, per[1:]):
        ta, tb = a[1].get("ticks") or [0], b[1].get("ticks") or [0]
        if tb[0] <= ta[-1]:
            print("  NOTE: %s overlaps the previous window (%d ticks) -- may be "
                  "the same hit, weaker as independent evidence" % (
                      os.path.basename(os.path.normpath(b[0])), ta[-1] - tb[0]))

    common = set(per[0][2])
    for _, _, c in per[1:]:
        common &= set(c)
    rows = []
    for addr in common:
        entries = [c[addr] for _, _, c in per]
        total = sum(e[0] for e in entries)
        rows.append((total, addr, entries))
    rows.sort(key=lambda r: -r[0])
    return rows, per


def transitions_in_ring(ring, direction, min_dwell, counter_max):
    """{addr: (score, before, after, nchanges)} for one ring array.

    Ring-level, not path-level, so the selftest can run the EXACT code the tool
    runs against a ring it has injected an event into. An earlier draft had the
    selftest re-implement the ranking inline; the two drifted apart and the test
    reported a failure that was the test's own scoring, not the tool's.
    """
    out = {}
    for addr in prefilter(ring, min_dwell, same_ends=False):
        got = stable_transition(ring[:, addr], min_dwell, direction)
        if got is None:
            continue
        before, after, _dwell, nchanges = got
        out[int(addr)] = (score_counter(before, after, nchanges, counter_max),
                          before, after, nchanges)
    return out


def transition_candidates(dump, mode, min_dwell, counter_max):
    """transitions_in_ring for a dump on disk; also reports the pre-filter size."""
    _meta, ring = load(dump)
    npre = prefilter(ring, min_dwell, same_ends=False).size
    return npre, transitions_in_ring(
        ring, "up" if mode == "up" else "any", min_dwell, counter_max)


def round_trip_in_ring(ring, min_dwell, counter_max):
    """v1 -> v2 -> v1 inside ONE window: [(score, addr, v1, v2, detail), ...]."""
    n = ring.shape[0]
    # A real visit HOLDS. Demand the middle value survive an eighth of the window
    # (or 3x min_dwell, whichever is longer) to earn the bonus -- measured on a
    # real untouched dump, ~70 bytes complete a round trip in 4 s of ALttP, so
    # without a dwell term the ranking is a coin flip among noise.
    hold = max(min_dwell * 3, n // 8)
    scored = []
    for addr in prefilter(ring, min_dwell, same_ends=True):
        got = within_round_trip(ring[:, addr], min_dwell)
        if got is None:
            continue
        v1, v2, nchanges, dwell = got
        score = score_counter(v1, v2, nchanges, counter_max)
        if dwell >= hold:
            score += 2
        scored.append((score, dwell, int(addr), v1, v2))
    # among equals prefer the one that held longest, not the lowest address
    scored.sort(key=lambda r: (-r[0], -r[1]))
    return [(score, addr, v1, v2,
             "%d -> %d -> %d  (held %d/%d snaps)" % (v1, v2, v1, dwell, n))
            for score, dwell, addr, v1, v2 in scored]


def round_trip_across_rings(rings, min_dwell, counter_max, verbose_names=None):
    """v1 -> v2 in the first ring, back in the next: the two-mark confirmation."""
    per = []
    for k, ring in enumerate(rings):
        addrs = prefilter(ring, min_dwell, same_ends=False)
        t = {}
        for addr in addrs:
            got = stable_transition(ring[:, addr], min_dwell, "any")
            if got is not None:
                t[int(addr)] = got
        if verbose_names:
            print("  %-26s %5d pre-filtered, %4d settled transitions" % (
                verbose_names[k], addrs.size, len(t)))
        per.append(t)

    common = set(per[0])
    for t in per[1:]:
        common &= set(t)
    rows = []
    for addr in common:
        chain = [t[addr] for t in per]
        # each dump must end where the next one begins, and the whole chain must
        # live on exactly two values -- that IS alternation, so with 2+ dumps it
        # has necessarily gone out and come back at least once
        if any(chain[k][1] != chain[k + 1][0] for k in range(len(chain) - 1)):
            continue
        seen = {v for e in chain for v in (e[0], e[1])}
        if len(seen) != 2:
            continue
        v1, v2 = chain[0][0], chain[0][1]
        nchanges = max(e[3] for e in chain)
        detail = " then ".join("%d->%d" % (e[0], e[1]) for e in chain)
        rows.append((score_counter(v1, v2, nchanges, counter_max), addr, v1, v2, detail))
    rows.sort(key=lambda r: (-r[0], abs(r[3] - r[2])))
    return rows, len(common)


def round_trip_rows(dumps, min_dwell, counter_max):
    """Addresses that went v1 -> v2 and came back, in one window or across dumps.

    This is the confirmation bar for a field with no HUD readout. Returning to
    the EXACT original value is the part that is hard to fake: bytes wander all
    the time, but a byte that goes somewhere and comes precisely home, in
    windows a human anchored on two specific game events, is the field.

    Across two dumps is FAR stronger than inside one -- measured, not assumed:
    on a real dump the one-window form leaves ~70 survivors while the two-dump
    form leaves 1. Prefer two marks whenever the events are more than 4 s apart.
    """
    if len(dumps) == 1:
        _meta, ring = load(dumps[0])
        return round_trip_in_ring(ring, min_dwell, counter_max), \
            prefilter(ring, min_dwell, same_ends=True).size

    rings = [load(d)[1] for d in dumps]
    names = [os.path.basename(os.path.normpath(d)) for d in dumps]
    return round_trip_across_rings(rings, min_dwell, counter_max, names)


def known_value_scan(dumps, observed):
    """THE method that actually works. Give it what you SAW on screen.

    Shape heuristics kept failing (2026-07-31): requiring a single clean step
    rejects real health, because damage drains over several frames. Ground truth
    beats cleverness -- read the hearts off each dump's mark.png, pass them in,
    and every byte in RAM that does not follow that sequence is eliminated.

    A death is worth more than any other sample: it pins the value to a hard 0
    at a known instant, which no unrelated byte survives. If you can die safely
    while capturing, do it.
    """
    last = []
    for d in dumps:
        _meta, ring = load(d)
        last.append(ring[-1].astype(np.int32))     # value at the mark
    V = np.stack(last)
    h = np.array(observed, dtype=float)

    ok = np.ones(V.shape[1], bool)
    for i in range(len(h)):
        for j in range(len(h)):
            if i == j:
                continue
            if h[i] == 0 and h[j] > 0:
                ok &= (V[i] == 0) & (V[j] > 0)
            elif h[i] < h[j]:
                ok &= V[i] < V[j]
            elif h[i] > h[j]:
                ok &= V[i] > V[j]

    cand = np.flatnonzero(ok)
    rows = []
    nz = h > 0
    for a in cand:
        v = V[:, a].astype(float)
        k = np.median(v[nz] / h[nz])               # implied units-per-heart
        if k <= 0:
            continue
        # Error in HEARTS, not raw units: an absolute-error ranking just favours
        # whichever byte happens to hold the smallest numbers.
        err = np.abs(v[nz] - k * h[nz]).max() / k
        rows.append((err, k, int(a), V[:, a].copy()))
    rows.sort()
    return rows


def report_transitions(dumps, mode, min_dwell, counter_max, max_hits):
    """up / any modes. Several dumps -> only addresses that moved in all of them."""
    per = []
    for d in dumps:
        npre, cands = transition_candidates(d, mode, min_dwell, counter_max)
        print("  %-26s %5d pre-filtered, %4d settled %s-transitions" % (
            os.path.basename(os.path.normpath(d)), npre, len(cands), mode))
        per.append(cands)

    common = set(per[0])
    for c in per[1:]:
        common &= set(c)
    if not common:
        print("\nNO address moved the right way in all %d dumps. Either a mark "
              "missed its event, or the dumps are not the same event." % len(dumps))
        return 1

    rows = sorted(((sum(c[a][0] for c in per), a) for a in common),
                  key=lambda r: -r[0])
    print("\n%-14s %5s  %s" % ("ADDR", "total", "per-dump  before->after"))
    for total, addr in rows[:max_hits]:
        print("%-14s %5d  %s" % (
            fmt_addr(addr), total,
            "  ".join("%d->%d" % (c[addr][1], c[addr][2]) for c in per)))

    best = rows[0][1]
    print("\ntop candidate: %s  %s" % (
        fmt_addr(best), "  ".join("%d->%d" % (c[best][1], c[best][2]) for c in per)))
    if len(dumps) == 1:
        print("CONFIRM IT: this is one dump, so it is a hypothesis. For a "
              "counter the decisive follow-up is the round trip -- spend what "
              "you picked up, mark that too, and re-run with --mode roundtrip.")
    return 0


def report_round_trip(dumps, min_dwell, counter_max, max_hits):
    if len(dumps) > 1:
        print("chaining %d dumps (each must end where the next begins):" % len(dumps))
    rows, npool = round_trip_rows(dumps, min_dwell, counter_max)
    if not rows:
        print("\nNO round trip found (pool of %d). Check that the two marks "
              "really bracket the pickup and the spend, and that each value "
              "held still for >= %d snapshots -- raise --min-dwell to demand "
              "more dwell, lower it if the marks were tight against the event."
              % (npool, min_dwell))
        return 1

    print("\n%-14s %5s %9s  %s" % ("ADDR", "score", "counter?", "round trip"))
    for score, addr, v1, v2, detail in rows[:max_hits]:
        counterish = "yes" if abs(v2 - v1) == 1 and max(v1, v2) <= counter_max else "-"
        print("%-14s %5d %9s  %s" % (fmt_addr(addr), score, counterish, detail))

    print("\ntop candidate: %s  %s" % (fmt_addr(rows[0][1]), rows[0][4]))
    print("%d address(es) completed the round trip." % len(rows))
    if len(rows) > 1:
        print("More than one survived, which is normal -- save fields cluster, "
              "so neighbours of the real field often move together. Prefer the "
              "counter-like row, and check its neighbours before wiring it in.")
    return 0


def selftest(dumps, min_dwell, counter_max):
    """Prove the detectors on REAL RAM with a synthetic event injected.

    Same discipline that validated the down-mode: take a real dump, plant a
    known transition at a known address, and require the ranking to put it on
    top. A detector that cannot find an event we planted ourselves has no
    business being pointed at Tim's captures.

    The address is chosen from bytes that are CONSTANT through the real window,
    so the injection is the only thing there is to find -- and every genuinely
    moving byte in the dump stays in as realistic competition.
    """
    if not dumps:
        print("selftest needs at least one real dump to inject into")
        return 1

    meta, ring = load(dumps[0])
    n = ring.shape[0]
    if n < 4 * min_dwell:
        print("dump too short (%d snapshots) to host a round trip" % n)
        return 1

    constant = np.flatnonzero(ring.max(axis=0) == ring.min(axis=0))
    # somewhere in EWRAM's save-field neighbourhood, away from the IWRAM churn
    addr = int(constant[constant > 32 * 1024][len(constant) // 3])
    third, twothirds = n // 3, (2 * n) // 3
    fails = []

    def rank_of(rows, want):
        for k, r in enumerate(rows):
            if r[1] == want:
                return k + 1
        return None

    def check(name, rows, want, require_top=True):
        """require_top=False asserts only that the detector FOUND the injection.

        The distinction matters: tests 2 and 3 exist to prove tolerance and
        detection. Demanding rank 1 there would be asserting that a deliberately
        messy injection outranks clean real bytes, which is not a property the
        tool should have -- a clean single step really is more counter-like.

        require_top means TOP SCORE, not first row. Ties at the maximum score
        are normal and their order within the tie is arbitrary -- the real key
        find (EWRAM 0x0234F, 2026-08-01) was tied at score 9 with two other
        bytes and would have "failed" a rank==1 assertion while being correct.
        Asserting first-row was the test claiming a precision the instrument
        does not have: the no-injection control on these dumps turns up ~28
        counter-like transitions by chance, which is the same reason the
        confirmation bar is two marks rather than one.
        """
        r = rank_of(rows, want)
        if require_top:
            top = rows[0][0] if rows else None
            # index, don't unpack: roundtrip rows carry extra fields beyond
            # (score, addr) and unpacking assumes a width they do not have.
            mine = next((r[0] for r in rows if r[1] == want), None)
            ok = mine is not None and mine == top
        else:
            ok = r is not None
        print("  %-24s %-6s rank=%s of %d" % (
            name, "PASS" if ok else "FAIL", r if r else "not found", len(rows)))
        if not ok:
            fails.append(name)
        return r

    print("selftest on %s (%d snapshots), injecting at %s" % (
        os.path.basename(os.path.normpath(dumps[0])), n, fmt_addr(addr)))
    print("  %d bytes are constant through this window, %d move on their own" % (
        constant.size, ring.shape[1] - constant.size))

    # 1. step UP: a key pickup, 0 -> 1 partway through
    up = ring.copy()
    up[:third, addr], up[third:, addr] = 0, 1
    found = transitions_in_ring(up, "up", min_dwell, counter_max)
    rows = sorted(((v[0], a) for a, v in found.items()), key=lambda r: -r[0])
    check("step-up (0->1)", rows, addr)

    # 2. step UP through junk: settles over 3 snapshots. This is the case the
    #    strict single-step detector cannot see, and seeing it at all is the
    #    whole reason the tolerant detector exists.
    ramp = ring.copy()
    ramp[:third, addr] = 0
    ramp[third:third + 3, addr] = (7, 3, 2)     # mid-transition junk, as a real
    ramp[third + 3:, addr] = 1                  # field can show while settling
    strict_sees_it = np.flatnonzero(
        ramp[1:, addr] != ramp[:-1, addr]).size == 1
    found = transitions_in_ring(ramp, "up", min_dwell, counter_max)
    rows = sorted(((v[0], a) for a, v in found.items()), key=lambda r: -r[0])
    check("step-up through junk", rows, addr, require_top=False)
    if strict_sees_it:
        print("    FAIL: the junk injection has only one change, so it does not "
              "actually test tolerance")
        fails.append("junk injection is not multi-step")
    else:
        print("    (strict single-step detector rejects this column, as intended)")

    # 3. round trip INSIDE one window: through a door and back
    rt = ring.copy()
    rt[:third, addr], rt[third:twothirds, addr], rt[twothirds:, addr] = 0, 1, 0
    rows = round_trip_in_ring(rt, min_dwell, counter_max)
    r3 = check("round trip, 1 window", rows, addr, require_top=False)

    # 4. round trip ACROSS two dumps: pick the key up, then spend it
    a_ring, b_ring = ring.copy(), ring.copy()
    a_ring[:third, addr], a_ring[third:, addr] = 0, 1      # pickup dump
    b_ring[:third, addr], b_ring[third:, addr] = 1, 0      # spend dump
    rows, _ = round_trip_across_rings([a_ring, b_ring], min_dwell, counter_max)
    check("round trip, 2 dumps", rows, addr)

    # 5. negative control: the same detectors on the UNTOUCHED dump. This is the
    #    number that says how much a result is worth, and it is why the one-
    #    window round trip is reported as weak rather than trusted.
    clean = round_trip_in_ring(ring, min_dwell, counter_max)
    counterish = [r for r in clean
                  if abs(r[3] - r[2]) == 1 and max(r[2], r[3]) <= counter_max]
    print("  %-24s %-6s %d round trips with no injection at all, %d counter-like"
          % ("no-injection control", "INFO", len(clean), len(counterish)))
    if r3 and r3 > 1:
        print("    -> so rank %d for a planted one-window round trip is the "
              "instrument being weak, not broken. Two marks, not one." % r3)

    print("\nselftest: %s" % ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hearts", default="",
                    help="comma-separated hearts observed in each dump's "
                         "mark.png, in the same order as the dumps (e.g. "
                         "5,2,0,6). Switches to a known-value scan, which is "
                         "far stronger than the shape heuristics -- and a 0 "
                         "(death) in the sequence is the best anchor there is.")
    ap.add_argument("dump", nargs="+",
                    help="ram_ring dump dir(s). Several from the SAME area -> "
                         "only addresses that stepped down in all of them are kept.")
    ap.add_argument("--max-hits", type=int, default=40,
                    help="stop printing after this many candidates per region")
    ap.add_argument("--mode", default="down",
                    choices=("down", "drain", "up", "any", "roundtrip"),
                    help="down: took damage ONCE, strict (default). drain: took "
                         "damage over several frames or several hits in one "
                         "window -- same health scoring, tolerant detector; use "
                         "this for 10s captures. up: picked something up. "
                         "any: changed and settled, e.g. a room ID. roundtrip: "
                         "v1->v2->v1, either inside one dump or across two "
                         "(pick a key up in the first, spend it in the second) "
                         "-- the strongest pattern for a field with no HUD.")
    ap.add_argument("--min-dwell", type=int, default=5,
                    help="snapshots the value must hold still either side of the "
                         "change (up/any/roundtrip only). Raise it if noise "
                         "floods the list, lower it if a mark was taken tight "
                         "against the event.")
    ap.add_argument("--counter-max", type=int, default=9,
                    help="largest value still plausible for a collectible "
                         "counter; used for scoring only, never to filter")
    ap.add_argument("--selftest", action="store_true",
                    help="inject a synthetic event into the given dump's real "
                         "RAM and check each detector ranks it first")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.dump, args.min_dwell, args.counter_max)

    if args.mode == "roundtrip":
        return report_round_trip(args.dump, args.min_dwell, args.counter_max,
                                 args.max_hits)

    if args.mode in ("up", "any"):
        return report_transitions(args.dump, args.mode, args.min_dwell,
                                  args.counter_max, args.max_hits)

    if args.hearts:
        observed = [float(x) for x in args.hearts.split(",")]
        if len(observed) != len(args.dump):
            print("need one heart value per dump (%d given, %d dumps)" % (
                len(observed), len(args.dump)))
            return 1
        rows = known_value_scan(args.dump, observed)
        print("observed hearts: %s" % observed)
        print("bytes matching the full ordering%s: %d" % (
            " + zero anchor" if 0 in observed else
            " (NO death in this set -- a 0 would narrow it far more)", len(rows)))
        if not rows:
            print("nothing matched. Re-check the heart counts against each "
                  "mark.png; a mark taken mid-damage-animation reads between "
                  "two heart values.")
            return 1
        print("\n%-14s %9s %7s  %s" % ("ADDR", "per-heart", "maxerr", "value at each mark"))
        for err, k, a, v in rows[:args.max_hits]:
            print("%-14s %9.2f %7.1f  %s" % (
                fmt_addr(a), k, err, " ".join("%3d" % x for x in v)))
        print("\nTOP: %s (implied %.1f units per heart)" % (
            fmt_addr(rows[0][2]), rows[0][1]))
        return 0

    if len(args.dump) > 1:
        print("intersecting %d dumps:" % len(args.dump))
        rows, per = intersect(args.dump, args.mode == "drain", args.min_dwell)
        if not rows:
            print("\nNO address stepped down in all %d dumps. Either a mark "
                  "missed its hit, or these are not all the same area." % len(args.dump))
            return 1
        print("\n%-14s %5s  %s" % ("ADDR", "total", "per-dump  before->after"))
        for total, addr, entries in rows[:args.max_hits]:
            detail = "  ".join("%d->%d" % (e[1], e[2]) for e in entries)
            print("%-14s %5d  %s" % (fmt_addr(addr), total, detail))
        best_total, best_addr, best_entries = rows[0]
        print("\nCONFIRMED CANDIDATE: %s" % fmt_addr(best_addr))
        print("  stepped down in all %d dumps: %s" % (
            len(args.dump), ", ".join("%d->%d" % (e[1], e[2]) for e in best_entries)))
        nb = best_entries[0][5]
        if nb:
            print("  adjacent constant byte %s = %d looks like max-HP" % (
                fmt_addr(nb[0]), nb[1]))
        if len(rows) > 1:
            print("  runner-up: %s (total score %d)" % (
                fmt_addr(rows[1][1]), rows[1][0]))
        return 0

    meta, ring = load(args.dump[0])
    n = ring.shape[0]
    print("dump %s: %d snaps, every %d ticks (%.1fs), note=%r" % (
        os.path.basename(os.path.normpath(args.dump[0])), n, meta["every"],
        meta.get("secs", 0.0), meta.get("note", "")))

    # only bytes that changed at all across the window are worth examining
    changed = np.flatnonzero(ring[0] != ring[-1])
    print("bytes differing start->end: %d" % changed.size)

    # Route through candidates() rather than re-implementing the scoring here.
    # This block used to duplicate it, which is how a ranking drifts from the
    # thing it is supposed to rank -- the same mistake the selftest made when it
    # re-implemented the ranking inline (fixed 2026-07-31).
    _m, _n, cands = candidates(args.dump[0], tolerant=(args.mode == "drain"),
                               min_dwell=args.min_dwell)
    ranked = [(sc, b, a, drop, i, addr, nb)
              for addr, (sc, b, a, drop, i, nb) in cands.items()]

    # best-looking first; among equals prefer the smaller, more plausible drop
    ranked.sort(key=lambda r: (-r[0], r[3]))

    if not ranked:
        print("\nNO single-step-down candidates. Either the mark missed the hit "
              "(ring only covers %.1fs) or HP is not a plain byte here. "
              "Re-mark faster after the hit, or raise --ram-keep." % meta.get("secs", 0))
        return 1

    print("\n%-6s %-8s %5s %6s %6s %5s %8s  %s" % (
        "REGION", "ADDR", "score", "before", "after", "drop", "maxHP?", "at_snap"))
    shown = {name: 0 for name, _, _ in REGIONS}
    for score, before, after, drop, i, addr, nb in ranked:
        for name, base, size in REGIONS:
            if base <= addr < base + size:
                if shown[name] >= args.max_hits:
                    break
                shown[name] += 1
                print("%-6s 0x%05X %5d %6d %6d %5d %8s  %d/%d" % (
                    name, addr - base, score, before, after, drop,
                    ("0x%05X=%d" % (nb[0] - base, nb[1])) if nb else "-", i, n))
                break

    best = ranked[0]
    print("\ntop candidate: %s  score=%d  %d -> %d" % (
        fmt_addr(best[5]), best[0], best[1], best[2]))
    if best[6]:
        print("  adjacent constant byte %s = %d looks like max-HP" % (
            fmt_addr(best[6][0]), best[6][1]))
    print("CONFIRM IT: mark a SECOND hit and re-run -- the same address must "
          "step down again. One dump is a hypothesis, two agreeing is evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
