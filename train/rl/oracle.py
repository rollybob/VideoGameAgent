"""Task 09 A0: RAM reward oracles for GBA games.

Each class reads reward-relevant state from a core's RAM each tick and emits
events as (tick, channel, delta, value) tuples.

-- Four Swords (4P link sessions, RewardOracle + FsOracle) --
In lockstep multiplayer every core simulates the whole world, so reading
core 0 suffices for all signals (per-player and team-level).

CHANNELS (validated 2026-07-08/28, see sessions/):
  rupees   IWRAM 0x6C88 (u16 le, team-shared). Proven write-probe (2026-07-08).
  health   IWRAM 0x03225 = player-0 current HP in eighths (8=1 heart, 4=half).
           Stride 0x80 to other players: P1=0x032A5, P2=0x03325, P3=0x033A5.
           Max HP at IWRAM 0x03224 (= 80 in p*_coop.state).
           Proven 2026-07-28: write-scan hit hud=True; writing 24->3 hearts,
           writing 8->1 heart confirmed eighths encoding via pixel diff.

Usage:
    oracle = RewardOracle()
    events = oracle.step(tick, core)   # after sess.tick(); [] most ticks

-- Solo ALttP (AlttpOracle) --
CHANNELS (validated 2026-07-09, see sessions/SESSION_2026-07-09_solo_mining.md):
  damage   EWRAM 0x00C93 negative delta (HP in eighths: 8=heart, 4=half).
           Full = 28 in alttp_ingame.state. Proven: 28->20 instant on a survived
           hit (take 012532); death-drain ticks HP down frame-by-frame to 0.
  death    EWRAM 0x00C93 transitions to 0. 4/6 combat tapes contain deaths.
           Respawn = instant jump from 0 to 28 (NOT a damage event).

Usage:
    oracle = AlttpOracle()
    events = oracle.step(tick, core)   # after core.run_frame(); [] most ticks
"""

import os

from mgba._pylib import ffi as _ffi


class RewardOracle:
    RUPEE_ADDR = 0x6C88

    def __init__(self):
        self.prev_rupees = None

    def read_rupees(self, core):
        iw = core.memory.iwram.u8
        return iw[self.RUPEE_ADDR] | (iw[self.RUPEE_ADDR + 1] << 8)

    def step(self, tick, core):
        """Returns a list of (tick, channel, delta, value) event tuples."""
        events = []
        rupees = self.read_rupees(core)
        if self.prev_rupees is not None and rupees != self.prev_rupees:
            events.append((tick, "rupees", rupees - self.prev_rupees, rupees))
        self.prev_rupees = rupees
        return events


class FsOracle:
    """RAM reward oracle for Four Swords (4P link session, read core 0).

    Tracks per-player health and team rupees.  All 4 players' data lives in
    core 0's IWRAM because lockstep makes every core identical.

    Health encoding: eighths (same as ALttP) -- 8 units = 1 full heart.
    Damage  = health decreases (negative delta).
    Death   = health transitions to 0.
    Respawn = health jumps up from 0 (NOT emitted -- reset, not reward).

    HEALTH IS GLOBAL AFTER ALL -- 0x00428 (corrected 2026-07-31).
    The earlier "addresses are per-checkpoint" conclusion (2026-07-30) was an
    artifact of two separate WRONG addresses, not a real property of the game:
      - 0x03225 (coop write-scan) reads a constant 0 in every other checkpoint.
      - 0x004A1 (the Sea of Trees "confirmed" address, and what fs-ppo-a1
        actually trained against) is NOT health at all. It sits constant at 31
        through a real player death captured 2026-07-31 -- it never moves.
    0x00428 came from human hit-marker dumps (link/ram_ring.py) anchored on a
    real death, and was verified to read a sane full 40/40 in ALL FOUR banks
    (coop, seaoftrees, taluscave, deathmountain) and to track damage / death /
    revive exactly across 7 independent marked hits in two different stages.
    0x01095 and 0x06CE0 mirror it byte-for-byte; 0x06CE1 is the constant max HP.

    N_PLAYERS defaults to 1: only player 0 is confirmed at this address. The old
    0x80 stride belonged to 0x03225 and does NOT carry over -- do not assume it.
    """

    HEALTH_BASE   = 0x00428  # IWRAM: player-0 current HP (eighths), all stages
    HEALTH_MAX    = 0x06CE1  # IWRAM: max HP, constant 40 (5 hearts) in all banks
    HEALTH_STRIDE = 0x80     # UNVERIFIED for 0x00428 -- only used if N_PLAYERS>1
    N_PLAYERS     = 1
    RUPEE_ADDR    = 0x6C88   # IWRAM: team rupees (u16 le). Re-confirmed
                             # 2026-07-31: read 375 matching the on-screen HUD,
                             # and dropped exactly 50 at a rupee-paid revive.

    def __init__(self, health_base=None, health_stride=None, n_players=None, rupee_addr=None):
        self.HEALTH_BASE = self.HEALTH_BASE if health_base is None else health_base
        self.HEALTH_STRIDE = self.HEALTH_STRIDE if health_stride is None else health_stride
        self.N_PLAYERS = self.N_PLAYERS if n_players is None else n_players
        self.RUPEE_ADDR = self.RUPEE_ADDR if rupee_addr is None else rupee_addr
        self.prev_health  = [None] * self.N_PLAYERS
        self.prev_rupees  = None

    def _iw(self, core):
        return _ffi.cast("uint8_t *", core._native.memory.iwram)

    def read_health(self, core, player=0):
        addr = self.HEALTH_BASE + player * self.HEALTH_STRIDE
        return int(self._iw(core)[addr])

    def read_rupees(self, core):
        iw = self._iw(core)
        return int(iw[self.RUPEE_ADDR]) | (int(iw[self.RUPEE_ADDR + 1]) << 8)

    def step(self, tick, core):
        """Returns list of (tick, channel, delta, value) tuples."""
        events = []
        for p in range(self.N_PLAYERS):
            h = self.read_health(core, p)
            prev = self.prev_health[p]
            if prev is not None:
                delta = h - prev
                if delta < 0:
                    events.append((tick, "health_p%d" % p, delta, h))
                if h == 0 and prev > 0:
                    events.append((tick, "death_p%d" % p, 0, 0))
            self.prev_health[p] = h

        rupees = self.read_rupees(core)
        if self.prev_rupees is not None and rupees != self.prev_rupees:
            events.append((tick, "rupees", rupees - self.prev_rupees, rupees))
        self.prev_rupees = rupees
        return events


class AlttpOracle:
    """RAM reward oracle for solo A Link to the Past (GBA, Four Swords cart).

    HEALTH ADDRESS CORRECTED 2026-08-01 to EWRAM 0x0234D.

    RETRACTION: this docstring previously claimed 0x00C93 was "re-confirmed
    2026-07-31 ... stepped 28 -> 20 -> 12 -> 4 -> 0" against the human capture.
    That sequence is not in the dumps and the claim was false. Pixel ground
    truth from the four mark.png HUD heart rows (1.5 / 1 / 0.5 / 0 hearts, max
    3) reads 0x0234D = 12 / 8 / 4 / 0 with 0x0234C = 24 = max HP, an exact
    match; 0x00C93 sat flat at 28 through three real hits, and 28 eighths is
    3.5 hearts, above the 3-heart maximum on screen.

    0x00C93 is not dead, it is a LAGGING LOW-RESOLUTION MIRROR: it tracks health
    roughly but collapses several hits into one step (20 -> 8) and holds stale
    out-of-range values. Both bytes reach 0, so death detection was fine, but
    damage COUNTS off the mirror undercount and the penalty lands on whatever
    action was taken when the mirror synced -- mistimed credit assignment, which
    is the part that actually hurts an RL agent.

    RUPEES ADDED 2026-07-31 (0x02340). Until now this oracle had ONLY damage and
    death -- a pure punishment signal with nothing to gain, which is the leading
    suspect for why alttp-ppo-a1 went flat: a policy that stands still in a
    corner is already near-optimal when the only feedback is "don't get hit".
    """

    # EWRAM offset; live HP in eighths (0x0234C alongside it is a constant max).
    # Env-overridable ONLY so the correction can be A/B'd against the old mirror
    # on identical seeds -- a2/a3 were both measured on 0x00C93, so comparability
    # has to be checkable rather than assumed. Default is the correct address.
    # `or` not a get() default: a set-but-empty var would otherwise reach
    # int("", 0) and crash the run rather than falling back.
    HEALTH_ADDR = int(os.environ.get("VGA_ALTTP_HEALTH_ADDR") or "0x0234D", 0)
    # EWRAM, u16 LE. Found by write-testing every byte holding the observed
    # on-screen value (find_rupee_addr.py): poking it moves the HUD digits and
    # the counter reads back exactly (231/300/999 all verified).
    # NOTE 0x02342 is the DISPLAYED counter, which animates toward 0x02340 at
    # ~1/frame. Rewarding off the display would smear one pickup across dozens
    # of ticks of +1, so always read the true value here instead.
    RUPEE_ADDR = 0x02340
    # ALttP rupees are COLLECTION-only (Tim 2026-08-02): reward the rise, never a
    # fall. Spending is a valid choice, and the 50-rupee death-revive must not
    # stack onto the death penalty. (Four Swords keeps the SIGNED spend penalty in
    # FsOracle -- rupees are contested/losable there.) Capped at 999; a single
    # legit pickup is at most a 300-rupee, so a larger delta is a garbage counter
    # read and is discarded -- the key-255-sentinel lesson.
    RUPEE_MAX = 999
    RUPEE_SANE_DELTA = 300
    # Max HP in eighths. NOT a constant 24 -- Tim flagged 2026-08-01 that heart
    # containers raise the maximum, so anything hard-wired to 3 hearts goes
    # stale the moment one is collected. Read it, do not assume it. A RISE here
    # is a heart-container pickup, which is a major collectible and a far
    # stronger positive signal than a recovery heart.
    HEALTH_MAX_ADDR = 0x0234C
    # Small-key count. Found 2026-08-01 from Tim's human round trip: 0 -> 1 on
    # the pickup dump, 1 -> 0 on the spend dump 4 s later, and flat across all
    # six surrounding dumps. Sits in the save-field cluster with rupees/HP.
    # NOTE none of the 5 candidates the automated write-test hunt produced on
    # 07-31 was this address -- the write-test cannot confirm a field with no
    # animated HUD, exactly as predicted.
    KEY_ADDR = 0x0234F
    # 0x0234F reads 255 in one capture (a non-dungeon area), so it is a sentinel
    # and not a count everywhere. Untreated, entering that area would look like
    # a +255 key delta and dwarf every other reward in the run. Any transition
    # touching a value above this bound is discarded rather than paid.
    KEY_SANE_MAX = 8
    # Magic meter, u8 in EWRAM. Found 2026-08-02 from Tim's two-jars-then-drain
    # capture, and it sits EXACTLY where the SNES SRAM structure predicts: this
    # port mirrors the SNES save block into EWRAM at (SNES - 0x7ECD20), which
    # already places rupees/maxHP/HP/keys, and SNES magic is 0x7EF36E -> 0x0234E,
    # one byte after HP. Confirmed against the real captures three ways: it
    # drains to EXACTLY 0 (16->12->8->4->0, the hard-zero anchor), it refills in
    # two clean +16 blocks (0->16 then 16->32, the two jars), and it is the ONLY
    # byte in RAM that both drains-to-0 and bulges-out-of-0 across the two
    # human-anchored windows (find_magic2.py). Full meter is 0x80=128 on SNES
    # ALttP (INFERRED, not observed at full here -- max seen was 112). Fills and
    # drains IN PLACE over several frames (no separate instant/displayed split
    # like rupees had), so a delta-based reward off this byte is correct.
    # There is NO separate "magic jar count": jars are meter refills, which show
    # up as +16 rises in this same byte -- the magic analogue of the heal channel.
    MAGIC_ADDR = 0x0234E
    # Bound like KEY_SANE_MAX: no 255-style sentinel was seen for magic in the
    # captures, but health/keys both hold garbage in non-normal game states, so
    # discard any transition touching a value above the real ceiling rather than
    # pay a fabricated +100 the first time the game tears down normal play.
    MAGIC_SANE_MAX = 128
    # Link's world coordinates, u16 LE in IWRAM. Found 2026-08-01 from two
    # SEPARATE human captures (walk left/right, then walk up/down), which is what
    # made the test decisive: X sweeps 112 units in the left/right capture and is
    # EXACTLY CONSTANT in the up/down one, and Y does the reverse. Not "small on
    # the other axis" -- zero. Mirrors exist at 0x0391C/0x03920 and 0x03838.
    POS_Y_ADDR = 0x038F0
    POS_X_ADDR = 0x038F4
    # Rooms sit on a 512-unit world grid: every captured room transition shows a
    # position discontinuity of exactly 512, while walking never jumps more than
    # 4. So the room index is derivable from position and needs no separate room
    # register -- three attempts to find one failed, and this is strictly better
    # because the same two addresses also give a DENSE per-step signal.
    ROOM_SHIFT = 9          # 1 << 9 == 512
    POS_JUMP_ROOM = 256     # a move larger than this is a transition, not walking
    # Dense exploration breadcrumb (2026-08-03). new_room pays a room CHANGE
    # (room = pos >> 9), but a cold policy never REACHES a boundary: it is ~85
    # directed steps away with zero intermediate signal, so exploration never
    # starts (new_room only ever fired in -03, where a boundary is a ledge
    # underfoot). This finer grid pays the first visit to each (pos >> EXPLORE_SHIFT)
    # cell per episode -- a continuous outward gradient that leads Link to the
    # boundaries the room bonus then pays off. Farming-safe (first-visit-per-
    # episode; cells_seen resets each episode). TUNED offline 2026-08-03 via
    # measure_dr coverage: shift 6 (64-unit) fired only 2-3/ep, shift 5 (32-unit)
    # gives a denser ~4-6/ep gradient without rewarding in-place jitter.
    EXPLORE_SHIFT = int(os.environ.get("VGA_EXPLORE_SHIFT") or "5", 0)

    # Enemy health, IWRAM. Found 2026-08-02 by Tim's kill-diff: a runs/BEST policy
    # kills the -04 enemy and at the death instant a cluster of bytes zeroes -- all
    # at slot 3 of the ALttP parallel 16-slot sprite arrays. 0x03253 is the health
    # field for that slot: it drains 6 -> 4 -> 2 -> 0 (three sword hits of -2) and
    # was byte-identical across two independent killer policies. This is the DENSE
    # ENGAGEMENT signal the reliability diagnosis called for: it pays Link for
    # DAMAGING the enemy (the one thing the avoidance attractor refuses to do),
    # fires 3x per kill and EARLIER than the key, and cannot be farmed because
    # health only ever drains. SLOT-SPECIFIC: 0x03253 is the enemy's slot only
    # because alttp_human-04's savestate fixes it there; other states assign other
    # slots, so this address is correct for -04 training and must be re-found
    # elsewhere. Env-overridable for exactly that reason.
    ENEMY_HP_ADDR = int(os.environ.get("VGA_ALTTP_ENEMY_HP_ADDR") or "0x03253", 0)  # old slot-3, kept for reference
    # The -04 enemy has 6 HP, -01's knights 4. Bounded finitely so a garbage/large
    # slot value cannot fabricate a huge "hit" (same pattern as KEY_SANE_MAX).
    ENEMY_HP_SANE_MAX = 32
    # GENERALIZED 2026-08-03 (multi-enemy capture, find_enemy_slots.py). The sprite
    # engine keeps a parallel 16-slot HEALTH array at base 0x03250 (slot i =
    # 0x03250+i). ANY slot whose health DRAINS is an enemy taking damage; static
    # (object) and empty (0) slots never fire under the decrease-only + prev>0
    # guards. Tim cleared -01's 3 knights one-by-one -> they occupied slots 2,3,4,
    # and summing decreases over ALL 16 slots counted exactly 12 dmg (3x4hp) with
    # ZERO false-fires from the static slots; broad-validated on the 18-min castle
    # tape (277 dmg / 122 events, no single drop > a heart). So enemy_dmg now sums
    # ALL slots and scores in EVERY room -- no -04 slot-3 restriction, no per-state
    # gate. Env-overridable base/count in case another game/version differs.
    ENEMY_HP_BASE = int(os.environ.get("VGA_ALTTP_ENEMY_HP_BASE") or "0x03250", 0)
    ENEMY_N_SLOTS = int(os.environ.get("VGA_ALTTP_ENEMY_SLOTS") or "16", 0)
    # NO type gate. 0x0316X was tried as an enemy-TYPE gate but is UNRELIABLE: the
    # multi-enemy capture showed it CYCLES 0x00/0x40/0x80/0xC0 (top 2 bits =
    # animation/direction), NOT a stable type -- exactly why the old ==0xC0 gate
    # suppressed real -04 hits. Health-decrease alone is the clean signal; do NOT
    # re-introduce an id gate on 0x0316X. Constants kept (unused) for old scripts.
    ENEMY_ID_ADDR = int(os.environ.get("VGA_ALTTP_ENEMY_ID_ADDR") or "0x03163", 0)
    ENEMY_ID_EXPECT = int(os.environ.get("VGA_ALTTP_ENEMY_ID_EXPECT") or "-1", 0)

    def read_pos(self, core):
        iw = _ffi.cast("uint8_t *", core._native.memory.iwram)
        x = int(iw[self.POS_X_ADDR]) | (int(iw[self.POS_X_ADDR + 1]) << 8)
        y = int(iw[self.POS_Y_ADDR]) | (int(iw[self.POS_Y_ADDR + 1]) << 8)
        return x, y

    def room_of(self, x, y):
        return (x >> self.ROOM_SHIFT, y >> self.ROOM_SHIFT)

    def __init__(self):
        self.prev_health = None
        self.prev_rupees = None
        self.prev_keys = None
        self.prev_health_max = None
        self.prev_magic = None
        self.prev_enemy_hp = None
        self.prev_enemy_id = None
        self.prev_room = None
        self.rooms_seen = set()
        self.cells_seen = set()

    def _ew(self, core):
        return _ffi.cast("uint8_t *", core._native.memory.wram)

    def read_health(self, core):
        return int(self._ew(core)[self.HEALTH_ADDR])

    def read_health_max(self, core):
        return int(self._ew(core)[self.HEALTH_MAX_ADDR])

    def read_keys(self, core):
        return int(self._ew(core)[self.KEY_ADDR])

    def read_magic(self, core):
        return int(self._ew(core)[self.MAGIC_ADDR])

    def read_enemy_hp(self, core):
        # IWRAM, not EWRAM -- the sprite table lives in IWRAM like position does.
        iw = _ffi.cast("uint8_t *", core._native.memory.iwram)
        return int(iw[self.ENEMY_HP_ADDR])

    def read_enemy_id(self, core):
        iw = _ffi.cast("uint8_t *", core._native.memory.iwram)
        return int(iw[self.ENEMY_ID_ADDR])

    def read_enemy_hp_slots(self, core):
        iw = _ffi.cast("uint8_t *", core._native.memory.iwram)
        return [int(iw[self.ENEMY_HP_BASE + s]) for s in range(self.ENEMY_N_SLOTS)]

    def read_rupees(self, core):
        ew = self._ew(core)
        return int(ew[self.RUPEE_ADDR]) | (int(ew[self.RUPEE_ADDR + 1]) << 8)

    def step(self, tick, core):
        """Returns list of (tick, channel, delta, value) event tuples.

        Emits 'damage' on any HP decrease (delta is negative). Deltas are in
        eighths, so a quarter-heart hit (bees) is -2 and needs no special case.
        Emits 'death' when HP first reaches 0 (often same tick as final damage).
        Emits 'heal' on HP rising while alive; 'heart_container' on MAX HP
        rising; 'key' on the key count rising and 'key_used' on it falling.
        Emits 'rupees' on any change (delta signed).
        Emits 'magic' on the magic meter RISING (a jar pickup) -- positive only,
        like 'heal'; casting (a fall) is using a tool, not a loss, so it is not
        penalised. A jar fills over several frames, so one pickup emits a run of
        small +deltas summing to +16, exactly as HP damage emits a run summing
        to the hit -- the delta-based reward totals correctly either way.
        Emits 'enemy_dmg' (delta POSITIVE = health removed) when the tracked
        enemy's health falls -- the dense engagement reward.
        Respawn (0->full) emits nothing -- that is a reset, not a reward signal.
        """
        events = []
        h = self.read_health(core)
        if self.prev_health is not None:
            delta = h - self.prev_health
            if delta < 0:
                events.append((tick, "damage", delta, h))
            elif delta > 0 and self.prev_health > 0:
                # Health going UP with the player alive = a heart pickup. Free
                # positive channel, no new address needed. The prev>0 guard is
                # what keeps a respawn (0 -> full) out of it: that is a reset,
                # not a reward, and paying for it would make dying profitable.
                events.append((tick, "heal", delta, h))
            if h == 0 and self.prev_health > 0:
                events.append((tick, "death", 0, 0))
        self.prev_health = h

        # Heart container: max HP rising. Only ever goes up in normal play, so
        # no direction guard is needed beyond ignoring the first observation.
        hm = self.read_health_max(core)
        if self.prev_health_max is not None and hm > self.prev_health_max:
            events.append((tick, "heart_container", hm - self.prev_health_max, hm))
        self.prev_health_max = hm

        # Keys. Split into two channels deliberately: 'key' is a collectible
        # (and a proxy for the kill that dropped it, since ALttP exposes no kill
        # counter), while 'key_used' is PROOF A DOOR WAS OPENED -- progress
        # through the dungeon, which nothing else in this oracle can observe.
        k = self.read_keys(core)
        if (self.prev_keys is not None and k != self.prev_keys
                and k <= self.KEY_SANE_MAX and self.prev_keys <= self.KEY_SANE_MAX):
            if k > self.prev_keys:
                events.append((tick, "key", k - self.prev_keys, k))
            else:
                events.append((tick, "key_used", k - self.prev_keys, k))
        self.prev_keys = k

        # Magic meter. Reward the RISE (a jar refill); casting drains it and is
        # NOT penalised -- spending a tool is not a loss. Same sane-bound guard
        # as keys so a tear-down garbage value cannot pay a fabricated pickup.
        # No respawn guard is needed here the way 'heal' needs one: the ALttP
        # env sets death_terminates=True, so the episode ends before any
        # death->continue magic refill can be misread as a pickup.
        m = self.read_magic(core)
        if (self.prev_magic is not None and m > self.prev_magic
                and m <= self.MAGIC_SANE_MAX
                and self.prev_magic <= self.MAGIC_SANE_MAX):
            events.append((tick, "magic", m - self.prev_magic, m))
        self.prev_magic = m

        # Enemy damage across ALL 16 sprite slots. Reward the FALL in any slot's
        # health (Link landed a hit on whatever enemy occupies it). Per-slot guards:
        # prev>0 so a fresh spawn (0->N rise) or an already-dead slot never pays;
        # prev/new within the sane bound so a garbage slot cannot fabricate a hit;
        # new<prev so only DAMAGE pays. Static (object) and empty (0) slots never
        # fire. Summing all slots scores combat in EVERY room (see ENEMY_HP_BASE);
        # prev_enemy_hp is now a per-slot LIST. The old slot-3-only + id-gate is gone.
        hp = self.read_enemy_hp_slots(core)
        if self.prev_enemy_hp is not None:
            for s in range(self.ENEMY_N_SLOTS):
                ph, h = self.prev_enemy_hp[s], hp[s]
                if h < ph and 0 < ph <= self.ENEMY_HP_SANE_MAX and h <= self.ENEMY_HP_SANE_MAX:
                    events.append((tick, "enemy_dmg", ph - h, h))
        self.prev_enemy_hp = hp

        # ALttP rupees: pay only the COLLECTION (a rise). 0<delta rejects
        # spending and no-change; delta<=SANE rejects a garbage counter read;
        # r<=MAX means a full meter never pays -- and if prev is already MAX the
        # delta is <=0 so it is excluded anyway (the "don't fire when maxed" rule).
        r = self.read_rupees(core)
        if (self.prev_rupees is not None and r > self.prev_rupees
                and r - self.prev_rupees <= self.RUPEE_SANE_DELTA
                and r <= self.RUPEE_MAX):
            events.append((tick, "rupees", r - self.prev_rupees, r))
        self.prev_rupees = r

        # Exploration. Paid on the FIRST visit to each room per episode, so each
        # room pays once and walking back and forth through a door cannot be
        # farmed. rooms_seen is reset with the oracle, i.e. once per episode.
        x, y = self.read_pos(core)
        # Dense breadcrumb: first visit to each fine cell this episode (see
        # EXPLORE_SHIFT). Emitted regardless of a room change -- this is the
        # continuous outward signal that leads Link toward boundaries; new_room
        # below is the coarse per-room payoff on top. cells_seen resets per
        # episode, so re-entering a cell never re-pays (farming-safe).
        cell = (x >> self.EXPLORE_SHIFT, y >> self.EXPLORE_SHIFT)
        if cell not in self.cells_seen:
            events.append((tick, "explore", 1, cell[0] * 100000 + cell[1]))
        self.cells_seen.add(cell)
        room = self.room_of(x, y)
        if self.prev_room is not None and room != self.prev_room:
            if room not in self.rooms_seen:
                events.append((tick, "new_room", 1, room[0] * 1000 + room[1]))
        self.rooms_seen.add(room)
        self.prev_room = room
        return events
