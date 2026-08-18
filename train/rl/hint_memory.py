"""hint_memory.py -- retain on-screen messages past the box closing (minimal hint persistence).

The pixel reader gives the agent what is on screen NOW; when a dialogue box closes, dialog_text
goes empty and any hint it contained ("...a prisoner in the castle dungeon...") is gone from the
VLM's context. This retains recent WHOLE messages so the agent can keep pursuing a destination it
was told about while walking. Deliberately minimal (no LLM distiller yet): assemble messages,
keep the last K distinct, surface them as soft background -- and A/B whether that helps before
adding classification. See [[project-vga-pixel-reader]] and the Task-07 caveat
([[project-vga-task07-refuted]]: durable task-state didn't move R3 when nav/perception was the
limit -- now we have a working reader AND walker, so it may finally have a substrate to act on).

MECHANISM
- observe(pix, step): fed the per-decision pixel read. While text is on screen, accumulate
  DISTINCT pages (a new page = low overlap with the last, so a static box isn't appended 30x, but
  a page-turn is). When the read goes empty for `close_after` decisions, the box has closed ->
  finalize the accumulated pages into one message.
- active(step): the last K finalized messages still within the recency window, newest first --
  what to inject into /act. Empty when disabled (the A/B OFF arm) so nothing is surfaced.

No emulator imports -- pure bookkeeping, unit-testable standalone.
"""
import difflib


def _ratio(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


class HintMemory:
    def __init__(self, enabled=True, k=3, decay=2400, close_after=2,
                 new_page_ratio=0.6, dedup_ratio=0.85, min_len=8):
        # k: max messages surfaced. decay: decisions a message stays active (~15/s, so 2400 ~160s
        # -- generous; the cap + supersession are the real control). close_after: consecutive empty
        # reads that mark a box closed. new_page_ratio: below this similarity a read is a NEW page.
        # dedup_ratio: above this an incoming message duplicates a stored one. min_len: drop
        # too-short finalized messages (stray fragments, not a hint).
        self.enabled = enabled
        self.k = k
        self.decay = decay
        self.close_after = close_after
        self.new_page_ratio = new_page_ratio
        self.dedup_ratio = dedup_ratio
        self.min_len = min_len
        self._pages = []          # pages of the message currently on screen
        self._empty = 0           # consecutive empty reads
        self._msgs = []           # [(message, step)] finalized, most-recent last
        self._events = []         # finalized messages (for the sidecar / inspection)

    def observe(self, pix, step):
        """Feed one decision's pixel read (the on-screen message text, or "")."""
        if not self.enabled:
            return
        pix = (pix or "").strip()
        if pix:
            self._empty = 0
            if not self._pages or _ratio(pix, self._pages[-1]) < self.new_page_ratio:
                self._pages.append(pix)
        else:
            self._empty += 1
            if self._empty == self.close_after and self._pages:
                self._finalize(step)

    def _finalize(self, step):
        message = " ".join(" ".join(self._pages).split()).strip()
        self._pages = []
        if len(message) < self.min_len:
            return
        for i, (m, _) in enumerate(self._msgs):
            if _ratio(message, m) > self.dedup_ratio:    # re-seen: refresh its recency, keep order
                self._msgs[i] = (m, step)
                return
        self._msgs.append((message, step))
        self._msgs = self._msgs[-self.k:]
        self._events.append((step, message))

    def active(self, step):
        """The messages to surface right now (newest first), or [] when disabled/empty."""
        if not self.enabled:
            return []
        live = [(m, s) for (m, s) in self._msgs if step - s <= self.decay]
        return [m for (m, s) in live[-self.k:]][::-1]

    def events(self):
        """All finalized messages [(step, message)] -- for the run sidecar."""
        return list(self._events)


def _selftest():
    m = HintMemory(k=2, decay=1000, close_after=2, min_len=8)
    # a box types/holds (same text repeated) then closes -> ONE message, appended once
    for s in range(5):
        m.observe("Link, I'm going out for a bit.", s)
    assert m._pages and len(m._pages) == 1, m._pages
    m.observe("", 5); m.observe("", 6)                       # 2 empty -> box closed
    assert m.active(6) == ["Link, I'm going out for a bit."], m.active(6)
    # a multi-page message: page turn is a NEW page -> assembled together
    m.observe("I am a prisoner in the", 20)
    m.observe("castle dungeon. Help me...", 21)              # distinct -> appended
    m.observe("", 22); m.observe("", 23)
    act = m.active(23)
    assert act[0] == "I am a prisoner in the castle dungeon. Help me...", act
    assert len(act) == 2 and "going out" in act[1], act      # cap k=2, newest first
    # recency decay drops the old one
    assert m.active(20 + 1000 + 1) == ["I am a prisoner in the castle dungeon. Help me..."] \
        or m.active(20 + 1000 + 1) == [], m.active(20 + 1000 + 1)
    # disabled -> inert
    d = HintMemory(enabled=False)
    for s in range(5):
        d.observe("some message here", s)
    d.observe("", 5); d.observe("", 6)
    assert d.active(6) == [] and d.events() == []
    print("hint_memory selftest OK")


if __name__ == "__main__":
    _selftest()
