"""
Emulator I/O. Console-level, not game-level: captures the emulator window and
sends button presses. This is the one place that knows about mGBA and the GBA
key mapping.

Platform split: the public GbaEmulator is unchanged, but the actual window
find / capture / input is delegated to a backend chosen by sys.platform.

  - Windows: pygetwindow for window geometry, pyautogui for capture + keys.
    This is the original code path, kept intact.
  - Linux  : xdotool for window discovery/geometry, mss for capture, and
    xdotool keydown/keyup --window <id> for input. Key events are addressed to
    the window id, so they land regardless of focus. That is why the Linux
    focus() is a deliberate no-op: it removes the old "inputs went to whatever
    window had focus" failure mode entirely, rather than papering over it.

Two design points carried over from the original:
  1. send() may focus the window before keys (Windows only; Linux does not need
     to, see above).
  2. capture() never forces focus -- it grabs by window bbox -- so capturing
     does not fight with the rest of the desktop.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time

import cv2
import numpy as np

from .contract import Action, Button

# GBA button -> logical key name. These are pyautogui key names (the Windows
# backend uses them directly). The Linux backend translates them to X11 keysyms
# via _X_KEYSYM below. They also match mGBA's *default* keyboard bindings.
GBA_KEYMAP: dict[Button, str] = {
    Button.UP: "up",
    Button.DOWN: "down",
    Button.LEFT: "left",
    Button.RIGHT: "right",
    Button.A: "x",
    Button.B: "z",
    Button.START: "enter",
    Button.SELECT: "backspace",
    Button.L: "a",
    Button.R: "s",
}

# Logical (pyautogui) key name -> X11 keysym name understood by `xdotool key`.
# Letters map to themselves; only the named keys differ.
_X_KEYSYM: dict[str, str] = {
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "enter": "Return",
    "backspace": "BackSpace",
    "x": "x",
    "z": "z",
    "a": "a",
    "s": "s",
}

# Window titles that are our own overlays, not the emulator.
_OVERLAY_KEYWORDS = ("vga", "agent", "ai", "control")


class EmulatorNotFound(RuntimeError):
    pass


class GbaEmulator:
    """mGBA window I/O. Platform-agnostic surface; backend does the real work."""

    def __init__(self, title_keyword: str = "mGBA", logger=print,
                 crop_chrome: bool = True) -> None:
        self.title_keyword = title_keyword
        self.logger = logger
        self.crop_chrome = crop_chrome  # strip host-emulator chrome from captures
        self._backend = _make_backend(logger)
        self._handle = self._backend.find(title_keyword, _OVERLAY_KEYWORDS)
        # The window we capture (largest-area match) is, under a real window manager,
        # an outer wrapper that DROPS synthetic key events; the child Qt window is what
        # actually consumes input. Resolve a separate input handle (no-op when they are
        # the same window -- e.g. the flat Xvfb+twm layout on :99). See input_handle().
        self._input_handle = self._backend.input_handle(self._handle)
        left, top, width, height = self._backend.geometry(self._handle)
        ih = f" input->{self._input_handle}" if self._input_handle != self._handle else ""
        self.logger(f"[emulator] using window: '{self._backend.title(self._handle)}' "
                    f"{width}x{height} @ {left},{top} (backend: {self._backend.name}){ih}")

    def refresh_window(self) -> None:
        """Re-locate the window (handles it being moved/resized between runs)."""
        self._handle = self._backend.find(self.title_keyword, _OVERLAY_KEYWORDS)
        self._input_handle = self._backend.input_handle(self._handle)

    def focus(self) -> bool:
        """Bring the emulator to the foreground so key events land in it.
        On Linux this is a no-op (keys are addressed to the window id)."""
        return self._backend.focus(self._handle)

    def is_foreground(self) -> bool:
        """True if the emulator is the active window. Used to skip frames where
        capture would grab whatever is occluding the emulator's screen rectangle
        (capture is a screen-region grab, not a true window grab)."""
        try:
            active = self._backend.active_title()
            return active is not None and self.title_keyword.lower() in active.lower()
        except Exception:
            return True  # if we cannot tell, do not block capture

    def capture(self) -> np.ndarray:
        """Grab the emulator window as a BGR frame. Does not force focus."""
        left, top, width, height = self._backend.geometry(self._handle)
        if width <= 0 or height <= 0:
            # Window was minimized/closed; try to recover once.
            self.refresh_window()
            left, top, width, height = self._backend.geometry(self._handle)
            if width <= 0 or height <= 0:
                raise EmulatorNotFound("Emulator window has no drawable area.")
        frame = self._backend.capture(left, top, width, height)
        if self.crop_chrome:
            frame = _crop_to_gba(frame)
        return frame

    def send(self, action: Action) -> None:
        """Execute one Action. A wait (button=None) just sleeps for its duration."""
        if action.button is None:
            time.sleep(max(action.duration, 0.0))
            return
        key = GBA_KEYMAP.get(action.button)
        if key is None:
            self.logger(f"[emulator] no key mapped for {action.button}")
            return
        self.focus()
        for i in range(max(action.repeats, 1)):
            self._backend.key_down(self._input_handle, key)
            time.sleep(action.duration)
            self._backend.key_up(self._input_handle, key)
            if i < action.repeats - 1:
                time.sleep(action.gap)


def _crop_to_gba(frame: np.ndarray) -> np.ndarray:
    """Crop a captured emulator-window frame to just the GBA framebuffer, removing
    host-emulator chrome (mGBA's menu bar). The GBA screen is a fixed 3:2 (240x160)
    aspect; mGBA renders it filling the window width and flush to the window bottom,
    with the menu bar occupying the top strip -- so the framebuffer is the bottom
    (width*2/3)-tall band. If the window is proportionally too wide (side letterbox)
    fall back to a centered 3:2 region. Deterministic; assumes the standard mGBA-qt
    layout (menu on top, no side/bottom chrome). No uniform-border trim -- it could
    not distinguish genuinely dark GAME content (e.g. the FRLG intro starfield) from
    a letterbox bar and ate the former."""
    h, w = frame.shape[:2]
    if h < 8 or w < 8:
        return frame
    gba_h = round(w * 2 / 3)
    if gba_h <= h:
        crop = frame[h - gba_h:h, 0:w]            # drop the top menu strip
    else:
        gba_w = round(h * 3 / 2)
        x0 = max(0, (w - gba_w) // 2)
        crop = frame[0:h, x0:x0 + gba_w]
    return crop


def _make_backend(logger):
    if sys.platform.startswith("win"):
        return _WindowsBackend(logger)
    return _LinuxBackend(logger)


class _WindowsBackend:
    """Original pygetwindow + pyautogui path. Handle is a pygetwindow Window."""

    name = "windows"

    def __init__(self, logger) -> None:
        self.logger = logger
        import pyautogui  # noqa: F401  (imported here so Linux never needs it)
        import pygetwindow as gw
        self._pyautogui = pyautogui
        self._gw = gw

    def find(self, keyword: str, overlay_keywords):
        matches = []
        for w in self._gw.getAllWindows():
            title = (w.title or "").lower()
            if keyword.lower() not in title:
                continue
            if any(k in title for k in overlay_keywords):
                continue
            if w.width > 100 and w.height > 100:
                matches.append(w)
        if not matches:
            raise EmulatorNotFound(
                f"No emulator window with '{keyword}' in its title. "
                f"Is mGBA running with a ROM loaded?")
        # Largest matching window is the real emulator, not a dialog.
        return max(matches, key=lambda w: w.width * w.height)

    def input_handle(self, handle):
        # Windows input is global/focus-based (pyautogui), not window-addressed, so
        # the capture window and the input target are the same object.
        return handle

    def title(self, handle) -> str:
        return handle.title or ""

    def geometry(self, handle):
        return handle.left, handle.top, handle.width, handle.height

    def focus(self, handle) -> bool:
        try:
            if handle.isMinimized:
                handle.restore()
            handle.activate()
            return True
        except Exception as e:  # activate() is flaky on Windows; do not crash the loop
            self.logger(f"[emulator] focus warning: {e}")
            return False

    def active_title(self):
        active = self._gw.getActiveWindow()
        return active.title if active is not None else None

    def capture(self, left, top, width, height) -> np.ndarray:
        shot = self._pyautogui.screenshot(region=(left, top, width, height))
        return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)

    def key_down(self, handle, key: str) -> None:
        self._pyautogui.keyDown(key)

    def key_up(self, handle, key: str) -> None:
        self._pyautogui.keyUp(key)


class _LinuxBackend:
    """xdotool (window/input) + mss (capture). Handle is an X window id (int)."""

    name = "linux"

    def __init__(self, logger) -> None:
        self.logger = logger
        import mss  # imported here so the module loads on Windows without mss
        self._sct = mss.mss()

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _run(args: list[str]) -> str:
        out = subprocess.run(args, capture_output=True, text=True, check=False)
        return out.stdout.strip()

    def _window_name(self, wid: int) -> str:
        return self._run(["xdotool", "getwindowname", str(wid)])

    def _geometry(self, wid: int):
        # Use xwininfo's ABSOLUTE coordinates. xdotool getwindowgeometry returns the
        # position relative to the (reparented) WM frame, so under a window manager
        # like twm it is off by the title-bar height -- which mis-aligned screen
        # grabs (verified: xdotool Y=239 vs true Y=212 under twm). xwininfo is
        # WM-independent.
        text = self._run(["xwininfo", "-id", str(wid)])
        vals: dict[str, int] = {}
        for key, name in (("X", "Absolute upper-left X"),
                          ("Y", "Absolute upper-left Y"),
                          ("WIDTH", "Width"), ("HEIGHT", "Height")):
            m = re.search(name + r":\s+(-?\d+)", text)
            if m:
                vals[key] = int(m.group(1))
        return (vals.get("X", 0), vals.get("Y", 0),
                vals.get("WIDTH", 0), vals.get("HEIGHT", 0))

    # -- backend interface ------------------------------------------------- #
    def find(self, keyword: str, overlay_keywords):
        # Enumerate all named, visible windows, then filter in Python -- mirrors
        # the Windows backend exactly (case-insensitive keyword, overlay skip,
        # min size, largest-area wins).
        ids_text = self._run(["xdotool", "search", "--onlyvisible", "--name", "."])
        candidates = []
        for line in ids_text.splitlines():
            line = line.strip()
            if not line.isdigit():
                continue
            wid = int(line)
            title = self._window_name(wid)
            tl = title.lower()
            if keyword.lower() not in tl:
                continue
            if any(k in tl for k in overlay_keywords):
                continue
            _, _, w, h = self._geometry(wid)
            if w > 100 and h > 100:
                candidates.append((wid, title, w * h))
        if not candidates:
            raise EmulatorNotFound(
                f"No emulator window with '{keyword}' in its title. "
                f"Is mGBA running with a ROM loaded (and DISPLAY set)?")
        wid, _, _ = max(candidates, key=lambda c: c[2])
        return wid

    def _child_windows(self, wid: int):
        # Immediate children of `wid` as (id, w, h), via xwininfo -children. Each child
        # line looks like: '0xc00006 "title": (...)  1024x807+28+98  +1585+461'.
        text = self._run(["xwininfo", "-id", str(wid), "-children"])
        kids = []
        in_children = False
        for line in text.splitlines():
            if re.search(r"\d+\s+child(ren)?:", line):
                in_children = True
                continue
            if not in_children:
                continue
            m = re.match(r"\s*(0x[0-9a-fA-F]+).*?\s(\d+)x(\d+)\+", line)
            if m:
                kids.append((int(m.group(1), 16), int(m.group(2)), int(m.group(3))))
        return kids

    def input_handle(self, handle):
        # `find()` returns the largest-area matching window. Under a real window
        # manager that is an outer wrapper whose synthetic (XSendEvent) key events are
        # dropped -- mGBA's Qt input handler lives on the child window (verified on :1:
        # keys to the wrapper do nothing; keys to its child advance the game). Target
        # the largest real child instead. When there is no such child (the flat single
        # -window layout under bare Xvfb+twm on :99, where the matched window already
        # accepts input) fall back to the handle -- so this is safe on both setups.
        kids = [k for k in self._child_windows(handle) if k[1] > 100 and k[2] > 100]
        if not kids:
            return handle
        return max(kids, key=lambda c: c[1] * c[2])[0]

    def title(self, handle) -> str:
        return self._window_name(handle)

    def geometry(self, handle):
        return self._geometry(handle)

    def focus(self, handle) -> bool:
        # No-op: xdotool key events below are addressed to the window id, so they
        # land regardless of which window is focused. Forcing focus under a bare
        # Xvfb (no window manager) would fail and buy us nothing.
        return True

    def active_title(self):
        # Best-effort; under a bare Xvfb there may be no active window. Caller
        # treats failure/None as "do not block capture".
        name = self._run(["xdotool", "getactivewindow", "getwindowname"])
        return name or None

    def capture(self, left, top, width, height) -> np.ndarray:
        # Clamp the requested rectangle to the virtual screen. Capture size is NOT
        # hard-coded -- the window geometry is read live each frame -- but mss raises
        # if asked for a region outside the framebuffer, which happens when a window
        # is larger than, or sits partly off, the display (e.g. fullscreen, or a
        # window shoved off an edge by the WM). Clamping to the live screen bounds
        # makes capture degrade gracefully (grab the visible intersection) for any
        # window size/position instead of crashing.
        screen = self._sct.monitors[0]  # union bounding box of all monitors
        sl, st = screen["left"], screen["top"]
        sr, sb = sl + screen["width"], st + screen["height"]
        l, t = max(left, sl), max(top, st)
        r, b = min(left + width, sr), min(top + height, sb)
        if r <= l or b <= t:
            raise EmulatorNotFound("Emulator window is entirely off-screen; cannot capture.")
        monitor = {"left": l, "top": t, "width": r - l, "height": b - t}
        shot = self._sct.grab(monitor)            # BGRA buffer
        frame = np.asarray(shot)                  # (h, w, 4), BGRA
        return np.ascontiguousarray(frame[:, :, :3])  # drop alpha -> BGR

    def key_down(self, handle, key: str) -> None:
        keysym = _X_KEYSYM.get(key, key)
        self._run(["xdotool", "keydown", "--window", str(handle), keysym])

    def key_up(self, handle, key: str) -> None:
        keysym = _X_KEYSYM.get(key, key)
        self._run(["xdotool", "keyup", "--window", str(handle), keysym])
