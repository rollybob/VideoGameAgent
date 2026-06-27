"""
Emulator I/O. Console-level, not game-level: captures the emulator window and
sends button presses. This is the one place that knows about mGBA and the GBA
key mapping.

Two deliberate fixes over the old controller:
  1. send() focuses the emulator window before issuing keys. The old code fired
     keystrokes at whatever window happened to have focus, so inputs silently went
     nowhere when focus drifted -- a prime "the loop runs but nothing happens"
     break.
  2. capture() does NOT force focus (it grabs by window bbox), so capturing never
     fights with the rest of the desktop.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw

from .contract import Action, Button

# GBA button -> mGBA default keyboard key.
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

# Window titles that are our own overlays, not the emulator.
_OVERLAY_KEYWORDS = ("vga", "agent", "ai", "control")


class EmulatorNotFound(RuntimeError):
    pass


class GbaEmulator:
    """mGBA window I/O."""

    def __init__(self, title_keyword: str = "mGBA", logger=print) -> None:
        self.title_keyword = title_keyword
        self.logger = logger
        self.window = self._find_window()
        self.logger(f"[emulator] using window: '{self.window.title}' "
                    f"{self.window.width}x{self.window.height} @ "
                    f"{self.window.left},{self.window.top}")

    def _find_window(self):
        matches = []
        for w in gw.getAllWindows():
            title = (w.title or "").lower()
            if self.title_keyword.lower() not in title:
                continue
            if any(k in title for k in _OVERLAY_KEYWORDS):
                continue
            if w.width > 100 and w.height > 100:
                matches.append(w)
        if not matches:
            raise EmulatorNotFound(
                f"No emulator window with '{self.title_keyword}' in its title. "
                f"Is mGBA running with a ROM loaded?")
        # Largest matching window is the real emulator, not a dialog.
        return max(matches, key=lambda w: w.width * w.height)

    def refresh_window(self) -> None:
        """Re-locate the window (handles it being moved/resized between runs)."""
        self.window = self._find_window()

    def focus(self) -> bool:
        """Bring the emulator to the foreground so key events land in it."""
        try:
            if self.window.isMinimized:
                self.window.restore()
            self.window.activate()
            return True
        except Exception as e:  # activate() is flaky on Windows; do not crash the loop
            self.logger(f"[emulator] focus warning: {e}")
            return False

    def is_foreground(self) -> bool:
        """True if the emulator is the active window. Used to skip frames where
        capture would grab whatever is occluding the emulator's screen rectangle
        (capture is a screen-region grab, not a true window grab)."""
        try:
            active = gw.getActiveWindow()
            return active is not None and self.title_keyword.lower() in (active.title or "").lower()
        except Exception:
            return True  # if we cannot tell, do not block capture

    def capture(self) -> np.ndarray:
        """Grab the emulator window as a BGR frame. Does not force focus."""
        left, top = self.window.left, self.window.top
        width, height = self.window.width, self.window.height
        if width <= 0 or height <= 0:
            # Window was minimized/closed; try to recover once.
            self.refresh_window()
            left, top = self.window.left, self.window.top
            width, height = self.window.width, self.window.height
            if width <= 0 or height <= 0:
                raise EmulatorNotFound("Emulator window has no drawable area.")
        shot = pyautogui.screenshot(region=(left, top, width, height))
        return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)

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
            pyautogui.keyDown(key)
            time.sleep(action.duration)
            pyautogui.keyUp(key)
            if i < action.repeats - 1:
                time.sleep(action.gap)
