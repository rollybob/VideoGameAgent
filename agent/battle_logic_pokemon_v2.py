
import time
import cv2
import numpy as np
from typing import Deque, Set
from collections import deque

from screen_reader import capture_window
from controllers.gba_controller import press, mash

def frame_similarity(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ah, aw = a.shape[:2]
    scale = 160.0 / min(ah, aw)
    na = cv2.resize(a, (int(aw*scale), int(ah*scale)), interpolation=cv2.INTER_NEAREST)
    nb = cv2.resize(b, (na.shape[1], na.shape[0]), interpolation=cv2.INTER_NEAREST)
    ga = cv2.cvtColor(na, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(nb, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(ga, gb).astype(np.float32) / 255.0
    mse = float(np.mean(diff**2))
    return 1.0 - mse

class PokemonBattleManagerV2:
    """
    FR/LG-compatible battle loop that:
      - cycles moves 1..4 and remembers PP-empty moves (no-OCR)
      - switches to another party member when all moves are empty
      - falls back to Struggle, then Run (once) if needed
      - resets memory at the start of each new battle
    """
    def __init__(self, window, logger=print):
        self.window = window
        self.logger = logger

        # --- timings/thresholds ---
        self.menu_settle = 0.12
        self.pace = 0.10
        self.mash_delay = 0.12
        self.fast_return_window = 1.5  # Give more time for battle animations
        self.fast_return_thresh = 0.98  # More forgiving - allows for battle animations
        self.switch_settle = 2.0      # time to wait for switch animation/menu transitions
        self.action_advance_taps = 12

        # --- battle memory ---
        self.turn_counter = 0
        self.order: Deque[int] = deque([1,2,3,4])
        self.pp_empty: Set[int] = set()
        self.move_fail_count: dict = {}  # Track failures per move before marking as empty
        self.empty_turns = 0
        self.party_tried: Set[int] = set()  # 1..6
        self.run_attempted = False

    # ---------- menu helpers ----------
    def ensure_main_menu(self):
        press("B", duration=self.pace); time.sleep(self.pace)
        press("B", duration=self.pace); time.sleep(self.pace)

    def reset_main_cursor_top_left(self):
        # force highlight to FIGHT (top-left)
        press("up", duration=0.05); time.sleep(0.05)
        press("left", duration=0.05); time.sleep(0.05)
        press("up", duration=0.05); time.sleep(0.05)
        press("left", duration=0.05); time.sleep(0.05)

    def open_moves_menu(self):
        self.reset_main_cursor_top_left()
        press("A", duration=self.pace)  # select FIGHT
        time.sleep(self.menu_settle)

    def open_party_menu(self):
        self.reset_main_cursor_top_left()
        press("down", duration=0.05)  # to POKéMON (bottom-left)
        time.sleep(0.05)
        press("A", duration=self.pace)
        time.sleep(self.menu_settle)

    def reset_moves_cursor_top_left(self):
        press("up", duration=0.05); time.sleep(0.05)
        press("up", duration=0.05); time.sleep(0.05)
        press("left", duration=0.05); time.sleep(0.05)
        press("left", duration=0.05); time.sleep(0.05)

    def cursor_to_move(self, idx: int):
        if idx == 1: return
        elif idx == 2: press("right", duration=0.05)
        elif idx == 3: press("down", duration=0.05)
        elif idx == 4:
            press("down", duration=0.05); time.sleep(0.05); press("right", duration=0.05)
        time.sleep(self.pace)

    def confirm(self):
        press("A", duration=self.pace)

    def advance_dialogue(self, taps=3):
        for _ in range(taps):
            press("A", duration=self.pace)
            time.sleep(self.mash_delay)

    # ---------- outcome tests ----------
    def action_failed_quickly(self, before, wait_s):
        time.sleep(wait_s)
        after = capture_window(self.window)
        sim = frame_similarity(before, after)
        self.logger(f"[Battle] Post-action similarity={sim:.4f} (thresh={self.fast_return_thresh})")
        
        # Be more conservative - high similarity suggests no change (might be out of PP)
        # But also check if we're potentially in a different state
        if sim >= self.fast_return_thresh:
            # Wait a bit more and check again to be sure
            time.sleep(0.5)
            after2 = capture_window(self.window)
            sim2 = frame_similarity(before, after2)
            self.logger(f"[Battle] Double-check similarity={sim2:.4f}")
            return sim2 >= self.fast_return_thresh
        
        return False

    # ---------- party switching ----------
    def party_reset_cursor_top(self):
        press("up", duration=0.05); time.sleep(0.05)
        press("up", duration=0.05); time.sleep(0.05)

    def party_move_to_slot(self, slot_idx: int):
        # Party screen is a vertical list (1..6). From top, move down (slot_idx-1) times.
        if slot_idx <= 1: return
        for _ in range(slot_idx - 1):
            press("down", duration=0.05)
            time.sleep(0.05)

    def attempt_switch_to_slot(self, slot_idx: int) -> bool:
        """Return True if we appear to have switched successfully."""
        self.logger(f"[Battle] Attempt switch to slot {slot_idx}")
        before = capture_window(self.window)
        self.confirm()  # select Pokémon
        time.sleep(0.15)
        self.confirm()  # confirm 'Switch' if prompted
        time.sleep(0.15)
        # Advance through confirmation text
        for _ in range(5):
            self.advance_dialogue(taps=1)
            time.sleep(0.08)

        # If switch succeeds, the party menu should close and animation plays
        if self.action_failed_quickly(before, self.switch_settle):
            self.logger("[Battle] Switch seems to have failed (still on party/menu).")
            return False
        self.logger("[Battle] Switch likely succeeded.")
        # reset move memory for the new active Pokémon
        self.pp_empty.clear()
        self.move_fail_count.clear()  # New Pokemon = reset failure tracking
        self.order = deque([1,2,3,4])
        return True

    def try_switch_party_member(self) -> bool:
        """Open party menu and try next unused slot. Return True if switched."""
        self.ensure_main_menu()
        self.open_party_menu()
        time.sleep(self.menu_settle)
        self.party_reset_cursor_top()

        # Try slots 2..6 first (avoid slot 1 which is often current)
        for slot in list(range(2,7)) + [1]:
            if slot in self.party_tried:
                continue
            self.party_reset_cursor_top()
            self.party_move_to_slot(slot)
            ok = self.attempt_switch_to_slot(slot)
            self.party_tried.add(slot)
            # Close any lingering menus/dialogue
            self.ensure_main_menu()
            if ok:
                return True
        return False

    # ---------- per-turn entry ----------
    def take_turn(self):
        self.turn_counter += 1
        self.logger(f"[Battle] Turn {self.turn_counter}")

        # Ensure we're on main battle menu and open Moves
        self.ensure_main_menu()
        self.open_moves_menu()
        time.sleep(self.menu_settle)
        self.reset_moves_cursor_top_left()

        # Try up to 4 moves
        tried = set()
        for _ in range(4):
            # rotate to next candidate
            while self.order and (self.order[0] in self.pp_empty or self.order[0] in tried):
                self.order.rotate(-1)
            if not self.order:
                break
            idx = self.order[0]
            tried.add(idx)

            self.logger(f"[Battle] Try move {idx}")
            self.cursor_to_move(idx)
            before = capture_window(self.window)
            self.confirm()
            time.sleep(0.25)
            self.advance_dialogue(taps=2)  # target selection dialogs, etc.

            if self.action_failed_quickly(before, self.fast_return_window):
                # Track failure count before marking as empty
                self.move_fail_count[idx] = self.move_fail_count.get(idx, 0) + 1
                fail_count = self.move_fail_count[idx]
                
                if fail_count >= 2:  # Require 2 failures before marking as empty
                    self.logger(f"[Battle] Move {idx} failed {fail_count} times -> marking OUT OF PP")
                    self.pp_empty.add(idx)
                else:
                    self.logger(f"[Battle] Move {idx} failed ({fail_count}/2) -> trying once more before marking empty")
                    
                press("B", duration=self.pace); time.sleep(self.menu_settle)
                self.reset_moves_cursor_top_left()
                continue

            # Move started; advance text
            for _ in range(self.action_advance_taps):
                self.advance_dialogue(taps=1)
                time.sleep(0.08)
            self.empty_turns = 0
            self.order.rotate(-1)
            return  # turn done

        # All moves failed -> try switching
        self.empty_turns += 1
        self.logger("[Battle] All moves exhausted -> try switching party")
        switched = self.try_switch_party_member()
        if switched:
            self.empty_turns = 0
            self.run_attempted = False
            return

        # If switching failed, attempt Struggle
        self.logger("[Battle] Switch failed or no options. Attempt Struggle.")
        self.ensure_main_menu()
        self.open_moves_menu()
        time.sleep(self.menu_settle)
        self.confirm()
        for _ in range(8):
            self.advance_dialogue(taps=1)
            time.sleep(0.08)

        # As a last resort once, attempt RUN
        if not self.run_attempted and self.empty_turns >= 2:
            self.logger("[Battle] Attempt RUN (last resort).")
            self.ensure_main_menu()
            self.reset_main_cursor_top_left()
            # Move to RUN (bottom-right): down + right
            press("down", duration=0.05); time.sleep(0.05)
            press("right", duration=0.05); time.sleep(0.05)
            self.confirm()
            for _ in range(6):
                self.advance_dialogue(taps=1)
                time.sleep(0.08)
            self.run_attempted = True

    # ---------- lifecycle ----------
    def on_battle_start(self):
        self.logger("[Battle] New battle -> reset memory")
        self.turn_counter = 0
        self.order = deque([1,2,3,4])
        self.pp_empty.clear()
        self.move_fail_count.clear()  # Reset failure tracking
        self.party_tried.clear()
        self.run_attempted = False
