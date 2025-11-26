
import time, cv2
from screen_reader import find_emulator_window, capture_window, detect_game_state
from controllers.gba_controller import press, mash
from battle_logic_pokemon_v2 import PokemonBattleManagerV2

DIRECTIONS = ["up","right","down","left"]

class PrototypeAgentV2:
    def __init__(self, window_title="mGBA", monitoring_bridge=None):
        self.window = find_emulator_window(window_title)
        self.dir_idx = 0
        self.prev_frame = None
        self.nochange_thresh = 0.992
        self.stuck_patience = 10
        self.nochange_count = 0
        self.battle = PokemonBattleManagerV2(self.window, logger=self._log_message)
        self.prev_state = "Unknown"
        self.monitoring_bridge = monitoring_bridge
        
        # Set up controller monitoring if bridge provided
        if self.monitoring_bridge:
            from controllers.gba_controller import set_monitoring_bridge
            set_monitoring_bridge(self.monitoring_bridge)
            self.monitoring_bridge.set_agent_info("Prototype Agent V2")
    
    def _log_message(self, message):
        """Internal logger that routes to monitoring bridge if available"""
        if self.monitoring_bridge:
            self.monitoring_bridge.log_message(message, "Battle")
        else:
            print(message)

    def _sim(self, a, b):
        if a is None or b is None: return 0.0
        import numpy as np, cv2
        ah, aw = a.shape[:2]
        scale = 160.0 / min(ah, aw)
        na = cv2.resize(a, (int(aw*scale), int(ah*scale)), interpolation=cv2.INTER_NEAREST)
        nb = cv2.resize(b, (na.shape[1], na.shape[0]), interpolation=cv2.INTER_NEAREST)
        ga = cv2.cvtColor(na, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(nb, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(ga, gb).astype(np.float32) / 255.0
        mse = float((diff**2).mean())
        return 1.0 - mse

    def explore(self):
        direction = DIRECTIONS[self.dir_idx]
        press(direction, duration=0.18)
        time.sleep(0.05)
        mash("A", count=1, delay=0.10)

    def run(self):
        self._log_message("[AgentV2] Running. Ctrl+C to stop.")
        if self.monitoring_bridge:
            self.monitoring_bridge.set_active(True)
        
        try:
            while True:
                frame = capture_window(self.window)
                state = detect_game_state(frame) or "Unknown"

                # Update monitoring bridge with state changes
                if state != self.prev_state and self.monitoring_bridge:
                    self.monitoring_bridge.update_state(state)

                # battle lifecycle edge
                if "Battle" in state and "Battle" not in self.prev_state:
                    self.battle.on_battle_start()

                if "Battle" in state:
                    self.battle.take_turn()
                else:
                    # anti-stuck explore
                    moved = True
                    if self.prev_frame is not None:
                        sim = self._sim(frame, self.prev_frame)
                        moved = sim < self.nochange_thresh
                        if not moved:
                            self.nochange_count += 1
                        else:
                            self.nochange_count = 0
                    if self.nochange_count >= self.stuck_patience:
                        self._log_message("[Explore] Stuck -> turning")
                        self.dir_idx = (self.dir_idx + 1) % len(DIRECTIONS)
                        self.nochange_count = 0
                    self.explore()

                self.prev_frame = frame
                self.prev_state = state
        except KeyboardInterrupt:
            self._log_message("\n[AgentV2] Stopped.")
        finally:
            if self.monitoring_bridge:
                self.monitoring_bridge.set_active(False)
