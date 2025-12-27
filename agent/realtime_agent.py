"""
Real-Time Game Agent - Using tiered decision architecture
==========================================================

This agent uses the tiered decision system for real-time gameplay:
- Fast layer: Behavior trees for immediate actions (<20ms)
- Medium layer: State analysis and goal evaluation (500ms)
- Slow layer: Strategic reasoning (async, 5s+)

Key improvements over proto_agent_v2:
1. Structured decision making with behavior trees
2. Proper state management with GameState
3. Goal-oriented behavior
4. Clean separation of concerns
5. Performance monitoring built-in
6. Ready for LLM integration when hardware allows
"""

import time
import cv2
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from screen_reader import find_emulator_window, capture_window, detect_game_state
from tiered_decision_system import TieredDecisionSystem, DecisionLayer
from behavior_tree import GameState
from debug_system import get_debugger, info, debug, warning, error, init_debugging
from config import get_config


@dataclass
class AgentStats:
    """Agent performance statistics"""
    frames_processed: int = 0
    battles_entered: int = 0
    battles_completed: int = 0
    stuck_events: int = 0
    context_switches: int = 0
    avg_frame_time_ms: float = 0.0
    start_time: float = 0.0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0


class RealTimeAgent:
    """
    Real-time game agent using tiered decision system.

    This agent runs at full frame rate, making decisions in <20ms
    while still having access to strategic reasoning through the
    slow layer.
    """

    def __init__(self, window_title: str = "mGBA", monitoring_bridge=None):
        self.config = get_config()
        self.debugger = get_debugger()

        # Window handling
        self.window_title = window_title
        self.window = None

        # Decision system
        self.decision_system = TieredDecisionSystem()

        # Frame comparison for stuck detection
        self.prev_frame = None
        self.frame_similarity_threshold = self.config.exploration.stuck_similarity_threshold

        # Battle integration
        self.battle_manager = None
        self.in_battle = False
        self.prev_state = "Unknown"

        # Monitoring
        self.monitoring_bridge = monitoring_bridge
        self.stats = AgentStats()

        # Control
        self.running = False
        self.paused = False

        # Callbacks
        self._setup_callbacks()

    def _setup_callbacks(self):
        """Set up decision system callbacks"""
        def on_state_change(new_state):
            self._log(f"Context changed to: {new_state}")
            if self.monitoring_bridge:
                self.monitoring_bridge.update_state(new_state)

        def on_goal_update(goal):
            self._log(f"New goal: {goal.description}")

        self.decision_system.on_state_change = on_state_change
        self.decision_system.on_goal_update = on_goal_update

    def _log(self, message: str, category: str = "Agent"):
        """Log message to monitoring bridge or console"""
        if self.monitoring_bridge:
            self.monitoring_bridge.log_message(message, category)
        else:
            print(f"[{category}] {message}")

    def _init_window(self):
        """Initialize emulator window connection"""
        try:
            self.window = find_emulator_window(self.window_title)
            self._log(f"Connected to emulator: {self.window.title}")
            return True
        except Exception as e:
            error("agent", f"Failed to find emulator window: {e}")
            return False

    def _init_battle_manager(self):
        """Initialize battle manager lazily"""
        if self.battle_manager is None:
            try:
                from battle_logic_pokemon_v2 import PokemonBattleManagerV2
                self.battle_manager = PokemonBattleManagerV2(
                    self.window,
                    logger=lambda msg: self._log(msg, "Battle")
                )
                debug("agent", "Battle manager initialized")
            except Exception as e:
                error("agent", f"Failed to initialize battle manager: {e}")

    def _calculate_frame_similarity(self, frame_a, frame_b) -> float:
        """Calculate similarity between two frames"""
        if frame_a is None or frame_b is None:
            return 0.0

        try:
            # Resize for faster comparison
            h, w = frame_a.shape[:2]
            scale = 160.0 / min(h, w)
            size = (int(w * scale), int(h * scale))

            a_resized = cv2.resize(frame_a, size, interpolation=cv2.INTER_NEAREST)
            b_resized = cv2.resize(frame_b, size, interpolation=cv2.INTER_NEAREST)

            # Convert to grayscale
            a_gray = cv2.cvtColor(a_resized, cv2.COLOR_BGR2GRAY)
            b_gray = cv2.cvtColor(b_resized, cv2.COLOR_BGR2GRAY)

            # Calculate MSE
            diff = cv2.absdiff(a_gray, b_gray).astype(np.float32) / 255.0
            mse = float(np.mean(diff ** 2))

            return 1.0 - mse

        except Exception as e:
            debug("agent", f"Frame similarity error: {e}")
            return 0.0

    def _update_stuck_detection(self, frame):
        """Update stuck detection based on frame similarity"""
        if self.prev_frame is not None:
            similarity = self._calculate_frame_similarity(frame, self.prev_frame)
            if similarity >= self.frame_similarity_threshold:
                # Frame didn't change much - might be stuck
                self.decision_system.game_state.stuck_count += 1
            else:
                # Frame changed - reset stuck counter
                self.decision_system.game_state.stuck_count = 0

            # Update decision system
            self.decision_system.update_stuck_counter(
                self.decision_system.game_state.stuck_count
            )

        self.prev_frame = frame

    def _handle_battle_transition(self, current_state: str):
        """Handle entering/exiting battles"""
        entering_battle = "Battle" in current_state and "Battle" not in self.prev_state
        exiting_battle = "Battle" not in current_state and "Battle" in self.prev_state

        if entering_battle:
            self._init_battle_manager()
            if self.battle_manager:
                self.battle_manager.on_battle_start()
                self.stats.battles_entered += 1
                self.in_battle = True
                self._log("Entering battle")

        elif exiting_battle:
            self.in_battle = False
            self.stats.battles_completed += 1
            self._log("Exiting battle")

        self.prev_state = current_state

    def _run_frame(self, frame) -> Optional[str]:
        """Process a single frame and return action taken"""
        start_time = time.perf_counter()

        # Detect game state
        state = detect_game_state(frame) or "Unknown"

        # Handle battle transitions
        self._handle_battle_transition(state)

        # Update decision system with battle state
        self.decision_system.update_battle_state(
            self.in_battle,
            1.0  # TODO: extract HP from frame
        )

        # Update stuck detection
        self._update_stuck_detection(frame)

        # Get decision from tiered system
        decision = self.decision_system.tick(frame)

        # Execute action
        action_taken = None
        if self.in_battle and self.battle_manager:
            # Delegate to battle manager for battles
            self.battle_manager.take_turn()
            action_taken = "battle_turn"
        elif decision:
            # Action was executed by behavior tree
            action_taken = decision.action

        # Update stats
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_frame_stats(elapsed_ms)

        return action_taken

    def _update_frame_stats(self, elapsed_ms: float):
        """Update frame processing statistics"""
        self.stats.frames_processed += 1
        n = self.stats.frames_processed
        self.stats.avg_frame_time_ms = (
            self.stats.avg_frame_time_ms * (n - 1) + elapsed_ms
        ) / n

    def run(self):
        """Main agent loop"""
        self._log("Starting Real-Time Agent")

        # Initialize
        if not self._init_window():
            self._log("Failed to initialize - exiting")
            return

        # Set up monitoring
        if self.monitoring_bridge:
            self.monitoring_bridge.set_active(True)
            self.monitoring_bridge.set_agent_info("Real-Time Agent v1.0")

            # Connect controller monitoring
            try:
                from controllers.gba_controller import set_monitoring_bridge
                set_monitoring_bridge(self.monitoring_bridge)
            except Exception as e:
                debug("agent", f"Could not set controller monitoring: {e}")

        # Start decision system
        self.decision_system.start()
        self.stats.start_time = time.time()
        self.running = True

        try:
            while self.running:
                # Handle pause
                if self.paused:
                    time.sleep(0.1)
                    continue

                try:
                    # Capture frame
                    frame = capture_window(self.window)

                    if frame is not None and frame.size > 0:
                        # Process frame
                        action = self._run_frame(frame)

                        # Log occasionally
                        if self.stats.frames_processed % 100 == 0:
                            debug("agent",
                                f"Frame {self.stats.frames_processed}: "
                                f"avg={self.stats.avg_frame_time_ms:.2f}ms, "
                                f"context={self.decision_system.game_state.current_context}"
                            )

                except Exception as e:
                    error("agent", f"Frame processing error: {e}")
                    time.sleep(0.1)

        except KeyboardInterrupt:
            self._log("Interrupted by user")

        finally:
            self._shutdown()

    def _shutdown(self):
        """Clean shutdown"""
        self.running = False
        self.decision_system.stop()

        if self.monitoring_bridge:
            self.monitoring_bridge.set_active(False)

        # Log final stats
        stats = self.get_stats()
        self._log(f"Agent stopped after {stats['uptime']:.1f}s")
        self._log(f"Frames: {stats['frames']}, Battles: {stats['battles_completed']}")

    def pause(self):
        """Pause agent execution"""
        self.paused = True
        self._log("Agent paused")

    def resume(self):
        """Resume agent execution"""
        self.paused = False
        self._log("Agent resumed")

    def stop(self):
        """Stop agent execution"""
        self.running = False
        self._log("Agent stop requested")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        decision_stats = self.decision_system.get_stats()

        return {
            'frames': self.stats.frames_processed,
            'uptime': self.stats.uptime_seconds,
            'avg_frame_ms': self.stats.avg_frame_time_ms,
            'battles_entered': self.stats.battles_entered,
            'battles_completed': self.stats.battles_completed,
            'stuck_events': self.stats.stuck_events,
            'context_switches': self.stats.context_switches,
            'current_context': decision_stats.get('current_context', 'unknown'),
            'current_tree': decision_stats.get('current_tree', 'none'),
            'active_goals': decision_stats.get('active_goals', 0),
            'decision_stats': decision_stats,
        }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

def main():
    """Run the real-time agent standalone"""
    import argparse

    parser = argparse.ArgumentParser(description="Real-Time Game Agent")
    parser.add_argument("--window", default="mGBA", help="Emulator window title")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Initialize debug system
    log_level = "DEBUG" if args.debug else "INFO"
    init_debugging(log_level=log_level)

    print("=" * 60)
    print("REAL-TIME GAME AGENT")
    print("=" * 60)
    print(f"Window: {args.window}")
    print(f"Debug: {args.debug}")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # Create and run agent
    agent = RealTimeAgent(window_title=args.window)
    agent.run()


if __name__ == "__main__":
    main()
