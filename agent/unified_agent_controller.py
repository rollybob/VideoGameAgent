"""
Unified Agent Controller - Manages all agent types with consistent control interface
"""

import threading
import time
from typing import Optional, Dict, Any
from enum import Enum

class AgentType(Enum):
    TRADITIONAL = "traditional"
    PROTOTYPE_V2 = "prototype_v2"
    FAST_MOVEMENT = "fast_movement"
    NEURAL = "neural"

class AgentState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class UnifiedAgentController:
    """
    Central controller that manages all agent types with unified pause/resume/status interface
    """
    
    def __init__(self, monitoring_bridge=None):
        self.monitoring_bridge = monitoring_bridge
        
        # Control state
        self.current_agent_type = None
        self.current_agent = None
        self.agent_thread = None
        self.agent_state = AgentState.STOPPED
        
        # Unified pause mechanism
        self._pause_requested = False
        self._shutdown_requested = False
        self._status_update_thread = None
        
        # Agent instances
        self._agents = {}
        
    def _create_agent(self, agent_type: AgentType, **kwargs) -> Any:
        """Create agent instance based on type"""
        if agent_type == AgentType.PROTOTYPE_V2:
            from proto_agent_v2 import PrototypeAgentV2
            return PrototypeAgentV2(monitoring_bridge=self.monitoring_bridge, **kwargs)
        elif agent_type == AgentType.FAST_MOVEMENT:
            from fast_movement_agent import FastMovementAgent
            return FastMovementAgent(**kwargs)
        elif agent_type == AgentType.TRADITIONAL:
            # Would integrate with existing GameAgent
            pass
        elif agent_type == AgentType.NEURAL:
            # Would integrate with neural agent
            pass
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def start_agent(self, agent_type: AgentType, **agent_kwargs) -> bool:
        """Start an agent with unified control interface"""
        try:
            # Stop any existing agent first
            if self.agent_state != AgentState.STOPPED:
                self.stop_agent()
                time.sleep(0.5)  # Brief pause for cleanup
            
            # Create new agent
            self.current_agent_type = agent_type
            self.current_agent = self._create_agent(agent_type, **agent_kwargs)
            
            # Reset control flags
            self._pause_requested = False
            self._shutdown_requested = False
            self.agent_state = AgentState.RUNNING
            
            # Start agent in controlled wrapper
            self.agent_thread = threading.Thread(target=self._controlled_agent_wrapper, daemon=True)
            self.agent_thread.start()
            
            # Start status monitoring
            self._status_update_thread = threading.Thread(target=self._status_monitor, daemon=True)
            self._status_update_thread.start()
            
            if self.monitoring_bridge:
                self.monitoring_bridge.set_active(True)
                self.monitoring_bridge.log_message(f"Started {agent_type.value} agent", "System")
            
            return True
            
        except Exception as e:
            self.agent_state = AgentState.ERROR
            if self.monitoring_bridge:
                self.monitoring_bridge.log_message(f"Failed to start agent: {e}", "Error")
            return False
    
    def _controlled_agent_wrapper(self):
        """Wrapper that adds pause/resume control to any agent"""
        try:
            # For PrototypeAgentV2, we need to modify its run loop
            if self.current_agent_type == AgentType.PROTOTYPE_V2:
                self._run_prototype_with_control()
            else:
                # For other agents, call their run method directly
                # (they would need to be modified to check pause state)
                self.current_agent.run()
                
        except Exception as e:
            self.agent_state = AgentState.ERROR
            if self.monitoring_bridge:
                self.monitoring_bridge.log_message(f"Agent error: {e}", "Error")
        finally:
            self.agent_state = AgentState.STOPPED
            if self.monitoring_bridge:
                self.monitoring_bridge.set_active(False)
    
    def _run_prototype_with_control(self):
        """Run prototype agent with unified pause/resume control"""
        from screen_reader import capture_window, detect_game_state
        from controllers.gba_controller import press, mash
        
        agent = self.current_agent
        agent._log_message("[Unified Controller] Prototype Agent V2 starting...")
        
        DIRECTIONS = ["up", "right", "down", "left"]
        
        while not self._shutdown_requested:
            # Handle pause state
            if self._pause_requested:
                self.agent_state = AgentState.PAUSED
                time.sleep(0.1)
                continue
            
            # Ensure we're in running state
            if self.agent_state != AgentState.RUNNING:
                self.agent_state = AgentState.RUNNING
            
            try:
                # Core prototype agent logic with pause checks
                frame = capture_window(agent.window)
                state = detect_game_state(frame) or "Unknown"

                # Update monitoring bridge with state changes
                if state != agent.prev_state and agent.monitoring_bridge:
                    agent.monitoring_bridge.update_state(state)

                # Battle lifecycle edge
                if "Battle" in state and "Battle" not in agent.prev_state:
                    agent.battle.on_battle_start()

                if "Battle" in state:
                    # Check for pause before battle turn
                    if not self._pause_requested and not self._shutdown_requested:
                        agent.battle.take_turn()
                else:
                    # Exploration with pause checks
                    if not self._pause_requested and not self._shutdown_requested:
                        # Anti-stuck explore logic
                        moved = True
                        if agent.prev_frame is not None:
                            sim = agent._sim(frame, agent.prev_frame)
                            moved = sim < agent.nochange_thresh
                            if not moved:
                                agent.nochange_count += 1
                            else:
                                agent.nochange_count = 0
                        
                        if agent.nochange_count >= agent.stuck_patience:
                            agent._log_message("[Explore] Stuck -> turning")
                            agent.dir_idx = (agent.dir_idx + 1) % len(DIRECTIONS)
                            agent.nochange_count = 0
                        
                        # Use fast movement if available
                        direction = DIRECTIONS[agent.dir_idx]
                        press(direction, duration=0.3)  # Longer press for faster movement
                        time.sleep(0.05)
                        mash("A", count=1, delay=0.10)

                agent.prev_frame = frame
                agent.prev_state = state
                
            except Exception as e:
                agent._log_message(f"Agent loop error: {e}")
                time.sleep(0.1)
    
    def _status_monitor(self):
        """Monitor and update agent status"""
        while not self._shutdown_requested and self.agent_thread and self.agent_thread.is_alive():
            if self.monitoring_bridge:
                status_text = f"Status: {self.agent_state.value.title()}"
                if self.current_agent_type:
                    status_text += f" ({self.current_agent_type.value})"
                
                # Update GUI status based on current state
                if self.agent_state == AgentState.RUNNING:
                    self.monitoring_bridge.gui.status_label.config(text=status_text, fg='#10B981')
                elif self.agent_state == AgentState.PAUSED:
                    self.monitoring_bridge.gui.status_label.config(text=status_text, fg='#F59E0B')
                elif self.agent_state == AgentState.ERROR:
                    self.monitoring_bridge.gui.status_label.config(text=status_text, fg='#EF4444')
            
            time.sleep(1)  # Update every second
    
    def pause_agent(self) -> bool:
        """Pause the currently running agent"""
        if self.agent_state == AgentState.RUNNING:
            self._pause_requested = True
            self.agent_state = AgentState.PAUSED
            if self.monitoring_bridge:
                self.monitoring_bridge.log_message("Agent paused", "Control")
            return True
        return False
    
    def resume_agent(self) -> bool:
        """Resume the currently paused agent"""
        if self.agent_state == AgentState.PAUSED:
            self._pause_requested = False
            self.agent_state = AgentState.RUNNING
            if self.monitoring_bridge:
                self.monitoring_bridge.log_message("Agent resumed", "Control")
            return True
        return False
    
    def toggle_pause(self) -> bool:
        """Toggle pause/resume state"""
        if self.agent_state == AgentState.RUNNING:
            return self.pause_agent()
        elif self.agent_state == AgentState.PAUSED:
            return self.resume_agent()
        return False
    
    def stop_agent(self) -> bool:
        """Stop the currently running agent"""
        self._shutdown_requested = True
        self._pause_requested = False
        
        if self.agent_thread and self.agent_thread.is_alive():
            self.agent_thread.join(timeout=2)  # Wait up to 2 seconds for graceful shutdown
        
        self.agent_state = AgentState.STOPPED
        self.current_agent = None
        self.current_agent_type = None
        
        if self.monitoring_bridge:
            self.monitoring_bridge.set_active(False)
            self.monitoring_bridge.log_message("Agent stopped", "System")
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status information"""
        return {
            'state': self.agent_state,
            'agent_type': self.current_agent_type,
            'is_running': self.agent_state in [AgentState.RUNNING, AgentState.PAUSED],
            'can_pause': self.agent_state == AgentState.RUNNING,
            'can_resume': self.agent_state == AgentState.PAUSED
        }