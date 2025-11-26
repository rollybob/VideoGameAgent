"""
Parallel Decision Manager - Coordinates multiple AI layers running in parallel
"""

import threading
import time
import queue
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import traceback

@dataclass
class DecisionResult:
    """Result from an AI decision layer"""
    action: str
    reasoning: str
    confidence: float
    layer_name: str
    processing_time: float
    success: bool = True
    error: Optional[str] = None

@dataclass
class GameState:
    """Cached game state for parallel processing"""
    frame: Any
    ml_frame: Any
    current_state: str
    screen_analysis: Dict
    timestamp: float
    confidence: float

class ParallelDecisionManager:
    """
    Manages parallel execution of multiple AI decision layers
    """
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="AI-Layer")
        self.decision_timeout = 2.0  # Maximum time to wait for decisions
        self.min_confidence_threshold = 0.3  # Minimum confidence to consider
        
        # Performance tracking
        self.decision_history = []
        self.layer_performance = {}
        self.parallel_stats = {
            'total_decisions': 0,
            'parallel_decisions': 0,
            'average_time': 0.0,
            'timeout_count': 0,
            'error_count': 0
        }
        
        # State caching
        self.cached_state = None
        self.state_cache_timeout = 0.1  # 100ms cache validity
        
    def cache_game_state(self, frame, ml_frame, current_state: str, screen_analysis: Dict, confidence: float = 1.0):
        """Cache current game state for AI layers"""
        self.cached_state = GameState(
            frame=frame,
            ml_frame=ml_frame,
            current_state=current_state,
            screen_analysis=screen_analysis,
            timestamp=time.time(),
            confidence=confidence
        )
    
    def get_cached_state(self) -> Optional[GameState]:
        """Get cached state if still valid"""
        if self.cached_state is None:
            return None
        
        age = time.time() - self.cached_state.timestamp
        if age > self.state_cache_timeout:
            return None
        
        return self.cached_state
    
    def run_fast_movement_layer(self, state: GameState, fast_movement_agent) -> DecisionResult:
        """Run Fast Movement layer in thread"""
        start_time = time.time()
        layer_name = "FastMovement"
        
        try:
            if fast_movement_agent is None:
                return DecisionResult("wait", "Fast movement not available", 0.0, layer_name, 0.0, False, "Agent not initialized")
            
            if not fast_movement_agent.should_use_fast_mode(state.current_state):
                return DecisionResult("wait", "Fast mode not appropriate for state", 0.0, layer_name, time.time() - start_time, False, "State not suitable")
            
            action, reasoning, confidence = fast_movement_agent.make_fast_decision(
                state.current_state, state.screen_analysis
            )
            
            processing_time = time.time() - start_time
            return DecisionResult(action, reasoning, confidence, layer_name, processing_time, True)
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Fast Movement error: {str(e)}"
            return DecisionResult("wait", error_msg, 0.0, layer_name, processing_time, False, str(e))
    
    def run_universal_ai_layer(self, state: GameState, universal_ai_bridge) -> DecisionResult:
        """Run Universal AI layer in thread"""
        start_time = time.time()
        layer_name = "UniversalAI"
        
        try:
            if universal_ai_bridge is None:
                return DecisionResult("wait", "Universal AI not available", 0.0, layer_name, 0.0, False, "Agent not initialized")
            
            action, reasoning, confidence = universal_ai_bridge.make_universal_decision(
                game_frame=state.ml_frame,
                current_state=state.current_state,
                screen_analysis=state.screen_analysis
            )
            
            processing_time = time.time() - start_time
            return DecisionResult(action, reasoning, confidence, layer_name, processing_time, True)
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Universal AI error: {str(e)}"
            return DecisionResult("wait", error_msg, 0.0, layer_name, processing_time, False, str(e))
    
    def run_neural_agent_layer(self, state: GameState, neural_agent, use_neural_agent: bool) -> DecisionResult:
        """Run Neural Agent layer in thread"""
        start_time = time.time()
        layer_name = "NeuralAgent"
        
        try:
            if not use_neural_agent or neural_agent is None:
                return DecisionResult("wait", "Neural agent not enabled", 0.0, layer_name, 0.0, False, "Agent not enabled")
            
            # Neural agent needs to be adapted to return confidence
            # For now, assume it returns action, reasoning, confidence format
            if hasattr(neural_agent, 'make_decision_with_confidence'):
                action, reasoning, confidence = neural_agent.make_decision_with_confidence(
                    state.frame, state.current_state
                )
            else:
                # Fallback to existing method with default confidence
                action = neural_agent.make_decision(state.frame, state.current_state)
                reasoning = f"Neural agent decision: {action}"
                confidence = 0.6  # Default confidence for neural decisions
            
            processing_time = time.time() - start_time
            return DecisionResult(action, reasoning, confidence, layer_name, processing_time, True)
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Neural Agent error: {str(e)}"
            return DecisionResult("wait", error_msg, 0.0, layer_name, processing_time, False, str(e))
    
    def arbitrate_decisions(self, decisions: List[DecisionResult]) -> Tuple[str, str, float, str]:
        """
        Arbitrate between multiple AI decisions using confidence and layer priority
        Returns: (action, reasoning, confidence, layer_name)
        """
        if not decisions:
            return "wait", "No decisions available", 0.0, "None"
        
        # Filter out failed decisions and low confidence
        valid_decisions = [
            d for d in decisions 
            if d.success and d.confidence >= self.min_confidence_threshold and d.action != "wait"
        ]
        
        if not valid_decisions:
            # If no valid decisions, use the best failed decision or wait
            best_failed = max(decisions, key=lambda d: d.confidence)
            return best_failed.action, f"Fallback: {best_failed.reasoning}", best_failed.confidence, best_failed.layer_name
        
        # Layer priority weights (higher = more trusted)
        layer_weights = {
            "FastMovement": 1.0,    # Good for exploration
            "UniversalAI": 1.2,     # Most sophisticated
            "NeuralAgent": 0.8      # Specialized but limited
        }
        
        # Calculate weighted confidence scores
        weighted_decisions = []
        for decision in valid_decisions:
            weight = layer_weights.get(decision.layer_name, 1.0)
            weighted_score = decision.confidence * weight
            weighted_decisions.append((decision, weighted_score))
        
        # Sort by weighted confidence
        weighted_decisions.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best decision
        best_decision = weighted_decisions[0][0]
        best_score = weighted_decisions[0][1]
        
        # Enhanced reasoning with parallel info
        reasoning = f"{best_decision.layer_name}: {best_decision.reasoning} (score: {best_score:.3f})"
        
        return best_decision.action, reasoning, best_decision.confidence, best_decision.layer_name
    
    def make_parallel_decision(self, frame, ml_frame, current_state: str, screen_analysis: Dict,
                             fast_movement_agent=None, universal_ai_bridge=None, neural_agent=None, 
                             use_neural_agent: bool = False) -> Tuple[str, str, float, str, Dict]:
        """
        Make decision using parallel AI layers
        Returns: (action, reasoning, confidence, layer_name, stats)
        """
        start_time = time.time()
        
        # Cache current state
        self.cache_game_state(frame, ml_frame, current_state, screen_analysis)
        state = self.cached_state
        
        # Submit parallel tasks
        futures = {}
        
        if fast_movement_agent is not None:
            future = self.thread_pool.submit(self.run_fast_movement_layer, state, fast_movement_agent)
            futures[future] = "FastMovement"
        
        if universal_ai_bridge is not None:
            future = self.thread_pool.submit(self.run_universal_ai_layer, state, universal_ai_bridge)
            futures[future] = "UniversalAI"
        
        if use_neural_agent and neural_agent is not None:
            future = self.thread_pool.submit(self.run_neural_agent_layer, state, neural_agent, use_neural_agent)
            futures[future] = "NeuralAgent"
        
        # Collect results with timeout
        decisions = []
        completed_layers = []
        timeout_layers = []
        error_layers = []
        
        try:
            for future in as_completed(futures, timeout=self.decision_timeout):
                layer_name = futures[future]
                try:
                    result = future.result()
                    decisions.append(result)
                    completed_layers.append(layer_name)
                    
                    if not result.success:
                        error_layers.append(layer_name)
                        
                except Exception as e:
                    error_msg = f"{layer_name} exception: {str(e)}"
                    error_result = DecisionResult(
                        "wait", error_msg, 0.0, layer_name, 
                        time.time() - start_time, False, str(e)
                    )
                    decisions.append(error_result)
                    error_layers.append(layer_name)
                    
        except Exception as timeout_error:
            # Handle timeout - collect any completed results
            for future, layer_name in futures.items():
                if not future.done():
                    timeout_layers.append(layer_name)
                    future.cancel()
        
        # Arbitrate between decisions
        action, reasoning, confidence, best_layer = self.arbitrate_decisions(decisions)
        
        # Calculate statistics
        total_time = time.time() - start_time
        stats = {
            'total_time': total_time,
            'layers_attempted': len(futures),
            'layers_completed': len(completed_layers),
            'layers_timeout': len(timeout_layers),
            'layers_error': len(error_layers),
            'completed_layers': completed_layers,
            'timeout_layers': timeout_layers,
            'error_layers': error_layers,
            'best_layer': best_layer,
            'decision_count': len(decisions),
            'parallel_speedup': max(0.1, max([d.processing_time for d in decisions if d.success], default=0.1)) / max(0.01, total_time)
        }
        
        # Update performance tracking
        self.update_performance_stats(stats, decisions)
        
        return action, reasoning, confidence, best_layer, stats
    
    def update_performance_stats(self, stats: Dict, decisions: List[DecisionResult]):
        """Update performance tracking statistics"""
        self.parallel_stats['total_decisions'] += 1
        if stats['layers_completed'] > 1:
            self.parallel_stats['parallel_decisions'] += 1
        
        # Running average of decision time
        current_avg = self.parallel_stats['average_time']
        total_decisions = self.parallel_stats['total_decisions']
        new_avg = (current_avg * (total_decisions - 1) + stats['total_time']) / total_decisions
        self.parallel_stats['average_time'] = new_avg
        
        # Track timeouts and errors
        if stats['layers_timeout']:
            self.parallel_stats['timeout_count'] += 1
        if stats['layers_error']:
            self.parallel_stats['error_count'] += 1
        
        # Update layer-specific performance
        for decision in decisions:
            layer = decision.layer_name
            if layer not in self.layer_performance:
                self.layer_performance[layer] = {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'average_time': 0.0,
                    'average_confidence': 0.0
                }
            
            perf = self.layer_performance[layer]
            perf['total_calls'] += 1
            
            if decision.success:
                perf['successful_calls'] += 1
                
                # Update running averages
                successful_calls = perf['successful_calls']
                perf['average_time'] = (perf['average_time'] * (successful_calls - 1) + decision.processing_time) / successful_calls
                perf['average_confidence'] = (perf['average_confidence'] * (successful_calls - 1) + decision.confidence) / successful_calls
    
    def get_performance_report(self) -> Dict:
        """Get detailed performance report"""
        report = {
            'parallel_stats': self.parallel_stats.copy(),
            'layer_performance': self.layer_performance.copy(),
            'cache_hit_rate': 0.0,  # Would need to track cache hits
            'average_speedup': 0.0
        }
        
        # Calculate average speedup
        if self.parallel_stats['parallel_decisions'] > 0:
            # Estimate speedup based on layer performance
            avg_layer_times = [
                perf['average_time'] for perf in self.layer_performance.values() 
                if perf['successful_calls'] > 0
            ]
            if len(avg_layer_times) > 1:
                sequential_time = sum(avg_layer_times)
                parallel_time = self.parallel_stats['average_time']
                if parallel_time > 0:
                    report['average_speedup'] = sequential_time / parallel_time
        
        return report
    
    def cleanup(self):
        """Clean up thread pool"""
        try:
            self.thread_pool.shutdown(wait=True)
        except Exception as e:
            print(f"Error during parallel decision manager cleanup: {e}")