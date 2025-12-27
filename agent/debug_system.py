"""
Debug System - Centralized debugging and logging for the AI Game Agent
======================================================================

Provides consistent logging, performance monitoring, and debug utilities
across all agent layers.
"""

import time
import logging
import functools
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

# Configure base logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)


@dataclass
class PerformanceMetrics:
    """Track performance metrics for a specific operation"""
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_time: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    def record(self, elapsed: float):
        self.call_count += 1
        self.total_time += elapsed
        self.last_time = elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)


class DebugSystem:
    """
    Centralized debug system for the AI agent.

    Features:
    - Hierarchical logging per layer
    - Performance monitoring with timing
    - Debug state inspection
    - Configurable verbosity levels
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.enabled = True
        self.log_level = logging.INFO

        # Layer-specific loggers
        self.loggers: Dict[str, logging.Logger] = {}

        # Performance tracking
        self.performance: Dict[str, Dict[str, PerformanceMetrics]] = defaultdict(
            lambda: defaultdict(PerformanceMetrics)
        )

        # Debug state storage
        self.debug_state: Dict[str, Any] = {}

        # Message history (for GUI display)
        self.message_history: list = []
        self.max_history = 1000

        # Callbacks for external systems (like GUI)
        self.message_callbacks: list = []

    def get_logger(self, layer_name: str) -> logging.Logger:
        """Get or create a logger for a specific layer"""
        if layer_name not in self.loggers:
            logger = logging.getLogger(f"agent.{layer_name}")
            logger.setLevel(self.log_level)
            self.loggers[layer_name] = logger
        return self.loggers[layer_name]

    def set_log_level(self, level: int):
        """Set global log level"""
        self.log_level = level
        for logger in self.loggers.values():
            logger.setLevel(level)

    def log(self, layer: str, level: int, message: str, details: Optional[Dict] = None):
        """Log a message with optional details"""
        if not self.enabled:
            return

        logger = self.get_logger(layer)

        # Format message with details
        if details:
            detail_str = " | ".join(f"{k}={v}" for k, v in details.items())
            full_message = f"{message} [{detail_str}]"
        else:
            full_message = message

        logger.log(level, full_message)

        # Store in history
        entry = {
            'timestamp': datetime.now(),
            'layer': layer,
            'level': logging.getLevelName(level),
            'message': message,
            'details': details
        }
        self.message_history.append(entry)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)

        # Notify callbacks
        for callback in self.message_callbacks:
            try:
                callback(entry)
            except Exception:
                pass  # Don't let callback errors break logging

    def record_performance(self, layer: str, operation: str, elapsed: float):
        """Record performance metrics for an operation"""
        self.performance[layer][operation].record(elapsed)

    def get_performance_summary(self, layer: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for one or all layers"""
        if layer:
            return {
                op: {
                    'calls': m.call_count,
                    'avg_ms': m.avg_time * 1000,
                    'min_ms': m.min_time * 1000 if m.min_time != float('inf') else 0,
                    'max_ms': m.max_time * 1000,
                    'last_ms': m.last_time * 1000
                }
                for op, m in self.performance[layer].items()
            }
        else:
            return {
                layer_name: self.get_performance_summary(layer_name)
                for layer_name in self.performance.keys()
            }

    def set_state(self, key: str, value: Any):
        """Store debug state for inspection"""
        self.debug_state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Retrieve debug state"""
        return self.debug_state.get(key, default)

    def add_message_callback(self, callback: Callable):
        """Add callback to be notified of new log messages"""
        self.message_callbacks.append(callback)

    def remove_message_callback(self, callback: Callable):
        """Remove a message callback"""
        if callback in self.message_callbacks:
            self.message_callbacks.remove(callback)


# Global debugger instance
_debugger: Optional[DebugSystem] = None


def get_debugger() -> DebugSystem:
    """Get the global debugger instance"""
    global _debugger
    if _debugger is None:
        _debugger = DebugSystem()
    return _debugger


def init_debugging(log_level: str = "INFO", enabled: bool = True):
    """Initialize the debug system with specified settings"""
    debugger = get_debugger()
    debugger.enabled = enabled

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    debugger.set_log_level(level_map.get(log_level.upper(), logging.INFO))

    return debugger


def monitor_performance(layer: str, operation: str):
    """
    Decorator to monitor function performance.

    Usage:
        @monitor_performance("perception", "frame_analysis")
        def analyze_frame(self, frame):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            debugger = get_debugger()
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                debugger.record_performance(layer, operation, elapsed)
        return wrapper
    return decorator


# Convenience logging functions
def debug(layer: str, message: str, details: Optional[Dict] = None):
    """Log debug message"""
    get_debugger().log(layer, logging.DEBUG, message, details)


def info(layer: str, message: str, details: Optional[Dict] = None):
    """Log info message"""
    get_debugger().log(layer, logging.INFO, message, details)


def warning(layer: str, message: str, details: Optional[Dict] = None):
    """Log warning message"""
    get_debugger().log(layer, logging.WARNING, message, details)


def error(layer: str, message: str, exception: Optional[Exception] = None,
          details: Optional[Dict] = None):
    """Log error message with optional exception"""
    if details is None:
        details = {}
    if exception:
        details['exception'] = str(exception)
        details['exception_type'] = type(exception).__name__
    get_debugger().log(layer, logging.ERROR, message, details)


def critical(layer: str, message: str, details: Optional[Dict] = None):
    """Log critical message"""
    get_debugger().log(layer, logging.CRITICAL, message, details)


# Testing
if __name__ == "__main__":
    # Initialize debugging
    init_debugging(log_level="DEBUG")

    # Test logging
    info("test", "Debug system initialized")
    debug("perception", "Processing frame", {"frame_id": 1, "size": "240x160"})
    warning("reasoning", "Slow decision", {"elapsed_ms": 250})
    error("memory", "Failed to save", exception=ValueError("Test error"))

    # Test performance monitoring
    @monitor_performance("test", "sample_operation")
    def sample_function():
        time.sleep(0.01)
        return "done"

    for _ in range(5):
        sample_function()

    # Print performance summary
    debugger = get_debugger()
    print("\nPerformance Summary:")
    print(debugger.get_performance_summary())

    print("\nDebug system test completed!")
