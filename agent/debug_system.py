"""
Debug System - Modular AI Game Agent
===================================

Comprehensive debugging and error tracing system for all modular components.
Provides layer-specific logging, error tracking, and performance monitoring.
"""

import logging
import time
import traceback
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import wraps
import os


@dataclass
class DebugEvent:
    """Represents a single debug event"""
    timestamp: float
    layer: str              # 'perception', 'text_understanding', 'reasoning', etc.
    level: str              # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    message: str
    details: Dict[str, Any] = None
    error_traceback: Optional[str] = None
    performance_data: Optional[Dict[str, float]] = None


@dataclass 
class LayerStats:
    """Performance statistics for a layer"""
    total_calls: int = 0
    total_errors: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None


class ModularDebugger:
    """
    Centralized debugging system for all modular components
    
    Features:
    - Layer-specific logging with color coding
    - Error tracking and statistics
    - Performance monitoring
    - Debug event history
    - Automatic log rotation
    - Real-time debugging output
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 max_events: int = 1000,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 log_level: str = "INFO"):
        
        self.log_dir = Path(log_dir)
        self.max_events = max_events
        self.enable_console = enable_console
        self.enable_file = enable_file
        
        # Create logs directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Debug event storage
        self.events: List[DebugEvent] = []
        self.layer_stats: Dict[str, LayerStats] = {}
        
        # Initialize logging system
        self.setup_logging(log_level)
        
        # Layer-specific loggers
        self.layer_loggers = {}
        self.init_layer_loggers()
        
        # Color codes for console output
        self.colors = {
            'DEBUG': '\033[36m',      # Cyan
            'INFO': '\033[32m',       # Green  
            'WARNING': '\033[33m',    # Yellow
            'ERROR': '\033[31m',      # Red
            'CRITICAL': '\033[35m',   # Magenta
            'RESET': '\033[0m'        # Reset
        }
        
        self.info("ModularDebugger", "Debug system initialized")
    
    def _safe_encode_message(self, message: str) -> str:
        """Safely encode message to avoid Unicode errors"""
        try:
            # Try to encode and decode to detect Unicode issues
            message.encode('cp1252')
            return message
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Remove problematic Unicode characters
            return message.encode('ascii', errors='ignore').decode('ascii')
    
    def setup_logging(self, log_level: str):
        """Setup Python logging system"""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(levelname)-8s | %(name)s | %(message)s'
        )
        
        # Setup root logger
        self.logger = logging.getLogger('ModularAgent')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(simple_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.enable_file:
            log_file = self.log_dir / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
    
    def init_layer_loggers(self):
        """Initialize layer-specific loggers"""
        layers = [
            'perception', 'text_understanding', 'reasoning', 
            'memory', 'goals', 'pathfinding', 'controllers', 'main'
        ]
        
        for layer in layers:
            logger = logging.getLogger(f'ModularAgent.{layer}')
            self.layer_loggers[layer] = logger
            self.layer_stats[layer] = LayerStats()
    
    def get_layer_logger(self, layer: str) -> logging.Logger:
        """Get logger for specific layer"""
        if layer not in self.layer_loggers:
            logger = logging.getLogger(f'ModularAgent.{layer}')
            self.layer_loggers[layer] = logger
            self.layer_stats[layer] = LayerStats()
        
        return self.layer_loggers[layer]
    
    def log_event(self, 
                  layer: str, 
                  level: str, 
                  message: str, 
                  details: Optional[Dict[str, Any]] = None,
                  performance_data: Optional[Dict[str, float]] = None):
        """Log a debug event"""
        
        # Create debug event
        event = DebugEvent(
            timestamp=time.time(),
            layer=layer,
            level=level.upper(),
            message=message,
            details=details or {},
            performance_data=performance_data
        )
        
        # Add to event history
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)  # Remove oldest event
        
        # Update layer stats
        if layer in self.layer_stats:
            stats = self.layer_stats[layer]
            stats.total_calls += 1
            
            if level.upper() in ['ERROR', 'CRITICAL']:
                stats.total_errors += 1
                stats.last_error = message
                stats.last_error_time = event.timestamp
        
        # Log through Python logging system
        logger = self.get_layer_logger(layer)
        log_method = getattr(logger, level.lower(), logger.info)
        
        # Format message with details
        full_message = message
        if details:
            detail_str = " | ".join(f"{k}={v}" for k, v in details.items())
            full_message += f" | {detail_str}"
        
        log_method(full_message)
        
        # Console output with colors (if enabled)
        if self.enable_console:
            self._console_output(event)
    
    def _console_output(self, event: DebugEvent):
        """Colored console output"""
        color = self.colors.get(event.level, '')
        reset = self.colors['RESET']
        
        timestamp_str = datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')
        
        print(f"{color}[{timestamp_str}] {event.layer.upper():<12} | {event.level:<8} | {event.message}{reset}")
        
        if event.details:
            for key, value in event.details.items():
                print(f"{color}    {key}: {value}{reset}")
    
    # Convenience methods for different log levels
    def debug(self, layer: str, message: str, **kwargs):
        """Log debug message"""
        self.log_event(layer, 'DEBUG', message, **kwargs)
    
    def info(self, layer: str, message: str, **kwargs):
        """Log info message"""
        self.log_event(layer, 'INFO', message, **kwargs)
    
    def warning(self, layer: str, message: str, **kwargs):
        """Log warning message"""
        self.log_event(layer, 'WARNING', message, **kwargs)
    
    def error(self, layer: str, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception"""
        details = kwargs.get('details', {})
        
        if exception:
            details['exception_type'] = type(exception).__name__
            details['exception_message'] = str(exception)
            
            # Add traceback
            event = DebugEvent(
                timestamp=time.time(),
                layer=layer,
                level='ERROR',
                message=message,
                details=details,
                error_traceback=traceback.format_exc()
            )
            
            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events.pop(0)
        
        self.log_event(layer, 'ERROR', message, details=details)
    
    def critical(self, layer: str, message: str, **kwargs):
        """Log critical message"""
        self.log_event(layer, 'CRITICAL', message, **kwargs)
    
    def performance_monitor(self, layer: str, operation: str = "operation"):
        """Decorator for performance monitoring"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    self.debug(layer, f"Starting {operation}: {func.__name__}")
                    result = func(*args, **kwargs)
                    
                    execution_time = time.time() - start_time
                    
                    # Update layer stats
                    if layer in self.layer_stats:
                        stats = self.layer_stats[layer]
                        stats.total_time += execution_time
                        stats.avg_time = stats.total_time / max(stats.total_calls, 1)
                    
                    self.info(layer, f"Completed {operation}: {func.__name__}", 
                             performance_data={'execution_time': execution_time})
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.error(layer, f"Failed {operation}: {func.__name__}", 
                              exception=e, 
                              details={'execution_time': execution_time})
                    raise
            
            return wrapper
        return decorator
    
    def get_layer_statistics(self) -> Dict[str, LayerStats]:
        """Get performance statistics for all layers"""
        return {layer: stats for layer, stats in self.layer_stats.items()}
    
    def get_recent_events(self, count: int = 50, layer: Optional[str] = None) -> List[DebugEvent]:
        """Get recent debug events"""
        events = self.events
        
        if layer:
            events = [e for e in events if e.layer == layer]
        
        return events[-count:]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        errors = [e for e in self.events if e.level in ['ERROR', 'CRITICAL']]
        
        return {
            'total_errors': len(errors),
            'recent_errors': errors[-10:],
            'error_by_layer': {
                layer: len([e for e in errors if e.layer == layer])
                for layer in self.layer_stats.keys()
            }
        }
    
    def save_debug_session(self, filename: Optional[str] = None):
        """Save debug session to file"""
        if not filename:
            filename = f"debug_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.log_dir / filename
        
        session_data = {
            'timestamp': time.time(),
            'events': [asdict(event) for event in self.events],
            'layer_stats': {layer: asdict(stats) for layer, stats in self.layer_stats.items()},
            'error_summary': self.get_error_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        self.info("debug_system", f"Debug session saved: {filepath}")
    
    def create_error_report(self) -> str:
        """Create detailed error report"""
        error_summary = self.get_error_summary()
        
        report = []
        report.append("MODULAR AGENT ERROR REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Errors: {error_summary['total_errors']}")
        report.append("")
        
        # Layer statistics
        report.append("LAYER STATISTICS:")
        for layer, stats in self.layer_stats.items():
            if stats.total_calls > 0:
                error_rate = (stats.total_errors / stats.total_calls) * 100
                report.append(f"  {layer.upper():<15} | Calls: {stats.total_calls:>4} | Errors: {stats.total_errors:>3} | Rate: {error_rate:>5.1f}% | Avg Time: {stats.avg_time:.3f}s")
        
        report.append("")
        
        # Recent errors
        recent_errors = [e for e in self.events[-20:] if e.level in ['ERROR', 'CRITICAL']]
        if recent_errors:
            report.append("RECENT ERRORS:")
            for event in recent_errors:
                timestamp_str = datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')
                report.append(f"  [{timestamp_str}] {event.layer.upper()}: {event.message}")
                if event.error_traceback:
                    # Show just the last line of traceback
                    last_line = event.error_traceback.strip().split('\n')[-1]
                    report.append(f"    {last_line}")
        
        return "\n".join(report)


# Global debug instance
_global_debugger: Optional[ModularDebugger] = None


def get_debugger() -> ModularDebugger:
    """Get global debugger instance"""
    global _global_debugger
    if _global_debugger is None:
        _global_debugger = ModularDebugger()
    return _global_debugger


def init_debugging(log_dir: str = "logs", 
                  log_level: str = "INFO",
                  enable_console: bool = True,
                  enable_file: bool = True) -> ModularDebugger:
    """Initialize global debugging system"""
    global _global_debugger
    _global_debugger = ModularDebugger(
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console,
        enable_file=enable_file
    )
    return _global_debugger


# Convenience functions for quick logging
def debug(layer: str, message: str, **kwargs):
    get_debugger().debug(layer, message, **kwargs)

def info(layer: str, message: str, **kwargs):
    get_debugger().info(layer, message, **kwargs)

def warning(layer: str, message: str, **kwargs):
    get_debugger().warning(layer, message, **kwargs)

def error(layer: str, message: str, exception: Optional[Exception] = None, **kwargs):
    get_debugger().error(layer, message, exception=exception, **kwargs)

def critical(layer: str, message: str, **kwargs):
    get_debugger().critical(layer, message, **kwargs)

def monitor_performance(layer: str, operation: str = "operation"):
    """Performance monitoring decorator"""
    return get_debugger().performance_monitor(layer, operation)


# Testing
if __name__ == "__main__":
    # Test the debugging system
    debugger = init_debugging(log_level="DEBUG")
    
    info("test", "Testing debug system initialization")
    debug("test", "Debug message with details", details={'param1': 'value1', 'param2': 42})
    warning("test", "Warning message")
    
    try:
        raise ValueError("Test exception")
    except Exception as e:
        error("test", "Test error handling", exception=e)
    
    # Test performance monitoring
    @monitor_performance("test", "sample_operation")
    def sample_function():
        time.sleep(0.1)
        return "completed"
    
    result = sample_function()
    
    # Show statistics
    stats = debugger.get_layer_statistics()
    print("\nStatistics:")
    for layer, stat in stats.items():
        if stat.total_calls > 0:
            print(f"  {layer}: {stat.total_calls} calls, {stat.total_errors} errors, {stat.avg_time:.3f}s avg")
    
    # Save session
    debugger.save_debug_session()
    
    print("\nDebug system test completed!")