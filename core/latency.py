import time
import logging

logger = logging.getLogger(__name__)

class LatencyTracker:
    """
    Context manager for measuring pipeline stage latency.
    
    Usage:
        with LatencyTracker("stage_name", call_sid="CA123") as tracker:
            await do_something()
        # Automatically logs: [CA123] stage_name: 1.23s
    """
    
    def __init__(self, stage_name: str, call_sid: str = "unknown", 
                 warn_threshold_ms: float = None):
        self.stage_name = stage_name
        self.call_sid = call_sid
        self.warn_threshold_ms = warn_threshold_ms
        self.elapsed_ms = None
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
        level = logging.WARNING if (
            self.warn_threshold_ms and self.elapsed_ms > self.warn_threshold_ms
        ) else logging.DEBUG
        logger.log(level, 
            f"[{self.call_sid}] {self.stage_name}: {self.elapsed_ms:.0f}ms"
        )
