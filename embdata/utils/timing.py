import functools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class FunctionTiming:
    """Store timing information for a single function call."""
    name: str
    duration: float
    start_time: float
    end_time: float

class TimingStats:
    """Class to track timing statistics for multiple function calls."""
    def __init__(self):
        self.timings: Dict[str, list[FunctionTiming]] = defaultdict(list)

    def add_timing(self, timing: FunctionTiming) -> None:
        """Add a new timing measurement."""
        self.timings[timing.name].append(timing)

    def get_latest_timings(self) -> Dict[str, float]:
        """Get the most recent timing for each function."""
        return {
            name: timings[-1].duration
            for name, timings in self.timings.items()
            if timings
        }

    def clear(self) -> None:
        """Clear all timing data."""
        self.timings.clear()

    def get_average_timing(self, func_name: str) -> float:
        """Get average timing for a specific function."""
        timings = self.timings.get(func_name, [])
        if not timings:
            return 0.0
        return sum(t.duration for t in timings) / len(timings)

    def get_timing_stats(self, func_name: str) -> Dict[str, float]:
        """Get detailed stats for a function."""
        timings = self.timings.get(func_name, [])
        if not timings:
            return {}

        durations = [t.duration for t in timings]
        return {
            "min": min(durations),
            "max": max(durations),
            "avg": sum(durations) / len(durations),
            "count": len(durations),
        }

def time_function(timer_getter: Callable[[], TimingStats]) -> Callable:
    """Decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            timer = timer_getter(args[0])
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                duration = end_time - start_time  # Keep in seconds
                timer.add_timing(FunctionTiming(
                    name=func.__name__,
                    duration=duration,
                    start_time=start_time,
                    end_time=end_time,
                ))
        return wrapper
    return decorator
