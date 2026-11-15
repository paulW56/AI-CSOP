"""Frame-rate measurement."""

import time


class FPSMeter:
    """Smoothed frames-per-second estimate.

    Uses an exponential moving average so the on-screen number doesn't jitter
    every frame. `alpha` is the weight given to the newest frame (0-1).
    """

    def __init__(self, alpha: float = 0.1, clock=time.perf_counter):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._clock = clock
        self._last = None
        self.fps = 0.0

    def tick(self) -> float:
        """Call once per frame. Returns the current smoothed FPS."""
        now = self._clock()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                instant = 1.0 / dt
                self.fps = instant if self.fps == 0 else (
                    self.alpha * instant + (1 - self.alpha) * self.fps
                )
        self._last = now
        return self.fps
