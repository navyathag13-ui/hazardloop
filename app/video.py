"""Video mode: decide per clip, not per frame.

A single frame can show a blurred head, a worker mid-turn or a helmet hidden
by a beam, so one-frame alerts flicker. SmoothedAlerts raises an alert only
when a violation is reported in `need` of the last `window` sampled frames
(default 2 of 3, which at 1 frame per second means it has to persist for about
2 seconds).
"""
from collections import deque


class SmoothedAlerts:
    def __init__(self, window: int = 3, need: int = 2):
        if not 1 <= need <= window:
            raise ValueError("need must be between 1 and window")
        self.need = need
        self.recent: deque[bool] = deque(maxlen=window)

    def update(self, violation: bool) -> bool:
        self.recent.append(bool(violation))
        return sum(self.recent) >= self.need
