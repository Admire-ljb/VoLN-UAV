from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from voln_uav.common.geometry import l2


Vec3 = Sequence[float]


@dataclass
class StationaryDetector:
    """Detect when the vehicle stays inside a small radius for too long."""

    timeout_sec: float
    radius_m: float
    anchor_position: list[float] | None = None
    anchor_time: float | None = None
    duration_sec: float = 0.0

    def update(self, position: Vec3, now: float) -> bool:
        if float(self.timeout_sec) <= 0.0:
            self.duration_sec = 0.0
            return False

        current = [float(v) for v in position[:3]]
        if self.anchor_position is None or self.anchor_time is None:
            self.anchor_position = current
            self.anchor_time = float(now)
            self.duration_sec = 0.0
            return False

        if l2(self.anchor_position, current) > max(float(self.radius_m), 0.0):
            self.anchor_position = current
            self.anchor_time = float(now)
            self.duration_sec = 0.0
            return False

        self.duration_sec = max(0.0, float(now) - float(self.anchor_time))
        return self.duration_sec >= float(self.timeout_sec)
