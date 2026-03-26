from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class RouteState:
    t: int
    position: list[float]
    yaw: float
    image: str
    imu: list[float]
    odometry: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BeaconAnnotation:
    beacon_id: str
    semantic_type: str
    relevant: bool
    visible: bool
    state_index: int | None = None
    template_image: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
