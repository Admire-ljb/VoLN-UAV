from __future__ import annotations

import math
from typing import Any, Sequence


PROPRIO_SCHEMA = "body_linear_angular_relative_v1"
PROPRIO_DIM = 9
DEFAULT_SAMPLE_INTERVAL_SEC = 2.0


def wrap_angle_rad(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _normalized_quaternion(
    orientation_xyzw: Sequence[float] | None,
) -> tuple[float, float, float, float] | None:
    if orientation_xyzw is None or len(orientation_xyzw) < 4:
        return None
    x, y, z, w = (float(orientation_xyzw[index]) for index in range(4))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return x / norm, y / norm, z / norm, w / norm


def world_vector_to_body(
    vector: Sequence[float],
    yaw_rad: float,
    orientation_xyzw: Sequence[float] | None = None,
) -> list[float]:
    """Rotate a NED/world displacement into the current UAV body frame."""
    x, y, z = (float(vector[index]) for index in range(3))
    quaternion = _normalized_quaternion(orientation_xyzw)
    if quaternion is not None:
        qx, qy, qz, qw = quaternion
        # AirSim reports the body-to-NED quaternion; transpose its rotation
        # matrix to map a world vector into the body frame.
        return [
            (1.0 - 2.0 * (qy * qy + qz * qz)) * x
            + 2.0 * (qx * qy + qz * qw) * y
            + 2.0 * (qx * qz - qy * qw) * z,
            2.0 * (qx * qy - qz * qw) * x
            + (1.0 - 2.0 * (qx * qx + qz * qz)) * y
            + 2.0 * (qy * qz + qx * qw) * z,
            2.0 * (qx * qz + qy * qw) * x
            + 2.0 * (qy * qz - qx * qw) * y
            + (1.0 - 2.0 * (qx * qx + qy * qy)) * z,
        ]
    cosine = math.cos(float(yaw_rad))
    sine = math.sin(float(yaw_rad))
    return [
        cosine * x + sine * y,
        -sine * x + cosine * y,
        z,
    ]


def body_vector_to_world(
    vector: Sequence[float],
    yaw_rad: float,
    orientation_xyzw: Sequence[float] | None = None,
) -> list[float]:
    """Rotate a UAV body-frame displacement into the NED/world frame."""
    x, y, z = (float(vector[index]) for index in range(3))
    quaternion = _normalized_quaternion(orientation_xyzw)
    if quaternion is not None:
        qx, qy, qz, qw = quaternion
        return [
            (1.0 - 2.0 * (qy * qy + qz * qz)) * x
            + 2.0 * (qx * qy - qz * qw) * y
            + 2.0 * (qx * qz + qy * qw) * z,
            2.0 * (qx * qy + qz * qw) * x
            + (1.0 - 2.0 * (qx * qx + qz * qz)) * y
            + 2.0 * (qy * qz - qx * qw) * z,
            2.0 * (qx * qz - qy * qw) * x
            + 2.0 * (qy * qz + qx * qw) * y
            + (1.0 - 2.0 * (qx * qx + qy * qy)) * z,
        ]
    cosine = math.cos(float(yaw_rad))
    sine = math.sin(float(yaw_rad))
    return [
        cosine * x - sine * y,
        sine * x + cosine * y,
        z,
    ]


def world_point_to_body(
    point_world: Sequence[float],
    origin_world: Sequence[float],
    yaw_rad: float,
    orientation_xyzw: Sequence[float] | None = None,
) -> list[float]:
    delta = [float(point_world[index]) - float(origin_world[index]) for index in range(3)]
    return world_vector_to_body(delta, yaw_rad, orientation_xyzw)


def body_point_to_world(
    point_body: Sequence[float],
    origin_world: Sequence[float],
    yaw_rad: float,
    orientation_xyzw: Sequence[float] | None = None,
) -> list[float]:
    delta = body_vector_to_world(point_body, yaw_rad, orientation_xyzw)
    return [float(origin_world[index]) + delta[index] for index in range(3)]


def _timestamp_seconds(state: dict[str, Any]) -> float | None:
    raw = state.get("raw")
    value = raw.get("timestamp") if isinstance(raw, dict) else state.get("timestamp")
    if value is None:
        return None
    timestamp = float(value)
    # AirSim timestamps are commonly reported in nanoseconds.
    if abs(timestamp) > 1e12:
        timestamp /= 1e9
    return timestamp


def _delta_time(
    state: dict[str, Any],
    previous_state: dict[str, Any] | None,
    default_interval_sec: float,
) -> float:
    if previous_state is None:
        return float(default_interval_sec)
    current_timestamp = _timestamp_seconds(state)
    previous_timestamp = _timestamp_seconds(previous_state)
    if current_timestamp is not None and previous_timestamp is not None:
        delta = current_timestamp - previous_timestamp
        if math.isfinite(delta) and delta > 1e-6:
            return delta
    return float(default_interval_sec)


def encode_proprioception(
    state: dict[str, Any],
    previous_state: dict[str, Any] | None = None,
    *,
    default_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
) -> list[float]:
    """Return the deployable 9-D proprio vector used by training and evaluation.

    Layout: body-frame linear velocity (3), body-frame angular velocity (3),
    and body-frame relative odometry since the previous observation (3).
    Absolute world position is never returned.
    """
    if state.get("proprio_schema") == PROPRIO_SCHEMA:
        imu = [float(value) for value in state.get("imu", [])]
        odometry = [float(value) for value in state.get("odometry", [])]
        vector = imu[:6] + odometry[:3]
        if len(vector) != PROPRIO_DIM or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"Invalid {PROPRIO_SCHEMA} proprioception payload")
        return vector

    position = state.get("position")
    previous_position = previous_state.get("position") if previous_state is not None else None
    if not isinstance(position, Sequence) or len(position) < 3:
        raise ValueError(
            "Legacy state lacks a world pose needed to derive non-leaking relative proprioception"
        )
    yaw = float(state.get("yaw", 0.0))
    if not isinstance(previous_position, Sequence) or len(previous_position) < 3:
        body_displacement = [0.0, 0.0, 0.0]
        angular_velocity = [0.0, 0.0, 0.0]
    else:
        world_displacement = [
            float(position[index]) - float(previous_position[index])
            for index in range(3)
        ]
        body_displacement = world_vector_to_body(
            world_displacement,
            yaw,
            state.get("orientation"),
        )
        delta_time = _delta_time(state, previous_state, default_interval_sec)
        previous_yaw = float(previous_state.get("yaw", yaw))
        angular_velocity = [0.0, 0.0, wrap_angle_rad(yaw - previous_yaw) / delta_time]
    delta_time = _delta_time(state, previous_state, default_interval_sec)
    linear_velocity = [value / delta_time for value in body_displacement]
    vector = linear_velocity + angular_velocity + body_displacement
    if len(vector) != PROPRIO_DIM or not all(math.isfinite(value) for value in vector):
        raise ValueError("Derived proprioception contains non-finite values")
    return vector
