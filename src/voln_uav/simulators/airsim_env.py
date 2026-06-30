from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voln_uav.common.geometry import l2
from voln_uav.common.io import ensure_dir


@dataclass
class AirSimStep:
    state: dict[str, Any]
    position: list[float]
    collision: bool


def yaw_to_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    half = float(yaw_rad) * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_target(src: list[float], dst: list[float]) -> float:
    return math.degrees(math.atan2(float(dst[1]) - float(src[1]), float(dst[0]) - float(src[0])))


class AirSimRouteEnv:
    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 41451,
        camera: str = "FrontCamera",
        image_type: str = "Scene",
        work_dir: str | Path = "work_dirs/airsim_eval",
        speed: float = 3.0,
        move_timeout_sec: float = 15.0,
        settle_sec: float = 0.05,
    ) -> None:
        try:
            import airsim
        except Exception as exc:  # pragma: no cover - depends on local AirSim install
            raise ImportError("Install the AirSim Python package to run environment evaluation.") from exc

        self.airsim = airsim
        self.ip = ip
        self.port = int(port)
        self.camera = camera
        self.image_type_name = image_type
        self.work_dir = ensure_dir(work_dir)
        self.speed = float(speed)
        self.move_timeout_sec = float(move_timeout_sec)
        self.settle_sec = float(settle_sec)
        self.client = airsim.MultirotorClient(ip=ip, port=int(port))

    def connect(self, timeout_sec: float = 60.0) -> None:
        deadline = time.time() + float(timeout_sec)
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                self.client.confirmConnection()
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                return
            except Exception as exc:  # pragma: no cover - depends on simulator timing
                last_error = exc
                time.sleep(1.0)
        raise RuntimeError(f"Could not connect to AirSim at {self.ip}:{self.port}") from last_error

    def close(self) -> None:
        try:
            self.client.hoverAsync().join()
        except Exception:
            pass
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except Exception:
            pass

    def reset_to_episode_start(self, episode: dict[str, Any]) -> None:
        first = episode["states"][0]
        x, y, z = [float(v) for v in first["position"]]
        yaw = float(first.get("yaw", 0.0))
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        pose = self.airsim.Pose(
            self.airsim.Vector3r(x, y, z),
            self.airsim.Quaternionr(qx, qy, qz, qw),
        )
        self.client.simSetVehiclePose(pose, True)
        time.sleep(self.settle_sec)

    def _image_type(self) -> int:
        return getattr(self.airsim.ImageType, self.image_type_name)

    def capture_image(self, episode_id: str, step_idx: int) -> str:
        out_dir = ensure_dir(self.work_dir / "observations" / episode_id)
        out_path = out_dir / f"{step_idx:06d}.png"
        request = self.airsim.ImageRequest(self.camera, self._image_type(), pixels_as_float=False, compress=True)
        responses = self.client.simGetImages([request])
        if not responses or not responses[0].image_data_uint8:
            raise RuntimeError(f"AirSim returned an empty image for camera {self.camera}")
        out_path.write_bytes(responses[0].image_data_uint8)
        return str(out_path)

    def current_state(self, episode_id: str, step_idx: int, previous_position: list[float] | None = None) -> AirSimStep:
        image = self.capture_image(episode_id, step_idx)
        state = self.client.getMultirotorState()
        kin = state.kinematics_estimated
        pos = kin.position
        ori = kin.orientation
        position = [float(pos.x_val), float(pos.y_val), float(pos.z_val)]
        yaw = yaw_from_quaternion(float(ori.x_val), float(ori.y_val), float(ori.z_val), float(ori.w_val))
        if previous_position is None:
            delta = [0.0, 0.0, 0.0]
        else:
            delta = [position[i] - previous_position[i] for i in range(3)]
        lin = kin.linear_velocity
        ang = kin.angular_velocity
        collision_info = self.client.simGetCollisionInfo()
        collision = bool(getattr(collision_info, "has_collided", False))
        item = {
            "position": position,
            "yaw": yaw,
            "image": image,
            "imu": [
                float(getattr(lin, "x_val", delta[0])),
                float(getattr(lin, "y_val", delta[1])),
                float(getattr(lin, "z_val", delta[2])),
                float(getattr(ang, "x_val", 0.0)),
                float(getattr(ang, "y_val", 0.0)),
                float(getattr(ang, "z_val", 0.0)),
            ],
            "odometry": position,
        }
        return AirSimStep(state=item, position=position, collision=collision)

    def move_to_waypoint(self, current_position: list[float], waypoint: list[float], control_mode: str = "move_to_position") -> None:
        x, y, z = [float(v) for v in waypoint[:3]]
        if control_mode == "teleport":
            yaw = math.radians(yaw_to_target(current_position, [x, y, z]))
            qx, qy, qz, qw = yaw_to_quaternion(yaw)
            pose = self.airsim.Pose(
                self.airsim.Vector3r(x, y, z),
                self.airsim.Quaternionr(qx, qy, qz, qw),
            )
            self.client.simSetVehiclePose(pose, True)
            time.sleep(self.settle_sec)
            return

        yaw_mode = self.airsim.YawMode(is_rate=False, yaw_or_rate=yaw_to_target(current_position, [x, y, z]))
        self.client.moveToPositionAsync(
            x,
            y,
            z,
            self.speed,
            timeout_sec=self.move_timeout_sec,
            drivetrain=self.airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=yaw_mode,
            lookahead=1.0,
        ).join()
        time.sleep(self.settle_sec)

    @staticmethod
    def reached_goal(position: list[float], episode: dict[str, Any], radius: float) -> bool:
        return l2(position, episode["states"][-1]["position"]) <= float(radius)
