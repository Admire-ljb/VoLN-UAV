from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voln_uav.common.geometry import l2
from voln_uav.common.io import ensure_dir
from voln_uav.simulators.beacon_placement import SIGN_ASSET_BASE, TARGET_TAG, plan_route_beacons, stable_episode_seed

TARGET_ASSET_ALIASES = (TARGET_TAG, "target", "people", "person")


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
        takeoff_timeout_sec: float = 10.0,
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
        self.takeoff_timeout_sec = float(takeoff_timeout_sec)
        self.client = airsim.MultirotorClient(ip=ip, port=int(port))
        self._beacon_object_cache: list[str] | None = None
        self._active_beacon_names: list[str] = []

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
            self.cleanup_beacons()
        except Exception:
            pass
        try:
            self.client.hoverAsync().join()
        except Exception:
            pass
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except Exception:
            pass

    def reset_to_episode_start(self, episode: dict[str, Any], ensure_flying: bool = True) -> None:
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
        if ensure_flying:
            self._ensure_flying_at_start([x, y, z])

    def _ensure_flying_at_start(self, position: list[float]) -> None:
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        x, y, z = [float(v) for v in position]

        def move_to_start() -> None:
            self.client.moveToPositionAsync(
                x,
                y,
                z,
                max(self.speed, 1.0),
                timeout_sec=max(self.move_timeout_sec, self.takeoff_timeout_sec),
                drivetrain=self.airsim.DrivetrainType.MaxDegreeOfFreedom,
            ).join()
            time.sleep(self.settle_sec)

        state = self.client.getMultirotorState()
        if int(getattr(state, "landed_state", 0)) == int(self.airsim.LandedState.Landed):
            self.client.takeoffAsync(timeout_sec=self.takeoff_timeout_sec).join()
            time.sleep(self.settle_sec)
        move_to_start()

        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        dist = math.sqrt((float(pos.x_val) - x) ** 2 + (float(pos.y_val) - y) ** 2 + (float(pos.z_val) - z) ** 2)
        if dist > 1.0 or int(getattr(state, "landed_state", 0)) == int(self.airsim.LandedState.Landed):
            self.client.takeoffAsync(timeout_sec=self.takeoff_timeout_sec).join()
            time.sleep(self.settle_sec)
            move_to_start()

    def _scene_beacon_objects(self) -> list[str]:
        if self._beacon_object_cache is None:
            bases = list(SIGN_ASSET_BASE.values()) + list(TARGET_ASSET_ALIASES)
            objects = self.client.simListSceneObjects()
            self._beacon_object_cache = [name for name in objects if any(base.lower() in name.lower() for base in bases)]
        return list(self._beacon_object_cache)

    def _pick_beacon_object(self, tag: str, used: set[str], rng: random.Random) -> str | None:
        names = [name for name in self._scene_beacon_objects() if name not in used]
        bases = list(TARGET_ASSET_ALIASES) if tag == TARGET_TAG else [SIGN_ASSET_BASE.get(tag, tag)]
        candidates = [name for name in names if any(base.lower() in name.lower() for base in bases)]
        rng.shuffle(candidates)
        return candidates[0] if candidates else None

    def current_position(self) -> list[float]:
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return [float(pos.x_val), float(pos.y_val), float(pos.z_val)]

    def _pose_from_plan(self, placement: dict[str, Any]) -> Any:
        x, y, z = [float(v) for v in placement["position"][:3]]
        qx, qy, qz, qw = yaw_to_quaternion(float(placement.get("yaw_rad", 0.0)))
        return self.airsim.Pose(
            self.airsim.Vector3r(x, y, z),
            self.airsim.Quaternionr(qx, qy, qz, qw),
        )

    def _hide_object(self, name: str, rng: random.Random, hidden_z: float) -> bool:
        pose = self.airsim.Pose(
            self.airsim.Vector3r(rng.uniform(-250.0, -150.0), rng.uniform(-250.0, -150.0), float(hidden_z)),
            self.airsim.Quaternionr(0.0, 0.0, 0.0, 1.0),
        )
        try:
            self.client.simSetObjectPose(name, pose)
            return True
        except Exception:
            return False

    def cleanup_beacons(self, all_available: bool = False, seed: int = 0, hidden_z: float = -500.0) -> None:
        names = self._scene_beacon_objects() if all_available else list(self._active_beacon_names)
        rng = random.Random(int(seed))
        for name in names:
            self._hide_object(name, rng, hidden_z)
        self._active_beacon_names = []

    def place_beacons_for_episode(self, episode: dict[str, Any], config: dict[str, Any] | None = None, seed: int = 0) -> list[dict[str, Any]]:
        cfg = config or {}
        if not bool(cfg.get("enabled", False)):
            return []
        episode_id = str(episode.get("episode_id", "episode"))
        rng_seed = stable_episode_seed(int(cfg.get("random_seed", seed)), episode_id)
        rng = random.Random(rng_seed)
        hidden_z = float(cfg.get("hidden_z", -500.0))
        if bool(cfg.get("hide_all_available", True)):
            self.cleanup_beacons(all_available=True, seed=rng_seed, hidden_z=hidden_z)
        else:
            self.cleanup_beacons(all_available=False, seed=rng_seed, hidden_z=hidden_z)

        used: set[str] = set()
        materialized: list[dict[str, Any]] = []
        for planned in plan_route_beacons(episode, cfg, base_seed=seed):
            item = dict(planned)
            obj_name = self._pick_beacon_object(str(item["tag"]), used, rng)
            item["object_name"] = obj_name
            item["placed"] = False
            if obj_name is None:
                item["error"] = "no matching AirSim scene object"
                materialized.append(item)
                continue
            try:
                self.client.simSetObjectPose(obj_name, self._pose_from_plan(item))
                used.add(obj_name)
                self._active_beacon_names.append(obj_name)
                item["placed"] = True
            except Exception as exc:
                item["error"] = repr(exc)
            materialized.append(item)
        return materialized

    def _image_type(self) -> int:
        return getattr(self.airsim.ImageType, self.image_type_name)

    def capture_image(self, episode_id: str, step_idx: int) -> str:
        out_dir = ensure_dir(self.work_dir / "observations" / episode_id)
        out_path = out_dir / f"{step_idx:06d}.png"
        image_bytes = self.client.simGetImage(self.camera, self._image_type())
        if not image_bytes:
            raise RuntimeError(f"AirSim returned an empty image for camera {self.camera}")
        out_path.write_bytes(image_bytes)
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
