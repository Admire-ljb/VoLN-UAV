from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voln_uav.common.geometry import l2, l2_xy
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
        max_teleport_step_m: float | None = None,
        max_teleport_vertical_step_m: float | None = None,
        teleport_keep_initial_height: bool = False,
        teleport_hover_after_setpose: bool = True,
        teleport_pause_after_setpose: bool = False,
        teleport_zero_velocity: bool = True,
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
        self.max_teleport_step_m = (
            float(max_teleport_step_m) if max_teleport_step_m is not None else max(float(speed), 0.1)
        )
        self.max_teleport_vertical_step_m = (
            float(max_teleport_vertical_step_m) if max_teleport_vertical_step_m is not None else 0.5
        )
        self.teleport_keep_initial_height = bool(teleport_keep_initial_height)
        self.teleport_hover_after_setpose = bool(teleport_hover_after_setpose)
        self.teleport_pause_after_setpose = bool(teleport_pause_after_setpose)
        self.teleport_zero_velocity = bool(teleport_zero_velocity)
        self.teleport_initial_z: float | None = None
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
                self._set_sim_pause(False)
                return
            except Exception as exc:  # pragma: no cover - depends on simulator timing
                last_error = exc
                time.sleep(1.0)
        raise RuntimeError(f"Could not connect to AirSim at {self.ip}:{self.port}") from last_error

    def close(self) -> None:
        self._set_sim_pause(False)
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

    def _set_sim_pause(self, paused: bool) -> bool:
        try:
            self.client.simPause(bool(paused))
            return True
        except Exception:
            return False

    def _zero_vehicle_motion(self, pose: Any) -> bool:
        try:
            kinematics = self.airsim.KinematicsState()
            kinematics.position = pose.position
            kinematics.orientation = pose.orientation
            kinematics.linear_velocity = self.airsim.Vector3r(0.0, 0.0, 0.0)
            kinematics.angular_velocity = self.airsim.Vector3r(0.0, 0.0, 0.0)
            kinematics.linear_acceleration = self.airsim.Vector3r(0.0, 0.0, 0.0)
            kinematics.angular_acceleration = self.airsim.Vector3r(0.0, 0.0, 0.0)
            self.client.simSetKinematics(kinematics, True)
            return True
        except Exception:
            return False

    def _stabilize_after_setpose(
        self,
        pose: Any,
        hover_after_setpose: bool | None = None,
        pause_after_setpose: bool | None = None,
        zero_velocity: bool | None = None,
    ) -> dict[str, bool]:
        hover = self.teleport_hover_after_setpose if hover_after_setpose is None else bool(hover_after_setpose)
        zero = self.teleport_zero_velocity if zero_velocity is None else bool(zero_velocity)
        status = {
            "zero_velocity": False,
            "hovered": False,
            "paused": False,
        }
        if zero:
            status["zero_velocity"] = self._zero_vehicle_motion(pose)
        if hover:
            try:
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                self.client.hoverAsync().join()
                status["hovered"] = True
            except Exception:
                status["hovered"] = False
        if self.settle_sec > 0.0:
            time.sleep(self.settle_sec)
        return status

    def reset_to_episode_start(self, episode: dict[str, Any], ensure_flying: bool = True) -> dict[str, bool]:
        first = episode["states"][0]
        x, y, z = [float(v) for v in first["position"]]
        yaw = float(first.get("yaw", 0.0))
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        pose = self.airsim.Pose(
            self.airsim.Vector3r(x, y, z),
            self.airsim.Quaternionr(qx, qy, qz, qw),
        )
        self._set_sim_pause(False)
        self.client.simSetVehiclePose(pose, True)
        self.teleport_initial_z = z
        reset_stabilization = {"zero_velocity": False, "hovered": False, "paused": False}
        if ensure_flying:
            if self.settle_sec > 0.0:
                time.sleep(self.settle_sec)
            self._set_sim_pause(False)
            self._ensure_flying_at_start([x, y, z])
        else:
            reset_stabilization = self._stabilize_after_setpose(pose)
        return reset_stabilization

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

    def _clamp_teleport_waypoint(
        self,
        current_position: list[float],
        waypoint: list[float],
        max_step_m: float | None = None,
        max_vertical_step_m: float | None = None,
        keep_initial_height: bool | None = None,
    ) -> dict[str, Any]:
        current = [float(v) for v in current_position[:3]]
        requested = [float(v) for v in waypoint[:3]]
        target = list(requested)
        invalid_waypoint = not all(math.isfinite(v) for v in target)
        if invalid_waypoint:
            target = list(current)

        step_limit = float(self.max_teleport_step_m if max_step_m is None else max_step_m)
        if not math.isfinite(step_limit) or step_limit <= 0.0:
            step_limit = max(float(self.speed), 0.1)

        vertical_limit = float(
            self.max_teleport_vertical_step_m if max_vertical_step_m is None else max_vertical_step_m
        )
        if not math.isfinite(vertical_limit) or vertical_limit <= 0.0:
            vertical_limit = 0.5

        keep_height = False
        height_locked = False
        height_adjusted = False
        requested_vertical_delta = requested[2] - current[2] if not invalid_waypoint else 0.0
        target_vertical_delta = target[2] - current[2]
        vertical_clipped = abs(target_vertical_delta) > vertical_limit
        if vertical_clipped:
            target[2] = current[2] + math.copysign(vertical_limit, target_vertical_delta)

        adjusted_delta = [target[i] - current[i] for i in range(3)]
        requested_distance = math.sqrt(sum(v * v for v in adjusted_delta))
        distance_clipped = requested_distance > step_limit > 0.0
        if distance_clipped:
            scale = step_limit / max(requested_distance, 1e-9)
            executed = [current[i] + adjusted_delta[i] * scale for i in range(3)]
        else:
            executed = target
        executed_distance = l2(current, executed)
        return {
            "requested_waypoint": requested,
            "height_limited_waypoint": target,
            "executed_waypoint": executed,
            "raw_requested_distance_m": l2(current, requested) if not invalid_waypoint else 0.0,
            "requested_distance_m": requested_distance,
            "executed_distance_m": executed_distance,
            "requested_vertical_delta_m": requested_vertical_delta,
            "executed_vertical_delta_m": executed[2] - current[2],
            "max_teleport_step_m": step_limit,
            "max_teleport_vertical_step_m": vertical_limit,
            "teleport_initial_z": self.teleport_initial_z,
            "teleport_keep_initial_height": keep_height,
            "teleport_height_locked": height_locked,
            "teleport_height_adjusted": height_adjusted,
            "teleport_vertical_clipped": vertical_clipped,
            "teleport_clipped": distance_clipped or vertical_clipped or height_adjusted or invalid_waypoint,
            "invalid_waypoint": invalid_waypoint,
        }

    def move_to_waypoint(
        self,
        current_position: list[float],
        waypoint: list[float],
        control_mode: str = "move_to_position",
        max_teleport_step_m: float | None = None,
        max_teleport_vertical_step_m: float | None = None,
        teleport_keep_initial_height: bool | None = None,
        teleport_hover_after_setpose: bool | None = None,
        teleport_pause_after_setpose: bool | None = None,
        teleport_zero_velocity: bool | None = None,
    ) -> dict[str, Any]:
        x, y, z = [float(v) for v in waypoint[:3]]
        if control_mode == "teleport":
            movement = self._clamp_teleport_waypoint(
                current_position,
                waypoint,
                max_step_m=max_teleport_step_m,
                max_vertical_step_m=max_teleport_vertical_step_m,
                keep_initial_height=teleport_keep_initial_height,
            )
            x, y, z = [float(v) for v in movement["executed_waypoint"]]
            yaw = math.radians(yaw_to_target(current_position, [x, y, z]))
            qx, qy, qz, qw = yaw_to_quaternion(yaw)
            pose = self.airsim.Pose(
                self.airsim.Vector3r(x, y, z),
                self.airsim.Quaternionr(qx, qy, qz, qw),
            )
            self._set_sim_pause(False)
            self.client.simSetVehiclePose(pose, True)
            movement["teleport_stabilization"] = self._stabilize_after_setpose(
                pose,
                hover_after_setpose=teleport_hover_after_setpose,
                pause_after_setpose=teleport_pause_after_setpose,
                zero_velocity=teleport_zero_velocity,
            )
            movement["control_mode"] = control_mode
            return movement

        self._set_sim_pause(False)
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
        return {
            "requested_waypoint": [x, y, z],
            "height_limited_waypoint": [x, y, z],
            "executed_waypoint": [x, y, z],
            "raw_requested_distance_m": l2(current_position, [x, y, z]),
            "requested_distance_m": l2(current_position, [x, y, z]),
            "executed_distance_m": l2(current_position, [x, y, z]),
            "requested_vertical_delta_m": z - float(current_position[2]),
            "executed_vertical_delta_m": z - float(current_position[2]),
            "max_teleport_step_m": None,
            "max_teleport_vertical_step_m": None,
            "teleport_initial_z": self.teleport_initial_z,
            "teleport_keep_initial_height": False,
            "teleport_height_locked": False,
            "teleport_vertical_clipped": False,
            "teleport_clipped": False,
            "invalid_waypoint": False,
            "control_mode": control_mode,
        }

    def teleport_reference_prefix(
        self,
        episode: dict[str, Any],
        count: int = 3,
        max_teleport_step_m: float | None = None,
        max_teleport_vertical_step_m: float | None = None,
        teleport_keep_initial_height: bool | None = None,
        teleport_hover_after_setpose: bool | None = None,
        teleport_pause_after_setpose: bool | None = None,
        teleport_zero_velocity: bool | None = None,
    ) -> list[dict[str, Any]]:
        states = list(episode.get("states", []))[: max(int(count), 0)]
        movements: list[dict[str, Any]] = []
        for ref_idx, state in enumerate(states):
            current = self.current_position()
            target = [float(v) for v in state["position"][:3]]
            movement = self.move_to_waypoint(
                current,
                target,
                control_mode="teleport",
                max_teleport_step_m=max_teleport_step_m,
                max_teleport_vertical_step_m=max_teleport_vertical_step_m,
                teleport_keep_initial_height=teleport_keep_initial_height,
                teleport_hover_after_setpose=teleport_hover_after_setpose,
                teleport_pause_after_setpose=teleport_pause_after_setpose,
                teleport_zero_velocity=teleport_zero_velocity,
            )
            movement.update(
                {
                    "controller": "reference_bootstrap",
                    "reference_index": ref_idx,
                    "position_before": current,
                    "position_after": self.current_position(),
                    "reference_waypoint": target,
                }
            )
            movements.append(movement)
        return movements

    @staticmethod
    def reached_goal(position: list[float], episode: dict[str, Any], radius: float) -> bool:
        return l2_xy(position, episode["states"][-1]["position"]) <= float(radius)
