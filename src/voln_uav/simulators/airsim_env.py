from __future__ import annotations

import math
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voln_uav.common.geometry import l2, l2_xy
from voln_uav.common.io import ensure_dir
from voln_uav.common.navigation_frames import PROPRIO_SCHEMA, world_vector_to_body
from voln_uav.simulators.beacon_placement import (
    SIGN_ASSET_ALIASES,
    SIGN_ASSET_BASE,
    TARGET_TAG,

    normalize_beacon_render_mode,
    plan_route_beacons,
    stable_episode_seed,
)

TARGET_ASSET_ALIASES = (TARGET_TAG, "target", "people", "person")


def _asset_name_matches_alias(name: str, alias: str) -> bool:
    """Match an Unreal asset token without matching inside unrelated words."""
    pattern = rf"(?<![a-z0-9]){re.escape(alias.casefold())}(?=$|[_-]|\d)"
    return re.search(pattern, name.casefold()) is not None


def _text_asset_matches_tag(name: str, tag: str) -> bool:
    """Match one exact text-sign family, including its optional white suffix."""
    pattern = rf"^label_{re.escape(str(tag).casefold())}(?:w)?(?=$|[_-]|\d)"
    return re.search(pattern, name.casefold()) is not None


def _target_asset_kind(name: str) -> str:
    """Classify a selected target scene object as a person or a sign."""
    people_aliases = ("target_people", "people", "person")
    if any(_asset_name_matches_alias(name, alias) for alias in people_aliases):
        return "people"
    return "sign"


def _adjust_target_pose_for_asset(
    placement: dict[str, Any],
    object_name: str,
    config: dict[str, Any],
) -> None:
    """Apply asset-specific target height and front-face orientation."""
    kind = _target_asset_kind(object_name)
    placement["target_asset_kind"] = kind
    if kind == "people":
        # People assets use physics and may settle downward. Spawn them above
        # the planned point so their collision capsule cannot begin underground.
        vertical_offset = float(config.get("target_people_vertical_ned_m", -100.0))
        position = [float(value) for value in placement["position"][:3]]
        position[2] += vertical_offset
        placement["position"] = position
        placement["target_asset_vertical_ned_offset_m"] = vertical_offset
        return

    # NED +Z points down, so a positive offset inserts a tall sign lower while
    # people remain at the planned target position.
    vertical_offset = float(config.get("target_sign_vertical_ned_m", 10.0))
    position = [float(value) for value in placement["position"][:3]]
    position[2] += vertical_offset
    placement["position"] = position
    placement["target_asset_vertical_ned_offset_m"] = vertical_offset

    # Sign meshes face along a lateral local axis rather than the people
    # asset's forward axis. Rebuild yaw from the recorded final-view heading.
    yaw_add_deg = float(config.get("target_sign_yaw_add_deg", 90.0))
    view_yaw_deg = float(placement.get("target_direction_yaw_deg", 0.0))
    yaw_rad = math.radians(view_yaw_deg + yaw_add_deg)
    placement["yaw_rad"] = (yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
    placement["target_asset_yaw_add_deg"] = yaw_add_deg


def _text_asset_yaw_correction_deg(
    tag: str,
    config: dict[str, Any],
) -> float:
    """Return the mesh-front correction for NED positive-yaw text signs."""
    should_flip = (
        bool(config.get("text_yaw_flip_right_turn_signs", False))
        and str(tag).startswith("right")
    )
    return float(config.get("text_yaw_flip_deg", 180.0)) if should_flip else 0.0


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


def split_waypoints_by_heading(
    points: list[list[float]],
    turn_threshold_deg: float = 15.0,
) -> list[list[list[float]]]:
    normalized = [[float(value) for value in point[:3]] for point in points]
    if len(normalized) < 2:
        return [normalized] if normalized else []

    def segment_heading(start: list[float], end: list[float]) -> float | None:
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        if math.hypot(dx, dy) <= 0.25:
            return None
        return math.degrees(math.atan2(dy, dx))

    chunks: list[list[list[float]]] = []
    start_index = 0
    baseline_heading: float | None = None
    for segment_index in range(len(normalized) - 1):
        heading = segment_heading(normalized[segment_index], normalized[segment_index + 1])
        if heading is None:
            continue
        if baseline_heading is None:
            baseline_heading = heading
            continue
        delta = (heading - baseline_heading + 180.0) % 360.0 - 180.0
        if abs(delta) >= float(turn_threshold_deg) and segment_index > start_index:
            chunks.append(normalized[start_index : segment_index + 1])
            start_index = segment_index
            baseline_heading = heading
    chunks.append(normalized[start_index:])
    return [chunk for chunk in chunks if len(chunk) >= 2]


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
        self._text_beacon_object_cache: list[str] | None = None
        self._active_beacon_names: list[str] = []
        self._passive_beacon_names: list[str] = []
        self._passive_scene_id: str | None = None

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

    def hover(self, position: list[float] | None = None) -> bool:
        self._set_sim_pause(False)
        try:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            hold = [float(v) for v in (position or self.current_position())[:3]]
            self.client.moveByVelocityAsync(
                0.0,
                0.0,
                0.0,
                1.0,
                drivetrain=self.airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=self.airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
            ).join()
            self.client.moveToPositionAsync(
                hold[0],
                hold[1],
                hold[2],
                max(self.speed, 1.0),
                timeout_sec=self.move_timeout_sec,
                drivetrain=self.airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=self.airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                lookahead=1.0,
            ).join()
            self.client.hoverAsync().join()
            if self.settle_sec > 0.0:
                time.sleep(self.settle_sec)
            return True
        except Exception:
            return False

    def close(self, keep_hovering: bool = False) -> None:
        self._set_sim_pause(False)
        if keep_hovering:
            self.hover()
            return
        try:
            self.cleanup_beacons(all_available=True)
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
        # A pose reset does not necessarily clear the velocity left by the
        # previous episode. Stabilize immediately so the vehicle cannot fall
        # while the scene cues are being placed.
        reset_stabilization = self._stabilize_after_setpose(pose)
        if ensure_flying:
            self._set_sim_pause(False)
            self._ensure_flying_at_start([x, y, z])
            # Takeoff/move-to-start can finish with a small tracking error.
            # Re-apply the exact dataset pose and hover before cue placement.
            self.client.simSetVehiclePose(pose, True)
            final_stabilization = self._stabilize_after_setpose(pose)
            for key, value in final_stabilization.items():
                reset_stabilization[key] = bool(reset_stabilization.get(key, False) or value)
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
            bases = [
                alias
                for aliases in SIGN_ASSET_ALIASES.values()
                for alias in aliases
            ] + list(TARGET_ASSET_ALIASES)
            objects = self.client.simListSceneObjects()
            self._beacon_object_cache = [
                name
                for name in objects
                if "label" not in name.casefold()
                and any(_asset_name_matches_alias(name, base) for base in bases)
            ]
        return list(self._beacon_object_cache)

    def _scene_text_beacon_objects(self) -> list[str]:
        if self._text_beacon_object_cache is None:
            self._text_beacon_object_cache = [
                name
                for name in self.client.simListSceneObjects()
                if "label" in name.casefold()
            ]
        return list(self._text_beacon_object_cache)

    def _pick_beacon_object(
        self,
        tag: str,
        used: set[str],
        rng: random.Random,
        render_mode: str = "direction",
    ) -> str | None:
        mode = normalize_beacon_render_mode(render_mode)
        if tag == TARGET_TAG:
            mode = "direction"
        if mode == "random":
            raise ValueError("Resolve random beacon mode before selecting a scene object")
        names = [
            name
            for name in (
                self._scene_text_beacon_objects()
                if mode == "text"
                else self._scene_beacon_objects()
            )
            if name not in used
        ]
        if tag == TARGET_TAG:
            candidates = [
                name for name in names if _target_asset_kind(name) == "people"
            ]
            if not candidates:
                candidates = [
                    name for name in names
                    if _asset_name_matches_alias(name, "target")
                ]
        else:
            if mode == "text":
                # The route planner already classifies cumulative turn angle;
                # preserve that exact class in the rendered text asset.
                candidates = [name for name in names if _text_asset_matches_tag(name, tag)]
            else:
                bases = list(SIGN_ASSET_ALIASES.get(tag, (SIGN_ASSET_BASE.get(tag, tag),)))
                candidates = [
                    name
                    for name in names
                    if any(_asset_name_matches_alias(name, base) for base in bases)
                ]
        rng.shuffle(candidates)
        return candidates[0] if candidates else None

    def _pick_active_beacon_object(
        self,
        tag: str,
        used: set[str],
        rng: random.Random,
        requested_mode: str,
    ) -> tuple[str | None, str]:
        """Select an active route cue while keeping random choices reproducible."""
        mode = normalize_beacon_render_mode(requested_mode)
        if tag == TARGET_TAG:
            return self._pick_beacon_object(tag, used, rng, "direction"), "target"

        modes = [mode]
        if mode == "random":
            modes = ["direction", "text"]
            rng.shuffle(modes)
        for candidate_mode in modes:
            name = self._pick_beacon_object(
                tag,
                used,
                rng,
                render_mode=candidate_mode,
            )
            if name is not None:
                return name, candidate_mode
        return None, modes[0]

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
        names = (
            self._scene_beacon_objects() + self._scene_text_beacon_objects()
            if all_available
            else list(self._active_beacon_names)
        )
        rng = random.Random(int(seed))
        for name in names:
            self._hide_object(name, rng, hidden_z)
        self._active_beacon_names = []
        if all_available:
            self._passive_beacon_names = []
            self._passive_scene_id = None

    def place_beacons_for_episode(self, episode: dict[str, Any], config: dict[str, Any] | None = None, seed: int = 0) -> list[dict[str, Any]]:
        cfg = config or {}
        if not bool(cfg.get("enabled", False)):
            return []
        episode_id = str(episode.get("episode_id", "episode"))
        rng_seed = stable_episode_seed(int(cfg.get("random_seed", seed)), episode_id)
        rng = random.Random(rng_seed)
        requested_render_mode = normalize_beacon_render_mode(cfg.get("render_mode"))
        hidden_z = float(cfg.get("hidden_z", -500.0))
        scene_id = str(episode.get("scene_id", "scene"))
        if self._passive_scene_id != scene_id:
            self.cleanup_beacons(all_available=True, seed=rng_seed, hidden_z=hidden_z)
        else:
            self.cleanup_beacons(all_available=False, seed=rng_seed, hidden_z=hidden_z)

        used: set[str] = set(self._passive_beacon_names)
        materialized: list[dict[str, Any]] = []
        if self._passive_scene_id != scene_id:
            for planned in episode.get("background_beacons", []) or []:
                item = {**planned, "kind": "passive_beacon"}
                obj_name = self._pick_beacon_object(
                    str(item.get("semantic_type", "road-sign")),
                    used,
                    rng,
                )
                item["object_name"] = obj_name
                item["placed"] = False
                if obj_name is not None and item.get("position") is not None:
                    try:
                        self.client.simSetObjectPose(obj_name, self._pose_from_plan(item))
                        used.add(obj_name)
                        self._passive_beacon_names.append(obj_name)
                        item["placed"] = True
                    except Exception as exc:
                        item["error"] = repr(exc)
                materialized.append(item)
            self._passive_scene_id = scene_id
        for planned in plan_route_beacons(episode, cfg, base_seed=seed):
            item = dict(planned)
            asset_identity = str(
                item.get(
                    "task_beacon_id",
                    f"{item.get('kind', 'beacon')}:{item.get('order', 0)}",
                )
            )
            asset_rng = random.Random(
                stable_episode_seed(rng_seed, f"{episode_id}:{asset_identity}")
            )
            obj_name, resolved_render_mode = self._pick_active_beacon_object(
                str(item["tag"]),
                used,
                asset_rng,
                requested_render_mode,
            )
            item["object_name"] = obj_name
            item["render_mode_requested"] = requested_render_mode
            item["render_mode"] = resolved_render_mode
            if item.get("kind") == "target" and obj_name is not None:
                _adjust_target_pose_for_asset(item, obj_name, cfg)
            if resolved_render_mode == "text":
                correction_deg = _text_asset_yaw_correction_deg(
                    str(item.get("tag", "")),
                    cfg,
                )
                item["asset_yaw_correction_deg"] = correction_deg
                if correction_deg:
                    item["yaw_rad"] = (
                        float(item.get("yaw_rad", 0.0))
                        + math.radians(correction_deg)
                        + math.pi
                    ) % (2.0 * math.pi) - math.pi
            item["placed"] = False
            if obj_name is None:
                item["error"] = (
                    "no matching AirSim scene object for "
                    f"beacon render mode {requested_render_mode!r}"
                )
                materialized.append(item)
                continue
            try:
                self.client.simSetObjectPose(obj_name, self._pose_from_plan(item))
                if item.get("kind") == "target" and cfg.get("target_scale") is not None:
                    scale_value = cfg.get("target_scale")
                    if isinstance(scale_value, (list, tuple)):
                        sx, sy, sz = [float(v) for v in scale_value[:3]]
                    else:
                        sx = sy = sz = float(scale_value)
                    item["scale"] = [sx, sy, sz]
                    item["scaled"] = bool(
                        self.client.simSetObjectScale(
                            obj_name,
                            self.airsim.Vector3r(sx, sy, sz),
                        )
                    )
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
        orientation = [
            float(ori.x_val),
            float(ori.y_val),
            float(ori.z_val),
            float(ori.w_val),
        ]
        body_linear_velocity = world_vector_to_body(
            [float(lin.x_val), float(lin.y_val), float(lin.z_val)],
            yaw,
            orientation,
        )
        body_odometry = world_vector_to_body(delta, yaw, orientation)
        collision_info = self.client.simGetCollisionInfo()
        collision = bool(getattr(collision_info, "has_collided", False))
        item = {
            "position": position,
            "yaw": yaw,
            "orientation": orientation,
            "image": image,
            "imu": [
                *body_linear_velocity,
                float(getattr(ang, "x_val", 0.0)),
                float(getattr(ang, "y_val", 0.0)),
                float(getattr(ang, "z_val", 0.0)),
            ],
            "odometry": body_odometry,
            "proprio_schema": PROPRIO_SCHEMA,
            "raw": {"timestamp": float(getattr(state, "timestamp", 0.0))},
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

    def set_reference_pose(
        self,
        state: dict[str, Any],
        zero_velocity: bool = True,
    ) -> dict[str, Any]:
        """Replay one recorded reference state as an exact AirSim vehicle pose."""
        position = [float(value) for value in state["position"][:3]]
        raw_orientation = state.get("orientation")
        orientation_source = "yaw"
        if isinstance(raw_orientation, (list, tuple)) and len(raw_orientation) >= 4:
            quaternion = [float(value) for value in raw_orientation[:4]]
            norm = math.sqrt(sum(value * value for value in quaternion))
            if all(math.isfinite(value) for value in quaternion) and norm > 1e-9:
                qx, qy, qz, qw = [value / norm for value in quaternion]
                orientation_source = "recorded_quaternion"
            else:
                yaw = float(state.get("yaw", 0.0))
                qx, qy, qz, qw = yaw_to_quaternion(yaw)
        else:
            yaw = float(state.get("yaw", 0.0))
            qx, qy, qz, qw = yaw_to_quaternion(yaw)

        pose = self.airsim.Pose(
            self.airsim.Vector3r(*position),
            self.airsim.Quaternionr(qx, qy, qz, qw),
        )
        self._set_sim_pause(False)
        self.client.simSetVehiclePose(pose, True)
        zeroed = self._zero_vehicle_motion(pose) if zero_velocity else False
        return {
            "requested_waypoint": position,
            "height_limited_waypoint": position,
            "executed_waypoint": position,
            "raw_requested_distance_m": 0.0,
            "requested_distance_m": 0.0,
            "executed_distance_m": 0.0,
            "requested_vertical_delta_m": 0.0,
            "executed_vertical_delta_m": 0.0,
            "max_teleport_step_m": None,
            "max_teleport_vertical_step_m": None,
            "teleport_keep_initial_height": False,
            "teleport_height_locked": False,
            "teleport_vertical_clipped": False,
            "teleport_clipped": False,
            "invalid_waypoint": False,
            "teleport_stabilization": {
                "zero_velocity": zeroed,
                "hovered": False,
                "paused": False,
            },
            "orientation_source": orientation_source,
            "control_mode": "setpose_replay",
        }

    def move_on_path(self, waypoints: list[list[float]], timeout_sec: float) -> dict[str, Any]:
        if not waypoints:
            return {"waypoint_count": 0, "control_mode": "move_on_path"}
        self._set_sim_pause(False)
        telemetry_client = self.airsim.MultirotorClient(ip=self.ip, port=self.port)

        def telemetry_position() -> list[float]:
            state = telemetry_client.getMultirotorState()
            position = state.kinematics_estimated.position
            return [
                float(position.x_val),
                float(position.y_val),
                float(position.z_val),
            ]

        initial_position = telemetry_position()
        telemetry_path = [initial_position]
        collision_samples = 0
        path = [
            self.airsim.Vector3r(float(point[0]), float(point[1]), float(point[2]))
            for point in waypoints
        ]
        future = self.client.moveOnPathAsync(
            path=path,
            velocity=self.speed,
            timeout_sec=max(float(timeout_sec), self.move_timeout_sec),
            drivetrain=self.airsim.DrivetrainType.ForwardOnly,
            yaw_mode=self.airsim.YawMode(is_rate=False),
            lookahead=3.0,
            adaptive_lookahead=1.0,
        )
        completed = threading.Event()
        motion_error: list[BaseException] = []

        def wait_for_motion() -> None:
            try:
                future.join()
            except BaseException as exc:  # pragma: no cover - depends on RPC failure mode
                motion_error.append(exc)
            finally:
                completed.set()

        waiter = threading.Thread(target=wait_for_motion, daemon=True)
        waiter.start()
        while not completed.wait(0.05):
            telemetry_path.append(telemetry_position())
            try:
                collision_samples += int(bool(telemetry_client.simGetCollisionInfo().has_collided))
            except Exception:
                pass
        waiter.join()
        final_position = telemetry_position()
        telemetry_path.append(final_position)
        if motion_error:
            raise RuntimeError("AirSim moveOnPath failed") from motion_error[0]

        endpoint_error_m = l2(final_position, waypoints[-1])
        path_follow_failed = endpoint_error_m > max(self.speed, 5.0)
        if self.settle_sec > 0.0:
            time.sleep(self.settle_sec)
        return {
            "waypoint_count": len(waypoints),
            "control_mode": "move_on_path",
            "path_api": "moveOnPathAsync",
            "speed_mps": self.speed,
            "drivetrain": "ForwardOnly",
            "lookahead": 3.0,
            "adaptive_lookahead": 1.0,
            "segment_count": 1,
            "failed_segment": 0 if path_follow_failed else None,
            "endpoint_error_m": endpoint_error_m,
            "telemetry_path": telemetry_path,
            "collision_samples": collision_samples,
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
