from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image

from voln_uav.common.config import load_config
from voln_uav.common.io import ensure_dir, read_jsonl, write_json
from voln_uav.evaluation.airsim_loop import check_airsim_readiness
from voln_uav.simulators.airsim_env import AirSimRouteEnv


class AirSimStreamState:
    def __init__(self, env: AirSimRouteEnv, episode: dict[str, Any], placements: list[dict[str, Any]], fps: float) -> None:
        self.env = env
        self.episode = episode
        self.placements = placements
        self.fps = max(float(fps), 0.5)
        self.lock = threading.Lock()
        self.started_at = time.time()

    def _png_to_jpeg(self, image_bytes: bytes) -> bytes:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=85)
            return out.getvalue()

    def frame_jpeg(self) -> bytes:
        with self.lock:
            image_bytes = self.env.client.simGetImage(self.env.camera, self.env._image_type())
        if not image_bytes:
            raise RuntimeError(f"AirSim returned an empty image for camera {self.env.camera}")
        return self._png_to_jpeg(image_bytes)

    def status(self) -> dict[str, Any]:
        with self.lock:
            state = self.env.client.getMultirotorState()
            collision = self.env.client.simGetCollisionInfo()
        kin = state.kinematics_estimated
        pos = kin.position
        vel = kin.linear_velocity
        return {
            "episode_id": self.episode.get("episode_id"),
            "scene_id": self.episode.get("scene_id"),
            "difficulty": self.episode.get("difficulty"),
            "camera": self.env.camera,
            "position": [float(pos.x_val), float(pos.y_val), float(pos.z_val)],
            "linear_velocity": [float(vel.x_val), float(vel.y_val), float(vel.z_val)],
            "landed_state": int(getattr(state, "landed_state", -1)),
            "collision": bool(getattr(collision, "has_collided", False)),
            "beacons_requested": len(self.placements),
            "beacons_placed": sum(1 for item in self.placements if item.get("placed")),
            "uptime_sec": time.time() - self.started_at,
        }

    def ensure_takeoff_at_start(self) -> dict[str, Any]:
        start_position = [float(v) for v in self.episode["states"][0]["position"]]
        with self.lock:
            self.env._ensure_flying_at_start(start_position)
        return {**self.status(), "command": "takeoff_at_start"}

    def reference_step_status(self, step: int) -> dict[str, Any]:
        states = self.episode["states"]
        idx = max(0, min(int(step), len(states) - 1))
        target = [float(v) for v in states[idx]["position"]]
        return {**self.status(), "command": "reference_step_status", "step": idx, "target_position": target}

    def move_to_reference_step(self, step: int, control_mode: str = "move_to_position") -> dict[str, Any]:
        states = self.episode["states"]
        idx = max(0, min(int(step), len(states) - 1))
        target = [float(v) for v in states[idx]["position"]]
        with self.lock:
            state = self.env.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            current = [float(pos.x_val), float(pos.y_val), float(pos.z_val)]
            self.env.move_to_waypoint(current, target, control_mode=control_mode)
        return {**self.status(), "command": "reference_step", "step": idx, "target_position": target}

    def start_readiness_watchdog(self, attempts: int = 3, delay_sec: float = 2.0) -> None:
        def worker() -> None:
            start = [float(v) for v in self.episode["states"][0]["position"]]
            for _ in range(max(int(attempts), 1)):
                time.sleep(float(delay_sec))
                status = self.status()
                pos = status["position"]
                dist = sum((float(pos[i]) - start[i]) ** 2 for i in range(3)) ** 0.5
                if int(status["landed_state"]) == 1 and dist <= 1.0:
                    return
                try:
                    self.ensure_takeoff_at_start()
                except Exception as exc:
                    print(f"[stream] readiness retry failed: {exc!r}", flush=True)

        threading.Thread(target=worker, name="airsim-stream-readiness", daemon=True).start()


def _write_json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any] | list[Any]) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(stream_state: AirSimStreamState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "VoLNAirSimStream/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}", flush=True)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)
            if path in ("/", "/index.html"):
                self._index()
                return
            if path == "/status.json":
                _write_json_response(self, stream_state.status())
                return
            if path == "/takeoff.json":
                _write_json_response(self, stream_state.ensure_takeoff_at_start())
                return
            if path == "/reference.json":
                step = int(params.get("step", ["0"])[0])
                control_mode = str(params.get("mode", ["status"])[0]).lower()
                if control_mode in {"status", "peek", "preview"}:
                    _write_json_response(self, stream_state.reference_step_status(step))
                    return
                if control_mode not in {"move_to_position", "teleport"}:
                    self.send_error(HTTPStatus.BAD_REQUEST, "mode must be status, move_to_position, or teleport")
                    return
                _write_json_response(self, stream_state.move_to_reference_step(step, control_mode=control_mode))
                return
            if path == "/beacons.json":
                _write_json_response(self, stream_state.placements)
                return
            if path == "/snapshot.jpg":
                self._snapshot()
                return
            if path == "/stream.mjpg":
                self._stream()
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

        def _index(self) -> None:
            body = f"""
<!doctype html>
<html>
<head><meta charset=\"utf-8\"><title>VoLN-UAV AirSim Stream</title></head>
<body style=\"margin:0;background:#111;color:#eee;font-family:Arial,sans-serif\">
  <div style=\"padding:12px 16px;background:#1d1d1d;display:flex;gap:16px;align-items:center\">
    <strong>VoLN-UAV AirSim Stream</strong>
    <a href=\"/status.json\" style=\"color:#8cc8ff\">status</a>
    <a href=\"/beacons.json\" style=\"color:#8cc8ff\">beacons</a>
    <a href=\"/snapshot.jpg\" style=\"color:#8cc8ff\">snapshot</a>
    <a href=\"/takeoff.json\" style=\"color:#8cc8ff\">takeoff</a>
    <a href=\"/reference.json?step=20&mode=status\" style=\"color:#8cc8ff\">peek20</a>
    <a href=\"/reference.json?step=20&mode=move_to_position\" style=\"color:#8cc8ff\">move20</a>
  </div>
  <img src=\"/stream.mjpg\" style=\"display:block;width:100vw;height:calc(100vh - 48px);object-fit:contain;background:#000\" />
</body>
</html>
""".encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _snapshot(self) -> None:
            try:
                frame = stream_state.frame_jpeg()
            except Exception as exc:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(frame)

        def _stream(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            delay = 1.0 / stream_state.fps
            while True:
                try:
                    frame = stream_state.frame_jpeg()
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    time.sleep(delay)
                except (BrokenPipeError, ConnectionResetError):
                    break
                except Exception as exc:
                    print(f"[stream] {exc!r}", flush=True)
                    time.sleep(delay)

    return Handler


def _select_episode(episodes: list[dict[str, Any]], episode_id: str | None, index: int) -> dict[str, Any]:
    if episode_id:
        for episode in episodes:
            if str(episode.get("episode_id")) == episode_id:
                return episode
        raise KeyError(f"episode_id not found: {episode_id}")
    if index < 0 or index >= len(episodes):
        raise IndexError(f"episode index {index} out of range for {len(episodes)} episodes")
    return episodes[index]


def _filter_episodes(config: dict[str, Any], episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scene_allowlist = set(config.get("scene_allowlist", []) or [])
    difficulty_allowlist = set(config.get("difficulty_allowlist", []) or [])
    filtered = []
    for episode in episodes:
        if scene_allowlist and episode.get("scene_id") not in scene_allowlist:
            continue
        if difficulty_allowlist and episode.get("difficulty") not in difficulty_allowlist:
            continue
        filtered.append(episode)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a local AirSim RGB stream after placing route-aware visual beacons.")
    parser.add_argument("--config", default="configs/eval_airsim_dataset_release.yaml")
    parser.add_argument("--episode-id")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--no-beacons", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    episodes = _filter_episodes(cfg, read_jsonl(Path(cfg["benchmark_root"]) / cfg["episodes_file"]))
    episode = _select_episode(episodes, args.episode_id, args.episode_index)
    issues = check_airsim_readiness(cfg, [episode])
    if issues:
        raise RuntimeError("AirSim stream is not ready:\n" + "\n".join(f"- {issue}" for issue in issues))

    env = AirSimRouteEnv(
        ip=str(cfg.get("ip", "127.0.0.1")),
        port=int(cfg.get("port", 41451)),
        camera=str(cfg.get("camera", "0")),
        image_type=str(cfg.get("image_type", "Scene")),
        work_dir=cfg.get("work_dir", "work_dirs/airsim_stream"),
        speed=float(cfg.get("speed", 3.0)),
        move_timeout_sec=float(cfg.get("move_timeout_sec", 15.0)),
        settle_sec=float(cfg.get("settle_sec", 0.05)),
        takeoff_timeout_sec=float(cfg.get("takeoff_timeout_sec", 10.0)),
    )
    placements: list[dict[str, Any]] = []
    try:
        env.connect(timeout_sec=float(cfg.get("connect_timeout_sec", 60.0)))
        env.reset_to_episode_start(episode)
        beacon_cfg = dict(cfg.get("beacon_placement", {}) or {})
        if args.no_beacons:
            beacon_cfg["enabled"] = False
        placements = env.place_beacons_for_episode(episode, beacon_cfg, seed=int(cfg.get("seed", 0)))
        env._ensure_flying_at_start([float(v) for v in episode["states"][0]["position"]])
        audit_dir = ensure_dir(Path(cfg.get("work_dir", "work_dirs/airsim_stream")) / "stream_setup")
        write_json({"episode_id": episode.get("episode_id"), "placements": placements}, audit_dir / "beacons.json")

        state = AirSimStreamState(env=env, episode=episode, placements=placements, fps=args.fps)
        state.start_readiness_watchdog()
        server = ThreadingHTTPServer((args.host, int(args.http_port)), make_handler(state))
        url = f"http://{args.host}:{args.http_port}/"
        print(json.dumps({
            "url": url,
            "stream": url + "stream.mjpg",
            "status": url + "status.json",
            "beacons": url + "beacons.json",
            "episode_id": episode.get("episode_id"),
            "beacons_placed": sum(1 for item in placements if item.get("placed")),
            "beacons_requested": len(placements),
        }, indent=2), flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
