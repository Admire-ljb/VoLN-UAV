#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from glob import glob
from typing import Optional

import numpy as np

try:
    import airsim
except Exception as e:  # pragma: no cover
    print("Please `pip install airsim` first.", e)
    sys.exit(1)


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _yaw_quat(deg: float):
    th = math.radians(deg) * 0.5
    return (math.cos(th), 0.0, 0.0, math.sin(th))


def pose_ahead(
    pose: "airsim.Pose",
    d: float,
    lateral_m: float,
    vertical_ned_m: float,
    yaw_add_deg: float,
    frame: str = "body",
) -> "airsim.Pose":
    q = pose.orientation
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val

    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + z * w)
    fz = 2.0 * (x * z - y * w)
    rx = 2.0 * (x * y - z * w)
    ry = 1.0 - 2.0 * (x * x + z * z)
    rz = 2.0 * (y * z + x * w)

    def _norm3(ax, ay, az):
        n = math.sqrt(ax * ax + ay * ay + az * az)
        return (ax / n, ay / n, az / n) if n > 1e-9 else (ax, ay, az)

    fx, fy, fz = _norm3(fx, fy, fz)
    rx, ry, rz = _norm3(rx, ry, rz)

    p = pose.position
    nx = p.x_val + d * fx + lateral_m * rx
    ny = p.y_val + d * fy + lateral_m * ry
    nz = p.z_val + vertical_ned_m

    dq = _yaw_quat(yaw_add_deg)
    if frame == "body":
        w2, x2, y2, z2 = _quat_mul((w, x, y, z), dq)
    else:
        w2, x2, y2, z2 = _quat_mul(dq, (w, x, y, z))

    return airsim.Pose(airsim.Vector3r(nx, ny, nz), airsim.Quaternionr(x2, y2, z2, w2))


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def yaw_from_quat(x, y, z, w):
    s1 = 2.0 * (w * z + x * y)
    s2 = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(s1, s2))


def _pick(d, paths, default=None):
    for p in paths:
        cur = d
        ok = True
        for k in p.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def _parse_pos(pos):
    if isinstance(pos, dict):
        def g(k):
            return pos.get(k) if k in pos else pos.get(k.upper())

        x, y, z = g("x"), g("y"), g("z")
        if x is not None and y is not None and z is not None:
            return float(x), float(y), float(z)
    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
        x, y, z = pos[:3]
        return float(x), float(y), float(z)
    raise ValueError(f"Unsupported position format: {type(pos)} -> {pos}")


def _parse_quat(ori) -> Optional[tuple[float, float, float, float]]:
    if isinstance(ori, dict):
        w = float(ori.get("w", ori.get("W", 1.0)))
        x = float(ori.get("x", ori.get("X", 0.0)))
        y = float(ori.get("y", ori.get("Y", 0.0)))
        z = float(ori.get("z", ori.get("Z", 0.0)))
        return (x, y, z, w)
    if isinstance(ori, (list, tuple)) and len(ori) >= 4:
        a, b, c, d = map(float, ori[:4])
        nx = math.sqrt(a * a + b * b + c * c + d * d)
        return (a, b, c, d) if 0.5 < nx < 2.0 else (b, c, d, a)
    return None


def load_waypoints_from_merged(merged_path: str):
    with open(merged_path, "r", encoding="utf-8") as r:
        m = json.load(r)
    frames = m.get("trajectory_raw_detailed") or m.get("trajectory_raw") or []
    if not frames:
        raise ValueError("merged_data.json lacks trajectory_raw_detailed/trajectory_raw")
    pts, quats, ts = [], [], []
    for d in frames:
        pos_raw = _pick(d, ["sensors.state.position", "state.position", "position", "pose.position", "pos"])
        ori_raw = _pick(d, ["sensors.state.orientation", "state.orientation", "orientation", "pose.orientation", "quat"])
        t_raw = _pick(d, ["sensors.state.timestamp", "timestamp"], None)

        x, y, z = _parse_pos(pos_raw)
        q = _parse_quat(ori_raw) or (0.0, 0.0, 0.0, 1.0)
        pts.append([x, y, z])
        quats.append(q)
        ts.append(float(t_raw) if t_raw is not None else None)
    if all(v is not None for v in ts):
        t0 = ts[0]
        if t0 > 1e14:
            ts = [(t - t0) / 1e9 for t in ts]
        elif t0 > 1e9:
            ts = [(t - t0) / 1e6 for t in ts]
        else:
            ts = [(t - t0) for t in ts]
    else:
        ts = None
    return pts, quats, ts


def capture_and_save(client, out_dir, idx, cam_names, save_depth=False):
    for name in cam_names:
        ensure_dir(os.path.join(out_dir, name))
        if save_depth:
            ensure_dir(os.path.join(out_dir, f"{name}_depth"))

    req = []
    for name in cam_names:
        req.append(airsim.ImageRequest(name, airsim.ImageType.Scene, pixels_as_float=False, compress=True))
        if save_depth:
            req.append(airsim.ImageRequest(name, airsim.ImageType.DepthPlanner, pixels_as_float=True, compress=False))

    resp = client.simGetImages(req)
    ridx = 0
    for name in cam_names:
        rgb = resp[ridx]
        ridx += 1
        if rgb and rgb.width > 0 and rgb.height > 0 and rgb.image_data_uint8:
            fn = os.path.join(out_dir, name, f"{idx:06d}.png")
            with open(fn, "wb") as w:
                w.write(rgb.image_data_uint8)
        if save_depth:
            dep = resp[ridx]
            ridx += 1
            if dep and dep.width > 0 and dep.height > 0 and dep.image_data_float:
                arr = airsim.get_pfm_array(dep)
                fn = os.path.join(out_dir, f"{name}_depth", f"{idx:06d}.pfm")
                airsim.write_pfm(fn, arr)


def save_state_log(client, out_dir, idx):
    state = client.getMultirotorState()
    kin = state.kinematics_estimated
    pos = kin.position
    ori = kin.orientation
    rec = {
        "step": int(idx),
        "timestamp": float(state.timestamp),
        "sensors": {
            "state": {
                "position": [pos.x_val, pos.y_val, pos.z_val],
                "orientation": [ori.x_val, ori.y_val, ori.z_val, ori.w_val],
            }
        },
    }
    ensure_dir(os.path.join(out_dir, "log"))
    with open(os.path.join(out_dir, "log", f"{idx:06d}.json"), "w", encoding="utf-8") as w:
        json.dump(rec, w, ensure_ascii=False, indent=2)


def _asset_base(sign: str) -> str:
    s = sign.strip().strip('"').strip("'")
    if s.lower().startswith("blueprint"):
        m = re.search(r"/([^/]+)\.([^']+)'", s)
        if m:
            return m.group(1)
    return s.split("/")[-1].split(".")[-1]


def _partner_base(base: str) -> str:
    return base[:-1] if base.endswith("w") else base + "w"


def _parse_sign_pairs(spec: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    if not spec.strip():
        return out
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"invalid --sign_at token: {tok}")
        i_str, s_str = tok.split(":", 1)
        if not i_str.strip().isdigit():
            raise ValueError(f"index is not numeric: {i_str}")
        out.append((int(i_str.strip()), s_str.strip()))
    return out


def replay(args: argparse.Namespace) -> None:
    sign_pairs = _parse_sign_pairs(args.sign_at)
    pts, quats, ts = load_waypoints_from_merged(args.merged_json)

    cam_names = [c.strip() for c in args.cams.split(",") if c.strip()]
    ensure_dir(args.out)

    client = airsim.MultirotorClient(ip=args.ip, port=args.port)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    x0, y0, z0 = pts[0]
    qx0, qy0, qz0, qw0 = quats[0]
    pose0 = airsim.Pose(airsim.Vector3r(x0, y0, z0), airsim.Quaternionr(qx0, qy0, qz0, qw0))
    client.simSetVehiclePose(pose0, True)
    time.sleep(0.001)

    if ts is not None and len(ts) >= 2:
        dts = [max(0.0, t2 - t1) for t1, t2 in zip(ts[:-1], ts[1:])]
        dts = [min(max(dt, 1.0 / args.hz), 0.5) for dt in dts]
    else:
        dts = [1.0 / args.hz] * (len(pts) - 1)

    preset_cfg = {
        "easy": {"d": 32.0, "lateral": (-5, -2), "vertical": 8.0},
        "normal": {"d": 45.0, "lateral": (-7, -3), "vertical": 10.0},
        "hard": {"d": 55.0, "lateral": (-10, -4), "vertical": 12.0},
    }[args.preset]

    scene_objs = client.simListSceneObjects()

    def pick_obj_name(sign_token: str) -> str | None:
        base = _asset_base(sign_token)
        cand = [n for n in scene_objs if base in n]
        if cand:
            return cand.pop()
        partner = _partner_base(base)
        cand2 = [n for n in scene_objs if partner in n]
        if cand2:
            return cand2.pop()
        return None

    used_names: list[str] = []
    for idx, signname in sign_pairs:
        obj_name = pick_obj_name(signname)
        if not obj_name or idx <= 20:
            continue
        ref_i = max(0, idx - 20)
        x, y, z = pts[ref_i]
        qx, qy, qz, qw = quats[ref_i]
        ref_pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.Quaternionr(qx, qy, qz, qw))
        if "target" in signname.lower():
            sign_pose = pose_ahead(ref_pose, d=30.0, lateral_m=0.0, vertical_ned_m=0.0, yaw_add_deg=180.0)
        else:
            l0, l1 = preset_cfg["lateral"]
            sign_pose = pose_ahead(
                ref_pose,
                d=preset_cfg["d"],
                lateral_m=float(np.random.randint(l0, l1)),
                vertical_ned_m=preset_cfg["vertical"],
                yaw_add_deg=90.0,
            )
        try:
            client.simSetObjectPose(obj_name, sign_pose)
            used_names.append(obj_name)
        except Exception as e:
            print(f"[WARN] simSetObjectPose({obj_name}) failed: {e}")

    idx_cnt = 0
    if args.mode == "kinematic":
        for i in range(0, len(pts)):
            x, y, z = pts[i]
            qx, qy, qz, qw = quats[i]
            pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.Quaternionr(qx, qy, qz, qw))
            client.simPause(False)
            client.simSetVehiclePose(pose, True)
            time.sleep(0.01)
            client.simPause(True)
            save_state_log(client, args.out, idx_cnt)
            if i % 5 == 0:
                capture_and_save(client, args.out, idx_cnt, cam_names, save_depth=args.depth)
            idx_cnt += 1
    else:
        for (x, y, z), (qx, qy, qz, qw), dt in zip(pts[1:], quats[1:], dts):
            yaw_deg = yaw_from_quat(qx, qy, qz, qw)
            yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)
            client.moveToPositionAsync(
                x,
                y,
                z,
                args.speed,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=yaw_mode,
                lookahead=1.0,
            ).join()
            save_state_log(client, args.out, idx_cnt)
            capture_and_save(client, args.out, idx_cnt, cam_names, save_depth=args.depth)
            time.sleep(dt)
            idx_cnt += 1

    try:
        for name in used_names:
            pose = client.simGetObjectPose(name)
            pose.position.z_val = -500
            client.simSetObjectPose(name, pose)
    except Exception as e:
        print(f"[WARN] cleanup labels failed: {e}")

    client.hoverAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print(f"Done. Saved to: {args.out}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_json", type=str, required=True, help="merged_data.json path")
    ap.add_argument("--out", type=str, required=True, help="output directory")
    ap.add_argument("--ip", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=41451)
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--cams", type=str, default="FrontCamera,RearCamera,LeftCamera,RightCamera,DownCamera")
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--mode", choices=["kinematic", "position"], default="kinematic")
    ap.add_argument("--sign_at", type=str, default="", help="comma-separated index:sign pairs")
    ap.add_argument("--preset", choices=["easy", "normal", "hard"], default="normal")
    return ap.parse_args()


def main() -> None:
    replay(parse_args())


if __name__ == "__main__":
    main()
