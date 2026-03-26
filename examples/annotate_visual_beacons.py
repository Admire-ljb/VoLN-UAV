#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate visual beacon placements from manifest scenes and optionally replay trajectories.

Workflow per scene:
1) parse `trajectory_raw` from merged json and convert quaternion to yaw
2) scan real images from front camera directory
3) build feature table for image-backed points
4) query LLM for sign placements and apply rule-based constraints
5) emit placements + target + audit CSV, then optionally invoke replay script
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

MODEL_NAME = os.getenv("XHANG_MODEL", "xhang")
XHANG_BASE_URL = os.getenv("XHANG_BASE_URL", "https://xhang.buaa.edu.cn/xhang-feijibiandui/v1")
XHANG_API_KEY = os.getenv("XHANG_API_KEY")

SIGN_POOL = [
    "left_yaw",
    "left_turn",
    "left90",
    "right_yaw",
    "right_turn",
    "right90",
    "up",
    "down",
]
TARGET_TAG = "target_people"
SIGN_MIN_GAP = 25

YAW_SMALL = 15.0
YAW_MED = 35.0
YAW_LARGE = 60.0
SLOPE_UP = 1.0
SLOPE_DOWN = -1.0

FRONT_DIR_CANDIDATES = ["frontcamera", "FrontCamera", "FrontCameraRecord"]
MERGED_FILE_CANDIDATES = ["merged_data.json", "merge_data.json"]

ASSET_BASE = {
    "left_yaw": "label_left_yaw",
    "left_turn": "label_left_turn",
    "left90": "label_left90",
    "right_yaw": "label_right_yaw",
    "right_turn": "label_right_turn",
    "right90": "label_right90",
    "up": "label_up",
    "down": "label_down",
}


@dataclass
class RowFeat:
    idx: int
    img: str
    x: float
    y: float
    z: float
    alt: float
    yaw_deg: float
    d_yaw_deg: float
    slope_m: float
    ds_m: float


def _resolve_session_dir(item: dict[str, Any], root: str | None) -> str | None:
    for k in ("session_dir", "dir", "path"):
        p = item.get(k)
        if p:
            p = str(p)
            if root and not os.path.isabs(p):
                p = os.path.join(root, p)
            return os.path.normpath(p)
    sid = item.get("session_id") or item.get("id")
    if sid and root:
        return os.path.normpath(os.path.join(root, str(sid)))
    return None


def _find_existing_path(base_dir: str, candidates: list[str], preferred: str | None = None, is_dir: bool = True) -> str | None:
    checks: list[str] = []
    if preferred:
        checks.append(preferred)
    checks.extend(candidates)
    for name in checks:
        p = os.path.join(base_dir, name)
        if is_dir and os.path.isdir(p):
            return p
        if not is_dir and os.path.isfile(p):
            return p
    return None


def load_scene_by_i(manifest_path: str, i: int, root: str | None = None, one_based: bool = False) -> tuple[dict[str, Any], list[str], str]:
    if i < 0:
        raise ValueError("i must be non-negative")
    target_idx = i - 1 if one_based else i

    line: str | None = None
    with open(manifest_path, "r", encoding="utf-8", errors="ignore") as f:
        for cur, ln in enumerate(f):
            if cur == target_idx:
                line = ln.strip()
                break
    if line is None:
        raise IndexError(f"manifest line index out of range: {i}")

    item = json.loads(line)
    session_dir = _resolve_session_dir(item, root)
    if not session_dir or not os.path.isdir(session_dir):
        raise FileNotFoundError(f"invalid session dir: {session_dir!r}")

    front_dir = _find_existing_path(session_dir, FRONT_DIR_CANDIDATES, item.get("front_dir") or item.get("frontcamera"), is_dir=True)
    if not front_dir:
        raise FileNotFoundError(f"front camera dir not found in {session_dir}")

    merged_path = _find_existing_path(session_dir, MERGED_FILE_CANDIDATES, item.get("merged_path") or item.get("merged"), is_dir=False)
    if not merged_path:
        raise FileNotFoundError(f"merged data file not found in {session_dir}")

    with open(merged_path, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    imgs = [
        os.path.join(front_dir, fn)
        for fn in os.listdir(front_dir)
        if fn.lower().endswith((".png", ".jpg", ".jpeg")) and os.path.isfile(os.path.join(front_dir, fn))
    ]
    if not imgs:
        raise FileNotFoundError(f"no images in {front_dir}")

    pat = re.compile(r"^(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
    imgs.sort(key=lambda p: int(pat.match(os.path.basename(p)).group(1)) if pat.match(os.path.basename(p)) else 10**12)
    return merged_data, imgs, session_dir


def _normalize_angle_rad(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _quat_to_yaw_rad(qx: float, qy: float, qz: float, qw: float) -> float:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0:
        qx = qy = qz = 0.0
        qw = 1.0
    else:
        qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return _normalize_angle_rad(math.atan2(siny_cosp, cosy_cosp))


def _extract_xyz_yaw(m: dict[str, Any]) -> tuple[list[float], list[float], list[float], list[float]]:
    traj = m.get("trajectory_raw")
    if not isinstance(traj, list) or not traj:
        raise ValueError("missing trajectory_raw")
    xs, ys, zs, yaws = [], [], [], []
    for row in traj:
        pos = row.get("position")
        ori = row.get("orientation")
        if not (isinstance(pos, (list, tuple)) and len(pos) >= 3 and isinstance(ori, (list, tuple)) and len(ori) >= 4):
            raise ValueError("trajectory_raw rows must contain position/orientation")
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        qx, qy, qz, qw = float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])
        xs.append(x)
        ys.append(y)
        zs.append(z)
        yaws.append(_quat_to_yaw_rad(qx, qy, qz, qw))
    return xs, ys, zs, yaws


def _norm_angle_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _unwrap_deg(deg_list: list[float]) -> list[float]:
    if not deg_list:
        return deg_list
    out = [deg_list[0]]
    for a in deg_list[1:]:
        prev = out[-1]
        delta = a - prev
        while delta > 180:
            a -= 360
            delta = a - prev
        while delta < -180:
            a += 360
            delta = a - prev
        out.append(a)
    return out


def build_img_index(imgs: list[str]) -> list[int]:
    pat = re.compile(r"^(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
    idxs: list[int] = []
    for p in imgs:
        m = pat.match(os.path.basename(p))
        if m:
            idxs.append(int(m.group(1)))
    idxs.sort()
    return idxs


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_features(m: dict[str, Any], imgs: list[str], frame_to_traj_ratio: int = 5) -> list[RowFeat]:
    xs, ys, zs, yaws_rad = _extract_xyz_yaw(m)
    yaws_deg = _unwrap_deg([r * 180.0 / math.pi for r in yaws_rad])
    idxs = build_img_index(imgs)
    path_by_idx = {int(Path(p).stem): p for p in imgs if Path(p).stem.isdigit()}

    feats: list[RowFeat] = []
    prev: RowFeat | None = None
    for k in idxs:
        tidx = min(len(xs) - 1, k // frame_to_traj_ratio)
        rf = RowFeat(
            idx=k,
            img=path_by_idx.get(k, ""),
            x=xs[tidx],
            y=ys[tidx],
            z=zs[tidx],
            alt=-zs[tidx],
            yaw_deg=yaws_deg[tidx],
            d_yaw_deg=0.0,
            slope_m=0.0,
            ds_m=0.0,
        )
        if prev is not None:
            rf.d_yaw_deg = _norm_angle_deg(rf.yaw_deg - prev.yaw_deg)
            rf.slope_m = rf.alt - prev.alt
            rf.ds_m = _dist((rf.x, rf.y), (prev.x, prev.y))
        feats.append(rf)
        prev = rf
    if not feats:
        raise ValueError("no usable features")
    return feats


def format_context_for_llm(feats: list[RowFeat], max_rows: int = 0) -> str:
    rows = feats
    if max_rows and len(feats) > max_rows:
        step = len(feats) / max_rows
        rows = []
        acc = 0.0
        while int(round(acc)) < len(feats) and len(rows) < max_rows:
            rows.append(feats[int(round(acc))])
            acc += step

    lines = ["index,x,y,altitude(m),yaw_deg,turn_deg_from_prev,slope_m_from_prev,ds_m,image"]
    for r in rows:
        if r.idx < 10:
            continue
        lines.append(
            f"{r.idx},{r.x:.2f},{r.y:.2f},{r.alt:.2f},{r.yaw_deg:.1f},{r.d_yaw_deg:.1f},{r.slope_m:.2f},{r.ds_m:.2f},{os.path.basename(r.img) or 'NA'}"
        )
    return "\n".join(lines)


def _safe_json_loads(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _too_close(idx: int, chosen: list[int], min_gap: int = SIGN_MIN_GAP) -> bool:
    return any(abs(idx - c) < min_gap for c in chosen)


def _legacy_to_new_tag(tag: str, rf: RowFeat) -> str | None:
    t = tag.lower()
    if t in SIGN_POOL:
        return t
    if t in ("right", "right_turn"):
        if rf.d_yaw_deg >= YAW_LARGE:
            return "right90"
        if rf.d_yaw_deg >= YAW_MED:
            return "right_turn"
        if rf.d_yaw_deg >= YAW_SMALL:
            return "right_yaw"
        return None
    if t in ("left", "left_turn"):
        if rf.d_yaw_deg <= -YAW_LARGE:
            return "left90"
        if rf.d_yaw_deg <= -YAW_MED:
            return "left_turn"
        if rf.d_yaw_deg <= -YAW_SMALL:
            return "left_yaw"
        return None
    if t == "up":
        return "up" if rf.slope_m >= SLOPE_UP else None
    if t == "down":
        return "down" if rf.slope_m <= SLOPE_DOWN else None
    return None


def call_llm_for_placements(client: OpenAI, context_csv: str, last_index: int) -> dict[str, Any]:
    sys = "You are a planning assistant. Choose where to place signs along sampled drone trajectory rows."
    user = f"""
Rules and goals:
- Available signs: {SIGN_POOL} plus mandatory '{TARGET_TAG}'.
- Always place '{TARGET_TAG}' at last index {last_index}.
- Besides target, place 3 to 4 signs from {SIGN_POOL}.
- Only choose indices shown in the table.
- Avoid duplicate or too-close placements (<{SIGN_MIN_GAP} frames).

Return strict JSON:
{{
  "placements": [{{"index": <int>, "sign": "<one of {SIGN_POOL}>", "reason": "<short>"}}],
  "target": {{"index": {last_index}, "sign": "{TARGET_TAG}", "reason": "end of route"}}
}}

Table:\n{context_csv}
"""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        stream=False,
    )
    content = resp.choices[0].message.content or "{}"
    return _safe_json_loads(content)


def enforce_constraints(feats: list[RowFeat], llm_out: dict[str, Any]) -> dict[str, Any]:
    feats_by_idx = {r.idx: r for r in feats}
    candidate_indices = set(feats_by_idx)
    last_idx = feats[-1].idx

    normalized: list[dict[str, Any]] = []
    for p in (llm_out.get("placements", []) or []):
        try:
            idx = int(p.get("index", -1))
        except Exception:
            continue
        if idx not in candidate_indices or idx == last_idx:
            continue
        sign = str(p.get("sign", "")).strip().lower()
        mapped = _legacy_to_new_tag(sign, feats_by_idx[idx])
        if mapped is None and sign in SIGN_POOL:
            mapped = sign
        if mapped:
            normalized.append({"index": idx, "sign": mapped, "reason": str(p.get("reason", ""))})

    normalized.sort(key=lambda x: x["index"])
    chosen: list[dict[str, Any]] = []
    chosen_idx: list[int] = []
    for p in normalized:
        if p["index"] in chosen_idx or _too_close(p["index"], chosen_idx):
            continue
        chosen.append(p)
        chosen_idx.append(p["index"])
        if len(chosen) >= 4:
            break

    def try_add(idx: int, tag: str, reason: str) -> bool:
        if idx not in candidate_indices or idx == last_idx:
            return False
        if idx in chosen_idx or _too_close(idx, chosen_idx):
            return False
        chosen.append({"index": idx, "sign": tag, "reason": reason})
        chosen_idx.append(idx)
        return True

    if len(chosen) < 3:
        turn_cands = sorted(feats[1:-1], key=lambda r: abs(r.d_yaw_deg), reverse=True)
        for r in turn_cands:
            tag = None
            if r.d_yaw_deg >= YAW_LARGE:
                tag = "right90"
            elif r.d_yaw_deg >= YAW_MED:
                tag = "right_turn"
            elif r.d_yaw_deg >= YAW_SMALL:
                tag = "right_yaw"
            elif r.d_yaw_deg <= -YAW_LARGE:
                tag = "left90"
            elif r.d_yaw_deg <= -YAW_MED:
                tag = "left_turn"
            elif r.d_yaw_deg <= -YAW_SMALL:
                tag = "left_yaw"
            if tag and try_add(r.idx, tag, f"turn {r.d_yaw_deg:.1f} deg") and len(chosen) >= 3:
                break

    if len(chosen) < 3:
        for r in feats[1:-1]:
            if r.slope_m >= SLOPE_UP and try_add(r.idx, "up", f"ascending {r.slope_m:.2f} m") and len(chosen) >= 3:
                break
            if r.slope_m <= SLOPE_DOWN and try_add(r.idx, "down", f"descending {r.slope_m:.2f} m") and len(chosen) >= 3:
                break

    chosen = chosen[:4]
    return {"placements": chosen, "target": {"index": last_idx, "sign": TARGET_TAG, "reason": "end of route"}}


def to_asset_name(tag: str, white: bool) -> str:
    base = ASSET_BASE.get(tag.lower())
    if not base:
        return tag
    return base + ("w" if white else "")


def write_audit_csv(out_csv: Path, feats: list[RowFeat]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(feats[0]).keys()))
        writer.writeheader()
        for r in feats:
            writer.writerow(asdict(r))


def decide_signs_with_llm(client: OpenAI, merged: dict[str, Any], imgs: list[str], max_rows: int = 0, frame_to_traj_ratio: int = 5) -> tuple[dict[str, Any], list[RowFeat], str]:
    feats = build_features(merged, imgs, frame_to_traj_ratio=frame_to_traj_ratio)
    table = format_context_for_llm(feats, max_rows=max_rows)
    try:
        llm_out = call_llm_for_placements(client, table, feats[-1].idx)
    except Exception:
        llm_out = {"placements": []}
    result = enforce_constraints(feats, llm_out)
    result["table"] = table
    return result, feats, table


def process_scene(
    client: OpenAI,
    manifest_path: str,
    dataset_root: str,
    scene_index: int,
    out_root: Path,
    replay_script: str,
    frame_to_traj_ratio: int,
    max_rows: int,
) -> None:
    merged, imgs, session_dir = load_scene_by_i(manifest_path=manifest_path, i=scene_index, root=dataset_root)
    result, feats, _ = decide_signs_with_llm(client, merged, imgs, max_rows=max_rows, frame_to_traj_ratio=frame_to_traj_ratio)

    scene_out = out_root / f"normal_{scene_index + 1:02d}"
    scene_out.mkdir(parents=True, exist_ok=True)
    with (scene_out / "sign_plan.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    write_audit_csv(scene_out / "audit_features.csv", feats)

    placements_sorted = sorted(result["placements"], key=lambda x: x["index"])
    use_white = False
    sign_pairs: list[tuple[int, str]] = []
    for p in placements_sorted:
        mapped = to_asset_name(p["sign"], white=use_white)
        sign_pairs.append((p["index"], mapped))
        use_white = not use_white
    sign_pairs.append((result["target"]["index"], result["target"]["sign"]))
    sign_str = ",".join(f"{idx}:{name}" for idx, name in sign_pairs)

    merged_path = _find_existing_path(session_dir, MERGED_FILE_CANDIDATES, preferred="merged_data.json", is_dir=False)
    cmd = [
        os.sys.executable,
        replay_script,
        "--merged_json",
        str(merged_path),
        "--out",
        str(scene_out),
        "--cams",
        "FrontCamera,RearCamera,LeftCamera,RightCamera,DownCamera",
        "--mode",
        "kinematic",
        "--hz",
        "10",
        "--sign_at",
        sign_str,
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plan visual beacons and replay trajectories from manifest scenes.")
    p.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    p.add_argument("--dataset_root", required=True, help="Root folder for manifest relative paths")
    p.add_argument("--start", type=int, default=0, help="Start scene index (0-based)")
    p.add_argument("--count", type=int, default=1, help="Number of scenes to process")
    p.add_argument("--out_root", required=True, help="Output root for generated plans/replays")
    p.add_argument("--replay_script", default="examples/replay_fix_trajectory_normal.py", help="Replay script path")
    p.add_argument("--max_rows", type=int, default=300, help="Max rows sent to LLM")
    p.add_argument("--frame_to_traj_ratio", type=int, default=5, help="Image index to trajectory index ratio")
    return p.parse_args()


def main() -> None:
    if not XHANG_API_KEY:
        raise RuntimeError("XHANG_API_KEY is required")
    args = parse_args()
    client = OpenAI(api_key=XHANG_API_KEY, base_url=XHANG_BASE_URL)
    out_root = Path(args.out_root)
    for i in range(args.start, args.start + args.count):
        process_scene(
            client=client,
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            scene_index=i,
            out_root=out_root,
            replay_script=args.replay_script,
            frame_to_traj_ratio=args.frame_to_traj_ratio,
            max_rows=args.max_rows,
        )


if __name__ == "__main__":
    main()
