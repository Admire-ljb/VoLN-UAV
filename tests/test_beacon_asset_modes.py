from __future__ import annotations

import math
import random

import pytest

from voln_uav.simulators.airsim_env import (
    AirSimRouteEnv,
    _adjust_target_pose_for_asset,
    _asset_name_matches_alias,
    _text_asset_matches_tag,
    _text_asset_yaw_correction_deg,
    _target_asset_kind,
)
from voln_uav.simulators.beacon_placement import normalize_beacon_render_mode


def _asset_selection_env() -> AirSimRouteEnv:
    env = object.__new__(AirSimRouteEnv)
    env._beacon_object_cache = [
        "left_turn_icon_1",
        "right_turn_icon_1",
        "up_icon_1",
        "target_people_1",
    ]
    env._text_beacon_object_cache = [
        "label_left_turn_1",
        "label_right_turn_1",
        "label_up_1",
    ]
    return env


def test_active_beacon_mode_selects_direction_or_text_assets() -> None:
    env = _asset_selection_env()
    direction_name, direction_mode = env._pick_active_beacon_object(
        "left_turn",
        set(),
        random.Random(7),
        "direction",
    )
    text_name, text_mode = env._pick_active_beacon_object(
        "left_turn",
        set(),
        random.Random(7),
        "text",
    )
    target_name, target_mode = env._pick_active_beacon_object(
        "target_people",
        set(),
        random.Random(7),
        "text",
    )

    assert direction_mode == "direction"
    assert direction_name == "left_turn_icon_1"
    assert text_mode == "text"
    assert text_name == "label_left_turn_1"
    assert target_mode == "target"
    assert target_name == "target_people_1"


def test_random_active_beacon_mode_is_reproducible_and_uses_available_style() -> None:
    env = _asset_selection_env()
    first = env._pick_active_beacon_object(
        "right_turn",
        set(),
        random.Random(23),
        "random",
    )
    second = env._pick_active_beacon_object(
        "right_turn",
        set(),
        random.Random(23),
        "random",
    )

    assert first == second
    assert first[1] in {"direction", "text"}
    assert first[0] is not None


def test_random_active_beacon_mode_falls_back_to_the_available_style() -> None:
    env = _asset_selection_env()
    env._text_beacon_object_cache = []

    name, mode = env._pick_active_beacon_object(
        "up",
        set(),
        random.Random(5),
        "random",
    )

    assert mode == "direction"
    assert name == "up_icon_1"


def test_beacon_render_mode_defaults_to_direction_and_rejects_unknown_values() -> None:
    assert normalize_beacon_render_mode() == "direction"
    assert normalize_beacon_render_mode("icons") == "direction"
    assert normalize_beacon_render_mode("labels") == "text"
    with pytest.raises(ValueError, match="Unsupported beacon render mode"):
        normalize_beacon_render_mode("blue")


def test_asset_alias_matching_rejects_substrings_inside_scene_object_names() -> None:
    assert _asset_name_matches_alias("here2_5", "here")
    assert _asset_name_matches_alias("right_turn4_11", "right_turn")
    assert _asset_name_matches_alias("Up_2", "up")
    assert _asset_name_matches_alias("target_people_2", "target_people")
    assert not _asset_name_matches_alias("BP_Sky_Sphere_8", "here")
    assert not _asset_name_matches_alias("SM_SupplyBox_10", "up")
    assert not _asset_name_matches_alias("personal_light", "person")


def test_text_asset_matching_preserves_exact_turn_angle_family() -> None:
    assert _text_asset_matches_tag("label_left90w2", "left90")
    assert _text_asset_matches_tag("label_left_turn2", "left_turn")
    assert _text_asset_matches_tag("label_right_yaw2_7", "right_yaw")
    assert not _text_asset_matches_tag("label_left_yaw2", "left90")
    assert not _text_asset_matches_tag("label_right_turn2", "right90")


def test_target_asset_kind_distinguishes_people_from_target_signs() -> None:
    assert _target_asset_kind("target_people_2") == "people"
    assert _target_asset_kind("People_7") == "people"
    assert _target_asset_kind("target_2") == "sign"
    assert _target_asset_kind("target2_5") == "sign"


def test_target_selection_prefers_explicit_people_over_target_signs() -> None:
    env = object.__new__(AirSimRouteEnv)
    env._beacon_object_cache = ["target2_5", "target_2", "target_people_5"]
    env._text_beacon_object_cache = []

    name, mode = env._pick_active_beacon_object(
        "target_people",
        set(),
        random.Random(7),
        "direction",
    )

    assert name == "target_people_5"
    assert mode == "target"


def test_target_sign_is_lowered_and_rotated_to_face_the_final_view() -> None:
    placement = {
        "kind": "target",
        "position": [20.0, 30.0, -40.0],
        "target_direction_yaw_deg": -45.0,
        "yaw_rad": math.radians(135.0),
    }
    _adjust_target_pose_for_asset(
        placement,
        "target2_5",
        {"target_sign_vertical_ned_m": 8.0, "target_sign_yaw_add_deg": 90.0},
    )
    assert placement["target_asset_kind"] == "sign"
    assert placement["position"] == pytest.approx([20.0, 30.0, -32.0])
    assert placement["yaw_rad"] == pytest.approx(math.radians(45.0))
    assert placement["target_asset_vertical_ned_offset_m"] == 8.0


def test_target_people_spawn_above_the_planned_pose_without_yaw_change() -> None:
    placement = {
        "kind": "target",
        "position": [20.0, 30.0, -40.0],
        "target_direction_yaw_deg": -45.0,
        "yaw_rad": math.radians(135.0),
    }
    _adjust_target_pose_for_asset(
        placement,
        "target_people_2",
        {
            "target_sign_vertical_ned_m": 8.0,
            "target_sign_yaw_add_deg": 90.0,
        },
    )
    assert placement["target_asset_kind"] == "people"
    assert placement["position"] == pytest.approx([20.0, 30.0, -140.0])
    assert placement["yaw_rad"] == pytest.approx(math.radians(135.0))
    assert placement["target_asset_vertical_ned_offset_m"] == -100.0


def test_text_sign_yaw_correction_defaults_to_zero_for_every_family() -> None:
    assert _text_asset_yaw_correction_deg("right90", {}) == 0.0
    assert _text_asset_yaw_correction_deg("left90", {}) == 0.0
    assert _text_asset_yaw_correction_deg("up", {}) == 0.0


def test_text_sign_yaw_correction_can_explicitly_flip_right_family() -> None:
    config: dict[str, object] = {
        "text_yaw_flip_right_turn_signs": True,
        "text_yaw_flip_deg": 180.0,
    }
    assert _text_asset_yaw_correction_deg("right90", config) == 180.0
    assert _text_asset_yaw_correction_deg("right_turn", config) == 180.0
    assert _text_asset_yaw_correction_deg("right_yaw", config) == 180.0
    assert _text_asset_yaw_correction_deg("left90", config) == 0.0
    assert _text_asset_yaw_correction_deg("up", config) == 0.0
