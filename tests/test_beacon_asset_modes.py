from __future__ import annotations

import random

import pytest

from voln_uav.simulators.airsim_env import AirSimRouteEnv
from voln_uav.simulators.beacon_placement import (
    normalize_beacon_render_mode,
)


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
