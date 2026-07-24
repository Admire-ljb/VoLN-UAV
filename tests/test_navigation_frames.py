import math
import pytest

from voln_uav.common.navigation_frames import (
    body_point_to_world,
    body_vector_to_world,
    encode_proprioception,
    world_point_to_body,
    world_vector_to_body,
)


def test_body_world_waypoint_round_trip_at_ninety_degree_yaw() -> None:
    origin = [10.0, -4.0, -20.0]
    world = [10.0, 1.0, -18.0]
    body = world_point_to_body(world, origin, math.pi / 2.0)

    assert body == pytest.approx([5.0, 0.0, 2.0])
    reconstructed = body_point_to_world(body, origin, math.pi / 2.0)
    assert all(abs(actual - expected) < 1e-8 for actual, expected in zip(reconstructed, world))


def test_proprioception_is_translation_invariant_and_body_relative() -> None:
    first_a = {"position": [100.0, 200.0, -30.0], "yaw": math.pi / 2.0}
    second_a = {"position": [100.0, 204.0, -28.0], "yaw": math.pi / 2.0}
    first_b = {"position": [-500.0, 700.0, -80.0], "yaw": math.pi / 2.0}
    second_b = {"position": [-500.0, 704.0, -78.0], "yaw": math.pi / 2.0}

    proprio_a = encode_proprioception(second_a, first_a, default_interval_sec=2.0)
    proprio_b = encode_proprioception(second_b, first_b, default_interval_sec=2.0)

    assert proprio_a == proprio_b
    assert proprio_a[:3] == pytest.approx([2.0, 0.0, 1.0])
    assert proprio_a[6:] == pytest.approx([4.0, 0.0, 2.0])


def test_body_frame_uses_full_quaternion_when_available() -> None:
    half = math.sqrt(0.5)
    orientation = [0.0, half, 0.0, half]
    body = world_vector_to_body([0.0, 0.0, -2.0], 0.0, orientation)

    assert body == pytest.approx([2.0, 0.0, 0.0], abs=1e-6)
    assert body_vector_to_world(body, 0.0, orientation) == pytest.approx(
        [0.0, 0.0, -2.0],
        abs=1e-6,
    )
