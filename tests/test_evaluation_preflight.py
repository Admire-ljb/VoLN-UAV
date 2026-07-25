from __future__ import annotations

from voln_uav.evaluation.airsim_loop import (
    check_airsim_readiness,
    episode_task_beacon_readiness_issues,
    filter_airsim_episodes,
)


def _episodes() -> list[dict[str, str]]:
    return [
        {"scene_id": "scene_a", "difficulty": "Easy"},
        {"scene_id": "scene_b", "difficulty": "Hard"},
        {"scene_id": "scene_a", "difficulty": "Normal"},
    ]


def test_filter_airsim_episodes_matches_scene_and_trial_selection() -> None:
    selected = filter_airsim_episodes(
        {"scene_allowlist": ["scene_a"], "episode_index": 1, "episode_stride": 1, "trials": 1},
        _episodes(),
    )
    assert [episode["difficulty"] for episode in selected] == ["Normal"]


def test_manual_airsim_preflight_rejects_multi_scene_selection(monkeypatch) -> None:
    monkeypatch.setattr("voln_uav.evaluation.airsim_loop.importlib.util.find_spec", lambda _name: object())
    monkeypatch.setattr("voln_uav.evaluation.airsim_loop._port_is_open", lambda _ip, _port: True)

    issues = check_airsim_readiness({"env": {"auto_launch": False}}, _episodes())

    assert any("only one scene at a time" in issue for issue in issues)


def test_manual_airsim_preflight_accepts_single_scene_selection(monkeypatch) -> None:
    monkeypatch.setattr("voln_uav.evaluation.airsim_loop.importlib.util.find_spec", lambda _name: object())
    monkeypatch.setattr("voln_uav.evaluation.airsim_loop._port_is_open", lambda _ip, _port: True)

    issues = check_airsim_readiness({"env": {"auto_launch": False}}, [_episodes()[0]])

    assert issues == []


def test_online_preflight_rejects_generated_method_specific_beacons() -> None:
    issues = episode_task_beacon_readiness_issues(
        {
            "beacon_placement": {
                "enabled": True,
                "source": "generated_from_route",
                "count": 4,
            }
        },
        [],
    )

    assert issues == [
        "AirSim online evaluation requires "
        "beacon_placement.source=episode_task_beacons"
    ]
