from voln_uav.evaluation.metrics import summarize_episode


def test_metrics_basic():
    ref = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    pred = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    out = summarize_episode(pred, ref, goal=[2, 0, 0], success_radius=0.5, shortest_path_length=2.0)
    assert out["SR"] == 1.0
    assert out["OSR"] == 1.0
    assert out["NE"] == 0.0
    assert out["SPL"] == 1.0
