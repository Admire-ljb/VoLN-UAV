from voln_uav.evaluation.metrics import aggregate_by_difficulty, summarize_episode


def test_metrics_basic():
    ref = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    pred = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    out = summarize_episode(pred, ref, goal=[2, 0, 0], success_radius=0.5, shortest_path_length=2.0)
    assert out["SR"] == 1.0
    assert out["OSR"] == 1.0
    assert out["NE"] == 0.0
    assert out["SPL"] == 1.0


def test_aggregate_by_difficulty_orders_paper_groups():
    items = [
        {"difficulty": "Hard", "NE": 3.0, "SR": 0.0, "OSR": 0.0, "nDTW": 0.3, "SPL": 0.0},
        {"difficulty": "Easy", "NE": 1.0, "SR": 1.0, "OSR": 1.0, "nDTW": 0.9, "SPL": 0.8},
        {"difficulty": "Easy", "NE": 3.0, "SR": 0.0, "OSR": 1.0, "nDTW": 0.7, "SPL": 0.0},
        {"difficulty": "Normal", "NE": 2.0, "SR": 0.5, "OSR": 0.5, "nDTW": 0.5, "SPL": 0.4},
    ]

    grouped = aggregate_by_difficulty(items)

    assert list(grouped) == ["Easy", "Normal", "Hard"]
    assert grouped["Easy"]["episodes"] == 2
    assert grouped["Easy"]["NE"] == 2.0
    assert grouped["Easy"]["SR"] == 0.5
