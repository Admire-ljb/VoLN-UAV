from voln_uav.evaluation.termination import StationaryDetector


def test_stationary_detector_times_out_inside_radius() -> None:
    detector = StationaryDetector(timeout_sec=5.0, radius_m=0.5)

    assert not detector.update([0.0, 0.0, 0.0], now=10.0)
    assert not detector.update([0.2, 0.0, 0.0], now=14.9)
    assert detector.update([0.1, 0.0, 0.0], now=15.0)
    assert detector.duration_sec == 5.0


def test_stationary_detector_resets_after_leaving_radius() -> None:
    detector = StationaryDetector(timeout_sec=5.0, radius_m=0.5)

    assert not detector.update([0.0, 0.0, 0.0], now=10.0)
    assert not detector.update([1.0, 0.0, 0.0], now=14.0)
    assert detector.duration_sec == 0.0
    assert not detector.update([1.1, 0.0, 0.0], now=18.0)
