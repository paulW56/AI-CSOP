import csv

import pytest

from csop.detect_webcam import parse_args
from csop.detections import Detection, DetectionLogger
from csop.fps import FPSMeter


class FakeClock:
    def __init__(self, step):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def test_fps_steady_rate():
    meter = FPSMeter(clock=FakeClock(0.05))  # 20 FPS
    for _ in range(50):
        fps = meter.tick()
    assert fps == pytest.approx(20.0)


def test_fps_first_tick_is_zero():
    assert FPSMeter(clock=FakeClock(0.1)).tick() == 0.0


def test_fps_rejects_bad_alpha():
    with pytest.raises(ValueError):
        FPSMeter(alpha=0)


def test_detection_center():
    d = Detection(0, 0.0, "bird", 0.9, 10, 20, 30, 60)
    assert d.center == (20, 40)


def test_logger_writes_csv(tmp_path):
    path = tmp_path / "out" / "dets.csv"
    dets = [Detection(0, 0.0, "person", 0.88, 1, 2, 3, 4),
            Detection(1, 0.1, "cup", 0.51, 5, 6, 7, 8)]
    with DetectionLogger(path) as log:
        log.write(dets)
    rows = list(csv.DictReader(path.open()))
    assert [r["label"] for r in rows] == ["person", "cup"]
    assert set(rows[0]) == set(DetectionLogger.FIELDS)


def test_parse_args_defaults():
    args = parse_args([])
    assert args.source == "0" and args.conf == 0.4 and args.log is None
