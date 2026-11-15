"""Detection data type and logging.

Kept separate from the model so later milestones (tracking, pan-tilt) can
consume detections without depending on YOLO directly.
"""

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class Detection:
    frame: int
    t: float            # seconds since start
    label: str
    confidence: float
    x1: float           # bounding box, pixels
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class DetectionLogger:
    """Appends detections to a CSV file for later analysis."""

    FIELDS: ClassVar[list[str]] = [f.name for f in fields(Detection)]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def write(self, dets: list[Detection]) -> None:
        for d in dets:
            self._writer.writerow(asdict(d))

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
