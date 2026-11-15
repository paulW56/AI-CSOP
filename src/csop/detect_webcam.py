"""Milestone 1: live object detection on a webcam feed.

Runs a pretrained YOLO model on each frame, draws labeled boxes, and shows FPS.

Usage:
    python -m csop.detect_webcam                  # default webcam, yolo11n
    python -m csop.detect_webcam --source 1       # second camera
    python -m csop.detect_webcam --source clip.mp4
    python -m csop.detect_webcam --log runs/detections.csv

Press q (with the video window focused) to quit.
"""

import argparse
import time

import cv2

from csop.detections import Detection, DetectionLogger
from csop.fps import FPSMeter


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live YOLO object detection.")
    p.add_argument("--source", default="0",
                   help="camera index (0, 1, ...) or path to a video file")
    p.add_argument("--model", default="yolo11n.pt",
                   help="YOLO weights; downloads automatically on first run")
    p.add_argument("--conf", type=float, default=0.4,
                   help="minimum confidence to show a detection (0-1)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="inference resolution; lower is faster, less accurate")
    p.add_argument("--log", default=None,
                   help="optional CSV path to record every detection")
    p.add_argument("--no-show", action="store_true",
                   help="run without a display window (benchmarking)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N frames (0 = run until q)")
    return p.parse_args(argv)


def open_source(source: str) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source!r}")
    return cap


def to_detections(result, frame_idx: int, t: float) -> list[Detection]:
    names = result.names
    out = []
    for box in result.boxes:
        x1, y1, x2, y2 = (round(float(v), 1) for v in box.xyxy[0])
        out.append(Detection(
            frame=frame_idx, t=round(t, 3),
            label=names[int(box.cls[0])],
            confidence=round(float(box.conf[0]), 3),
            x1=x1, y1=y1, x2=x2, y2=y2,
        ))
    return out


def main(argv=None) -> None:
    args = parse_args(argv)

    # Imported here so --help works even before the heavy dependency loads.
    from ultralytics import YOLO

    model = YOLO(args.model)
    cap = open_source(args.source)
    fps = FPSMeter()
    logger = DetectionLogger(args.log) if args.log else None
    start = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                   verbose=False)[0]
            dets = to_detections(result, frame_idx, time.perf_counter() - start)
            if logger:
                logger.write(dets)

            current_fps = fps.tick()
            if not args.no_show:
                annotated = result.plot()
                cv2.putText(annotated, f"FPS {current_fps:4.1f}  objects {len(dets)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("AI-CSOP detection (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if logger:
            logger.close()

    elapsed = time.perf_counter() - start
    if frame_idx:
        print(f"Processed {frame_idx} frames in {elapsed:.1f}s "
              f"(avg {frame_idx / elapsed:.1f} FPS)")


if __name__ == "__main__":
    main()
