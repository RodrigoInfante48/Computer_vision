"""
Level 2 — People Counter with Virtual Zones
Tracks persons crossing IN/OUT lines and exports a CSV log.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from zone_config import DEFAULT_ZONES, scale_zones

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_SIZE       = "yolov8n.pt"
CONFIDENCE       = 0.4
CAMERA_INDEX     = 0
ZONE_COLOR_IN    = (0, 255, 0)    # green
ZONE_COLOR_OUT   = (0, 0, 255)    # red
TEXT_COLOR       = (255, 255, 255)
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="People counter with virtual zones")
    p.add_argument("--source", default=str(CAMERA_INDEX),
                   help="Video source: camera index (int) or file path")
    p.add_argument("--save", action="store_true",
                   help="Save annotated video to output/")
    p.add_argument("--zone-config", default=None,
                   help="Path to JSON file with custom zone definitions")
    return p.parse_args()


def load_zones(zone_config_path: str | None) -> dict:
    if zone_config_path:
        with open(zone_config_path) as f:
            return json.load(f)
    return DEFAULT_ZONES


def get_centroid(box) -> tuple[int, int]:
    x1, y1, x2, y2 = map(int, box)
    return (x1 + x2) // 2, (y1 + y2) // 2


def crossed_line(prev_cy: int, curr_cy: int, line_y: int) -> bool:
    """True when centroid moves from one side of line_y to the other."""
    return (prev_cy < line_y <= curr_cy) or (curr_cy < line_y <= prev_cy)


def draw_zones(frame, scaled_zones):
    for name, pts in scaled_zones.items():
        color = ZONE_COLOR_IN if name == "IN" else ZONE_COLOR_OUT
        if len(pts) == 2:
            cv2.line(frame, pts[0], pts[1], color, 2)
            label_pos = (pts[0][0] + 10, pts[0][1] - 10)
            cv2.putText(frame, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)


def draw_overlay(frame, count_in: int, count_out: int, fps: float):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (220, 90), (0, 0, 0), -1)
    cv2.putText(frame, f"IN:  {count_in}",  (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, ZONE_COLOR_IN, 2)
    cv2.putText(frame, f"OUT: {count_out}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, ZONE_COLOR_OUT, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}",  (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)


def main():
    args = parse_args()

    # Source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Zones
    zones = load_zones(args.zone_config)
    scaled = scale_zones(zones, width, height)
    line_in_y  = scaled["IN"][0][1]
    line_out_y = scaled["OUT"][0][1]

    # Model
    model = YOLO(MODEL_SIZE)

    # Output setup
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"counts_{timestamp_str}.csv"

    writer_out = None
    if args.save:
        video_path = output_dir / f"counts_{timestamp_str}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_out = cv2.VideoWriter(str(video_path), fourcc, orig_fps,
                                     (width, height))

    # State
    count_in  = 0
    count_out = 0
    counted_in:  set[int] = set()
    counted_out: set[int] = set()
    prev_centroids: dict[int, tuple[int, int]] = {}
    events: list[dict] = []
    frame_num = 0
    fps = 0.0
    tick = cv2.getTickCount()

    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["timestamp", "event", "person_id", "frame_number"]
        )
        csv_writer.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Track
            results = model.track(
                frame,
                persist=True,
                classes=[0],        # person only
                conf=CONFIDENCE,
                verbose=False,
            )

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids   = results[0].boxes.id.cpu().numpy().astype(int)

                for box, pid in zip(boxes, ids):
                    cx, cy = get_centroid(box)

                    if pid in prev_centroids:
                        _, prev_cy = prev_centroids[pid]

                        if pid not in counted_in and crossed_line(prev_cy, cy, line_in_y):
                            count_in += 1
                            counted_in.add(pid)
                            event = {
                                "timestamp": datetime.now().isoformat(),
                                "event": "IN",
                                "person_id": pid,
                                "frame_number": frame_num,
                            }
                            csv_writer.writerow(event)
                            csv_file.flush()

                        if pid not in counted_out and crossed_line(prev_cy, cy, line_out_y):
                            count_out += 1
                            counted_out.add(pid)
                            event = {
                                "timestamp": datetime.now().isoformat(),
                                "event": "OUT",
                                "person_id": pid,
                                "frame_number": frame_num,
                            }
                            csv_writer.writerow(event)
                            csv_file.flush()

                    prev_centroids[pid] = (cx, cy)

                    # Draw bounding box + ID
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    cv2.putText(frame, f"ID {pid}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 200, 0), 1)
                    cv2.circle(frame, (cx, cy), 4, (255, 200, 0), -1)

            # FPS calculation (rolling)
            tock = cv2.getTickCount()
            elapsed = (tock - tick) / cv2.getTickFrequency()
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if elapsed > 0 else fps
            tick = tock

            draw_zones(frame, scaled)
            draw_overlay(frame, count_in, count_out, fps)

            if writer_out:
                writer_out.write(frame)

            cv2.imshow("People Counter — press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer_out:
        writer_out.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Events saved to: {csv_path}")
    print(f"Total IN: {count_in} | Total OUT: {count_out}")


if __name__ == "__main__":
    main()
