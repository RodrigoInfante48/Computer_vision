import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from dwell_analyzer import DwellAnalyzer
from report_generator import generate_report

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "yolov8n.pt"
CONFIDENCE = 0.4
MAX_AGE = 30       # frames before a lost track is dropped
MIN_HITS = 3       # minimum detections before a track is confirmed
PERSON_CLASS = 0   # COCO class id for "person"

# ── Color thresholds (BGR) ────────────────────────────────────────────────────
COLOR_SHORT  = (0, 200, 0)    # green  — < 30s
COLOR_MEDIUM = (0, 200, 255)  # yellow — 30-120s
COLOR_LONG   = (0, 0, 220)    # red    — > 120s


def _dwell_color(seconds: float) -> tuple:
    if seconds < 30:
        return COLOR_SHORT
    elif seconds <= 120:
        return COLOR_MEDIUM
    return COLOR_LONG


def _point_in_zone(cx: int, cy: int, zone: list) -> bool:
    """zone is [[x1,y1],[x2,y2]] bounding rectangle."""
    (x1, y1), (x2, y2) = zone[0], zone[1]
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _draw_panel(frame, stats_rows: list, total_unique: int, avg_dwell: float) -> None:
    h, w = frame.shape[:2]
    panel_w = 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 24
    cv2.putText(frame, "TOP DWELL TIMES", (w - panel_w + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 22
    for rank, (tid, secs) in enumerate(stats_rows, 1):
        color = _dwell_color(secs)
        cv2.putText(frame, f"  {rank}. ID {tid:>3}  {secs:>6.1f}s",
                    (w - panel_w + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += 18

    y += 10
    cv2.putText(frame, f"Unique persons: {total_unique}", (w - panel_w + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += 18
    cv2.putText(frame, f"Avg dwell: {avg_dwell:.1f}s", (w - panel_w + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def run(source, save: bool, zones: dict) -> None:
    model = YOLO(MODEL)
    tracker = DeepSort(max_age=MAX_AGE, n_init=MIN_HITS)

    cap_source = 0 if source is None else source

    # Try DirectShow first on Windows (avoids MSMF errors), then default backend
    cap = None
    if isinstance(cap_source, int):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            _cap = cv2.VideoCapture(cap_source, backend)
            if _cap.isOpened():
                cap = _cap
                print(f"[INFO] Camera opened with backend {backend}")
                break
            _cap.release()
    else:
        cap = cv2.VideoCapture(cap_source)

    if cap is None or not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open source: {cap_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    analyzer = DwellAnalyzer(fps)

    output_dir = Path(__file__).parent / "output"
    writer = None
    if save:
        output_dir.mkdir(exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"dwell_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))

    frame_num = 0
    print("[INFO] Running — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        results = model(frame, verbose=False)[0]

        # Build DeepSort detections: ([x,y,w,h], conf, class)
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls != PERSON_CLASS:
                continue
            conf = float(box.conf[0])
            if conf < CONFIDENCE:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        # Panel data
        panel_entries = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cx, cy = (l + r) // 2, (t + b) // 2

            # Zone detection
            zone_name = None
            for zname, zcoords in zones.items():
                if _point_in_zone(cx, cy, zcoords):
                    zone_name = zname
                    break

            analyzer.update(tid, frame_num, (l, t, r, b), zone_name)
            dwell = analyzer.get_dwell_seconds(tid)
            color = _dwell_color(dwell)

            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            label = f"ID {tid}  {dwell:.0f}s"
            if zone_name:
                label += f" [{zone_name}]"
            cv2.putText(frame, label, (l, t - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            panel_entries.append((tid, dwell))

        # Draw zones
        for zname, zcoords in zones.items():
            (x1, y1), (x2, y2) = zcoords[0], zcoords[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
            cv2.putText(frame, zname, (x1 + 4, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        # Side panel
        df = analyzer.get_all_stats()
        total_unique = len(df)
        avg_dwell = df["dwell_seconds"].mean() if not df.empty else 0.0
        top5 = sorted(panel_entries, key=lambda x: x[1], reverse=True)[:5]
        _draw_panel(frame, top5, total_unique, avg_dwell)

        if writer:
            writer.write(frame)

        cv2.imshow("Dwell Time Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    generate_report(analyzer, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dwell time tracker — Level 3")
    parser.add_argument("--source", default=None,
                        help="Video file path or camera index (default: webcam 0)")
    parser.add_argument("--save", action="store_true",
                        help="Save processed video to output/")
    parser.add_argument("--zones", default=None,
                        help='JSON file with named zones, e.g. {"entrance": [[0,0],[320,480]]}')
    args = parser.parse_args()

    zones = {}
    if args.zones:
        with open(args.zones) as f:
            zones = json.load(f)

    source = args.source
    if source is not None and source.isdigit():
        source = int(source)

    run(source, args.save, zones)
