import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from heatmap_generator import HeatmapGenerator

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "yolov8n.pt"
CONFIDENCE = 0.4
MAX_AGE = 30
MIN_HITS = 3
PERSON_CLASS = 0

COLOR_SHORT  = (0, 200, 0)
COLOR_MEDIUM = (0, 200, 255)
COLOR_LONG   = (0, 0, 220)


def _dwell_color(seconds: float) -> tuple:
    if seconds < 30:
        return COLOR_SHORT
    elif seconds <= 120:
        return COLOR_MEDIUM
    return COLOR_LONG


def _draw_hud(frame, total_unique: int, avg_dwell: float, show_heatmap: bool) -> None:
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 36), (340, h), (30, 30, 30), -1)
    status = "ON" if show_heatmap else "OFF"
    cv2.putText(frame, f"Persons: {total_unique}  Avg dwell: {avg_dwell:.1f}s  Heatmap: {status}",
                (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    cv2.putText(frame, "[H] toggle heatmap  [S] snapshot  [Q] quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


def run(source, save: bool) -> None:
    model = YOLO(MODEL)
    tracker = DeepSort(max_age=MAX_AGE, n_init=MIN_HITS)

    cap_source = 0 if source is None else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open source: {cap_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    heatmap = HeatmapGenerator(frame_w, frame_h)
    show_heatmap = True

    # Dwell tracking: {track_id: {"first_frame", "last_frame", "cx", "cy"}}
    track_data: dict[int, dict] = {}
    frame_num = 0
    session_start = time.time()

    writer = None
    if save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"heatmap_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))

    print("[INFO] Running — H: toggle heatmap | S: snapshot | Q: quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        results = model(frame, verbose=False)[0]

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

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cx, cy = (l + r) // 2, (t + b) // 2

            heatmap.update(cx, cy)

            if tid not in track_data:
                track_data[tid] = {"first_frame": frame_num, "cx": cx, "cy": cy}
            track_data[tid]["last_frame"] = frame_num
            track_data[tid]["cx"] = cx
            track_data[tid]["cy"] = cy

            dwell = (frame_num - track_data[tid]["first_frame"]) / fps
            color = _dwell_color(dwell)

            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, f"ID {tid}  {dwell:.0f}s", (l, t - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Compute stats
        dwells = [(frame_num - v["first_frame"]) / fps for v in track_data.values()]
        avg_dwell = float(np.mean(dwells)) if dwells else 0.0

        display = frame.copy()
        if show_heatmap:
            display = heatmap.get_heatmap_overlay(display)

        _draw_hud(display, len(track_data), avg_dwell, show_heatmap)

        if writer:
            writer.write(display)

        cv2.imshow("Heatmap Tracker — Level 4", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("h"):
            show_heatmap = not show_heatmap
        elif key == ord("s"):
            snap_path = output_dir / f"heatmap_{datetime.now().strftime('%H%M%S')}.png"
            heatmap.save_heatmap(snap_path)
            print(f"[INFO] Snapshot saved: {snap_path}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    session_duration = time.time() - session_start
    _save_outputs(heatmap, track_data, fps, frame_num, session_duration, output_dir)


def _save_outputs(heatmap: HeatmapGenerator, track_data: dict, fps: float,
                  last_frame: int, session_duration: float, output_dir: Path) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Heatmap PNG
    heatmap_path = output_dir / f"heatmap_final_{ts}.png"
    heatmap.save_heatmap(heatmap_path)
    print(f"[INFO] Heatmap saved: {heatmap_path}")

    # Dwell CSV
    rows = []
    for tid, data in track_data.items():
        dwell = (data.get("last_frame", last_frame) - data["first_frame"]) / fps
        rows.append({"track_id": tid, "dwell_seconds": round(dwell, 2),
                     "last_cx": data["cx"], "last_cy": data["cy"]})

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"dwell_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Dwell CSV saved: {csv_path}")

    # JSON metrics
    avg_dwell = float(df["dwell_seconds"].mean()) if not df.empty else 0.0
    peak = heatmap.peak_zone()
    metrics = {
        "total_persons": len(track_data),
        "avg_dwell_seconds": round(avg_dwell, 2),
        "peak_zone": peak,
        "session_duration_seconds": round(session_duration, 2),
    }
    json_path = output_dir / f"metrics_{ts}.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    print(f"[INFO] Metrics JSON saved: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heatmap tracker — Level 4")
    parser.add_argument("--source", default=None,
                        help="Video file path or camera index (default: webcam 0)")
    parser.add_argument("--save", action="store_true",
                        help="Save processed video to output/")
    args = parser.parse_args()

    source = args.source
    if source is not None and source.isdigit():
        source = int(source)

    run(source, args.save)
