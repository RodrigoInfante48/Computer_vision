import sys
import time
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from config import (
    CAMERA_INDEX, MODEL_SIZE, CONFIDENCE_THRESHOLD, OUTPUT_DIR
)
from database import init_db, SessionLocal, Session as DBSession, Detection
from alert_engine import AlertEngine
from api import set_live_state, clear_live_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0


def run(video_source=None):
    init_db()
    db = SessionLocal()

    source = video_source if video_source else CAMERA_INDEX
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", source)
        sys.exit(1)

    source_label = str(source) if isinstance(source, str) else "webcam"
    db_session = DBSession(start_time=datetime.utcnow(), video_source=source_label)
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    session_id = db_session.id
    logger.info("Session %d started — source: %s", session_id, source_label)

    model = YOLO(MODEL_SIZE)
    tracker = DeepSort(max_age=30)
    alert_engine = AlertEngine(session_id=session_id, db=db)

    track_data: dict[int, dict] = defaultdict(lambda: {
        "first_seen": datetime.utcnow(),
        "last_seen": datetime.utcnow(),
        "positions": [],
    })
    unique_ids: set[int] = set()

    output_path = OUTPUT_DIR / f"session_{session_id}.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(str(output_path), fourcc, 20, (w, h))

            results = model(frame, verbose=False)[0]
            detections_raw = []
            for box in results.boxes:
                if int(box.cls[0]) != PERSON_CLASS_ID:
                    continue
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections_raw.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

            tracks = tracker.update_tracks(detections_raw, frame=frame)
            now = datetime.utcnow()
            current_count = 0

            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                l, t, r, b = map(int, track.to_ltrb())
                cx = (l + r) / 2 / w
                cy = (t + b) / 2 / h

                unique_ids.add(tid)
                current_count += 1
                td = track_data[tid]
                td["last_seen"] = now
                td["positions"].append((cx, cy))

                dwell = (now - td["first_seen"]).total_seconds()
                alert_engine.check_dwell(tid, dwell)

                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"ID:{tid} {dwell:.0f}s",
                    (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )

            alert_engine.check_occupancy(current_count)

            set_live_state({
                "session_id": session_id,
                "current_count": current_count,
                "unique_ids": len(unique_ids),
                "timestamp": now.isoformat(),
            })

            cv2.putText(frame, f"Count: {current_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Full System — press Q to quit", frame)
            writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        for tid, td in track_data.items():
            positions = td["positions"]
            avg_x = float(np.mean([p[0] for p in positions])) if positions else 0.0
            avg_y = float(np.mean([p[1] for p in positions])) if positions else 0.0
            dwell = (td["last_seen"] - td["first_seen"]).total_seconds()
            det = Detection(
                session_id=session_id,
                track_id=tid,
                first_seen=td["first_seen"],
                last_seen=td["last_seen"],
                dwell_seconds=dwell,
                avg_x=avg_x,
                avg_y=avg_y,
            )
            db.add(det)

        db_session.end_time = datetime.utcnow()
        db_session.total_persons = len(unique_ids)
        db.commit()
        db.close()
        clear_live_state()
        logger.info("Session %d closed — %d unique persons tracked.", session_id, len(unique_ids))


if __name__ == "__main__":
    video_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run(video_source=video_arg)
