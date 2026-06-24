import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# --- Config ---
MODEL_SIZE = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4
CAMERA_INDEX = 0
VIDEO_PATH = ""

OUTPUT_DIR = Path(__file__).parent / "output"


def parse_args():
    parser = argparse.ArgumentParser(description="Basic person detection with YOLOv8")
    parser.add_argument(
        "--source",
        default=str(CAMERA_INDEX),
        help="Camera index (0) or path to video file",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save processed video to output/detection_output.mp4",
    )
    return parser.parse_args()


def open_source(source: str):
    try:
        return cv2.VideoCapture(int(source))
    except ValueError:
        return cv2.VideoCapture(source)


def main():
    args = parse_args()
    model = YOLO(MODEL_SIZE)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "detection_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (width, height))
        print(f"Saving output to {out_path}")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONFIDENCE_THRESHOLD, classes=[0], verbose=False)

        person_count = 0
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person {conf:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )
            person_count += 1

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
        )
        cv2.putText(
            frame, f"Persons: {person_count}", (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
        )

        cv2.imshow("Person Detection — press Q to quit", frame)

        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
