from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
MODEL_SIZE = os.getenv("MODEL_SIZE", "yolov8n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.4))

MAX_OCCUPANCY = int(os.getenv("MAX_OCCUPANCY", 10))
DWELL_ALERT_SECONDS = int(os.getenv("DWELL_ALERT_SECONDS", 300))
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")

DB_PATH = os.getenv("DB_PATH", str(OUTPUT_DIR / "sessions.db"))
