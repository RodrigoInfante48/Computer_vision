import logging
from datetime import datetime, timezone

import requests

from config import ALERT_WEBHOOK_URL, MAX_OCCUPANCY, DWELL_ALERT_SECONDS

logger = logging.getLogger(__name__)


class AlertEngine:
    def __init__(self, session_id: int, db=None):
        self.session_id = session_id
        self.db = db
        self._occupancy_alerted = False
        self._dwell_alerted: set[int] = set()

    def check_occupancy(self, current_count: int) -> bool:
        """Fire an alert if current_count exceeds MAX_OCCUPANCY. Returns True if alerted."""
        if current_count > MAX_OCCUPANCY and not self._occupancy_alerted:
            self._occupancy_alerted = True
            payload = self._build_payload(
                alert_type="occupancy_exceeded",
                value=current_count,
                threshold=MAX_OCCUPANCY,
            )
            self._persist_alert("occupancy_exceeded", payload)
            self.send_webhook(payload)
            return True
        if current_count <= MAX_OCCUPANCY:
            self._occupancy_alerted = False
        return False

    def check_dwell(self, track_id: int, seconds: float) -> bool:
        """Fire a one-time alert per track_id when dwell exceeds DWELL_ALERT_SECONDS."""
        if seconds >= DWELL_ALERT_SECONDS and track_id not in self._dwell_alerted:
            self._dwell_alerted.add(track_id)
            payload = self._build_payload(
                alert_type="dwell_exceeded",
                value=round(seconds, 1),
                threshold=DWELL_ALERT_SECONDS,
                extra={"track_id": track_id},
            )
            self._persist_alert("dwell_exceeded", payload)
            self.send_webhook(payload)
            return True
        return False

    def send_webhook(self, payload: dict) -> bool:
        """POST payload to ALERT_WEBHOOK_URL. Compatible with Make.com / n8n / Slack."""
        if not ALERT_WEBHOOK_URL:
            logger.debug("No webhook URL configured — skipping send.")
            return False
        try:
            response = requests.post(ALERT_WEBHOOK_URL, json=payload, timeout=5)
            response.raise_for_status()
            logger.info("Webhook sent: %s → %s", payload["alert_type"], response.status_code)
            return True
        except requests.RequestException as exc:
            logger.warning("Webhook delivery failed: %s", exc)
            return False

    def _build_payload(self, alert_type: str, value, threshold, extra: dict | None = None) -> dict:
        payload = {
            "alert_type": alert_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "value": value,
            "threshold": threshold,
            "session_id": self.session_id,
        }
        if extra:
            payload.update(extra)
        return payload

    def _persist_alert(self, alert_type: str, payload: dict):
        if self.db is None:
            return
        from database import Alert
        alert = Alert(
            session_id=self.session_id,
            alert_type=alert_type,
            triggered_at=datetime.utcnow(),
            details=payload,
        )
        self.db.add(alert)
        self.db.commit()
