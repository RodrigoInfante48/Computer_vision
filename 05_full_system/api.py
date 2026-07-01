import io
import threading
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session as DBSession

from database import get_db, Session, Detection, Alert, init_db

app = FastAPI(title="Computer Vision System API", version="1.0.0")

# Shared state written by main.py during a live session
_live_state: dict = {}
_live_lock = threading.Lock()


def set_live_state(state: dict):
    with _live_lock:
        _live_state.clear()
        _live_state.update(state)


def clear_live_state():
    with _live_lock:
        _live_state.clear()


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/sessions")
def list_sessions(db: DBSession = Depends(get_db)):
    sessions = db.query(Session).order_by(Session.start_time.desc()).all()
    return [
        {
            "id": s.id,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "video_source": s.video_source,
            "total_persons": s.total_persons,
        }
        for s in sessions
    ]


@app.get("/sessions/{session_id}/stats")
def session_stats(session_id: int, db: DBSession = Depends(get_db)):
    s = db.query(Session).filter(Session.id == session_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    detections = db.query(Detection).filter(Detection.session_id == session_id).all()
    alerts = db.query(Alert).filter(Alert.session_id == session_id).all()

    avg_dwell = (
        sum(d.dwell_seconds for d in detections) / len(detections) if detections else 0
    )
    duration = (
        (s.end_time - s.start_time).total_seconds() if s.end_time else None
    )

    return {
        "session_id": session_id,
        "start_time": s.start_time,
        "end_time": s.end_time,
        "duration_seconds": duration,
        "total_persons": s.total_persons,
        "unique_tracks": len(detections),
        "avg_dwell_seconds": round(avg_dwell, 2),
        "total_alerts": len(alerts),
    }


@app.get("/sessions/{session_id}/detections")
def session_detections(session_id: int, db: DBSession = Depends(get_db)):
    s = db.query(Session).filter(Session.id == session_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    detections = db.query(Detection).filter(Detection.session_id == session_id).all()
    return [
        {
            "track_id": d.track_id,
            "first_seen": d.first_seen,
            "last_seen": d.last_seen,
            "dwell_seconds": d.dwell_seconds,
            "avg_x": d.avg_x,
            "avg_y": d.avg_y,
        }
        for d in detections
    ]


@app.get("/sessions/{session_id}/heatmap")
def session_heatmap(
    session_id: int,
    width: int = 640,
    height: int = 480,
    db: DBSession = Depends(get_db),
):
    s = db.query(Session).filter(Session.id == session_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    detections = db.query(Detection).filter(Detection.session_id == session_id).all()
    if not detections:
        raise HTTPException(status_code=404, detail="No detections for this session")

    heatmap = np.zeros((height, width), dtype=np.float32)
    for d in detections:
        x = int(min(max(d.avg_x * width, 0), width - 1))
        y = int(min(max(d.avg_y * height, 0), height - 1))
        radius = 40
        cv2.circle(heatmap, (x, y), radius, 1.0, -1)

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=20)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    _, buffer = cv2.imencode(".png", colored)
    return Response(content=buffer.tobytes(), media_type="image/png")


@app.get("/live/stats")
def live_stats():
    with _live_lock:
        if not _live_state:
            return {"active": False}
        return {"active": True, **_live_state}
