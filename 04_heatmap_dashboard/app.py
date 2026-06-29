import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from heatmap_generator import HeatmapGenerator

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "yolov8n.pt"
CONFIDENCE = 0.4
MAX_AGE = 30
MIN_HITS = 3
PERSON_CLASS = 0
HEATMAP_REFRESH_SECONDS = 5

st.set_page_config(page_title="Heatmap Dashboard", layout="wide", page_icon="🔥")

# ── Session state initialization ──────────────────────────────────────────────
for key, default in {
    "running": False,
    "track_data": {},
    "frame_rgb": None,
    "heatmap_img": None,
    "dwell_df": pd.DataFrame(),
    "metrics": {},
    "last_heatmap_update": 0.0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _dwell_color_hex(seconds: float) -> str:
    if seconds < 30:
        return "#00c800"
    elif seconds <= 120:
        return "#ffc800"
    return "#dc0000"


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔥 Heatmap Dashboard")
st.sidebar.markdown("**Level 4** — Computer Vision")

source_type = st.sidebar.radio("Input source", ["Webcam", "Video file"])
source = 0
uploaded = None
if source_type == "Video file":
    uploaded = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        st.sidebar.info("Upload a video to enable analysis.")

col_start, col_stop = st.sidebar.columns(2)
start_btn = col_start.button("▶ Start", disabled=st.session_state.running)
stop_btn  = col_stop.button("⏹ Stop",  disabled=not st.session_state.running)

if stop_btn:
    st.session_state.running = False

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("Heatmap Analytics Dashboard")

# Metrics row
m1, m2, m3 = st.columns(3)
unique_ph   = m1.empty()
avg_dwell_ph = m2.empty()
peak_ph      = m3.empty()

unique_ph.metric("Unique persons", st.session_state.metrics.get("total_persons", 0))
avg_dwell_ph.metric("Avg dwell (s)", st.session_state.metrics.get("avg_dwell_seconds", 0.0))
peak = st.session_state.metrics.get("peak_zone", {})
peak_ph.metric("Peak zone", f"({peak.get('x','-')}, {peak.get('y','-')})" if peak else "—")

tab_live, tab_heatmap, tab_report, tab_export = st.tabs(
    ["📷 Live Feed", "🔥 Heatmap", "📊 Report", "⬇ Export"]
)

with tab_live:
    live_frame_ph = st.empty()
    live_frame_ph.info("Press ▶ Start to begin analysis.")

with tab_heatmap:
    heatmap_ph = st.empty()
    heatmap_ph.info("Heatmap will appear here after analysis starts.")

with tab_report:
    report_ph = st.empty()
    hist_ph   = st.empty()

with tab_export:
    st.subheader("Download results")
    csv_dl_ph  = st.empty()
    json_dl_ph = st.empty()

# ── Analysis loop ─────────────────────────────────────────────────────────────
if start_btn and not st.session_state.running:
    if source_type == "Video file" and uploaded is None:
        st.sidebar.error("Please upload a video first.")
    else:
        st.session_state.running = True
        st.session_state.track_data = {}
        st.session_state.dwell_df = pd.DataFrame()
        st.session_state.metrics = {}

        # Resolve source
        tmp_file = None
        if source_type == "Video file" and uploaded is not None:
            tmp_file = tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False)
            tmp_file.write(uploaded.read())
            tmp_file.flush()
            cap_source = tmp_file.name
        else:
            cap_source = 0

        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            st.error(f"Cannot open source: {cap_source}")
            st.session_state.running = False
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            model   = YOLO(MODEL)
            tracker = DeepSort(max_age=MAX_AGE, n_init=MIN_HITS)
            heatmap = HeatmapGenerator(frame_w, frame_h)

            track_data: dict[int, dict] = {}
            frame_num = 0

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.session_state.running = False
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
                    color_bgr = (0, 200, 0) if dwell < 30 else ((0, 200, 255) if dwell <= 120 else (0, 0, 220))
                    cv2.rectangle(frame, (l, t), (r, b), color_bgr, 2)
                    cv2.putText(frame, f"ID {tid}  {dwell:.0f}s", (l, t - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

                # Overlay heatmap on display frame
                display = heatmap.get_heatmap_overlay(frame)
                display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                live_frame_ph.image(display_rgb, channels="RGB", use_container_width=True)

                # Heatmap tab — refresh every N seconds
                now = time.time()
                if now - st.session_state.last_heatmap_update > HEATMAP_REFRESH_SECONDS:
                    heatmap_bgr = heatmap._colorized()
                    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
                    heatmap_ph.image(heatmap_rgb, caption="Accumulated heatmap", use_container_width=True)
                    st.session_state.last_heatmap_update = now

                # Build dwell dataframe
                rows = []
                for tid, data in track_data.items():
                    dwell = (data.get("last_frame", frame_num) - data["first_frame"]) / fps
                    rows.append({"track_id": tid, "dwell_seconds": round(dwell, 2),
                                 "cx": data["cx"], "cy": data["cy"]})
                df = pd.DataFrame(rows)

                # Update metrics
                avg_dwell = float(df["dwell_seconds"].mean()) if not df.empty else 0.0
                peak = heatmap.peak_zone()
                metrics = {
                    "total_persons": len(track_data),
                    "avg_dwell_seconds": round(avg_dwell, 2),
                    "peak_zone": peak,
                    "session_duration_seconds": round(frame_num / fps, 2),
                }
                st.session_state.metrics = metrics
                st.session_state.dwell_df = df

                unique_ph.metric("Unique persons", metrics["total_persons"])
                avg_dwell_ph.metric("Avg dwell (s)", f"{metrics['avg_dwell_seconds']:.1f}")
                peak_ph.metric("Peak zone", f"({peak['x']}, {peak['y']})")

                # Report tab
                if not df.empty:
                    report_ph.dataframe(
                        df.sort_values("dwell_seconds", ascending=False),
                        use_container_width=True, hide_index=True,
                    )
                    fig = px.histogram(df, x="dwell_seconds", nbins=20,
                                       title="Dwell time distribution",
                                       labels={"dwell_seconds": "Dwell time (s)"},
                                       color_discrete_sequence=["#FF4B4B"])
                    hist_ph.plotly_chart(fig, use_container_width=True)

                # Export tab
                if not df.empty:
                    csv_dl_ph.download_button(
                        "⬇ Download CSV", df.to_csv(index=False).encode(),
                        file_name="dwell_report.csv", mime="text/csv", key=f"csv_{frame_num}"
                    )
                    json_dl_ph.download_button(
                        "⬇ Download JSON metrics", json.dumps(metrics, indent=2).encode(),
                        file_name="metrics.json", mime="application/json", key=f"json_{frame_num}"
                    )

            cap.release()
            if tmp_file:
                Path(tmp_file.name).unlink(missing_ok=True)

        st.session_state.running = False
        st.success("Analysis complete.")
