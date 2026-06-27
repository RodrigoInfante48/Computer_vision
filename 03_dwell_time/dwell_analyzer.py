import pandas as pd


class DwellAnalyzer:
    def __init__(self, fps: float):
        self.fps = fps
        # track_id -> {first_frame, last_frame, frame_count, zones_visited}
        self._tracks: dict[int, dict] = {}

    def update(self, track_id: int, frame_number: int, bbox, zone_name: str = None):
        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "first_frame": frame_number,
                "last_frame": frame_number,
                "frame_count": 1,
                "zones_visited": [],
            }
        else:
            entry = self._tracks[track_id]
            entry["last_frame"] = frame_number
            entry["frame_count"] += 1

        if zone_name is not None:
            self._tracks[track_id]["zones_visited"].append(zone_name)

    def get_dwell_seconds(self, track_id: int) -> float:
        if track_id not in self._tracks:
            return 0.0
        entry = self._tracks[track_id]
        span_frames = entry["last_frame"] - entry["first_frame"] + 1
        return span_frames / self.fps

    def get_all_stats(self) -> pd.DataFrame:
        rows = []
        for track_id, entry in self._tracks.items():
            span_frames = entry["last_frame"] - entry["first_frame"] + 1
            dwell_seconds = span_frames / self.fps
            zones = entry["zones_visited"]
            avg_zone = max(set(zones), key=zones.count) if zones else None
            rows.append(
                {
                    "track_id": track_id,
                    "first_seen_frame": entry["first_frame"],
                    "last_seen_frame": entry["last_frame"],
                    "total_frames": entry["frame_count"],
                    "dwell_seconds": round(dwell_seconds, 2),
                    "avg_zone": avg_zone,
                }
            )
        return pd.DataFrame(rows, columns=[
            "track_id", "first_seen_frame", "last_seen_frame",
            "total_frames", "dwell_seconds", "avg_zone",
        ])
