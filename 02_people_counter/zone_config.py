"""
Zone configuration for people counter.
Coordinates are proportional (0.0–1.0) — resolution-agnostic.
"""

# Default zones: horizontal line at 40% (IN) and 60% (OUT) of frame height.
DEFAULT_ZONES = {
    "IN": [
        (0.0, 0.4),
        (1.0, 0.4),
    ],
    "OUT": [
        (0.0, 0.6),
        (1.0, 0.6),
    ],
}


def scale_zones(zones: dict, width: int, height: int) -> dict:
    """Convert proportional zone coordinates to absolute pixel values."""
    scaled = {}
    for name, points in zones.items():
        scaled[name] = [(int(x * width), int(y * height)) for x, y in points]
    return scaled
