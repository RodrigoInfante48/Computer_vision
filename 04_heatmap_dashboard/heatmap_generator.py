from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_BLUR_RADIUS = 21   # Gaussian blur to smooth the heatmap
DEFAULT_ALPHA = 0.4        # overlay transparency


class HeatmapGenerator:
    def __init__(self, frame_width: int, frame_height: int):
        self.width = frame_width
        self.height = frame_height
        self._accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)

    def update(self, centroid_x: int, centroid_y: int, weight: float = 1.0) -> None:
        x = int(np.clip(centroid_x, 0, self.width - 1))
        y = int(np.clip(centroid_y, 0, self.height - 1))
        self._accumulator[y, x] += weight

    def _colorized(self) -> np.ndarray:
        blurred = cv2.GaussianBlur(self._accumulator, (DEFAULT_BLUR_RADIUS, DEFAULT_BLUR_RADIUS), 0)
        max_val = blurred.max()
        if max_val == 0:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        normalized = (blurred / max_val * 255).astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

    def get_heatmap_overlay(self, frame: np.ndarray, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
        colorized = self._colorized()
        mask = cv2.cvtColor(colorized, cv2.COLOR_BGR2GRAY)
        result = frame.copy()
        result[mask > 0] = cv2.addWeighted(frame, 1 - alpha, colorized, alpha, 0)[mask > 0]
        return result

    def reset(self) -> None:
        self._accumulator[:] = 0

    def save_heatmap(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self._colorized())

    def peak_zone(self, radius: int = 40) -> dict:
        blurred = cv2.GaussianBlur(self._accumulator, (DEFAULT_BLUR_RADIUS, DEFAULT_BLUR_RADIUS), 0)
        if blurred.max() == 0:
            return {"x": 0, "y": 0, "radius": radius}
        _, _, _, max_loc = cv2.minMaxLoc(blurred)
        return {"x": int(max_loc[0]), "y": int(max_loc[1]), "radius": radius}

    @property
    def accumulator(self) -> np.ndarray:
        return self._accumulator.copy()


if __name__ == "__main__":
    gen = HeatmapGenerator(640, 480)
    for _ in range(200):
        gen.update(np.random.randint(100, 540), np.random.randint(100, 380))
    out = Path("output")
    out.mkdir(exist_ok=True)
    gen.save_heatmap(out / f"test_heatmap_{datetime.now().strftime('%H%M%S')}.png")
    print("[INFO] Test heatmap saved.")
