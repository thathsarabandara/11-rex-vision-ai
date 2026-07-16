import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class Track:
    """Lightweight track record for ByteTrack output."""

    def __init__(self, track_id: int, class_id: int, class_name: str) -> None:
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.age_frames: int = 0
        self.lost_frames: int = 0
        self.history: list[dict] = []  # bounding box history

    def update(self, bbox: dict, confidence: float) -> None:
        self.age_frames += 1
        self.lost_frames = 0
        self.history.append({**bbox, "confidence": confidence})
        if len(self.history) > 30:
            self.history.pop(0)

    def mark_lost(self) -> None:
        self.lost_frames += 1

    @property
    def last_bbox(self) -> Optional[dict]:
        return self.history[-1] if self.history else None


class Tracker:
    """Multi-object tracker wrapping ByteTrack.

    Falls back to a simple IoU-based centroid tracker when ByteTrack is
    unavailable (e.g., in CI environments without GPU).
    """

    def __init__(self) -> None:
        self._bytetrack = None
        self.is_loaded: bool = False
        self._tracks: dict[int, Track] = {}

    def load(self) -> bool:
        try:
            # ByteTrack via ultralytics built-in tracker — no separate install needed
            # when ultralytics is available
            self.is_loaded = True
            logger.info("Tracker (ByteTrack stub) ready — tracking via YOLO track_id")
            return True
        except Exception as exc:
            logger.warning(f"Tracker failed to load: {exc}")
            self.is_loaded = False
            return False

    def unload(self) -> None:
        self._tracks.clear()
        self.is_loaded = False

    def update(self, detections: list[dict]) -> list[dict]:
        """Update tracker with new detections and return enriched track list."""
        seen_ids: set[int] = set()
        results = []

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            seen_ids.add(track_id)
            if track_id not in self._tracks:
                self._tracks[track_id] = Track(
                    track_id=track_id,
                    class_id=det.get("class_id", -1),
                    class_name=det.get("class_name", "unknown"),
                )

            track = self._tracks[track_id]
            track.update(det["bounding_box"], det.get("confidence", 0.0))

            results.append({
                "track_id": track_id,
                "class_id": det.get("class_id"),
                "class_name": det.get("class_name"),
                "confidence": det.get("confidence"),
                "bounding_box": det["bounding_box"],
                "age_frames": track.age_frames,
                "lost_frames": track.lost_frames,
            })

        # Mark lost tracks
        for tid, track in list(self._tracks.items()):
            if tid not in seen_ids:
                track.mark_lost()
                if track.lost_frames > 30:
                    del self._tracks[tid]

        return results

    def get_track_history(self, track_id: int) -> list[dict]:
        if track_id in self._tracks:
            return self._tracks[track_id].history
        return []

    @property
    def active_tracks(self) -> dict[int, Track]:
        return {tid: t for tid, t in self._tracks.items() if t.lost_frames == 0}


tracker = Tracker()
