"""Persistent object positions expressed in a shared world coordinate frame.

The desktop fallback uses the existing visual-SLAM heading.  A mobile ARCore
or VIO client can post metric camera poses through the API; the same memory
and query code then works without any platform-specific branches.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass
class CameraPose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0
    source: str = "slam_heading"
    timestamp: float = 0.0
    translation_timestamp: float = 0.0


class CameraPoseTracker:
    """Pose adapter for ARCore/VIO, with a safe heading-only desktop fallback."""

    def __init__(self):
        self._pose = CameraPose(timestamp=time.time())

    def update_slam_heading(self, yaw_deg: float) -> None:
        # Prefer fresh ARCore/VIO metric poses.  Monocular desktop SLAM has no
        # trustworthy metric translation without IMU/scale input, so preserve
        # its origin and update only orientation.
        now = time.time()
        if self._pose.source in {"arcore", "vio"} and now - self._pose.translation_timestamp < 2.0:
            return
        self._pose.yaw_deg = float(yaw_deg) % 360.0
        if (self._pose.source == "landmark_slam" and
                now - self._pose.translation_timestamp > 2.0):
            self._pose.source = "slam_heading"
        elif self._pose.source not in {"landmark_slam", "arcore", "vio"}:
            self._pose.source = "slam_heading"
        self._pose.timestamp = now

    def update_external_pose(self, x: float, y: float, z: float, yaw_deg: float,
                             source: str = "arcore") -> None:
        if source not in {"arcore", "vio"}:
            raise ValueError("source must be 'arcore' or 'vio'")
        self._pose = CameraPose(
            x=float(x), y=float(y), z=float(z), yaw_deg=float(yaw_deg) % 360.0,
            source=source, timestamp=time.time(), translation_timestamp=time.time(),
        )

    def update_landmark_translation(self, x: float, y: float, z: float,
                                    landmark_count: int) -> bool:
        """Fuse metric depth landmarks into the heading-only desktop pose."""
        if self._pose.source in {"arcore", "vio"}:
            return False
        dx, dy, dz = float(x) - self._pose.x, float(y) - self._pose.y, float(z) - self._pose.z
        step = math.hypot(dx, dz)
        if step < 0.025:
            return False
        # Reject/cap one-frame depth spikes while allowing normal walking.
        if step > 0.6:
            scale = 0.6 / step
            dx, dz = dx * scale, dz * scale
        alpha = 0.55 if landmark_count >= 2 else 0.38
        self._pose.x += alpha * dx
        self._pose.y += alpha * dy
        self._pose.z += alpha * dz
        self._pose.source = "landmark_slam"
        self._pose.timestamp = time.time()
        self._pose.translation_timestamp = self._pose.timestamp
        return True

    @property
    def pose(self) -> CameraPose:
        return CameraPose(**asdict(self._pose))


@dataclass
class StoredWorldObject:
    id: str
    label: str
    track_id: int
    world_x: float
    world_y: float
    world_z: float
    confidence: float
    last_seen: float
    visible: bool = False


class PersistentSpatialMemory:
    """Tiny JSON-backed world-memory store for the hackathon prototype."""

    VISIBILITY_GRACE_SECONDS = 1.5
    ASSOCIATION_DISTANCE_M = 0.8
    MOVE_CONFIRM_FRAMES = 3
    MOVE_CONSISTENCY_M = 0.45
    TRANSIENT_EXPIRY_SECONDS = 24 * 3600

    def __init__(self, path: str):
        self._path = Path(path)
        self._objects: Dict[str, StoredWorldObject] = {}
        self._pending_moves: Dict[str, dict] = {}
        self._load()

    def update_tracks(self, tracks: Iterable, pose: CameraPose) -> list[dict]:
        """Update visible objects and return confirmed, meaningful changes."""
        now = time.time()
        changed = False
        events: list[dict] = []
        seen_object_ids: set[str] = set()
        # Keep an old confirmed observation available for cautious verbal
        # context, but remove transient/abandoned map points after a day.
        expired = [key for key, obj in self._objects.items()
                   if now - obj.last_seen > self.TRANSIENT_EXPIRY_SECONDS]
        for key in expired:
            del self._objects[key]
            self._pending_moves.pop(key, None)
            changed = True
        for track in tracks:
            if getattr(track, "frames_since_seen", 0) != 0:
                continue
            wx, wy, wz = self._camera_to_world(track, pose)
            previous = self._match_object(track.label, track.id, wx, wz, now)
            key = previous.id if previous else self._new_id(track.label, track.id, now)
            old_position = None
            if previous is not None:
                old_position = (previous.world_x, previous.world_y, previous.world_z)
                displacement = math.dist(old_position, (wx, wy, wz))
                if displacement > self.ASSOCIATION_DISTANCE_M:
                    confirmed = self._confirm_move(key, (wx, wy, wz), now)
                    if confirmed is None:
                        wx, wy, wz = old_position
                    else:
                        wx, wy, wz = confirmed
                        events.append({
                            "type": "moved", "object_id": key,
                            "label": previous.label,
                            "from": {"x": round(old_position[0], 3),
                                     "y": round(old_position[1], 3),
                                     "z": round(old_position[2], 3)},
                            "to": {"x": round(wx, 3), "y": round(wy, 3),
                                   "z": round(wz, 3)},
                            "distance_m": round(displacement, 2),
                        })
                else:
                    self._pending_moves.pop(key, None)
                    if pose.source in {"slam_heading", "landmark_slam"}:
                        # In desktop fallback these objects are the landmarks
                        # used to estimate camera motion, so keep them anchored.
                        wx, wy, wz = old_position
                    else:
                        # Metric ARCore/VIO can safely refine landmark position.
                        wx = previous.world_x * 0.65 + wx * 0.35
                        wy = previous.world_y * 0.65 + wy * 0.35
                        wz = previous.world_z * 0.65 + wz * 0.35
            self._objects[key] = StoredWorldObject(
                id=key, label=track.label, track_id=track.id,
                world_x=round(wx, 3), world_y=round(wy, 3), world_z=round(wz, 3),
                confidence=round(float(track.det.confidence), 3), last_seen=now,
                visible=True,
            )
            seen_object_ids.add(key)
            changed = True

        for key, obj in self._objects.items():
            if (key not in seen_object_ids and obj.visible and
                    now - obj.last_seen >= self.VISIBILITY_GRACE_SECONDS):
                obj.visible = False
                events.append({
                    "type": "disappeared", "object_id": key, "label": obj.label,
                    "direction": self._direction_for_object(obj, pose),
                    "last_seen": obj.last_seen,
                })
                changed = True
        if changed:
            self._save()
        return events

    def estimate_camera_position(self, tracks: Iterable,
                                 pose: CameraPose) -> Optional[tuple[float, float, float, int]]:
        """Solve camera translation from stable world landmarks and live depth.

        Each known landmark provides `camera = world - R(yaw) * local`. The
        median across visible static objects rejects individual depth outliers.
        """
        dynamic_labels = {"person", "dog", "cat", "car", "bicycle", "motorcycle"}
        candidates = []
        yaw = math.radians(pose.yaw_deg)
        by_track = {obj.track_id: obj for obj in self._objects.values()}
        for track in tracks:
            obj = by_track.get(int(track.id))
            if obj is None or obj.label.lower() in dynamic_labels:
                continue
            confidence = float(getattr(getattr(track, "det", None), "confidence", 0.0))
            local_z = float(getattr(track, "translation_z", 0.0))
            if confidence < 0.5 or local_z <= 0.15:
                continue
            local_x = float(getattr(track, "translation_x", 0.0))
            local_y = -float(getattr(track, "translation_y", 0.0))
            offset_x = math.cos(yaw) * local_x + math.sin(yaw) * local_z
            offset_z = -math.sin(yaw) * local_x + math.cos(yaw) * local_z
            candidates.append((
                obj.world_x - offset_x,
                obj.world_y - local_y,
                obj.world_z - offset_z,
            ))
        if not candidates:
            return None
        xs = sorted(item[0] for item in candidates)
        ys = sorted(item[1] for item in candidates)
        zs = sorted(item[2] for item in candidates)
        middle = len(candidates) // 2
        median = lambda values: values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
        return median(xs), median(ys), median(zs), len(candidates)

    def find_object(self, label: str, pose: CameraPose,
                    visible_track_ids: Optional[Iterable[int]] = None) -> Optional[dict]:
        target = label.lower().strip()
        candidates = [obj for obj in self._objects.values() if obj.label.lower() == target]
        if not candidates:
            return None
        obj = max(candidates, key=lambda item: (item.last_seen, item.confidence))
        return self._locate_object(obj, pose, visible_track_ids)

    def find_object_by_id(self, object_id: str, pose: CameraPose,
                          visible_track_ids: Optional[Iterable[int]] = None) -> Optional[dict]:
        """Locate one locked memory instance without switching by label."""
        obj = self._objects.get(object_id)
        if obj is None:
            return None
        return self._locate_object(obj, pose, visible_track_ids)

    def _locate_object(self, obj: StoredWorldObject, pose: CameraPose,
                       visible_track_ids: Optional[Iterable[int]]) -> dict:
        dx, dy, dz = obj.world_x - pose.x, obj.world_y - pose.y, obj.world_z - pose.z
        # Navigation distance is horizontal walking distance. Height is kept in
        # world memory but must not make a tabletop object appear farther away.
        distance = math.hypot(dx, dz)
        direction = self._relative_direction(dx, dz, pose.yaw_deg)
        relative_x, relative_z = self._world_delta_to_camera(dx, dz, pose.yaw_deg)
        heading_error = self._heading_error(relative_x, relative_z)
        world_bearing = (math.degrees(math.atan2(dx, dz)) + 360.0) % 360.0
        age_s = max(0, int(time.time() - obj.last_seen))
        visible_ids = set(visible_track_ids) if visible_track_ids is not None else None
        visible = obj.visible and (visible_ids is None or obj.track_id in visible_ids)
        return {
            "id": obj.id,
            "object": obj.label,
            "tracking_id": obj.track_id,
            "visible": visible,
            "distance": round(distance, 2),
            "direction": direction,
            "bearing_deg": round(world_bearing, 1),
            "heading_error_deg": round(heading_error, 1),
            "last_seen": self._time_ago(age_s),
            "last_seen_timestamp": obj.last_seen,
            "world_coordinates": {"x": obj.world_x, "y": obj.world_y, "z": obj.world_z},
            "relative_coordinates": {"x": round(relative_x, 3), "y": round(dy, 3),
                                     "z": round(relative_z, 3)},
            "confidence": obj.confidence,
            "reliability": self._reliability(obj.confidence, age_s),
            "pose_source": pose.source,
        }

    def snapshot(self, pose: CameraPose,
                 visible_track_ids: Optional[Iterable[int]] = None) -> list[dict]:
        """Return every persistent object relative to the latest camera pose."""
        visible_ids = set(visible_track_ids) if visible_track_ids is not None else None
        result = []
        for obj in sorted(self._objects.values(), key=lambda item: item.last_seen, reverse=True):
            dx, dy, dz = obj.world_x - pose.x, obj.world_y - pose.y, obj.world_z - pose.z
            relative_x, relative_z = self._world_delta_to_camera(dx, dz, pose.yaw_deg)
            age_s = max(0, int(time.time() - obj.last_seen))
            result.append({
                "id": obj.id, "object": obj.label, "label": obj.label,
                "tracking_id": obj.track_id, "track_id": obj.track_id,
                "visible": obj.visible and (visible_ids is None or obj.track_id in visible_ids),
                "distance": round(math.hypot(dx, dz), 2),
                "distance_m": round(math.hypot(dx, dz), 2),
                "direction": self._relative_direction(dx, dz, pose.yaw_deg),
                "bearing_deg": round((math.degrees(math.atan2(dx, dz)) + 360.0) % 360.0, 1),
                "heading_error_deg": round(self._heading_error(relative_x, relative_z), 1),
                "last_seen": self._time_ago(age_s), "time_ago": self._time_ago(age_s),
                "last_seen_timestamp": obj.last_seen,
                "world_coordinates": {"x": obj.world_x, "y": obj.world_y, "z": obj.world_z},
                "relative_coordinates": {"x": round(relative_x, 3), "y": round(dy, 3),
                                         "z": round(relative_z, 3)},
                "translation_x": round(relative_x, 3),
                "translation_z": round(relative_z, 3),
                "confidence": obj.confidence,
                "reliability": self._reliability(obj.confidence, age_s),
            })
        return result

    def summary(self) -> dict:
        return {
            "stored_objects": len(self._objects),
            "visible_objects": sum(1 for obj in self._objects.values() if obj.visible),
            "path": str(self._path),
        }

    def _match_object(self, label: str, track_id: int, wx: float, wz: float,
                      now: float) -> Optional[StoredWorldObject]:
        normalised = label.lower().strip()
        candidates = [obj for obj in self._objects.values()
                      if obj.label.lower().strip() == normalised]
        exact = next((obj for obj in candidates if obj.track_id == track_id), None)
        if exact:
            return exact
        if not candidates:
            return None
        nearest = min(candidates, key=lambda obj: math.hypot(obj.world_x - wx, obj.world_z - wz))
        separation = math.hypot(nearest.world_x - wx, nearest.world_z - wz)
        # A tracker commonly assigns a new ID after an object briefly leaves view.
        if separation <= self.ASSOCIATION_DISTANCE_M:
            return nearest
        if len(candidates) == 1 and now - nearest.last_seen < 30.0:
            return nearest
        return None

    def _confirm_move(self, key: str, position: tuple[float, float, float],
                      now: float) -> Optional[tuple[float, float, float]]:
        pending = self._pending_moves.get(key)
        if (pending is None or now - pending["timestamp"] > 2.0 or
                math.dist(pending["position"], position) > self.MOVE_CONSISTENCY_M):
            self._pending_moves[key] = {"position": position, "count": 1, "timestamp": now}
            return None
        count = pending["count"] + 1
        averaged = tuple((a * pending["count"] + b) / count
                         for a, b in zip(pending["position"], position))
        self._pending_moves[key] = {"position": averaged, "count": count, "timestamp": now}
        if count < self.MOVE_CONFIRM_FRAMES:
            return None
        self._pending_moves.pop(key, None)
        return averaged

    def _new_id(self, label: str, track_id: int, now: float) -> str:
        base = f"{label}:{track_id}"
        return base if base not in self._objects else f"{base}:{int(now * 1000)}"

    @classmethod
    def _direction_for_object(cls, obj: StoredWorldObject, pose: CameraPose) -> str:
        return cls._relative_direction(obj.world_x - pose.x, obj.world_z - pose.z,
                                       pose.yaw_deg)

    @staticmethod
    def _reliability(confidence: float, age_s: int) -> str:
        effective = confidence * math.pow(0.5, age_s / 7200.0)
        if effective >= 0.65 and age_s < 300:
            return "reliable"
        if effective >= 0.35 and age_s < 3600:
            return "likely"
        return "uncertain"

    @staticmethod
    def _camera_to_world(track, pose: CameraPose) -> tuple[float, float, float]:
        # Camera coordinates: +X right, +Y down, +Z forward.  World uses +Y up.
        local_x = float(getattr(track, "translation_x", 0.0))
        local_y = -float(getattr(track, "translation_y", 0.0))
        local_z = float(getattr(track, "translation_z", 0.0))
        yaw = math.radians(pose.yaw_deg)
        return (
            pose.x + math.cos(yaw) * local_x + math.sin(yaw) * local_z,
            pose.y + local_y,
            pose.z - math.sin(yaw) * local_x + math.cos(yaw) * local_z,
        )

    @staticmethod
    def _relative_direction(dx: float, dz: float, yaw_deg: float) -> str:
        local_x, local_z = PersistentSpatialMemory._world_delta_to_camera(
            dx, dz, yaw_deg
        )
        angle = math.degrees(math.atan2(local_x, local_z))
        labels = ("front", "front-right", "right", "behind-right",
                  "behind", "behind-left", "left", "front-left")
        return labels[int((angle + 22.5) % 360 // 45)]

    @staticmethod
    def _world_delta_to_camera(dx: float, dz: float,
                               yaw_deg: float) -> tuple[float, float]:
        yaw = math.radians(yaw_deg)
        return (
            math.cos(yaw) * dx - math.sin(yaw) * dz,
            math.sin(yaw) * dx + math.cos(yaw) * dz,
        )

    @staticmethod
    def _heading_error(local_x: float, local_z: float) -> float:
        angle = math.degrees(math.atan2(local_x, local_z))
        return (angle + 180.0) % 360.0 - 180.0

    @staticmethod
    def _time_ago(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds} seconds ago"
        return f"{seconds // 60} minutes ago"

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._objects = {}
            for item in raw.get("objects", []):
                # A process restart has no live frame yet. Preserve the object,
                # but require a fresh track before calling it visible again.
                item["visible"] = False
                obj = StoredWorldObject(**item)
                self._objects[obj.id] = obj
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            self._objects = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(json.dumps({"objects": [asdict(obj) for obj in self._objects.values()]}, indent=2),
                             encoding="utf-8")
        os.replace(temporary, self._path)
