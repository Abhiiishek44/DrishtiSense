"""Deterministic goal navigation and meaningful spatial-scene changes.

This module intentionally has no model or database dependencies.  It consumes
the existing pose, persistent-memory, detector-track, and safe-path outputs so
it can later be replaced by a full planner without changing those producers.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

from spatial_memory import CameraPose, PersistentSpatialMemory


_ALIASES = {
    "phone": "cell phone", "mobile": "cell phone", "photo frame": "picture frame",
    "photo picture": "picture frame",
    "photograph frame": "picture frame", "key": "keys", "keychain": "keys",
    "sofa": "couch", "television": "tv",
    "eye glasses": "eyeglasses", "glasses": "eyeglasses", "spectacles": "eyeglasses",
}
_GOAL_PREFIXES = (
    r"take me to\s+", r"guide me to\s+", r"navigate me to\s+",
    r"lead me to\s+", r"walk me to\s+", r"go to\s+",
)
_LOCATION_PREFIXES = (
    r"where (?:is|are)\s+", r"where did i (?:put|leave)\s+",
    r"where can i find\s+", r"find\s+", r"locate\s+", r"show me\s+",
    r"help me find\s+",
)


def _normalise_target(value: str) -> str:
    target = re.sub(r"[^a-z0-9 ]+", " ", value.lower()).strip()
    target = re.sub(r"^(?:my|the|a|an)\s+", "", target)
    target = re.sub(r"\s+(?:please|for me)$", "", target).strip()
    return _ALIASES.get(target, target)


def parse_goal_command(text: str) -> Optional[str]:
    normalised = text.lower().strip()
    for pattern in _GOAL_PREFIXES:
        match = re.search(pattern, normalised)
        if match:
            target = _normalise_target(normalised[match.end():])
            return target or None
    return None


def parse_location_query(text: str) -> Optional[str]:
    normalised = text.lower().strip()
    for pattern in _LOCATION_PREFIXES:
        match = re.search(pattern, normalised)
        if match:
            target = _normalise_target(normalised[match.end():])
            return target or None
    return None


def natural_direction(direction: str) -> str:
    return {
        "front": "directly ahead", "front-right": "ahead and slightly to your right",
        "right": "to your right", "behind-right": "behind you and to your right",
        "behind": "behind you", "behind-left": "behind you and to your left",
        "left": "to your left", "front-left": "ahead and slightly to your left",
    }.get(direction, direction.replace("-", " "))


@dataclass
class NavigationGoal:
    id: str
    label: str
    target_object_id: str
    status: str
    started_at: float
    updated_at: float
    last_visual_confirmation: float
    tracking_state: str = "remembered"
    phase: str = "rotate"
    aligned: bool = False
    smoothed_heading_error: Optional[float] = None
    smoothed_distance: Optional[float] = None
    last_command: str = ""
    last_emitted_distance: Optional[float] = None
    last_emitted_error: Optional[float] = None


class GoalNavigator:
    """A small state machine that turns world-memory geometry into commands."""

    DEFAULT_ARRIVAL_DISTANCE_M = 0.65
    ALIGN_ENTER_DEG = 10.0
    ALIGN_EXIT_DEG = 18.0
    TURN_IN_PLACE_DEG = 25.0
    SLIGHT_ADJUST_DEG = 10.0
    ANGLE_SMOOTHING = 0.45
    DISTANCE_SMOOTHING = 0.55
    UPDATE_INTERVAL_SECONDS = 0.5
    # Large/solid targets are destinations and obstacles at the same time.
    # Complete guidance at a safe stand-off instead of walking the user into
    # the object to satisfy the generic small-item arrival threshold.
    TARGET_STANDOFF_M = {
        "person": 1.2, "chair": 0.9, "bench": 0.9,
        "couch": 0.9, "bed": 0.9, "dining table": 0.9,
        "toilet": 0.8, "refrigerator": 0.9,
    }

    def __init__(self, arrival_distance_m: float = DEFAULT_ARRIVAL_DISTANCE_M):
        self._goal: Optional[NavigationGoal] = None
        self._arrival_distance_m = max(0.3, float(arrival_distance_m))

    @property
    def active_label(self) -> Optional[str]:
        return self._goal.label if self._goal and self._goal.status == "active" else None

    def start(self, label: str, memory: PersistentSpatialMemory, pose: CameraPose,
              visible_track_ids: Iterable[int] = ()) -> dict:
        if self._goal and self._goal.status == "active":
            current = memory.find_object_by_id(
                self._goal.target_object_id, pose, visible_track_ids
            )
            if current is not None:
                event = self._build_update(
                    current, safe_path=None, obstacle=None, force=True
                )
                if event and label.lower().strip() != self._goal.label.lower():
                    event["text"] = (
                        f"Navigation is locked to {self._goal.label}. "
                        "Cancel it before choosing another target."
                    )
                return event
        target = memory.find_object(label, pose, visible_track_ids)
        if target is None:
            return self._event(
                status="not_found", label=label,
                text=f"I have not seen {label} yet. Scan the room first.",
            )
        now = time.time()
        if target.get("reliability") == "uncertain" and not target.get("visible"):
            return self._event(
                status="unreliable", label=target["object"],
                text=f"My memory of {target['object']} is too old to guide you safely.",
                target=target,
            )
        self._goal = NavigationGoal(
            id=str(uuid.uuid4()), label=target["object"], status="active",
            target_object_id=target["id"], started_at=now, updated_at=0.0,
            last_visual_confirmation=float(target.get("last_seen_timestamp", now)),
            tracking_state="visible" if target.get("visible") else "remembered",
        )
        return self._build_update(target, safe_path=None, obstacle=None, force=True)

    def cancel(self) -> dict:
        if not self._goal or self._goal.status != "active":
            return self._event("idle", "", "There is no active navigation goal.")
        self._goal.status = "cancelled"
        return self._event("cancelled", self._goal.label, "Navigation cancelled.")

    def update(self, memory: PersistentSpatialMemory, pose: CameraPose,
               visible_track_ids: Iterable[int], safe_path: Optional[dict],
               obstacle: Optional[dict] = None,
               force: bool = False) -> Optional[dict]:
        if not self._goal or self._goal.status != "active":
            return None
        target = memory.find_object_by_id(
            self._goal.target_object_id, pose, visible_track_ids
        )
        if target is None:
            self._goal.status = "lost"
            return self._event(
                "lost", self._goal.label,
                f"I can no longer locate {self._goal.label}. Stop and scan around.",
            )
        if target.get("reliability") == "uncertain" and not target.get("visible"):
            self._goal.status = "unreliable"
            self._goal.tracking_state = "unreliable"
            return self._event(
                "unreliable", self._goal.label,
                f"My memory of {self._goal.label} is no longer reliable. Stop and scan around.",
                target,
            )
        if target.get("visible"):
            self._goal.last_visual_confirmation = float(
                target.get("last_seen_timestamp", time.time())
            )
            self._goal.tracking_state = "visible"
        else:
            self._goal.tracking_state = (
                "remembered" if target.get("pose_source") in {"arcore", "vio", "landmark_slam"}
                else "orientation_only"
            )
        return self._build_update(target, safe_path, obstacle, force=force)

    def snapshot(self) -> dict:
        return asdict(self._goal) if self._goal else {"status": "idle"}

    def _build_update(self, target: dict, safe_path: Optional[dict],
                      obstacle: Optional[dict],
                      force: bool = False) -> Optional[dict]:
        assert self._goal is not None
        raw_distance = float(target["distance"])
        raw_error = float(target.get("heading_error_deg", 0.0))
        now = time.time()

        if self._goal.smoothed_heading_error is None:
            heading_error = raw_error
        else:
            delta = self._wrap_angle(raw_error - self._goal.smoothed_heading_error)
            # A deliberate turn can change bearing by tens of degrees between
            # pose packets. Snap to it; smoothing is only for small pose noise.
            heading_error = raw_error if abs(delta) > 45.0 else self._wrap_angle(
                self._goal.smoothed_heading_error + self.ANGLE_SMOOTHING * delta
            )
        if self._goal.smoothed_distance is None:
            distance = raw_distance
        else:
            distance = (
                self._goal.smoothed_distance * (1.0 - self.DISTANCE_SMOOTHING) +
                raw_distance * self.DISTANCE_SMOOTHING
            )
        self._goal.smoothed_heading_error = heading_error
        self._goal.smoothed_distance = distance

        arrival_distance = max(
            self._arrival_distance_m,
            self.TARGET_STANDOFF_M.get(str(target.get("object", "")).lower(), 0.0),
        )
        if raw_distance <= arrival_distance and abs(raw_error) <= self.ALIGN_EXIT_DEG:
            self._goal.status = "complete"
            self._goal.tracking_state = "arrived"
            uses_standoff = arrival_distance > self._arrival_distance_m
            text = (
                f"Stop here. Your {self._goal.label} is directly ahead, "
                f"about {raw_distance:.1f} metres away."
                if uses_standoff else
                f"You are here. Your {self._goal.label} is directly ahead."
            )
            return self._event("complete", self._goal.label, text, target,
                               "reach", "◎ STOP HERE · TARGET AHEAD" if uses_standoff
                               else "◎ YOU'RE HERE", distance, heading_error)

        abs_error = abs(heading_error)
        was_aligned = self._goal.aligned
        if self._goal.aligned:
            self._goal.aligned = abs_error <= self.ALIGN_EXIT_DEG
        else:
            self._goal.aligned = abs_error <= self.ALIGN_ENTER_DEG

        obstacle_is_critical = bool(
            obstacle and float(obstacle.get("distance_m", 99.0)) <= 0.8
        )
        if obstacle_is_critical or (safe_path and safe_path.get("status") == "blocked"):
            label = obstacle.get("label", "obstacle") if obstacle else "obstacle"
            command, status = "stop", "blocked"
            hud = "STOP · PATH BLOCKED"
            text = f"Stop. A {label} is blocking your path."
        elif abs_error > self.TURN_IN_PLACE_DEG:
            self._goal.phase = "rotate"
            degrees = int(round(abs_error / 5.0) * 5)
            if abs_error >= 165.0:
                command, hud, text = "turn_around", "↻ TURN AROUND", "Turn around."
            else:
                side = "right" if heading_error > 0 else "left"
                continuing = self._goal.last_command in {f"turn_{side}", f"continue_{side}"}
                command = f"continue_{side}" if continuing else f"turn_{side}"
                verb = "CONTINUE TURNING" if continuing else "TURN"
                arrow = "↻" if side == "right" else "↺"
                hud = f"{arrow} {verb} {side.upper()} · {degrees}°"
                text = f"{'Continue turning' if continuing else 'Turn'} {side} about {degrees} degrees."
            status = "active"
        elif obstacle and safe_path and safe_path.get("direction") in {"left", "right"}:
            self._goal.phase = "move"
            side = safe_path["direction"]
            command, status = f"avoid_{side}", "active"
            arrow = "↖" if side == "left" else "↗"
            hud = f"{arrow} SLIGHTLY {side.upper()} · {distance:.1f}m"
            text = (
                f"Move slightly {side} to avoid the {obstacle['label']}, "
                f"then continue toward the {self._goal.label}."
            )
        elif abs_error >= self.SLIGHT_ADJUST_DEG:
            self._goal.phase = "move"
            side = "right" if heading_error > 0 else "left"
            command, status = f"slight_{side}", "active"
            arrow = "↗" if side == "right" else "↖"
            hud = f"{arrow} SLIGHTLY {side.upper()} · {distance:.1f}m"
            text = f"Move slightly {side}. About {distance:.1f} metres to go."
        else:
            self._goal.phase = "move"
            command, status = ("aligned" if not was_aligned else "walk_straight"), "active"
            hud = f"↑ {'ALIGNED' if command == 'aligned' else 'WALK STRAIGHT'} · {distance:.1f}m"
            text = (
                f"Aligned. Walk straight. About {distance:.1f} metres to go."
                if command == "aligned" else f"Walk straight. About {distance:.1f} metres to go."
            )

        should_emit = (
            force or command != self._goal.last_command or
            self._goal.last_emitted_distance is None or
            abs(distance - self._goal.last_emitted_distance) >= 0.05 or
            self._goal.last_emitted_error is None or
            abs(self._wrap_angle(heading_error - self._goal.last_emitted_error)) >= 3.0 or
            now - self._goal.updated_at >= self.UPDATE_INTERVAL_SECONDS
        )
        self._goal.last_command = command
        if not should_emit:
            return None
        self._goal.updated_at = now
        self._goal.last_emitted_distance = distance
        self._goal.last_emitted_error = heading_error
        if self._goal.tracking_state == "orientation_only" and command in {
            "walk_straight", "aligned", "slight_left", "slight_right"
        }:
            text += " Live walking distance needs an ARCore or VIO pose."
        return self._event(status, self._goal.label, text, target, command, hud,
                           distance, heading_error)

    def _event(self, status: str, label: str, text: str,
               target: Optional[dict] = None, command: str = "",
               hud: str = "", distance: Optional[float] = None,
               heading_error: Optional[float] = None) -> dict:
        return {
            "type": "goal_update", "goal_id": self._goal.id if self._goal else None,
            "target": label, "status": status, "text": text,
            "target_object_id": self._goal.target_object_id if self._goal else None,
            "command": command, "hud": hud,
            "distance_m": round(distance, 2) if distance is not None else (
                target.get("distance") if target else None
            ),
            "direction": target.get("direction") if target else None,
            "bearing_deg": target.get("bearing_deg") if target else None,
            "heading_error_deg": round(heading_error, 1) if heading_error is not None else (
                target.get("heading_error_deg") if target else None
            ),
            "visible": target.get("visible") if target else False,
            "last_visual_confirmation": (
                self._goal.last_visual_confirmation if self._goal else
                target.get("last_seen_timestamp") if target else None
            ),
            "tracking_state": self._goal.tracking_state if self._goal else "idle",
            "target_world_position": target.get("world_coordinates") if target else None,
            "target_tracking_id": target.get("tracking_id") if target else None,
            "pose_source": target.get("pose_source") if target else None,
        }

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (float(value) + 180.0) % 360.0 - 180.0


class EnvironmentalChangeDetector:
    """Debounces detector/memory changes before producing accessibility alerts."""

    OBSTACLE_DISTANCE_M = 1.5
    OBSTACLE_CONFIRM_FRAMES = 2
    EVENT_COOLDOWN_SECONDS = 8.0
    NON_BLOCKING = {
        "picture frame", "mirror", "tv", "television", "clock", "cup", "bottle",
        "cell phone", "remote", "keyboard", "mouse", "book", "keys", "eyeglasses",
    }

    def __init__(self):
        self._initialised = False
        self._blocking_track_ids: set[int] = set()
        self._pending_obstacles: dict[int, dict] = {}
        self._last_event: dict[str, float] = {}
        self._path_was_clear = False

    def evaluate(self, tracks: Iterable, memory_events: Iterable[dict],
                 safe_path: Optional[dict], active_goal_label: Optional[str]) -> list[dict]:
        now = time.time()
        tracks = list(tracks)
        output: list[dict] = []
        current_ids = {int(track.id) for track in tracks}

        if not self._initialised:
            self._blocking_track_ids = {
                int(track.id) for track in tracks if self._is_blocking(track)
            }
            self._path_was_clear = bool(safe_path and safe_path.get("status") == "clear")
            self._initialised = True
        else:
            for track in tracks:
                track_id = int(track.id)
                label = str(getattr(track, "label", "object"))
                if active_goal_label and label.lower() == active_goal_label.lower():
                    self._blocking_track_ids.discard(track_id)
                    self._pending_obstacles.pop(track_id, None)
                    continue
                blocking = self._is_blocking(track)
                if track_id in self._blocking_track_ids and blocking:
                    continue
                if not blocking:
                    self._blocking_track_ids.discard(track_id)
                    self._pending_obstacles.pop(track_id, None)
                    continue
                distance = float(getattr(track, "smoothed_distance", 99.0))
                pending = self._pending_obstacles.setdefault(track_id, {
                    "count": 0, "was_clear": self._path_was_clear,
                })
                pending["count"] += 1
                if pending["count"] >= self.OBSTACLE_CONFIRM_FRAMES:
                    self._blocking_track_ids.add(track_id)
                    self._pending_obstacles.pop(track_id, None)
                    if pending["was_clear"] and self._allow(f"obstacle:{label}", now):
                        level = "critical" if distance <= 1.0 else "warning"
                        message = f"Stop. A {label} is now blocking your path."
                        output.append(self._alert(level, label, distance, message, "environment_change"))

            for event in memory_events:
                label = str(event.get("label", "object"))
                if event.get("type") == "moved" and self._allow(f"moved:{label}", now):
                    message = f"The {label} has moved from where I last saw it."
                    if active_goal_label and label.lower() == active_goal_label.lower():
                        message = f"The {label} moved. I am updating your route."
                    output.append(self._alert("warning", label, None, message, "object_moved"))
                elif (event.get("type") == "disappeared" and active_goal_label and
                      label.lower() == active_goal_label.lower() and
                      str(event.get("direction", "")).startswith("front") and
                      self._allow(f"missing:{label}", now)):
                    output.append(self._alert(
                        "warning", label, None,
                        f"The {label} is no longer where I last saw it.", "object_missing",
                    ))

        near_centre = any(
            str(getattr(track, "label", "")).lower() not in self.NON_BLOCKING and
            (not active_goal_label or str(getattr(track, "label", "")).lower() != active_goal_label.lower()) and
            float(getattr(track, "smoothed_distance", 99.0)) <= self.OBSTACLE_DISTANCE_M and
            abs(float(getattr(track, "azimuth_deg", 90.0))) <= 28.0
            for track in tracks
        )
        self._path_was_clear = bool(
            safe_path and safe_path.get("status") == "clear" and not near_centre
        )
        self._blocking_track_ids.intersection_update(current_ids)
        return output

    def _is_blocking(self, track) -> bool:
        label = str(getattr(track, "label", "object"))
        distance = float(getattr(track, "smoothed_distance", 99.0))
        azimuth = abs(float(getattr(track, "azimuth_deg", 90.0)))
        confidence = float(getattr(getattr(track, "det", None), "confidence", 0.0))
        return (
            label.lower() not in self.NON_BLOCKING and
            distance <= self.OBSTACLE_DISTANCE_M and azimuth <= 28.0 and
            confidence >= 0.5
        )

    def _allow(self, key: str, now: float) -> bool:
        if now - self._last_event.get(key, 0.0) < self.EVENT_COOLDOWN_SECONDS:
            return False
        self._last_event[key] = now
        return True

    @staticmethod
    def _alert(level: str, label: str, distance_m: Optional[float], message: str,
               change_type: str) -> dict:
        return {
            "type": "safety_alert", "level": level, "label": label,
            "distance_m": round(distance_m, 2) if distance_m is not None else None,
            "clock_direction": "12 o'clock" if distance_m is not None else "",
            "message": message, "source": change_type,
        }
