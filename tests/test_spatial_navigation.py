import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from goal_navigation import EnvironmentalChangeDetector, GoalNavigator
from models import BoundingBox, Detection
from spatial_memory import CameraPose, CameraPoseTracker, PersistentSpatialMemory
from vision import DepthFusionEngine, HybridTargetDetector, Track, YOLODetector


def track(track_id, label, x, z, confidence=0.9, azimuth=0.0):
    return SimpleNamespace(
        id=track_id,
        label=label,
        translation_x=x,
        translation_y=0.0,
        translation_z=z,
        smoothed_distance=z,
        azimuth_deg=azimuth,
        frames_since_seen=0,
        det=SimpleNamespace(confidence=confidence),
    )


class SpatialNavigationScenarioTest(unittest.TestCase):
    def test_cropped_close_person_uses_safe_width_distance(self):
        detection = Detection(
            label="person", confidence=0.95,
            bbox=BoundingBox(x1=0, y1=0, x2=500, y2=478),
            frame_width=640, frame_height=480,
        )
        person = Track(id=4, label="person", det=detection, hits=3)
        depth = DepthFusionEngine(fov_h_deg=62.0)
        depth.calibrate(640, 480)

        distance, _velocity = depth.update(person)

        self.assertLess(distance, 0.8)

    def test_chair_goal_stops_at_safe_standoff(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentSpatialMemory(str(Path(directory) / "world.json"))
            pose = CameraPose()
            memory.update_tracks([track(9, "chair", 0.0, 0.8)], pose)

            event = GoalNavigator(arrival_distance_m=0.3).start(
                "chair", memory, pose, [9]
            )

            self.assertEqual(event["status"], "complete")
            self.assertEqual(event["command"], "reach")
            self.assertIn("Stop here", event["text"])
            self.assertIn("STOP HERE", event["hud"])

    def test_bottle_memory_goal_obstacle_and_arrival(self):
        """Requested demo: remember bottle, guide, warn, and complete."""
        with tempfile.TemporaryDirectory() as directory:
            memory_path = Path(directory) / "world.json"
            memory = PersistentSpatialMemory(str(memory_path))
            navigator = GoalNavigator()
            changes = EnvironmentalChangeDetector()
            facing_forward = CameraPose(yaw_deg=0.0, timestamp=100.0)

            # 1. Detect and persist a bottle three metres ahead.
            with patch("spatial_memory.time.time", return_value=100.0):
                memory.update_tracks([track(1, "bottle", 0.0, 3.0)], facing_forward)
            self.assertTrue(memory_path.exists())

            # 2. Lock this specific bottle as the target while it is visible.
            with patch("goal_navigation.time.time", return_value=100.0):
                started = navigator.start("bottle", memory, facing_forward, [1])
            self.assertEqual(started["status"], "active")
            self.assertEqual(started["command"], "aligned")
            locked_id = started["target_object_id"]

            # 3. Turn 180 degrees. It leaves view but remains world-anchored.
            facing_away = CameraPose(yaw_deg=180.0, timestamp=102.0)
            with patch("spatial_memory.time.time", return_value=102.0):
                memory.update_tracks([], facing_away)
                remembered = memory.find_object("bottle", facing_away, [])
            self.assertIsNotNone(remembered)
            self.assertFalse(remembered["visible"])
            self.assertEqual(remembered["direction"], "behind")
            self.assertAlmostEqual(remembered["distance"], 3.0)
            with patch("goal_navigation.time.time", return_value=102.0):
                turn = navigator.update(memory, facing_away, [], None, force=True)
            self.assertEqual(turn["command"], "turn_around")
            self.assertAlmostEqual(abs(turn["heading_error_deg"]), 180.0)

            # 4. Rotate back toward the target; movement starts only when aligned.
            with patch("goal_navigation.time.time", return_value=104.0):
                guidance = navigator.update(
                    memory, facing_forward, [],
                    {"status": "clear", "direction": "center", "clearance_m": 3.0},
                    force=True,
                )
            self.assertIn("Walk straight", guidance["text"])
            self.assertEqual(guidance["command"], "aligned")

            # 5. Metric VIO translation moves toward the fixed world point.
            first_step = CameraPose(z=1.0, yaw_deg=0.0, source="vio", timestamp=105.0)
            second_step = CameraPose(z=1.5, yaw_deg=0.0, source="vio", timestamp=106.0)
            with patch("goal_navigation.time.time", return_value=105.0):
                distance_one = navigator.update(
                    memory, first_step, [],
                    {"status": "clear", "direction": "center", "clearance_m": 3.0},
                    force=True,
                )
            with patch("goal_navigation.time.time", return_value=106.0):
                distance_two = navigator.update(
                    memory, second_step, [],
                    {"status": "clear", "direction": "center", "clearance_m": 3.0},
                    force=True,
                )
            self.assertLess(distance_two["distance_m"], distance_one["distance_m"])

            # 6. A chair blocks the target line; immediate movement shifts left.
            clear_path = {"status": "clear", "direction": "center", "clearance_m": 3.0}
            changes.evaluate([], [], clear_path, "bottle")
            chair = track(2, "chair", 0.0, 0.8, confidence=0.9)
            self.assertEqual(changes.evaluate([chair], [], clear_path, "bottle"), [])
            alerts = changes.evaluate([chair], [], clear_path, "bottle")
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]["level"], "critical")
            self.assertEqual(alerts[0]["message"], "Stop. A chair is now blocking your path.")
            detour_path = {"status": "clear", "direction": "left", "clearance_m": 1.8}
            with patch("goal_navigation.time.time", return_value=107.0):
                detour = navigator.update(
                    memory, second_step, [], detour_path,
                    obstacle={"label": "chair", "distance_m": 1.0}, force=True,
                )
            self.assertEqual(detour["command"], "avoid_left")
            self.assertIn("avoid the chair", detour["text"])

            # 7. Re-detection with a new tracker ID corrects the locked world point.
            with patch("spatial_memory.time.time", return_value=108.0):
                memory.update_tracks([track(3, "bottle", 0.0, 1.7)], second_step)
                corrected = memory.find_object_by_id(locked_id, second_step, [3])
            self.assertIsNotNone(corrected)
            self.assertTrue(corrected["visible"])
            self.assertEqual(corrected["tracking_id"], 3)
            self.assertGreater(corrected["world_coordinates"]["z"], 3.0)

            # 8. Continue to the corrected position and stop inside the threshold.
            near_bottle = CameraPose(z=2.55, yaw_deg=0.0, source="vio", timestamp=110.0)
            with patch("spatial_memory.time.time", return_value=110.0):
                memory.update_tracks([track(3, "bottle", 0.0, 0.5)], near_bottle)
            with patch("goal_navigation.time.time", return_value=110.0):
                completed = navigator.update(memory, near_bottle, [3], clear_path, force=True)
            self.assertEqual(completed["status"], "complete")
            self.assertIn("You are here", completed["text"])

            # Persistence survives a process restart.
            reloaded = PersistentSpatialMemory(str(memory_path))
            with patch("spatial_memory.time.time", return_value=110.0):
                persisted = reloaded.find_object("bottle", near_bottle)
            self.assertIsNotNone(persisted)
            self.assertFalse(persisted["visible"])
            self.assertLessEqual(persisted["distance"], 0.65)

    def test_camera_to_world_transform_tracks_rotation_and_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentSpatialMemory(str(Path(directory) / "world.json"))
            pose = CameraPose(x=1.0, z=1.0, yaw_deg=90.0, source="arcore")
            with patch("spatial_memory.time.time", return_value=10.0):
                memory.update_tracks([track(4, "bottle", 0.0, 2.0)], pose)
                location = memory.find_object("bottle", pose, [4])
            self.assertAlmostEqual(location["world_coordinates"]["x"], 3.0, places=2)
            self.assertAlmostEqual(location["world_coordinates"]["z"], 1.0, places=2)
            self.assertAlmostEqual(location["heading_error_deg"], 0.0, places=2)
            moved_pose = CameraPose(x=2.0, z=1.0, yaw_deg=90.0, source="arcore")
            with patch("spatial_memory.time.time", return_value=11.0):
                moved = memory.find_object("bottle", moved_pose, [])
            self.assertAlmostEqual(moved["distance"], 1.0, places=2)

    def test_goal_lock_and_turn_hysteresis(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentSpatialMemory(str(Path(directory) / "world.json"))
            forward = CameraPose(yaw_deg=0.0)
            with patch("spatial_memory.time.time", return_value=20.0):
                memory.update_tracks([
                    track(1, "bottle", 0.0, 3.0),
                    track(2, "door", 1.5, 4.0),
                ], forward)
            navigator = GoalNavigator()
            facing_right = CameraPose(yaw_deg=90.0)
            with patch("goal_navigation.time.time", return_value=20.0):
                first = navigator.start("bottle", memory, facing_right, [1, 2])
            self.assertEqual(first["command"], "turn_left")
            with patch("goal_navigation.time.time", return_value=20.2):
                continuing = navigator.update(memory, CameraPose(yaw_deg=80.0), [1, 2], None,
                                              force=True)
            self.assertEqual(continuing["command"], "continue_left")
            with patch("goal_navigation.time.time", return_value=20.4):
                refused = navigator.start("door", memory, CameraPose(yaw_deg=70.0), [1, 2])
            self.assertEqual(refused["target"], "bottle")
            self.assertIn("locked to bottle", refused["text"])

    def test_landmark_fallback_updates_distance_when_camera_moves(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentSpatialMemory(str(Path(directory) / "world.json"))
            pose_tracker = CameraPoseTracker()
            with patch("spatial_memory.time.time", return_value=30.0):
                memory.update_tracks([track(8, "bottle", 0.0, 3.0)], pose_tracker.pose)
                before = memory.find_object("bottle", pose_tracker.pose, [8])

            # The same anchored bottle now measures only 2.5m away. Solving
            # world = camera + R*local yields forward camera translation.
            closer_track = track(8, "bottle", 0.0, 2.5)
            estimate = memory.estimate_camera_position([closer_track], pose_tracker.pose)
            self.assertIsNotNone(estimate)
            x, y, z, count = estimate
            with patch("spatial_memory.time.time", return_value=30.2):
                self.assertTrue(pose_tracker.update_landmark_translation(x, y, z, count))
                memory.update_tracks([closer_track], pose_tracker.pose)
                after = memory.find_object("bottle", pose_tracker.pose, [8])
            self.assertEqual(pose_tracker.pose.source, "landmark_slam")
            self.assertLess(after["distance"], before["distance"])
            self.assertAlmostEqual(after["world_coordinates"]["z"], 3.0, places=2)

    def test_large_position_change_requires_three_consistent_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentSpatialMemory(str(Path(directory) / "world.json"))
            pose = CameraPose()
            with patch("spatial_memory.time.time", return_value=1.0):
                memory.update_tracks([track(1, "bottle", 0.0, 2.0)], pose)
            moved_events = []
            for frame_number in range(3):
                with patch("spatial_memory.time.time", return_value=2.0 + frame_number * 0.1):
                    moved_events = memory.update_tracks(
                        [track(1, "bottle", 1.2, 2.0)], pose
                    )
            self.assertEqual([event["type"] for event in moved_events], ["moved"])

    def test_existing_object_entering_clear_path_is_a_change(self):
        changes = EnvironmentalChangeDetector()
        clear_path = {"status": "clear", "direction": "center", "clearance_m": 3.0}
        chair_at_side = track(7, "chair", 0.8, 2.5, confidence=0.9, azimuth=45.0)
        changes.evaluate([chair_at_side], [], clear_path, None)
        chair_ahead = track(7, "chair", 0.0, 0.9, confidence=0.9, azimuth=0.0)
        self.assertEqual(changes.evaluate([chair_ahead], [], clear_path, None), [])
        alerts = changes.evaluate([chair_ahead], [], clear_path, None)
        self.assertEqual(alerts[0]["source"], "environment_change")


class OpenVocabularyPromptTest(unittest.TestCase):
    def test_photo_frame_and_shirt_expand_to_high_detail_prompts(self):
        detector = YOLODetector.__new__(YOLODetector)
        frame_prompts, frame_labels = detector._open_vocab_prompts(["photo frame"])
        shirt_prompts, shirt_labels = detector._open_vocab_prompts(["shirt"])

        self.assertIn("picture frame", frame_prompts)
        self.assertIn("framed photograph", frame_prompts)
        self.assertEqual(frame_labels["photo frame"], "picture frame")
        self.assertEqual(detector._canonical_target("photo picture"), "picture frame")
        self.assertIn("polo shirt", shirt_prompts)
        self.assertEqual(shirt_labels["shirt"], "shirt")

    def test_eyeglasses_aliases_use_a_focused_small_object_scan(self):
        detector = YOLODetector.__new__(YOLODetector)
        prompts, labels = detector._open_vocab_prompts(["eye glasses"])
        self.assertIn("eyeglasses", prompts)
        self.assertIn("spectacles", prompts)
        self.assertEqual(labels["reading glasses"], "eyeglasses")
        self.assertEqual(detector._canonical_target("spectacles"), "eyeglasses")

    def test_ambient_phone_guess_needs_a_stronger_score(self):
        detector = YOLODetector.__new__(YOLODetector)
        detector._confidence = 0.40
        detector._coco_model = object()
        detector._world_model = None
        weak_phone = Detection(
            label="cell phone", confidence=0.52,
            bbox=BoundingBox(x1=0, y1=0, x2=20, y2=40), frame_width=100, frame_height=100,
        )
        strong_bottle = Detection(
            label="bottle", confidence=0.52,
            bbox=BoundingBox(x1=30, y1=0, x2=50, y2=50), frame_width=100, frame_height=100,
        )
        detector._run = lambda *_args, **_kwargs: [weak_phone, strong_bottle]

        labels = [item.label for item in detector.detect(object())]
        self.assertEqual(labels, ["bottle"])

    def test_hybrid_eyeglasses_variants_and_head_roi(self):
        detector = HybridTargetDetector("unused", "unused")
        self.assertIn("pair of glasses", detector._VARIANTS["eyeglasses"])
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person = Detection(
            label="person", confidence=0.9,
            bbox=BoundingBox(x1=160, y1=40, x2=480, y2=440),
            frame_width=640, frame_height=480,
        )
        rois = detector._person_head_rois(frame, [person])
        self.assertEqual(len(rois), 1)
        crop, offset_x, offset_y = rois[0]
        self.assertGreater(crop.shape[0], 100)
        self.assertEqual(offset_y, 40)
        self.assertLessEqual(offset_x, 160)

    def test_offline_eyeglasses_cascades_and_eye_pair_geometry(self):
        detector = HybridTargetDetector("unused", "unused")
        self.assertTrue(detector._load_eyeglasses_cascades())

        aligned = detector._best_eyeglasses_pair(
            [(20, 30, 28, 20), (78, 31, 29, 21)], region_w=140, region_h=100,
        )
        misaligned = detector._best_eyeglasses_pair(
            [(20, 15, 28, 20), (78, 70, 29, 21)], region_w=140, region_h=100,
        )
        self.assertIsNotNone(aligned)
        self.assertGreaterEqual(aligned[-1], 0.65)
        self.assertIsNone(misaligned)

    def test_worn_eyeglasses_fallback_runs_before_grounding_dino(self):
        import numpy as np
        detector = HybridTargetDetector("unused", "unused")
        detector.select_target("eye glasses")
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        local = Detection(
            label="eyeglasses", confidence=0.76,
            bbox=BoundingBox(x1=45, y1=35, x2=115, y2=65),
            frame_width=160, frame_height=120, source="opencv-eyeglasses",
        )
        detector._detect_worn_eyeglasses = lambda *_args: [local]
        detector._ensure_loaded = lambda: self.fail("Grounding DINO should not load")

        results = detector.search(frame, "eyeglasses", [])

        self.assertEqual(results, [local])
        self.assertEqual(detector.status()["target"], "eyeglasses")
        self.assertTrue(detector.status()["locked"])


if __name__ == "__main__":
    unittest.main()
