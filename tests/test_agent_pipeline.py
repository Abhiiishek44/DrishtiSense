"""Integration coverage for the EventBus-driven cognitive route."""
import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from agents import CoordinatorAgent, CriticAgent, LibrarianAgent, WorldModel, _deterministic_parse
from event_bus import EventBus, QueryPayload
from main import ConnectionManager
from models import BoundingBox, Detection, MemorySearchResult, SpatialMemory
from orchestrator import LuminaOrchestrator
from goal_navigation import GoalNavigator
from spatial_memory import CameraPoseTracker, PersistentSpatialMemory
from vision import Track, YOLODetector


class _MemoryDatabase:
    def __init__(self, result):
        self._result = result

    def search(self, query, **_kwargs):
        return [self._result] if query == "bottle" else []


class AgentPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_visible_bottle_query_answers_immediately_without_agent_wait(self):
        messages = []

        async def capture(message):
            messages.append(message)

        bottle = Detection(
            label="bottle", confidence=0.82,
            bbox=BoundingBox(x1=10, y1=10, x2=40, y2=90),
            frame_width=100, frame_height=100,
        )
        track = Track(id=7, label="bottle", det=bottle, hits=3,
                      smoothed_distance=2.3, azimuth_deg=0.0)
        orchestrator = LuminaOrchestrator.__new__(LuminaOrchestrator)
        orchestrator._broadcast = capture
        orchestrator._tracker = SimpleNamespace(tracks={track.id: track})
        orchestrator._detector = None
        orchestrator._focused_detector = None
        orchestrator._active_target = None
        orchestrator._active_target_label = ""
        orchestrator._open_vocab_targets = []
        orchestrator._use_open_vocab = False
        orchestrator._target_min_frames = 3
        orchestrator._pending_query_text = ""
        orchestrator._pending_query_target = ""
        orchestrator._pending_query_started = 0.0
        orchestrator._pending_live_response_sent = False
        orchestrator._last_response_query = ""
        orchestrator._last_response_at = 0.0

        await orchestrator.query("where is my bottle")

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["type"], "response")
        self.assertEqual(messages[0]["object"], "bottle")
        self.assertIn("12 o'clock", messages[0]["text"])
        self.assertNotIn("searching", messages[0]["text"].lower())

    async def test_missing_object_scan_times_out_with_final_response(self):
        messages = []

        async def capture(message):
            messages.append(message)

        async def stalled_scan(_classes):
            await asyncio.sleep(1.0)

        orchestrator = LuminaOrchestrator.__new__(LuminaOrchestrator)
        orchestrator._broadcast = capture
        orchestrator._tracker = SimpleNamespace(tracks={})
        orchestrator._pending_query_text = "where is my bottle"
        orchestrator._pending_query_target = "bottle"
        orchestrator._pending_live_response_sent = False
        orchestrator.DIRECT_FIND_TIMEOUT_SECONDS = 0.01
        orchestrator.detect_once = stalled_scan
        orchestrator.find_object = lambda _target: None

        await orchestrator._complete_direct_find("where is my bottle", "bottle")

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["type"], "response")
        self.assertIn("can't see the bottle", messages[0]["text"])
        self.assertEqual(orchestrator._pending_query_text, "")

    async def test_worn_eyeglasses_response_says_they_are_on_the_face(self):
        messages = []

        async def capture(message):
            messages.append(message)

        orchestrator = LuminaOrchestrator.__new__(LuminaOrchestrator)
        orchestrator._broadcast = capture
        orchestrator._last_response_query = ""
        orchestrator._last_response_at = 0.0
        orchestrator._active_target = {
            "label": "eyeglasses", "source": "opencv-eyeglasses",
            "distance": 0.6, "relativeAngle": 0.0, "direction": "MOVE FORWARD",
            "visible": True, "confidence": 0.81,
        }

        await orchestrator._broadcast_active_target_response("where are my glasses")

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["object"], "eyeglasses")
        self.assertIn("on your face", messages[0]["text"])

    async def test_websocket_broadcasts_are_serialized_per_client(self):
        class FakeWebSocket:
            def __init__(self):
                self.active_sends = 0
                self.max_active_sends = 0

            async def accept(self):
                pass

            async def send_text(self, _payload):
                self.active_sends += 1
                self.max_active_sends = max(self.max_active_sends, self.active_sends)
                await asyncio.sleep(0.01)
                self.active_sends -= 1

        manager = ConnectionManager()
        socket = FakeWebSocket()
        await manager.connect(socket)
        await asyncio.gather(manager.broadcast({"sequence": 1}),
                             manager.broadcast({"sequence": 2}))

        self.assertEqual(socket.max_active_sends, 1)

    def test_only_explicit_location_requests_enter_find_mode(self):
        self.assertEqual(_deterministic_parse("Where is my bottle?")["intent"], "find")
        self.assertEqual(_deterministic_parse("What can you see?")["intent"], "inventory")
        self.assertEqual(_deterministic_parse("How are you?")["intent"], "unknown")

    def test_eye_glasses_does_not_parse_as_wine_glass(self):
        parsed = _deterministic_parse("Where are my eye glasses?")
        self.assertEqual(parsed["target"], "eyeglasses")

    def test_photo_picture_uses_one_canonical_navigation_label(self):
        parsed = _deterministic_parse("Where is my photo picture?")
        self.assertEqual(parsed["target"], "picture frame")
        self.assertEqual(YOLODetector._canonical_target("photo picture"), "picture frame")

    async def test_successful_visible_find_starts_navigation(self):
        messages = []

        async def capture(message):
            messages.append(message)

        bottle = Detection(
            label="bottle", confidence=0.82,
            bbox=BoundingBox(x1=10, y1=10, x2=40, y2=90),
            frame_width=100, frame_height=100,
        )
        track = Track(id=7, label="bottle", det=bottle, hits=3,
                      smoothed_distance=2.3, translation_z=2.3, azimuth_deg=0.0)
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = LuminaOrchestrator.__new__(LuminaOrchestrator)
            orchestrator._broadcast = capture
            orchestrator._tracker = SimpleNamespace(tracks={track.id: track})
            orchestrator._persistent_memory = PersistentSpatialMemory(
                str(Path(directory) / "world.json")
            )
            orchestrator._pose_tracker = CameraPoseTracker()
            orchestrator._goal_navigator = GoalNavigator()
            orchestrator._active_target = None
            orchestrator._active_target_label = "bottle"
            orchestrator._target_min_frames = 3
            orchestrator._pending_query_text = ""
            orchestrator._pending_query_target = ""
            orchestrator._pending_query_started = 0.0
            orchestrator._pending_live_response_sent = False
            orchestrator._last_response_query = ""
            orchestrator._last_response_at = 0.0
            orchestrator._update_active_target([track])

            await orchestrator._complete_visible_find("where is my bottle", track)

            self.assertEqual([message["type"] for message in messages],
                             ["response", "goal_update"])
            self.assertEqual(messages[1]["status"], "active")
            self.assertEqual(messages[1]["command"], "aligned")
            self.assertEqual(orchestrator.get_goal()["label"], "bottle")

    def test_focused_scan_keeps_exact_base_detector_result(self):
        bottle = Detection(
            label="bottle", confidence=0.82,
            bbox=BoundingBox(x1=10, y1=10, x2=40, y2=90),
            frame_width=100, frame_height=100,
        )
        person = Detection(
            label="person", confidence=0.95,
            bbox=BoundingBox(x1=45, y1=5, x2=95, y2=98),
            frame_width=100, frame_height=100,
        )

        matches = LuminaOrchestrator._matching_target_detections([person, bottle], "bottle")
        self.assertEqual(matches, [bottle])

    def test_standard_bottle_query_does_not_use_heavy_hybrid_model(self):
        detector = YOLODetector.__new__(YOLODetector)
        detector._coco_model = SimpleNamespace(names={0: "person", 39: "bottle"})
        orchestrator = LuminaOrchestrator.__new__(LuminaOrchestrator)
        orchestrator._detector = detector
        orchestrator._focused_detector = SimpleNamespace()

        self.assertTrue(detector.supports_standard_target("bottle"))
        self.assertFalse(orchestrator._should_use_hybrid_target("bottle"))
        self.assertTrue(orchestrator._should_use_hybrid_target("eyeglasses"))

    def test_chair_search_keeps_general_person_hazards_in_live_frame(self):
        person = Detection(
            label="person", confidence=0.91,
            bbox=BoundingBox(x1=0, y1=0, x2=70, y2=100),
            frame_width=100, frame_height=100,
        )
        chair = Detection(
            label="chair", confidence=0.76,
            bbox=BoundingBox(x1=72, y1=30, x2=99, y2=100),
            frame_width=100, frame_height=100,
        )

        class SceneDetector:
            def detect(self, _frame):
                return [person, chair]

            def supports_standard_target(self, target):
                return target == "chair"

            def detect_open(self, *_args):
                raise AssertionError("standard chair must not replace scene detection")

        orchestrator = LuminaOrchestrator.__new__(LuminaOrchestrator)
        orchestrator._detector = SceneDetector()
        orchestrator._use_open_vocab = True
        orchestrator._open_vocab_targets = ["chair"]

        detections = orchestrator._detect_live_frame(object())

        self.assertEqual([detection.label for detection in detections],
                         ["person", "chair"])

    async def test_general_question_returns_without_starting_camera_search(self):
        messages = []

        async def capture(message):
            messages.append(message)

        orchestrator = LuminaOrchestrator.__new__(LuminaOrchestrator)
        orchestrator._broadcast = capture
        orchestrator._pending_query_text = "old search"
        orchestrator._pending_query_target = "bottle"
        orchestrator._pending_live_response_sent = True
        orchestrator._active_target_label = "bottle"
        orchestrator._active_target = {"label": "bottle"}
        orchestrator._open_vocab_targets = ["bottle"]
        orchestrator._use_open_vocab = True
        orchestrator._focused_detector = None
        orchestrator._goal_navigator = SimpleNamespace(snapshot=lambda: {"status": "idle"})
        orchestrator._world_model = None

        await orchestrator.query("How are you?")

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["type"], "response")
        self.assertIn("find objects", messages[0]["text"])
        self.assertFalse(orchestrator._use_open_vocab)
        self.assertEqual(orchestrator._pending_query_text, "")

    async def test_location_query_reaches_librarian_coordinator_and_critic(self):
        memory = SpatialMemory(
            id="bottle-1", label="bottle", confidence=0.95,
            original_confidence=0.95, angle_abs=0.0, distance_m=2.0,
            frame_x_norm=0.5, frame_y_norm=0.5, timestamp=time.time(),
            session_id="test",
        )
        result = MemorySearchResult(memory=memory, score=1.0, match_type="exact")
        bus = EventBus()
        coordinator = CoordinatorAgent(bus, llm=None, world_model=WorldModel())
        librarian = LibrarianAgent(bus, _MemoryDatabase(result))
        critic = CriticAgent(bus, confidence_threshold=0.60, coordinator_ref=coordinator)
        finals = []
        final_ready = asyncio.Event()

        async def capture_final(event):
            finals.append(event.payload)
            final_ready.set()

        librarian.register()
        coordinator.register()
        critic.register()
        bus.subscribe("navigation/route_final", capture_final)
        await bus.start()
        try:
            await bus.publish(
                "system/query_received", QueryPayload(raw_text="where is my bottle?"),
                publisher="TEST",
            )
            # User-response topics use the EventBus low-latency lane and are
            # dispatched as tasks rather than through the normal queue.
            await asyncio.wait_for(final_ready.wait(), timeout=1.0)
        finally:
            await bus.stop()

        self.assertEqual(len(finals), 1)
        self.assertIn("Your bottle", finals[0].response_text)
        self.assertTrue(finals[0].verdict.approved)
