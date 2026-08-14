"""Integration coverage for the EventBus-driven cognitive route."""
import asyncio
import time
import unittest

from agents import CoordinatorAgent, CriticAgent, LibrarianAgent, WorldModel, _deterministic_parse
from event_bus import EventBus, QueryPayload
from models import MemorySearchResult, SpatialMemory


class _MemoryDatabase:
    def __init__(self, result):
        self._result = result

    def search(self, query, **_kwargs):
        return [self._result] if query == "bottle" else []


class AgentPipelineTest(unittest.IsolatedAsyncioTestCase):
    def test_eye_glasses_does_not_parse_as_wine_glass(self):
        parsed = _deterministic_parse("Where are my eye glasses?")
        self.assertEqual(parsed["target"], "eyeglasses")

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

        async def capture_final(event):
            finals.append(event.payload)

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
            await asyncio.wait_for(bus._queue.join(), timeout=1.0)
        finally:
            await bus.stop()

        self.assertEqual(len(finals), 1)
        self.assertIn("Your bottle", finals[0].response_text)
        self.assertTrue(finals[0].verdict.approved)
