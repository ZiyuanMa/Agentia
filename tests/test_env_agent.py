"""Unit tests for EnvironmentAgent using mocked LLM with JSON output mode."""
import pytest
import json
from unittest.mock import MagicMock
from pydantic import ValidationError
from env_agent import EnvironmentAgent
from world import World
from schemas import WorldObject, Location, ModifyObjectState, CreateObject, TransferObject
from utils import LLMClient


class MockMessage:
    """Mock object for OpenAI chat completion response with JSON content."""
    def __init__(self, content: str):
        self.content = content


class TestEnvironmentAgent:
    """Test EnvironmentAgent functionality with JSON output mode."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        return MagicMock(spec=LLMClient)
    
    @pytest.fixture
    def env_agent(self, mock_llm):
        """Create a test environment agent."""
        return EnvironmentAgent(mock_llm)
    
    @pytest.fixture
    def world(self):
        """Create a simple test world."""
        config = {
            "locations": [
                {"id": "room_a", "name": "Room A", "description": "A test room", "connected_to": []}
            ],
            "objects": [
                {
                    "id": "test_obj",
                    "name": "Test Object",
                    "location": "room_a",
                    "state": "normal",
                    "description": "A test object",
                    "properties": ["interactive"]
                }
            ]
        }
        return World(config)
    
    @pytest.fixture
    def location(self, world):
        """Get test location."""
        return world.get_location("room_a")
    
    @pytest.fixture
    def target_object(self, world):
        """Get test object."""
        return world.get_object("test_obj")

    def test_build_context(self, env_agent, target_object, location):
        """Test context building for LLM."""
        context = env_agent._build_context(
            agent_name="Alice",
            target_object=target_object,
            action_description="use the object",
            location=location,
            witnesses=["Bob", "Alice"]
        )
        
        assert "Alice" in context
        assert "Test Object" in context
        assert "test_obj" in context
        assert "Room A" in context
        assert "use the object" in context
        # Bob should be in witnesses
        assert "Bob" in context

    def test_resolve_interaction_result_action(self, env_agent, mock_llm, world, location, target_object):
        """Test resolving interaction with result action in JSON mode."""
        json_response = json.dumps({
            "reasoning": "The object can be used safely.",
            "decisions": [
                {"action": "result", "success": True, "message": "You successfully used the object.", "reasoning": "It worked fine."}
            ]
        })
        mock_llm.chat_completion = MagicMock(return_value=MockMessage(json_response))
        
        result = env_agent.resolve_interaction(
            agent_name="Alice",
            target_object=target_object,
            action_description="use",
            location=location,
            witnesses=[],
            world=world
        )
        
        assert result["success"] is True
        assert "successfully used" in result["message"]

    def test_resolve_interaction_modify_object_state(self, env_agent, mock_llm, world, location, target_object):
        """Test ModifyState executes immediately without LockAgent."""
        json_response = json.dumps({
            "reasoning": "The object will break from this action.",
            "decisions": [
                {"action": "modify_state", "object_id": "test_obj", "new_state": "broken"},
                {"action": "result", "success": True, "message": "The object broke.", "reasoning": "It shattered."}
            ]
        })
        mock_llm.chat_completion = MagicMock(return_value=MockMessage(json_response))
        
        # Object starts as "normal"
        assert target_object.state == "normal"
        
        result = env_agent.resolve_interaction(
            agent_name="Alice",
            target_object=target_object,
            action_description="break",
            location=location,
            witnesses=[],
            world=world
        )
        
        # Without LockAgent, effect should execute immediately
        assert target_object.state == "broken"
        assert result["success"] is True

    def test_resolve_interaction_with_lock_defers_effects(self, env_agent, mock_llm, world, location, target_object):
        """Test that object effects are deferred when LockAgent is used."""
        json_response = json.dumps({
            "reasoning": "Repairing takes time.",
            "decisions": [
                {"action": "lock_agent", "agent_name": "Alice", "duration_minutes": 10, "reason": "repairing", "completion_message": "Repair complete."},
                {"action": "modify_state", "object_id": "test_obj", "new_state": "repaired"},
                {"action": "result", "success": True, "message": "You started repairing.", "reasoning": "Began work."}
            ]
        })
        mock_llm.chat_completion = MagicMock(return_value=MockMessage(json_response))
        
        result = env_agent.resolve_interaction(
            agent_name="Alice",
            target_object=target_object,
            action_description="repair",
            location=location,
            witnesses=[],
            world=world
        )
        
        # Object state should NOT change yet (deferred)
        assert target_object.state == "normal"
        
        # Agent should be locked
        lock = world.agent_locks.get("Alice")
        assert lock is not None
        assert lock["reason"] == "repairing"
        assert len(lock["pending_effects"]) == 1

    def test_resolve_interaction_create_object(self, env_agent, mock_llm, world, location, target_object):
        """Test CreateObject action creates new object."""
        json_response = json.dumps({
            "reasoning": "Making coffee produces a coffee cup.",
            "decisions": [
                {"action": "create_object", "object_id": "coffee_cup_1", "name": "Coffee Cup", 
                 "location_id": "room_a", "state": "hot", "description": "A hot cup of coffee", "properties": ["consumable"]},
                {"action": "result", "success": True, "message": "You made coffee.", "reasoning": "Coffee ready."}
            ]
        })
        mock_llm.chat_completion = MagicMock(return_value=MockMessage(json_response))
        
        result = env_agent.resolve_interaction(
            agent_name="Alice",
            target_object=target_object,
            action_description="make coffee",
            location=location,
            witnesses=[],
            world=world
        )
        
        assert result["success"] is True
        coffee = world.get_object("coffee_cup_1")
        assert coffee is not None
        assert coffee.name == "Coffee Cup"
        assert coffee.state == "hot"

    def test_resolve_interaction_destroy_object(self, env_agent, mock_llm, world, location, target_object):
        """Test DestroyObject action removes object."""
        # First create an object to destroy
        world.create_object("temp_obj", "Temporary Object", "room_a")
        assert world.get_object("temp_obj") is not None
        
        json_response = json.dumps({
            "reasoning": "The object is consumed.",
            "decisions": [
                {"action": "destroy_object", "object_id": "temp_obj", "reason": "consumed"},
                {"action": "result", "success": True, "message": "Object consumed.", "reasoning": "Gone."}
            ]
        })
        mock_llm.chat_completion = MagicMock(return_value=MockMessage(json_response))
        
        result = env_agent.resolve_interaction(
            agent_name="Alice",
            target_object=target_object,
            action_description="consume",
            location=location,
            witnesses=[],
            world=world
        )
        
        assert result["success"] is True
        assert world.get_object("temp_obj") is None

    def test_resolve_interaction_broadcast(self, env_agent, mock_llm, world, location, target_object):
        """Test BroadcastAction notifies agents in location."""
        # Add an agent to the location
        world.place_agent("Bob", "room_a")
        
        json_response = json.dumps({
            "reasoning": "This makes a loud noise.",
            "decisions": [
                {"action": "broadcast", "location_id": "room_a", "message": "A loud noise echoed!"},
                {"action": "result", "success": True, "message": "You made a noise.", "reasoning": "Bang!"}
            ]
        })
        mock_llm.chat_completion = MagicMock(return_value=MockMessage(json_response))
        
        result = env_agent.resolve_interaction(
            agent_name="Alice",
            target_object=target_object,
            action_description="bang on",
            location=location,
            witnesses=["Bob"],
            world=world
        )
        
        # Bob should have received the event
        events = world.get_pending_events("Bob")
        assert len(events) > 0
        assert "loud noise" in events[0].lower()

    def test_resolve_interaction_llm_failure(self, env_agent, mock_llm, world, location, target_object):
        """Test graceful handling of LLM failure."""
        mock_llm.chat_completion = MagicMock(return_value=None)
        
        result = env_agent.resolve_interaction(
            agent_name="Alice",
            target_object=target_object,
            action_description="use",
            location=location,
            witnesses=[],
            world=world
        )
        
        assert result["success"] is False
        assert "no effect" in result["message"].lower()

    def test_resolve_interaction_invalid_json(self, env_agent, mock_llm, world, location, target_object):
        """Test handling of invalid JSON response falls back gracefully."""
        mock_llm.chat_completion = MagicMock(return_value=MockMessage("not valid json at all"))
        
        result = env_agent.resolve_interaction(
            agent_name="Alice",
            target_object=target_object,
            action_description="use",
            location=location,
            witnesses=[],
            world=world
        )
        
        # Should still return a result (fallback behavior)
        assert "message" in result


class TestToolValidation:
    """Test Pydantic validation for tool arguments."""
    
    def test_invalid_modify_object_state(self):
        """Test validation error for invalid ModifyObjectState args."""
        with pytest.raises(ValidationError):
            ModifyObjectState(object_id="test")  # Missing required fields
    
    def test_valid_create_object(self):
        """Test valid CreateObject parsing."""
        obj = CreateObject(
            object_id="new_obj",
            name="New Object",
            location_id="room_a",
            description="A new object"
        )
        assert obj.state == "normal"  # Default value
        assert obj.properties == []  # Default factory

    def test_valid_transfer_object(self):
        """Test TransferObject with optional fields."""
        transfer = TransferObject(
            object_id="item",
            from_location="room_a",
            to_agent="Alice"
        )
        assert transfer.to_location is None
        assert transfer.from_agent is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
