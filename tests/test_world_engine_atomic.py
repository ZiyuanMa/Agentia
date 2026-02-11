"""Unit tests for WorldEngine using Atomic Tools logic."""
import pytest
import json
from unittest.mock import MagicMock
from world_engine import WorldEngine
from world import World
from schemas import (
    InteractionResult, 
    UpdateObjectAction, 
    CreateObjectAction, 
    DestroyObjectAction, 
    TransferObjectAction,
    WorldObject,
    Location
)
from utils import LLMClient

# --- Mocks for Tool Calls ---

class MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = json.dumps(arguments)

class MockToolCall:
    def __init__(self, name, arguments, call_id="call_123"):
        self.id = call_id
        self.function = MockFunction(name, arguments)

class MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

# --- Test Fixtures ---

class TestWorldEngineAtomic:
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        return MagicMock(spec=LLMClient)
    
    @pytest.fixture
    def world_engine(self, mock_llm):
        """Create a test world engine."""
        return WorldEngine(mock_llm)
    
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
                    "location_id": "room_a",
                    "state": "normal",
                    "description": "A test object",
                    "internal_state": {"interactive": True},
                    "mechanics": "Can be toggled."
                }
            ],
            "agents": []
        }
        return World(config)
    
    @pytest.fixture
    def location(self, world):
        return world.get_location("room_a")
    
    @pytest.fixture
    def target_object(self, world):
        return world.get_object("test_obj")

    # --- Tests ---

    def test_inquiry_tool_execution(self, world_engine, mock_llm, world, location, target_object):
        """Test that inquiry tools are executed immediately and return results to LLM."""
        
        # Turn 1: Call query_object
        msg1 = MockMessage(tool_calls=[
            MockToolCall("query_object", {"object_id": "test_obj"})
        ])
        
        # Turn 2: Finalize interaction based on inquiry
        msg2 = MockMessage(tool_calls=[
            MockToolCall("interaction_result", {
                "success": True, 
                "message": "Object is interactive.", 
                "duration": 0
            })
        ])
        
        mock_llm.chat_completion = MagicMock(side_effect=[msg1, msg2])
        
        result = world_engine.resolve_interaction(
            agent_name="Alice",
            target_object=target_object,
            action_description="inspect",
            location=location,
            witnesses=[],
            world=world
        )
        
        assert result["message"] == "Object is interactive."
        # Verify call history
        assert mock_llm.chat_completion.call_count == 2
        
        # Check that the tool output was fed back to LLM (we need to inspect the 'messages' list passed to call 2)
        # call_args_list[1] is the second call. args[0] is 'messages'.
        messages_arg = mock_llm.chat_completion.call_args_list[1][0][0]
        # The messages list is mutable and updated as loop progresses.
        # Structure at end: [Sys, User, AIMsg(query), ToolMsg(query_result), AIMsg(result), ToolMsg(result_ack)]
        # We want the ToolMsg(query_result) which is at index -3.
        tool_msg = messages_arg[-3]
        
        assert tool_msg["role"] == "tool"
        content = json.loads(tool_msg["content"])
        
        assert content.get("state") == "normal" # From object dump

    def test_immediate_action_application(self, world_engine, mock_llm, world, location, target_object):
        """Test that staged action tools are applied immediately when duration is 0."""
        
        # Turn 1: Update object + Finalize
        msg1 = MockMessage(tool_calls=[
            MockToolCall("update_object", {"object_id": "test_obj", "state": "broken"}),
            MockToolCall("interaction_result", {
                "success": True, 
                "message": "You broke it.", 
                "duration": 0
            })
        ])
        
        mock_llm.chat_completion = MagicMock(side_effect=[msg1])
        
        result = world_engine.resolve_interaction(
            agent_name="Alice",
            target_object=target_object,
            action_description="break",
            location=location,
            witnesses=[],
            world=world
        )
        
        assert result["message"] == "You broke it."
        # Verify object state changed immediately
        assert target_object.state == "broken"

    def test_deferred_action_application(self, world_engine, mock_llm, world, location, target_object):
        """Test that actions are deferred when duration > 0."""
        
        # Turn 1: Update object + Finalize with duration
        msg1 = MockMessage(tool_calls=[
            MockToolCall("update_object", {"object_id": "test_obj", "state": "repaired"}),
            MockToolCall("interaction_result", {
                "success": True, 
                "message": "Repair complete.", 
                "duration": 10,
                "task_description": "repairing"
            })
        ])
        
        mock_llm.chat_completion = MagicMock(side_effect=[msg1])
        
        result = world_engine.resolve_interaction(
            agent_name="Alice",
            target_object=target_object,
            action_description="repair",
            location=location,
            witnesses=[],
            world=world
        )
        
        # Message should indicate start
        assert "Started: repairing" in result["message"]
        
        # Object state should NOT change yet
        assert target_object.state == "normal"
        
        # Check lock
        lock = world.check_agent_lock("Alice")
        assert lock["reason"] == "repairing"
        
        # Check pending effects in internal lock structure
        internal_lock = world.agent_locks["Alice"]
        assert len(internal_lock["pending_effects"]) == 1
        assert internal_lock["pending_effects"][0]["type"] == "UpdateObject"

    def test_create_and_transfer_atomic(self, world_engine, mock_llm, world, location, target_object):
        """Test multiple atomic actions in one turn."""
        
        # Alice is in room_a
        world.place_agent("Alice", "room_a")
        
        # Turn 1: Create Coffee -> Transfer to Alice -> Finalize
        msg1 = MockMessage(tool_calls=[
            MockToolCall("create_object", {
                "object_id": "coffee_1", 
                "name": "Coffee", 
                "location_id": "room_a",
                "description": "Hot coffee"
            }),
            MockToolCall("transfer_object", {
                "object_id": "coffee_1",
                "to_id": "Alice",
                "from_id": "room_a"
            }),
            MockToolCall("interaction_result", {
                "success": True, 
                "message": "You made coffee and picked it up.", 
                "duration": 0
            })
        ])
        
        mock_llm.chat_completion = MagicMock(side_effect=[msg1])
        
        world_engine.resolve_interaction(
            agent_name="Alice",
            target_object=target_object, 
            action_description="make coffee",
            location=location,
            witnesses=[],
            world=world
        )
        
        # Check creation
        coffee = world.get_object("coffee_1")
        assert coffee is not None
        assert coffee.name == "Coffee"
        
        # Check transfer (Alice should have it)
        inventory = world.get_agent_inventory("Alice")
        # Inventory items are formatted strings: "Name (id: ID)"
        found = False
        for item in inventory:
            if "coffee_1" in item:
                found = True
                break
        assert found, f"Expected coffee_1 in inventory, got: {inventory}"
