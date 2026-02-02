"""Unit tests for World object operations and deferred effects."""
import pytest
from datetime import datetime, timedelta
from world import World
from schemas import WorldObject, Location


class TestWorldObjectOperations:
    """Test create/destroy/transfer object functionality."""
    
    @pytest.fixture
    def world(self):
        """Create a simple world for testing."""
        config = {
            "locations": [
                {
                    "id": "room_a",
                    "name": "Room A",
                    "description": "A test room",
                    "connected_to": ["room_b"]
                },
                {
                    "id": "room_b", 
                    "name": "Room B",
                    "description": "Another test room",
                    "connected_to": ["room_a"]
                }
            ],
            "objects": [
                {
                    "id": "test_object",
                    "name": "Test Object",
                    "location": "room_a",
                    "state": "normal",
                    "description": "A test object",
                    "properties": ["portable"]
                }
            ]
        }
        return World(config)

    def test_create_object(self, world):
        """Test creating a new object."""
        success = world.create_object(
            object_id="new_obj",
            name="New Object",
            location_id="room_a",
            state="fresh",
            description="A newly created object",
            properties=["consumable"]
        )
        
        assert success is True
        obj = world.get_object("new_obj")
        assert obj is not None
        assert obj.name == "New Object"
        assert obj.state == "fresh"
        assert obj.location_id == "room_a"
        assert "consumable" in obj.properties

    def test_create_duplicate_object_fails(self, world):
        """Test that creating an object with existing ID fails."""
        success = world.create_object(
            object_id="test_object",  # Already exists
            name="Duplicate",
            location_id="room_a"
        )
        assert success is False

    def test_destroy_object(self, world):
        """Test destroying an object."""
        assert world.get_object("test_object") is not None
        
        success = world.destroy_object("test_object")
        
        assert success is True
        assert world.get_object("test_object") is None

    def test_destroy_nonexistent_object_fails(self, world):
        """Test that destroying non-existent object fails."""
        success = world.destroy_object("fake_object")
        assert success is False

    def test_transfer_object_between_locations(self, world):
        """Test transferring object from one location to another."""
        obj = world.get_object("test_object")
        assert obj.location_id == "room_a"
        
        success = world.transfer_object(
            object_id="test_object",
            from_location="room_a",
            to_location="room_b"
        )
        
        assert success is True
        assert obj.location_id == "room_b"


class TestDeferredEffects:
    """Test deferred effects with agent locks."""
    
    @pytest.fixture
    def world(self):
        """Create a simple world for testing."""
        config = {
            "locations": [
                {"id": "room", "name": "Room", "description": "A room", "connected_to": []}
            ],
            "objects": [
                {"id": "machine", "name": "Machine", "location": "room", "state": "working", "description": "A machine"}
            ]
        }
        return World(config)

    def test_lock_with_no_pending_effects(self, world):
        """Test basic lock without pending effects."""
        world.set_agent_lock("Alice", 10, "working")
        
        lock = world.check_agent_lock("Alice")
        assert lock is not None
        assert lock["expired"] is False
        assert lock["reason"] == "working"

    def test_lock_expires_and_executes_pending_effects(self, world):
        """Test that pending effects execute when lock expires."""
        # Set lock with pending CreateObject effect
        pending = [
            {
                "type": "CreateObject",
                "args": {
                    "object_id": "coffee_cup",
                    "name": "Coffee Cup",
                    "location_id": "room",
                    "state": "hot",
                    "description": "A fresh cup of coffee"
                }
            }
        ]
        
        world.set_agent_lock("Bob", 5, "making coffee", pending_effects=pending)
        
        # Object doesn't exist yet
        assert world.get_object("coffee_cup") is None
        
        # Advance time past lock expiration
        world.sim_time += timedelta(minutes=10)
        
        # Check lock - should execute pending effects
        lock_status = world.check_agent_lock("Bob")
        
        assert lock_status["expired"] is True
        
        # Object should now exist
        coffee = world.get_object("coffee_cup")
        assert coffee is not None
        assert coffee.name == "Coffee Cup"
        assert coffee.state == "hot"

    def test_pending_modify_object_state(self, world):
        """Test deferred ModifyObjectState."""
        pending = [
            {
                "type": "ModifyObjectState",
                "args": {
                    "object_id": "machine",
                    "new_state": "repaired"
                }
            }
        ]
        
        world.set_agent_lock("Charlie", 20, "repairing", pending_effects=pending)
        
        # State unchanged while locked
        assert world.get_object("machine").state == "working"
        
        # Expire lock
        world.sim_time += timedelta(minutes=25)
        world.check_agent_lock("Charlie")
        
        # State should now be updated
        assert world.get_object("machine").state == "repaired"

    def test_pending_destroy_object(self, world):
        """Test deferred DestroyObject."""
        # First create an object to destroy
        world.create_object("food", "Food", "room")
        assert world.get_object("food") is not None
        
        pending = [
            {"type": "DestroyObject", "args": {"object_id": "food"}}
        ]
        
        world.set_agent_lock("Diana", 5, "eating", pending_effects=pending)
        
        # Food still exists
        assert world.get_object("food") is not None
        
        # Expire lock
        world.sim_time += timedelta(minutes=10)
        world.check_agent_lock("Diana")
        
        # Food consumed
        assert world.get_object("food") is None

    def test_multiple_pending_effects(self, world):
        """Test multiple pending effects execute in order."""
        pending = [
            {"type": "CreateObject", "args": {"object_id": "result", "name": "Result", "location_id": "room"}},
            {"type": "ModifyObjectState", "args": {"object_id": "machine", "new_state": "idle"}}
        ]
        
        world.set_agent_lock("Eve", 10, "processing", pending_effects=pending)
        world.sim_time += timedelta(minutes=15)
        world.check_agent_lock("Eve")
        
        # Both effects executed
        assert world.get_object("result") is not None
        assert world.get_object("machine").state == "idle"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
