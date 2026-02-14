"""
Agentia Schema Definitions

All Pydantic models for the simulation are centralized here:
- World models (WorldObject, Location)
- Agent action models (Move, Talk, etc.)
- WorldEngine action models (UpdateObject, CreateObject, etc.)
- Decision models (AgentDecision, WorldEngineDecision)
"""
import json
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field


# =============================================================================
# World Models
# =============================================================================

class WorldObject(BaseModel):
    """An object that exists in the world."""
    id: str
    name: str
    location_id: str
    description: str                                  # Visible description
    state: str = "normal"                             # Visible state (e.g. "open", "closed", "on", "off")
    mechanics: str = Field(default="", description="Rules and physics of the object (WorldEngine only)")
    internal_state: dict = Field(default_factory=dict, description="Hidden internal state (locked, contents, etc) - WorldEngine only")


class Location(BaseModel):
    """A location in the world that agents can visit."""
    id: str
    name: str
    description: str
    connected_to: List[str] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)  # List of object IDs
    agents_present: List[str] = Field(default_factory=list)  # List of agent names


class Task(BaseModel):
    """A task in the agent's daily plan."""
    id: str = Field(description="Unique ID for the task (e.g., '1', 'task_a')")
    description: str = Field(description="Description of the task")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(default="pending")


class Plan(BaseModel):
    """Internal model for the plan tool structure (used for schema generation)."""
    tasks: List[Task] = Field(description="The full list of tasks for your plan.")


class Task(BaseModel):
    """A task in the agent's daily plan."""
    id: str = Field(description="Unique ID for the task (e.g., '1', 'task_a')")
    description: str = Field(description="Description of the task")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(default="pending")


class Plan(BaseModel):
    """Internal model for the plan tool structure (used for schema generation)."""
    tasks: List[Task] = Field(description="The full list of tasks for your plan.")


# =============================================================================
# SimAgent Action Models
# =============================================================================

class Move(BaseModel):
    """Move to a location connected to your current location"""
    location_id: str = Field(description="The ID of the location to move to")


class Talk(BaseModel):
    """Say something to people in the same location"""
    message: str = Field(description="What you want to say")
    target_agent: Optional[str] = Field(None, description="Specific agent to address")


class Interact(BaseModel):
    """Interact with an object in the current location"""
    object_id: str = Field(description="The ID of the object to interact with")
    action: str = Field(description="The detailed description of interaction")


class Wait(BaseModel):
    """Wait and observe the surroundings"""
    reason: str = Field(default="observing", description="Why you are waiting")


class AgentDecision(BaseModel):
    """
    The complete output format for an agent's decision.
    Agents must output this exact JSON structure.
    """
    reasoning: str = Field(description="Your step-by-step thought process analyzing the situation")
    action_type: Literal["move", "talk", "interact", "wait"] = Field(
        description="The type of action to take"
    )
    action: Union[Move, Talk, Interact, Wait] = Field(
        description="Action parameters object"
    )

    def get_validated_action(self):
        """Validate and return the typed action based on action_type."""
        if isinstance(self.action, BaseModel):
            return self.action
        
        action_map = {
            "move": Move,
            "talk": Talk,
            "interact": Interact,
            "wait": Wait,
        }
        model = action_map.get(self.action_type)
        if model and isinstance(self.action, dict):
            return model(**self.action)
        return self.action

    @staticmethod
    def fallback(reasoning: str = "Parsing failed") -> "AgentDecision":
        """Create a default wait action for error/fallback scenarios."""
        return AgentDecision(
            reasoning=reasoning,
            action_type="wait",
            action=Wait(reason="fallback")
        )

# =============================================================================
# WorldEngine Inquiry Models
# =============================================================================

class QueryEntityParams(BaseModel):
    """Parameters for querying any entity (object or agent) in the world."""
    entity_id: str = Field(description="The ID of the entity to query (object ID or agent name)")


# =============================================================================
# WorldEngine Action Models
# =============================================================================

class InteractionResult(BaseModel):
    """
    The outcome of an interaction - immediate or time-delayed.
    If duration > 0, the agent is locked and effects are deferred until completion.
    """
    message: str = Field(description="Result message. Shown immediately if duration=0, or after completion if duration>0.")
    
    # Time cost (optional - for actions that take time)
    duration: int = Field(0, description="Action duration in minutes. If > 0, agent is locked.")
    task_description: Optional[str] = Field(default=None, description="Description of the ongoing task if duration > 0 (e.g. 'repairing')")


class UpdateObject(BaseModel):
    """Update any field(s) of an object. Only provided fields will be updated."""
    object_id: str
    state: Optional[str] = Field(None, description="New visible state (e.g. 'open', 'broken', 'active')")
    description: Optional[str] = Field(None, description="New visible description")
    internal_state: Optional[dict] = Field(None, description="Updates to internal state (merged with existing)")


class CreateObject(BaseModel):
    """Create a new object in the world."""
    object_id: str
    name: str
    location_id: str
    state: str = "normal"
    description: str = ""
    mechanics: str = ""
    internal_state: dict = Field(default_factory=dict)


class DestroyObject(BaseModel):
    """Remove an object from the world."""
    object_id: str


class TransferObject(BaseModel):
    """Transfer an object from one container/location/agent to another."""
    object_id: str
    from_id: str = Field(description="Source ID (Room ID, Agent ID, or Container ID)")
    to_id: str = Field(description="Destination ID (Room ID, Agent ID, or Container ID)")


# Union of world effect actions (changes to the world state)
WorldEffect = Union[
    UpdateObject,
    CreateObject,
    DestroyObject,
    TransferObject,
]

# =============================================================================
# Schema Generation Utilities
# =============================================================================

def get_agent_decision_schema() -> str:
    """Get a simplified JSON schema string for the agent decision format."""
    schema_dict = AgentDecision.model_json_schema()
    schema_str = json.dumps(schema_dict, indent=2)
    return schema_str.replace("{", "{{").replace("}", "}}")


def get_update_plan_tool_schema() -> dict:
    """Get the tool definition for updating the daily plan."""
    return {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": "Create or update your daily plan. Use this to set your schedule, mark tasks as complete, or replan when circumstances change.",
            "parameters": Plan.model_json_schema()
        }
    }

