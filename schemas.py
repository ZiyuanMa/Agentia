"""
Agentia Schema Definitions

All Pydantic models for the simulation are centralized here:
- World models (WorldObject, Location)
- Agent action models (MoveAction, TalkAction, etc.)
- WorldEngine action models (ResultAction, ModifyStateAction, etc.)
- Decision models (AgentDecision, WorldEngineDecision)
"""
import json
from typing import List, Optional, Literal, Union, Any
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


# =============================================================================
# SimAgent Action Models
# =============================================================================

class MoveAction(BaseModel):
    """Move to a connected location"""
    location_id: str = Field(description="The ID of the location to move to")


class TalkAction(BaseModel):
    """Say something to people in the same location"""
    message: str = Field(description="What you want to say")
    target_agent: Optional[str] = Field(None, description="Specific agent to address")


class InteractAction(BaseModel):
    """Interact with an object in the current location"""
    object_id: str = Field(description="The ID of the object to interact with")
    action: str = Field(description="The detailed description of interaction")


class WaitAction(BaseModel):
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
    action: Union[MoveAction, TalkAction, InteractAction, WaitAction] = Field(
        description="Action parameters object"
    )

    def get_validated_action(self):
        """Validate and return the typed action based on action_type."""
        if isinstance(self.action, BaseModel):
            return self.action
        
        action_map = {
            "move": MoveAction,
            "talk": TalkAction,
            "interact": InteractAction,
            "wait": WaitAction,
        }
        model = action_map.get(self.action_type)
        if model and isinstance(self.action, dict):
            return model(**self.action)
        return self.action

# =============================================================================
# WorldEngine Action Models
# =============================================================================

class ResultAction(BaseModel):
    """Report the outcome of the interaction to the actor."""
    action: Literal["result"] = "result"
    success: bool
    message: str = Field(description="Narrative of what happened")


class ModifyStateAction(BaseModel):
    """Change the VISIBLE state of an object (e.g. open/closed, on/off)."""
    action: Literal["modify_state"] = "modify_state"
    object_id: str
    new_state: str
    new_description: Optional[str] = Field(None, description="Updated visible description if appearance changes")


class ModifyInternalStateAction(BaseModel):
    """Update a specific key in an object's HIDDEN internal_state."""
    action: Literal["modify_internal_state"] = "modify_internal_state"
    object_id: str
    key: str
    value: Any


class LockAgentAction(BaseModel):
    """Lock an agent for a duration (for time-consuming actions)."""
    action: Literal["lock_agent"] = "lock_agent"
    agent_name: str
    duration_minutes: int
    description: str
    completion_message: str = ""


class CreateObjectAction(BaseModel):
    """Create a new object in the world."""
    action: Literal["create_object"] = "create_object"
    object_id: str
    name: str
    location_id: str
    state: str = "normal"
    description: str = ""
    mechanics: str = ""
    internal_state: dict = Field(default_factory=dict)


class DestroyObjectAction(BaseModel):
    """Remove an object from the world."""
    action: Literal["destroy_object"] = "destroy_object"
    object_id: str


class TransferObjectAction(BaseModel):
    """Transfer an object from one container/location/agent to another."""
    action: Literal["transfer_object"] = "transfer_object"
    object_id: str
    from_id: str = Field(description="Source ID (Room ID, Agent ID, or Container ID)")
    to_id: str = Field(description="Destination ID (Room ID, Agent ID, or Container ID)")


# Union of all WorldEngine action types
WorldEngineAction = Union[
    ResultAction,
    ModifyStateAction,
    ModifyInternalStateAction,

    LockAgentAction,
    CreateObjectAction,
    DestroyObjectAction,
    TransferObjectAction,
]


class WorldEngineDecision(BaseModel):
    """
    The complete output format for WorldEngine's decision.
    WorldEngine validates actions using this model.
    """
    reasoning: str = Field(description="Step-by-step analysis of the situation and physics before deciding outcomes")
    decisions: List[WorldEngineAction] = Field(description="List of actions to execute, starting with a ResultAction")


# =============================================================================
# Schema Generation Utilities
# =============================================================================

def get_agent_decision_schema() -> str:
    """Get a simplified JSON schema string for the agent decision format."""
    schema_dict = AgentDecision.model_json_schema()
    schema_str = json.dumps(schema_dict, indent=2)
    return schema_str.replace("{", "{{").replace("}", "}}")


def get_world_engine_decision_schema() -> str:
    """Get a simplified JSON schema string for the WorldEngine output format."""
    schema_dict = WorldEngineDecision.model_json_schema()
    schema_str = json.dumps(schema_dict, indent=2)
    return schema_str.replace("{", "{{").replace("}", "}}")
