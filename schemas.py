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
    """Move to a location connected to your current location"""
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

    @staticmethod
    def fallback(reasoning: str = "Parsing failed") -> "AgentDecision":
        """Create a default wait action for error/fallback scenarios."""
        return AgentDecision(
            reasoning=reasoning,
            action_type="wait",
            action=WaitAction(reason="fallback")
        )

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
    task_description: Optional[str] = Field(None, description="Description of the ongoing task (e.g. 'repairing')")


class UpdateObjectAction(BaseModel):
    """Update any field(s) of an object. Only provided fields will be updated."""
    action: Literal["update_object"] = "update_object"
    object_id: str
    state: Optional[str] = Field(None, description="New visible state (e.g. 'open', 'broken', 'active')")
    description: Optional[str] = Field(None, description="New visible description")
    internal_state: Optional[dict] = Field(None, description="Updates to internal state (merged with existing)")


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


# Union of world effect actions (changes to the world state)
WorldEffect = Union[
    UpdateObjectAction,
    CreateObjectAction,
    DestroyObjectAction,
    TransferObjectAction,
]


class WorldEngineDecision(BaseModel):
    """
    The complete output format for WorldEngine's decision via tool call.
    Separates the interaction result from world effects for clarity.
    """
    result: InteractionResult = Field(description="The outcome of the interaction")
    effects: List[WorldEffect] = Field(default_factory=list, description="Optional list of world state changes")


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
