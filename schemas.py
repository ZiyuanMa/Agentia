"""
Agentia Schema Definitions

All Pydantic models for the simulation are centralized here:
- World models (WorldObject, Location)
- Agent action models (MoveAction, TalkAction, etc.)
- EnvAgent action models (ResultAction, ModifyStateAction, etc.)
- Decision models (AgentDecision, EnvAgentDecision)
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
    state: str = "normal"
    description: str
    properties: List[str] = Field(default_factory=list)


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





# Backwards compatibility aliases
Move = MoveAction
Talk = TalkAction
Interact = InteractAction
Wait = WaitAction



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
# EnvAgent Action Models
# =============================================================================

class ResultAction(BaseModel):
    """Report the outcome of the interaction to the actor."""
    action: Literal["result"] = "result"
    success: bool
    message: str = Field(description="Narrative of what happened")
    reasoning: str = Field(description="Your analysis of the action", default="")


class ModifyStateAction(BaseModel):
    """Change an object's state."""
    action: Literal["modify_state"] = "modify_state"
    object_id: str
    new_state: str


class BroadcastAction(BaseModel):
    """Broadcast an event message to everyone in a location."""
    action: Literal["broadcast"] = "broadcast"
    location_id: str
    message: str


class LockAgentAction(BaseModel):
    """Lock an agent for a duration (for time-consuming actions)."""
    action: Literal["lock_agent"] = "lock_agent"
    agent_name: str
    duration_minutes: int
    reason: str
    completion_message: str = ""


class CreateObjectAction(BaseModel):
    """Create a new object in the world."""
    action: Literal["create_object"] = "create_object"
    object_id: str
    name: str
    location_id: str
    state: str = "normal"
    description: str = ""
    properties: List[str] = Field(default_factory=list)


class DestroyObjectAction(BaseModel):
    """Remove an object from the world."""
    action: Literal["destroy_object"] = "destroy_object"
    object_id: str
    reason: str = ""


class TransferObjectAction(BaseModel):
    """Transfer an object between locations or agents."""
    action: Literal["transfer_object"] = "transfer_object"
    object_id: str
    from_location: Optional[str] = None
    to_location: Optional[str] = None
    from_agent: Optional[str] = None
    to_agent: Optional[str] = None


# Union of all EnvAgent action types
EnvAgentAction = Union[
    ResultAction,
    ModifyStateAction,
    BroadcastAction,
    LockAgentAction,
    CreateObjectAction,
    DestroyObjectAction,
    TransferObjectAction,
]


class EnvAgentDecision(BaseModel):
    """
    The complete output format for EnvAgent's decision.
    EnvAgent validates actions using this model.
    """
    reasoning: str = Field(description="Step-by-step analysis of the situation and physics before deciding outcomes")
    decisions: List[EnvAgentAction] = Field(description="List of actions to execute, starting with a ResultAction")


# =============================================================================
# Legacy Tool Models (for OpenAI function calling compatibility)
# =============================================================================

class ModifyObjectState(BaseModel):
    """Change an object's state in the world"""
    object_id: str = Field(description="The ID of the object to modify")
    new_state: str = Field(description="The new state of the object (e.g., 'broken', 'working', 'empty')")
    description: str = Field(description="A brief description of what happened")


class BroadcastEvent(BaseModel):
    """Announce an event to everyone in a location"""
    location_id: str = Field(description="The location where the event occurs")
    message: str = Field(description="The event message that everyone will perceive")


class LockAgent(BaseModel):
    """Lock an agent for a duration (they cannot act during this time)"""
    agent_name: str = Field(description="Name of the agent to lock")
    duration_minutes: int = Field(description="How long the agent is busy (in minutes)")
    reason: str = Field(description="Why the agent is busy")
    completion_message: Optional[str] = Field(None, description="Message shown when the action completes")


class ActionResult(BaseModel):
    """Report the immediate result of the action to the actor"""
    success: bool = Field(description="Whether the action succeeded")
    message: str = Field(description="Feedback message for the actor")


class CreateObject(BaseModel):
    """Create a new object in the world (e.g., making coffee creates a coffee cup)"""
    object_id: str = Field(description="Unique ID for the new object")
    name: str = Field(description="Display name of the object")
    location_id: str = Field(description="Location where the object is created")
    state: str = Field(default="normal", description="Initial state of the object")
    description: str = Field(description="Description of the object")
    properties: List[str] = Field(default_factory=list, description="Object properties (e.g., 'consumable', 'portable')")


class DestroyObject(BaseModel):
    """Remove an object from the world (e.g., consuming food, breaking something beyond repair)"""
    object_id: str = Field(description="ID of the object to destroy")
    reason: str = Field(description="Why the object was destroyed")


class TransferObject(BaseModel):
    """Transfer an object between locations or to/from an agent's inventory"""
    object_id: str = Field(description="ID of the object to transfer")
    from_location: Optional[str] = Field(None, description="Source location ID (None if from agent inventory)")
    to_location: Optional[str] = Field(None, description="Destination location ID (None if to agent inventory)")
    from_agent: Optional[str] = Field(None, description="Agent giving the object (None if from location)")
    to_agent: Optional[str] = Field(None, description="Agent receiving the object (None if to location)")


# =============================================================================
# Schema Generation Utilities
# =============================================================================

def get_agent_decision_schema() -> str:
    """Get a simplified JSON schema string for the agent decision format."""
    schema_dict = AgentDecision.model_json_schema()
    schema_str = json.dumps(schema_dict, indent=2)
    return schema_str.replace("{", "{{").replace("}", "}}")


def get_env_agent_decision_schema() -> str:
    """Get a simplified JSON schema string for the EnvAgent output format."""
    schema_dict = EnvAgentDecision.model_json_schema()
    schema_str = json.dumps(schema_dict, indent=2)
    return schema_str.replace("{", "{{").replace("}", "}}")


def pydantic_to_openai_tool(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model to OpenAI function calling format."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": model.__name__,
            "description": model.__doc__ or "",
            "parameters": schema
        }
    }


# List of tool models for backwards compatibility
ENV_AGENT_TOOL_MODELS: List[type[BaseModel]] = [
    ModifyObjectState,
    BroadcastEvent,
    LockAgent,
    ActionResult,
    CreateObject,
    DestroyObject,
    TransferObject,
]

SIM_AGENT_TOOL_MODELS: List[type[BaseModel]] = [
    MoveAction,
    TalkAction,
    InteractAction,
    WaitAction,
]

# Generate OpenAI tools format
ENV_AGENT_TOOLS = [pydantic_to_openai_tool(model) for model in ENV_AGENT_TOOL_MODELS]
SIM_AGENT_TOOLS = [pydantic_to_openai_tool(model) for model in SIM_AGENT_TOOL_MODELS]
