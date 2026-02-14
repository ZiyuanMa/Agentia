from typing import List, Dict, Any, Optional, TYPE_CHECKING
import json
import logging
from pydantic import BaseModel, Field, ValidationError

from schemas import (
    InteractionResult,
    WorldObject,
    Location,
    QueryEntityParams,
    UpdateObject,
    CreateObject,
    DestroyObject,
    TransferObject
)
from prompts import WORLD_ENGINE_SYSTEM_PROMPT, WORLD_ENGINE_CONTEXT_TEMPLATE
from utils import LLMClient

if TYPE_CHECKING:
    from world import World

# =============================================================================
# Constants
# =============================================================================

# ReAct loop configuration
MAX_REACT_TURNS = 15

# Tool result status codes
TOOL_STATUS_OK = "ok"
TOOL_STATUS_STAGED = "effect_staged"
TOOL_STATUS_RECEIVED = "received"

# Special keys in tool results
RESULT_KEY = "_result"  # Key for passing InteractionResult through tool response

class WorldEngine:
    """
    LLM-powered Game Master for resolving agent-object interactions.
    
    The WorldEngine orchestrates a ReAct (Reasoning + Acting) loop where an LLM:
    1. Investigates world state using query tools
    2. Validates and stages modifications using action tools
    3. Produces a final narrative outcome with optional world effects
    
    This ensures interactions are:
    - Contextually aware (queries before deciding)
    - Logically consistent (validation prevents impossible actions)
    - Rich and emergent (LLM generates creative outcomes)
    
    Attributes:
        llm (LLMClient): Client for making LLM API calls
        tools (List[Dict]): Available tools (query_entity, update_object, etc.)
        logger (Logger): For debugging and interaction tracking
    
    Example:
        >>> engine = WorldEngine(llm_client)
        >>> result = engine.resolve_interaction(
        ...     "Alice", coffee_cup, "drink coffee", kitchen, [], world
        ... )
        >>> print(result['message'])
        'Alice takes a sip of the warm coffee, feeling energized.'
    """
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
        self.logger = logging.getLogger("Agentia.WorldEngine")
        
        # Define the tools available to the GM
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_entity",
                    "description": """Query any entity in the world by its ID. This works for:
- Objects: Returns full object state (id, name, state, description, mechanics, internal_state)
- Agents: Returns agent info including current location and inventory

Use this to investigate objects, check agent inventories, or inspect any entity before making decisions.""",
                    "parameters": QueryEntityParams.model_json_schema()
                }
            },
            # --- New Atomic Action Tools ---
            {
                "type": "function",
                "function": {
                    "name": "interaction_result",
                    "description": "Finalize the interaction. Call this to return the narrative outcome and duration. This ends your turn.",
                    "parameters": InteractionResult.model_json_schema()
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_object",
                    "description": "Update an object's state, description, or internal state.",
                    "parameters": UpdateObject.model_json_schema()
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_object",
                    "description": "Create a new object in the world.",
                    "parameters": CreateObject.model_json_schema()
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "destroy_object",
                    "description": "Permanently remove an object from the world.",
                    "parameters": DestroyObject.model_json_schema()
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "transfer_object",
                    "description": "Move an object between containers, locations, or agents.",
                    "parameters": TransferObject.model_json_schema()
                }
            }
        ]

    def resolve_interaction(self, agent_name: str, target_object: WorldObject,
                           action_description: str, location: Location,
                           witnesses: List[str], world: 'World',
                           inventory: List[str] = None) -> Dict[str, Any]:
        """
        Resolve an agent's interaction with an object using LLM-powered reasoning.
        
        The method runs a ReAct loop where the LLM can query world state,
        stage modifications, and finalize with a narrative result.
        
        Args:
            agent_name: Name of the agent performing the action
            target_object: The world object being interacted with
            action_description: Natural language description (e.g., "drink coffee")
            location: Current location context
            witnesses: List of agent names who can observe this interaction
            world: World instance for querying and modifying state
            inventory: Optional list of agent's current inventory items
        
        Returns:
            Dict with 'message' (str) describing the outcome. May also contain
            'success' (bool) on errors, or be empty on deferred effects (duration > 0).
        
        Notes:
            - The loop runs for a maximum of MAX_REACT_TURNS iterations
            - Effects are staged during reasoning and applied at finalization
            - Long-duration actions defer effect application via agent locks
        """
        # Record stats if available
        self._record_world_engine_call()
        
        self.logger.info(f"WorldEngine: Resolving '{action_description}' for {agent_name}...")
        
        context = self._build_context(agent_name, target_object, action_description, 
                                      location, witnesses, inventory)
        
        messages = [
            {"role": "system", "content": WORLD_ENGINE_SYSTEM_PROMPT}, 
            {"role": "user", "content": context}
        ]
        
        # Collect effects staged during reasoning; applied at finalization
        pending_effects = []
        final_result = None
        
        # ReAct Loop (Max turns configured by constant)
        for turn in range(MAX_REACT_TURNS):
            response = self.llm.chat_completion(
                messages,
                tools=self.tools
            )
            
            if not response:
                return {"success": False, "message": "The Game Master is silent (LLM Error)."}
            
            llm_response = response
            messages.append(llm_response)
            
            if llm_response.tool_calls:
                count = len(llm_response.tool_calls)
                if llm_response.content:
                    self.logger.info(f"GM Turn {turn+1} Thought: {llm_response.content}")
                self.logger.info(f"GM Turn {turn+1}: Emitting {count} tool call(s)...")
                
                for i, tool_call in enumerate(llm_response.tool_calls):
                    func_name = tool_call.function.name
                    args_str = tool_call.function.arguments
                    call_id = tool_call.id
                    
                    self.logger.info(f"  [{i+1}/{count}] {func_name} | args: {args_str}")
                    
                    try:
                        args = json.loads(args_str)
                        
                        # Dispatch to appropriate handler
                        tool_result = self._dispatch_tool(func_name, args, world, pending_effects)
                        
                        # Check if this was the final result
                        if RESULT_KEY in tool_result:
                            final_result = tool_result.pop(RESULT_KEY)
                            # Append tool response first
                            messages.append({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": json.dumps(tool_result)
                            })
                            # Then immediately return with finalization
                            return self._finalize_interaction(final_result, pending_effects, world, agent_name)

                        # Append tool output
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps(tool_result)
                        })
                        
                    except Exception as e:
                        error_msg = f"Tool execution failed: {str(e)}"
                        self.logger.error(error_msg)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps({"error": error_msg})
                        })
                
                # Continue to next turn
                continue
            
            # No tool calls - handle appropriately  
            if not self._handle_no_tool_call(llm_response, messages):
                return {"success": False, "message": "GM Error: Empty response."}
            continue
        
        
        return {"success": False, "message": "The interaction took too long to resolve."}
    
    def _handle_no_tool_call(self, llm_response: Any, messages: List[Dict]) -> bool:
        """
        Handle when LLM responds without calling tools.
        
        Args:
            llm_response: The LLM's response object (ChatCompletion)
            messages: Conversation history to append prompt to
        
        Returns:
            False if response is invalid (empty), True if handled successfully
        """
        content = llm_response.content or ""
        if not content:
            return False  # Signal invalid response
        
        # LLM wrote text but didn't call tools - remind it
        self.logger.warning("GM output text without tool call.")
        messages.append({
            "role": "user", 
            "content": "Please call 'interaction_result' to finalize the outcome."
        })
        return True
    
    # =========================================================================
    # Tool Dispatch and Handling
    # =========================================================================
    
    def _dispatch_tool(self, func_name: str, args: Dict, world: 'World', 
                       pending_effects: List[Dict]) -> Dict:
        """
        Dispatch tool call to appropriate handler.
        Returns tool result dict to send back to LLM.
        """
        # --- Query Tools ---
        if func_name == "query_entity":
            return self._execute_query_entity(args.get("entity_id"), world)
        
        # --- Action Tools (validate + stage) ---
        action_tools = {
            "update_object": (self._validate_update_object, "UpdateObject"),
            "create_object": (self._validate_create_object, "CreateObject"),
            "destroy_object": (self._validate_destroy_object, "DestroyObject"),
            "transfer_object": (self._validate_transfer_object, "TransferObject"),
        }
        
        if func_name in action_tools:
            validator, effect_type = action_tools[func_name]
            error = validator(args, world)
            if error:
                return error
            
            pending_effects.append({"type": effect_type, "args": args})
            self.logger.info(f"  -> Staged effect: {effect_type}")
            return {"status": TOOL_STATUS_STAGED, "message": f"{func_name} staged."}
        
        # --- Final Result Tool ---
        if func_name == "interaction_result":
            decision = InteractionResult.model_validate(args)
            self.logger.info(f"  -> Interaction Finalized: {decision.message}")
            return {"status": TOOL_STATUS_RECEIVED, "message": "Interaction finalized.", RESULT_KEY: decision}
        
        # --- Unknown Tool ---
        return {"error": f"Unknown tool: {func_name}"}
    
    # =========================================================================
    # Validation Helper Methods
    # =========================================================================
    
    def _validate_update_object(self, args: Dict, world: 'World') -> Optional[Dict]:
        """Validate update_object arguments. Returns error dict if invalid, None if valid."""
        object_id = args.get("object_id")
        obj = world.get_object(object_id)
        if not obj:
            return {"error": f"Cannot update: object '{object_id}' does not exist"}
        return None
    
    def _validate_create_object(self, args: Dict, world: 'World') -> Optional[Dict]:
        """Validate create_object arguments. Returns error dict if invalid, None if valid."""
        object_id = args.get("object_id")
        location_id = args.get("location_id")
        
        if object_id in world.objects:
            return {"error": f"Cannot create: object '{object_id}' already exists"}
        elif location_id and location_id not in world.locations:
            return {"error": f"Cannot create: location '{location_id}' does not exist"}
        return None
    
    def _validate_destroy_object(self, args: Dict, world: 'World') -> Optional[Dict]:
        """Validate destroy_object arguments. Returns error dict if invalid, None if valid."""
        object_id = args.get("object_id")
        obj = world.get_object(object_id)
        if not obj:
            return {"error": f"Cannot destroy: object '{object_id}' does not exist"}
        return None
    
    def _validate_transfer_object(self, args: Dict, world: 'World') -> Optional[Dict]:
        """Validate transfer_object arguments. Returns error dict if invalid, None if valid."""
        object_id = args.get("object_id")
        to_id = args.get("to_id")
        
        obj = world.get_object(object_id)
        if not obj:
            return {"error": f"Cannot transfer: object '{object_id}' does not exist"}
        elif to_id not in world.locations and to_id not in world.agent_locations and to_id not in world.objects:
            return {"error": f"Cannot transfer: destination '{to_id}' does not exist"}
        return None
    
    # =========================================================================
    # Tool Execution Methods
    # =========================================================================

    def _execute_query_entity(self, entity_id: str, world: 'World') -> Dict:
        """
        Unified query tool that checks objects and agents.
        Returns the first match found.
        """
        # Try as object
        obj = world.get_object(entity_id)
        if obj:
            return {
                "type": "object",
                "data": obj.model_dump()
            }
        
        # Try as agent (check if agent exists in world.agent_locations)
        if entity_id in world.agent_locations:
            location_id = world.get_agent_location(entity_id)
            inventory = world.get_agent_inventory(entity_id)
            return {
                "type": "agent",
                "data": {
                    "name": entity_id,
                    "location_id": location_id,
                    "inventory": inventory
                }
            }
        
        return {"error": f"Entity '{entity_id}' not found"}

    def _finalize_interaction(self, result_model: Any, 
                              pending_effects: List[Dict], 
                              world: 'World', agent_name: str) -> Dict:
        """Apply effects and return final dict."""
        output = {"message": result_model.message}
        duration = result_model.duration or 0
        
        if duration > 0:
            task = result_model.task_description or "busy"
            output["message"] = f"Started: {task} ({duration} min)..."
            
            # Defer effects
            world.set_agent_lock(
                agent_name,
                duration,
                task,
                result_model.message,
                pending_effects=pending_effects
            )
        else:
            # Apply immediately
            for effect in pending_effects:
                world.execute_effect(effect)
                
        return output


    def _build_context(self, agent_name: str, target_object: WorldObject,
                       action_description: str, location: Location,
                       witnesses: List[str], inventory: List[str] = None) -> str:
        return WORLD_ENGINE_CONTEXT_TEMPLATE.format(
            agent_name=agent_name,
            inventory=str(inventory) if inventory else "[]",
            object_name=target_object.name,
            object_id=target_object.id,
            object_state=target_object.state,
            object_description=target_object.description,
            object_internal_state=str(target_object.internal_state) if target_object.internal_state else "{}",
            object_mechanics=target_object.mechanics if target_object.mechanics else "None", 
            location_name=location.name if location else 'Unknown',
            location_id=location.id if location else 'unknown',
            location_description=location.description if location else '',
            witnesses=[w for w in witnesses if w != agent_name] if witnesses else 'None',
            action_description=action_description if action_description else 'interact with the object'
        )

    def _record_world_engine_call(self) -> None:
        try:
            from logger_config import get_stats
            stats = get_stats()
            if stats:
                stats.record_world_engine_call()
        except ImportError:
            pass
