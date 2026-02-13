from typing import List, Dict, Any, Optional, TYPE_CHECKING
import json
import logging
from pydantic import BaseModel, Field, ValidationError

from schemas import (
    WorldEngineDecision,
    InteractionResult,
    WorldObject,
    Location
)
from prompts import WORLD_ENGINE_SYSTEM_PROMPT, WORLD_ENGINE_CONTEXT_TEMPLATE
from utils import LLMClient

if TYPE_CHECKING:
    from world import World

class WorldEngine:
    """
    The World Engine acts as a Game Master, resolving complex interactions.
    It uses a ReAct loop (Tool Calling) to investigate the world state before making a decision.
    """
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
        self.logger = logging.getLogger("Agentia.WorldEngine")
        
        # Define the tools available to the GM
        from schemas import (
            InteractionResult,
            UpdateObjectAction,
            CreateObjectAction,
            DestroyObjectAction,
            TransferObjectAction
        )
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_object",
                    "description": "Get the full state of ANY object in the world by its ID. Use this to check remote switches, hidden items, or internal states.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "object_id": {"type": "string", "description": "The ID of the object to inspect."}
                        },
                        "required": ["object_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_inventory",
                    "description": "Check what an agent is holding. Use this to verify keys, tools, or keycards.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_name": {"type": "string", "description": "The name of the agent to check."}
                        },
                        "required": ["agent_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_location",
                    "description": "Get details about a specific location, including who is there and what objects are visible.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location_id": {"type": "string", "description": "The ID of the location."}
                        },
                        "required": ["location_id"]
                    }
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
                    "parameters": UpdateObjectAction.model_json_schema()
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_object",
                    "description": "Create a new object in the world.",
                    "parameters": CreateObjectAction.model_json_schema()
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "destroy_object",
                    "description": "Permanently remove an object from the world.",
                    "parameters": DestroyObjectAction.model_json_schema()
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "transfer_object",
                    "description": "Move an object between containers, locations, or agents.",
                    "parameters": TransferObjectAction.model_json_schema()
                }
            }
        ]

    def resolve_interaction(self, agent_name: str, target_object: WorldObject,
                           action_description: str, location: Location,
                           witnesses: List[str], world: 'World',
                           inventory: List[str] = None) -> Dict[str, Any]:
        """
        Resolve interaction using a loop of Tool Calls.
        Ends when 'interaction_result' is called.
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
        
        # Accumulate effects to be applied (if duration > 0) or apply immediately
        pending_effects = []
        final_result = None
        
        # ReAct Loop (Max 10 turns)
        for turn in range(10):
            response = self.llm.chat_completion(
                messages,
                tools=self.tools
            )
            
            if not response:
                return {"success": False, "message": "The Game Master is silent (LLM Error)."}
            
            msg = response
            messages.append(msg)
            
            if msg.tool_calls:
                count = len(msg.tool_calls)
                if msg.content:
                    self.logger.info(f"GM Turn {turn+1} Thought: {msg.content}")
                self.logger.info(f"GM Turn {turn+1}: Emitting {count} tool call(s)...")
                
                # Flag to check if we are finishing this turn
                finished = False
                
                for i, tool_call in enumerate(msg.tool_calls):
                    func_name = tool_call.function.name
                    args_str = tool_call.function.arguments
                    call_id = tool_call.id
                    
                    self.logger.info(f"  [{i+1}/{count}] {func_name}")
                    
                    try:
                        args = json.loads(args_str)
                        tool_result = {"status": "ok"} # Default response
                        
                        # --- Inquiry Tools ---
                        if func_name in ["query_object", "check_inventory", "query_location"]:
                            tool_result = self._execute_inquiry_tool(func_name, args, world)
                            
                        # --- Action Tools ---
                        elif func_name in ["update_object", "create_object", "destroy_object", "transfer_object"]:
                            # Standardize mapping to internal effect format
                            effect_type_map = {
                                "update_object": "UpdateObject",
                                "create_object": "CreateObject",
                                "destroy_object": "DestroyObject",
                                "transfer_object": "TransferObject"
                            }
                            internal_type = effect_type_map.get(func_name)
                            # Remove 'action' field which is part of the schema but not needed for internal args
                            if "action" in args:
                                del args["action"]
                                
                            pending_effects.append({"type": internal_type, "args": args})
                            tool_result = {"status": "effect_staged", "message": f"{func_name} staged."}
                            self.logger.info(f"  -> Staged effect: {internal_type}")

                        # --- Final Result Tool ---
                        elif func_name == "interaction_result":
                            # Validate against schema
                            from schemas import InteractionResult
                            decision = InteractionResult.model_validate(args)
                            final_result = decision
                            finished = True
                            tool_result = {"status": "received", "message": "Interaction finalized."}
                            self.logger.info(f"  -> Interaction Finalized: {decision.message}")

                        else:
                            tool_result = {"error": f"Unknown tool: {func_name}"}

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
                
                if finished and final_result:
                    return self._finalize_interaction(final_result, pending_effects, world, agent_name)
                
                continue
            
            # No tool call
            content = msg.content or ""
            if not content:
                 return {"success": False, "message": "GM Error: Empty response."}
            
            # If GM just talks without calling tools
            self.logger.warning("GM output text without tool call.")
            messages.append({"role": "user", "content": "Please call 'interaction_result' to finalize the outcome."})
            continue
        
        return {"success": False, "message": "The interaction took too long to resolve."}

    def _execute_inquiry_tool(self, name: str, args: Dict, world: 'World') -> Dict:
        """Execute read-only inquiry tools."""
        if name == "query_object":
            obj = world.get_object(args.get("object_id"))
            return obj.model_dump() if obj else {"error": "Object not found"}
        
        elif name == "check_inventory":
            items = world.get_agent_inventory(args.get("agent_name"))
            return {"inventory": items}
            
        elif name == "query_location":
            loc = world.get_location(args.get("location_id"))
            return loc.model_dump() if loc else {"error": "Location not found"}
        
        return {"error": "Unknown inquiry tool"}

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
