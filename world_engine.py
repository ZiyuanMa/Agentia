from typing import List, Dict, Any, TYPE_CHECKING
from pydantic import BaseModel, ValidationError
import logging
import json

from schemas import (
    InteractionResult,
    UpdateObjectAction,
    CreateObjectAction,
    DestroyObjectAction,
    TransferObjectAction,
    WorldEngineDecision,
)
from prompts import WORLD_ENGINE_SYSTEM_PROMPT, WORLD_ENGINE_CONTEXT_TEMPLATE

if TYPE_CHECKING:
    from world import World, WorldObject, Location

from utils import LLMClient


def _record_world_engine_call() -> None:
    """Record a WorldEngine call to stats if available."""
    try:
        from logger_config import get_stats
        stats = get_stats()
        if stats:
            stats.record_world_engine_call()
    except ImportError:
        pass


class WorldEngine:
    """
    The World Engine acts as a Game Master, resolving complex interactions
    using JSON output mode with a list of typed actions.
    """
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
        self.logger = logging.getLogger("Agentia.WorldEngine")
    
    def resolve_interaction(self, agent_name: str, target_object: 'WorldObject',
                           action_description: str, location: 'Location',
                           witnesses: List[str], world: 'World',
                           inventory: List[str] = None) -> Dict[str, Any]:
        """
        Call LLM to determine the outcome of a complex interaction.
        Returns a result dict with success, message, and optionally triggers world effects.
        """
        _record_world_engine_call()
        
        # Build context prompt
        context = self._build_context(agent_name, target_object, action_description, 
                                      location, witnesses, inventory)
        
        messages = [
            {"role": "system", "content": WORLD_ENGINE_SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ]
        
        # Call LLM with JSON response format
        response = self.llm.chat_completion(
            messages, 
            response_format={"type": "json_object"}
        )
        
        if not response or not response.content:
            self.logger.error("WorldEngine LLM call failed")
            return {"success": False, "message": "The action had no effect."}
        
        content = response.content.strip()
        result = {"success": False, "message": "Parsing failed", "reasoning": None}
        
        try:
            # Use WorldEngineDecision for unified parsing
            decision = WorldEngineDecision.model_validate_json(content)
            
            # Log reasoning
            self.logger.info(f"WorldEngine reasoning: {decision.reasoning}")
            
            # Extract result directly (no need to search through a list)
            interaction = decision.result
            result["success"] = interaction.success
            result["message"] = interaction.message
            msg_preview = (interaction.message or "")[:100]
            self.logger.info(f"WorldEngine result: success={interaction.success}, duration={interaction.duration}min, message={msg_preview}...")
            
            # Collect effects
            pending_effects = []
            for effect in decision.effects:
                action_type = effect.action  # e.g. "update_object", "create_object"
                pending_effects.append((action_type, effect))
                self.logger.info(f"WorldEngine effect: {action_type}({effect.model_dump(exclude={'action'})})")
            
            # If interaction has duration > 0, defer effects until completion
            if interaction.duration > 0:
                task = interaction.task_description or "busy"
                
                # Override immediate message so Agent knows it started
                result["message"] = f"Started: {task} ({interaction.duration} min)..."
                
                deferred = [self._action_to_effect_dict(t, a) for t, a in pending_effects]
                world.set_agent_lock(
                    agent_name,
                    interaction.duration,
                    task,
                    interaction.message,  # Original message becomes completion message
                    pending_effects=deferred
                )
                self.logger.info(f"Agent locked for {interaction.duration}min, task={task}")
            else:
                # Execute effects immediately
                for action_type, action in pending_effects:
                    self._execute_action(action_type, action, world)
                    
        except (ValidationError, json.JSONDecodeError) as e:
            self.logger.error(f"WorldEngine Parse Error: {e}")
            # Try to extract message from raw content
            result["message"] = f"Action completed. {content[:100]}"
            result["success"] = True
        
        return result
    
    def _build_context(self, agent_name: str, target_object: 'WorldObject',
                       action_description: str, location: 'Location',
                       witnesses: List[str], inventory: List[str] = None) -> str:
        """Build the context prompt for the WorldEngine."""
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
    
    def _action_to_effect_dict(self, action_type: str, action: BaseModel) -> Dict[str, Any]:
        """Convert a typed action to a dict format for pending effects."""
        type_map = {
            "update_object": "UpdateObject",
            "create_object": "CreateObject",
            "destroy_object": "DestroyObject",
            "transfer_object": "TransferObject",
        }
        
        effect_type = type_map.get(action_type, action_type)
        args = action.model_dump(exclude={"action"})
        
        return {"type": effect_type, "args": args}
    
    def _execute_action(self, action_type: str, action: BaseModel, world: 'World') -> None:
        """Execute a single action immediately."""
        self.logger.info(f"Executing action: {action_type}")
        
        match action_type:
            case "update_object":
                world.update_object(
                    object_id=action.object_id,
                    state=action.state,
                    description=action.description,
                    internal_state=action.internal_state
                )
            
            case "create_object":
                world.create_object(
                    object_id=action.object_id,
                    name=action.name,
                    location_id=action.location_id,
                    state=action.state,
                    description=action.description,
                    mechanics=action.mechanics,
                    internal_state=action.internal_state
                )
            
            case "destroy_object":
                world.destroy_object(action.object_id)
            
            case "transfer_object":
                world.transfer_object(
                    object_id=action.object_id,
                    from_id=action.from_id,
                    to_id=action.to_id
                )
            
            case _:
                self.logger.warning(f"Unknown action type: {action_type}")
