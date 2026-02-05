from typing import List, Dict, Any, TYPE_CHECKING, Union
from pydantic import BaseModel, ValidationError
import logging
import json

from schemas import (
    ResultAction,
    ModifyStateAction,
    ModifyInternalStateAction,
    LockAgentAction,
    CreateObjectAction,
    DestroyObjectAction,
    TransferObjectAction,
    WorldEngineDecision,
)
from prompts import WORLD_ENGINE_SYSTEM_PROMPT, WORLD_ENGINE_CONTEXT_TEMPLATE

if TYPE_CHECKING:
    from world import World, WorldObject, Location

from utils import LLMClient


def _record_world_engine_call():
    """Record a WorldEngine call to stats if available."""
    try:
        from logger_config import get_stats
        stats = get_stats()
        if stats:
            # Legacy stat name
            stats.record_env_agent_call()
    except ImportError:
        pass


class WorldEngine:
    """
    The World Engine (formerly EnvironmentAgent) acts as a Game Master, 
    resolving complex interactions using JSON output mode with a list of typed actions.
    """
    def __init__(self, llm_client: LLMClient):
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
            
            # Log the top-level reasoning
            self.logger.info(f"WorldEngine reasoning: {decision.reasoning}")
            
            actions = decision.decisions
            
            # Process each action
            pending_effects = []
            lock_action = None
            
            for action in actions:
                # Use isinstance to check action types since they are already parsed models
                if isinstance(action, ResultAction):
                    result["success"] = action.success
                    result["message"] = action.message
                    # Top-level reasoning already logged, no need to duplicate
                    self.logger.info(f"WorldEngine result: success={action.success}, message={action.message[:100]}...")
                    
                elif isinstance(action, LockAgentAction):
                    lock_action = action
                    self.logger.info(f"WorldEngine action: lock_agent(agent={lock_action.agent_name}, duration={lock_action.duration_minutes}min, desc={lock_action.description})")
                    
                elif isinstance(action, ModifyStateAction):
                    pending_effects.append(("modify_state", action))
                    self.logger.info(f"WorldEngine action: modify_state(object={action.object_id}, new_state={action.new_state})")

                elif isinstance(action, ModifyInternalStateAction):
                    pending_effects.append(("modify_internal_state", action))
                    self.logger.info(f"WorldEngine action: modify_internal_state(object={action.object_id}, key={action.key}, value={action.value})")

                elif isinstance(action, CreateObjectAction):
                    pending_effects.append(("create_object", action))
                    self.logger.info(f"WorldEngine action: create_object(id={action.object_id}, name={action.name}, location={action.location_id}, state={action.state})")
                    
                elif isinstance(action, DestroyObjectAction):
                    pending_effects.append(("destroy_object", action))
                    self.logger.info(f"WorldEngine action: destroy_object(id={action.object_id})")
                    
                elif isinstance(action, TransferObjectAction):
                    pending_effects.append(("transfer_object", action))
                    self.logger.info(f"WorldEngine action: transfer_object(id={action.object_id}, from={action.from_id}, to={action.to_id})")
            
            # If there's a lock, defer effects until lock expires
            if lock_action:
                deferred = [self._action_to_effect_dict(t, a) for t, a in pending_effects]
                world.set_agent_lock(
                    lock_action.agent_name,
                    lock_action.duration_minutes,
                    lock_action.description,
                    lock_action.completion_message,
                    pending_effects=deferred
                )
                self.logger.info(f"LockAgent set with {len(deferred)} pending effects")
            else:
                # Execute effects immediately
                for action_type, action in pending_effects:
                    self._execute_action(action_type, action, world, location)
                    
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
            "modify_state": "ModifyState",
            "modify_internal_state": "ModifyInternalState",
            "create_object": "CreateObject",
            "destroy_object": "DestroyObject",
            "transfer_object": "TransferObject",
        }
        
        effect_type = type_map.get(action_type, action_type)
        args = action.model_dump(exclude={"action"})
        
        return {"type": effect_type, "args": args}
    
    def _execute_action(self, action_type: str, action: BaseModel, world: 'World', location: 'Location'):
        """Execute a single action immediately."""
        self.logger.info(f"Executing action: {action_type}")
        
        if action_type == "modify_state":
            action: ModifyStateAction
            world.modify_object_state(action.object_id, action.new_state, action.new_description)

        elif action_type == "modify_internal_state":
            action: ModifyInternalStateAction
            world.modify_object_internal_state(action.object_id, action.key, action.value)
        
        elif action_type == "create_object":
            action: CreateObjectAction
            world.create_object(
                object_id=action.object_id,
                name=action.name,
                location_id=action.location_id,
                state=action.state,
                description=action.description,
                mechanics=action.mechanics,
                internal_state=action.internal_state
            )
        
        elif action_type == "destroy_object":
            action: DestroyObjectAction
            world.destroy_object(action.object_id)
        
        elif action_type == "transfer_object":
            action: TransferObjectAction
            world.transfer_object(
                object_id=action.object_id,
                from_id=action.from_id,
                to_id=action.to_id
            )
