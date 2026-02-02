from typing import List, Dict, Any, TYPE_CHECKING, Union
from pydantic import BaseModel, ValidationError
import logging
import json

from schemas import (
    ResultAction,
    ModifyStateAction,
    BroadcastAction,
    LockAgentAction,
    CreateObjectAction,
    DestroyObjectAction,
    TransferObjectAction,
    EnvAgentDecision,
)
from prompts import ENV_AGENT_SYSTEM_PROMPT, ENV_AGENT_CONTEXT_TEMPLATE

if TYPE_CHECKING:
    from world import World, WorldObject, Location

from utils import LLMClient


def _record_env_agent_call():
    """Record an EnvAgent call to stats if available."""
    try:
        from logger_config import get_stats
        stats = get_stats()
        if stats:
            stats.record_env_agent_call()
    except ImportError:
        pass


class EnvironmentAgent:
    """
    The Environment Agent acts as a Game Master, resolving complex interactions
    using JSON output mode with a list of typed actions.
    """
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.logger = logging.getLogger("Agentia.EnvAgent")
    
    def resolve_interaction(self, agent_name: str, target_object: 'WorldObject',
                           action_description: str, location: 'Location',
                           witnesses: List[str], world: 'World') -> Dict[str, Any]:
        """
        Call LLM to determine the outcome of a complex interaction.
        Returns a result dict with success, message, and optionally triggers world effects.
        """
        _record_env_agent_call()
        
        # Build context prompt
        context = self._build_context(agent_name, target_object, action_description, 
                                      location, witnesses)
        
        messages = [
            {"role": "system", "content": ENV_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ]
        
        # Call LLM with JSON response format
        response = self.llm.chat_completion(
            messages, 
            response_format={"type": "json_object"}
        )
        
        if not response or not response.content:
            self.logger.error("EnvAgent LLM call failed")
            return {"success": False, "message": "The action had no effect."}
        
        content = response.content.strip()
        result = {"success": False, "message": "Parsing failed", "reasoning": None}
        
        try:
            # Use EnvAgentDecision for unified parsing
            decision = EnvAgentDecision.model_validate_json(content)
            
            # Log the top-level reasoning
            self.logger.info(f"EnvAgent reasoning: {decision.reasoning}")
            
            actions = decision.decisions
            
            # Process each action
            pending_effects = []
            lock_action = None
            
            for action in actions:
                # Use isinstance to check action types since they are already parsed models
                if isinstance(action, ResultAction):
                    result["success"] = action.success
                    result["message"] = action.message
                    result["reasoning"] = action.reasoning
                    self.logger.info(f"EnvAgent reasoning: {action.reasoning}")
                    self.logger.info(f"EnvAgent result: success={action.success}, message={action.message[:100]}...")
                    
                elif isinstance(action, LockAgentAction):
                    lock_action = action
                    self.logger.info(f"EnvAgent action: lock_agent(agent={lock_action.agent_name}, duration={lock_action.duration_minutes}min, reason={lock_action.reason})")
                    
                elif isinstance(action, ModifyStateAction):
                    pending_effects.append(("modify_state", action))
                    self.logger.info(f"EnvAgent action: modify_state(object={action.object_id}, new_state={action.new_state})")
                    
                elif isinstance(action, BroadcastAction):
                    pending_effects.append(("broadcast", action))
                    self.logger.info(f"EnvAgent action: broadcast(location={action.location_id}, message={action.message[:50]}...)")
                    
                elif isinstance(action, CreateObjectAction):
                    pending_effects.append(("create_object", action))
                    self.logger.info(f"EnvAgent action: create_object(id={action.object_id}, name={action.name}, location={action.location_id}, state={action.state}, props={action.properties})")
                    
                elif isinstance(action, DestroyObjectAction):
                    pending_effects.append(("destroy_object", action))
                    self.logger.info(f"EnvAgent action: destroy_object(id={action.object_id}, reason={action.reason})")
                    
                elif isinstance(action, TransferObjectAction):
                    pending_effects.append(("transfer_object", action))
                    self.logger.info(f"EnvAgent action: transfer_object(id={action.object_id}, from={action.from_location or action.from_agent}, to={action.to_location or action.to_agent})")
            
            # If there's a lock, defer effects until lock expires
            if lock_action:
                deferred = [self._action_to_effect_dict(t, a) for t, a in pending_effects]
                world.set_agent_lock(
                    lock_action.agent_name,
                    lock_action.duration_minutes,
                    lock_action.reason,
                    lock_action.completion_message,
                    pending_effects=deferred
                )
                self.logger.info(f"LockAgent set with {len(deferred)} pending effects")
            else:
                # Execute effects immediately
                for action_type, action in pending_effects:
                    self._execute_action(action_type, action, world, location)
                    
        except (ValidationError, json.JSONDecodeError) as e:
            self.logger.error(f"EnvAgent Parse Error: {e}")
            # Try to extract message from raw content
            result["message"] = f"Action completed. {content[:100]}"
            result["success"] = True
        
        return result
    
    def _build_context(self, agent_name: str, target_object: 'WorldObject',
                      action_description: str, location: 'Location',
                      witnesses: List[str]) -> str:
        """Build the context prompt for the EnvAgent."""
        return ENV_AGENT_CONTEXT_TEMPLATE.format(
            agent_name=agent_name,
            object_name=target_object.name,
            object_id=target_object.id,
            object_state=target_object.state,
            object_properties=target_object.properties,
            object_description=target_object.description,
            location_name=location.name if location else 'Unknown',
            location_id=location.id if location else 'unknown',
            location_description=location.description if location else '',
            witnesses=[w for w in witnesses if w != agent_name] if witnesses else 'None',
            action_description=action_description if action_description else 'interact with the object'
        )
    
    def _action_to_effect_dict(self, action_type: str, action: BaseModel) -> Dict[str, Any]:
        """Convert a typed action to a dict format for pending effects."""
        type_map = {
            "modify_state": "ModifyObjectState",
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
            obj = world.get_object(action.object_id)
            if obj:
                old_state = obj.state
                obj.state = action.new_state
                self.logger.info(f"Object {action.object_id} state: {old_state} -> {action.new_state}")
        
        elif action_type == "broadcast":
            action: BroadcastAction
            world.broadcast_to_location(action.location_id, action.message)
        
        elif action_type == "create_object":
            action: CreateObjectAction
            world.create_object(
                object_id=action.object_id,
                name=action.name,
                location_id=action.location_id,
                state=action.state,
                description=action.description,
                properties=action.properties
            )
        
        elif action_type == "destroy_object":
            action: DestroyObjectAction
            world.destroy_object(action.object_id)
        
        elif action_type == "transfer_object":
            action: TransferObjectAction
            world.transfer_object(
                object_id=action.object_id,
                from_location=action.from_location,
                to_location=action.to_location,
                from_agent=action.from_agent,
                to_agent=action.to_agent
            )
