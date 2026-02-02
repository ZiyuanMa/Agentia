from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import networkx as nx
import logging
import json

from config import TICK_DURATION_MINUTES
from schemas import WorldObject, Location
from utils import LLMClient
from env_agent import EnvironmentAgent

logger = logging.getLogger("Agentia.World")


class World:
    def __init__(self, path_or_config, llm_client: LLMClient = None):
        self.logger = logging.getLogger("Agentia.World")
        self.graph = nx.Graph()
        self.locations: Dict[str, Location] = {}
        self.objects: Dict[str, WorldObject] = {}
        self.pending_events: Dict[str, List[str]] = {}  # agent_name -> list of events
        self.agent_locks: Dict[str, Dict] = {}  # agent_name -> lock info
        self.agent_locations: Dict[str, str] = {}  # agent_name -> location_id (single source of truth)
        
        # Time management - World is the single source of truth for simulation time
        self.sim_time = datetime(2024, 1, 1, 8, 0)  # Start Monday 8:00 AM
        
        # Initialize EnvAgent if LLM client provided
        self.env_agent = EnvironmentAgent(llm_client) if llm_client else None
        
        if isinstance(path_or_config, str):
            with open(path_or_config, 'r') as f:
                config = json.load(f)
        else:
            config = path_or_config
            
        self._load_from_config(config)

    def _load_from_config(self, config: Dict):
        # Load Locations
        for loc_data in config.get("locations", []):
            loc = Location(
                id=loc_data["id"],
                name=loc_data["name"],
                description=loc_data["description"],
                connected_to=loc_data.get("connected_to", []),
                objects=loc_data.get("objects", [])
            )
            self.locations[loc.id] = loc
            self.graph.add_node(loc.id, data=loc)
            
            # Add edges
            for target_id in loc.connected_to:
                self.graph.add_edge(loc.id, target_id)

        # Load Objects
        for obj_data in config.get("objects", []):
            obj = WorldObject(
                id=obj_data["id"],
                name=obj_data["name"],
                location_id=obj_data["location"], # Note: config uses 'location' but class uses 'location_id' for clarity, assuming mapping
                state=obj_data.get("state", "normal"),
                description=obj_data["description"],
                properties=obj_data.get("properties", [])
            )
            self.objects[obj.id] = obj

    def get_location(self, location_id: str) -> Optional[Location]:
        return self.locations.get(location_id)

    def get_object(self, object_id: str) -> Optional[WorldObject]:
        return self.objects.get(object_id)

    def place_agent(self, agent_name: str, location_id: str):
        """Place an agent in a location (used during initialization)."""
        loc = self.get_location(location_id)
        if loc:
            self.agent_locations[agent_name] = location_id
            if agent_name not in loc.agents_present:
                loc.agents_present.append(agent_name)
            return True
        return False
    
    def get_agent_location(self, agent_name: str) -> Optional[str]:
        """Get the current location of an agent."""
        return self.agent_locations.get(agent_name)

    def move_agent(self, agent_name: str, to_loc: str) -> bool:
        """Move an agent to a new location. Returns success status."""
        from_loc = self.agent_locations.get(agent_name)
        if not from_loc:
            return False
            
        if not self.graph.has_edge(from_loc, to_loc):
            return False
            
        location_from = self.get_location(from_loc)
        location_to = self.get_location(to_loc)
        
        if location_from and agent_name in location_from.agents_present:
            location_from.agents_present.remove(agent_name)
        
        if location_to:
            location_to.agents_present.append(agent_name)
            self.agent_locations[agent_name] = to_loc
            return True
        return False
        
    def get_connected_locations(self, location_id: str) -> List[str]:
         if location_id in self.locations:
             return self.locations[location_id].connected_to
         return []

    def create_object(self, object_id: str, name: str, location_id: str, 
                     state: str = "normal", description: str = "", 
                     properties: List[str] = None) -> bool:
        """Create a new object in the world. Returns success status."""
        if object_id in self.objects:
            self.logger.warning(f"Object {object_id} already exists")
            return False
        
        obj = WorldObject(
            id=object_id,
            name=name,
            location_id=location_id,
            state=state,
            description=description,
            properties=properties or []
        )
        self.objects[object_id] = obj
        
        # Add to location's object list if valid location
        if location_id and location_id in self.locations:
            self.locations[location_id].objects.append(object_id)
            
        self.logger.info(f"Created object: {name} ({object_id}) at {location_id}")
        return True

    def destroy_object(self, object_id: str) -> bool:
        """Remove an object from the world. Returns success status."""
        if object_id not in self.objects:
            self.logger.warning(f"Object {object_id} not found")
            return False
        
        obj = self.objects.pop(object_id)
        
        # Remove from location's object list
        if obj.location_id and obj.location_id in self.locations:
            if object_id in self.locations[obj.location_id].objects:
                self.locations[obj.location_id].objects.remove(object_id)
                
        self.logger.info(f"Destroyed object: {obj.name} ({object_id})")
        return True

    def transfer_object(self, object_id: str, 
                       from_location: str = None, to_location: str = None,
                       from_agent: str = None, to_agent: str = None) -> bool:
        """
        Transfer an object between locations or agent inventories.
        Returns success status.
        """
        obj = self.objects.get(object_id)
        if not obj:
            self.logger.warning(f"Object {object_id} not found for transfer")
            return False
        
        # Handle from_agent -> to_location (agent drops item)
        # Handle from_agent -> to_agent (agent gives item)
        # Handle from_location -> to_agent (agent picks up item)
        # Handle from_location -> to_location (move object)
        
        if from_agent and to_location:
            # Agent drops item to location
            obj.location_id = to_location
            if to_location in self.locations:
                self.locations[to_location].objects.append(object_id)
            self.logger.info(f"{from_agent} dropped {obj.name} at {to_location}")
            
        elif from_location and to_agent:
            # Agent picks up item from location
            if from_location in self.locations:
                if object_id in self.locations[from_location].objects:
                    self.locations[from_location].objects.remove(object_id)
            obj.location_id = None  # No longer at a location
            self.logger.info(f"{to_agent} picked up {obj.name} from {from_location}")
            
        elif from_agent and to_agent:
            # Agent gives item to another agent
            self.logger.info(f"{from_agent} gave {obj.name} to {to_agent}")
            
        elif from_location and to_location:
            # Move object between locations
            if from_location in self.locations:
                if object_id in self.locations[from_location].objects:
                    self.locations[from_location].objects.remove(object_id)
            
            obj.location_id = to_location
            if to_location in self.locations:
                self.locations[to_location].objects.append(object_id)
                
            self.logger.info(f"Moved {obj.name} from {from_location} to {to_location}")
        else:
            self.logger.warning(f"Invalid transfer parameters for {object_id}")
            return False
        
        return True

    def broadcast_to_location(self, location_id: str, message: str, exclude_agent: str = None):
        """
        Broadcast an event message to all agents in a location.
        The message will be added to each agent's pending events queue.
        """
        loc = self.get_location(location_id)
        if not loc:
            return
        
        for agent_name in loc.agents_present:
            if agent_name != exclude_agent:
                if agent_name not in self.pending_events:
                    self.pending_events[agent_name] = []
                self.pending_events[agent_name].append(message)
                logger.info(f"Event queued for {agent_name}: {message}")

    def get_pending_events(self, agent_name: str) -> List[str]:
        """
        Get and clear all pending events for an agent.
        """
        events = self.pending_events.pop(agent_name, [])
        return events

    def get_agent_context_data(self, agent_name: str, location_id: str) -> Dict[str, Any]:
        """
        Build the dictionary of context data for an agent's decision-making.
        """
        time_str = self.get_time_str()
        loc = self.get_location(location_id)
        
        data = {
            "time": time_str,
            "location_name": "Unknown",
            "location_description": "You are in an unknown void.",
            "people": "No one else",
            "objects": "None",
            "connections": "None",
            "events": "None"
        }
        
        if not loc:
             return data
        
        data["location_name"] = loc.name
        data["location_description"] = loc.description
        
        # People present (excluding self)
        people = [p for p in loc.agents_present if p != agent_name]
        data["people"] = str(people) if people else "No one else"
        
        # Objects in location with their states
        if loc.objects:
            objects_info = []
            for obj_id in loc.objects:
                obj = self.get_object(obj_id)
                if obj:
                    objects_info.append(f"  - {obj.name} (id: {obj.id}, state: {obj.state})")
            if objects_info:
                data["objects"] = "\n".join(objects_info)
        
        # Connected locations
        data["connections"] = str(loc.connected_to)
        
        # Recent events
        pending_events = self.get_pending_events(agent_name)
        if pending_events:
            data["events"] = "\n".join([f"- {event}" for event in pending_events])
            
        return data


    def process_action(self, agent_name: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes an agent's decision against the world state.
        Returns a result dictionary with success status and message.
        """
        action_type = decision.get("action_type")
        target = decision.get("target")
        content = decision.get("content")
        current_location_id = self.get_agent_location(agent_name)
        
        result = {"success": False, "message": ""}
        
        if action_type == "move":
            if target:
                success = self.move_agent(agent_name, target)
                if success:
                    result["success"] = True
                    result["message"] = f"Successfully moved to {target}."
                else:
                    result["message"] = f"Failed to move to {target}. It might not be connected or valid."
            else:
                 result["message"] = "Move action requires a target."

        elif action_type == "talk":
            if content:
                logger.info(f"{agent_name} says: '{content}'")
                # Broadcast to others in the same location
                self.broadcast_to_location(
                    current_location_id,
                    f"You heard {agent_name} say: '{content}'",
                    exclude_agent=agent_name
                )
                result["success"] = True
                result["message"] = f"You said: '{content}'"
            else:
                result["message"] = "Talk action requires content."
                
        elif action_type == "wait":
            result["success"] = True
            result["message"] = "Waited for one tick."
        
        elif action_type == "interact":
            if target:
                obj = self.get_object(target)
                if obj:
                    # Always use EnvAgent for richer interaction handling
                    action_desc = content or ""
                    loc = self.get_location(current_location_id)
                    witnesses = loc.agents_present if loc else []
                    result = self.env_agent.resolve_interaction(
                        agent_name=agent_name,
                        target_object=obj,
                        action_description=action_desc,
                        location=loc,
                        witnesses=witnesses,
                        world=self
                    )
                else:
                    result["message"] = f"Object '{target}' not found."
            else:
                result["message"] = "Interact action requires a target object."
            
        else:
            result["message"] = f"Unknown action type: {action_type}"
            
        return result

    def advance_time(self):
        """Advance simulation time by one tick."""
        self.sim_time += timedelta(minutes=TICK_DURATION_MINUTES)
    
    def get_time_str(self) -> str:
        """Get formatted time string for agent context."""
        return self.sim_time.strftime("%A, %I:%M %p")

    def set_agent_lock(self, agent_name: str, duration_minutes: int, reason: str, 
                       completion_message: str = None, pending_effects: List[Dict] = None):
        """Set a lock on an agent for a duration with optional deferred effects."""
        until_time = self.sim_time + timedelta(minutes=duration_minutes)
        
        self.agent_locks[agent_name] = {
            "until_time": until_time,
            "reason": reason,
            "completion_message": completion_message or f"Finished {reason}.",
            "pending_effects": pending_effects or []
        }
        logger.info(f"Agent {agent_name} locked until {until_time.strftime('%I:%M %p')}: {reason}")

    def check_agent_lock(self, agent_name: str) -> Optional[Dict]:
        """Check if agent is locked. Returns lock info or None. Executes pending effects if lock expired."""
        lock = self.agent_locks.get(agent_name)
        if not lock:
            return None
        
        if self.sim_time >= lock["until_time"]:
            # Lock expired - execute pending effects
            for effect in lock.get("pending_effects", []):
                self._execute_pending_effect(effect)
            
            del self.agent_locks[agent_name]
            return {"expired": True, "message": lock["completion_message"]}
        
        return {"expired": False, "reason": lock["reason"]}

    def _execute_pending_effect(self, effect: Dict):
        """Execute a deferred effect when an agent lock expires."""
        effect_type = effect.get("type")
        args = effect.get("args", {})
        
        if effect_type == "CreateObject":
            self.create_object(**args)
        elif effect_type == "DestroyObject":
            self.destroy_object(args.get("object_id"))
        elif effect_type == "TransferObject":
            self.transfer_object(**args)
        elif effect_type == "ModifyObjectState":
            obj = self.get_object(args.get("object_id"))
            if obj:
                obj.state = args.get("new_state")
                logger.info(f"Deferred: Object {args.get('object_id')} state -> {args.get('new_state')}")


