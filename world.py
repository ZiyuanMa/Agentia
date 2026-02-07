from typing import List, Dict, Optional, Any
from datetime import timedelta
import networkx as nx
import logging
import json

from config import TICK_DURATION_MINUTES, SIMULATION_START_TIME
from schemas import WorldObject, Location
from utils import LLMClient
from world_engine import WorldEngine


class World:
    """The simulation world containing locations, objects, and agents."""
    
    def __init__(self, path_or_config, llm_client: LLMClient = None) -> None:
        self.logger = logging.getLogger("Agentia.World")
        self.graph = nx.Graph()
        self.locations: Dict[str, Location] = {}
        self.objects: Dict[str, WorldObject] = {}
        self.pending_events: Dict[str, List[str]] = {}
        self.agent_locks: Dict[str, Dict] = {}
        self.agent_locations: Dict[str, str] = {}
        
        # Time management - World is the single source of truth for simulation time
        self.sim_time = SIMULATION_START_TIME
        
        # Initialize WorldEngine if LLM client provided
        self.world_engine = WorldEngine(llm_client) if llm_client else None
        
        if isinstance(path_or_config, str):
            with open(path_or_config, 'r') as f:
                config = json.load(f)
        else:
            config = path_or_config
            
        self._load_from_config(config)

    def _load_from_config(self, config: Dict) -> None:
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
                location_id=obj_data.get("location_id"),
                state=obj_data.get("state", "normal"),
                description=obj_data["description"],
                mechanics=obj_data.get("mechanics", ""),
                internal_state=obj_data.get("internal_state", {})
            )
            self.objects[obj.id] = obj

    def get_location(self, location_id: str) -> Optional[Location]:
        return self.locations.get(location_id)

    def get_object(self, object_id: str) -> Optional[WorldObject]:
        return self.objects.get(object_id)

    def place_agent(self, agent_name: str, location_id: str) -> bool:
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

    def get_agent_inventory(self, agent_name: str) -> List[str]:
        """Get list of object names held by an agent."""
        inventory = []
        for obj in self.objects.values():
            if obj.location_id == agent_name:
                inventory.append(f"{obj.name} (id: {obj.id})")
        return inventory

    def create_object(self, object_id: str, name: str, location_id: str, 
                     state: str = "normal", description: str = "", 
                     mechanics: str = "",
                     internal_state: Dict[str, Any] = None) -> bool:
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
            mechanics=mechanics,
            internal_state=internal_state or {}
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

    def transfer_object(self, object_id: str, from_id: str, to_id: str) -> bool:
        """
        Transfer an object between IDs (Locations, Agents, or Containers).
        Returns success status.
        """
        obj = self.objects.get(object_id)
        if not obj:
            self.logger.warning(f"Object {object_id} not found for transfer")
            return False
        
        # --- REMOVE from Source ---
        if from_id and from_id in self.locations:
            if object_id in self.locations[from_id].objects:
                self.locations[from_id].objects.remove(object_id)
        
        # Note: If from_id is an Agent or Container, we don't need to update a list 
        # because we only track location_id on the object itself for those cases.

        # --- ADD to Destination ---
        
        # Update object's location pointer
        obj.location_id = to_id
        
        # If destination is a Location (Room), update its list
        if to_id in self.locations:
            self.locations[to_id].objects.append(object_id)
            self.logger.info(f"Transferred {obj.name} to location {to_id}")
            
        elif to_id in self.agent_locations: # It's an Agent
            self.logger.info(f"Transferred {obj.name} to agent {to_id}")
             
        elif to_id in self.objects: # It's a Container Object
            self.logger.info(f"Transferred {obj.name} into container {to_id}")
             
        else:
            self.logger.warning(f"Transferred {obj.name} to unknown ID {to_id} (assuming external/agent)")
        
        return True

    def update_object(self, object_id: str, state: str = None, 
                       description: str = None, internal_state: dict = None) -> bool:
        """
        Update any field(s) of an object. Only provided (non-None) fields will be updated.
        internal_state is merged with existing values, not replaced entirely.
        """
        obj = self.objects.get(object_id)
        if not obj:
            self.logger.warning(f"Object {object_id} not found for update")
            return False
        
        updates = []
        
        if state is not None:
            obj.state = state
            updates.append(f"state={state}")
            
        if description is not None:
            obj.description = description
            updates.append(f"description updated")
            
        if internal_state is not None:
            obj.internal_state.update(internal_state)
            updates.append(f"internal_state+={internal_state}")
        
        if updates:
            self.logger.info(f"Object {object_id} updated: {', '.join(updates)}")
        
        return True

    def broadcast_to_location(self, location_id: str, message: str, exclude_agent: str = None) -> None:
        """
        Broadcast an event message to all agents in a location.
        The message will be added to each agent's pending events queue.
        """
        loc = self.get_location(location_id)
        if not loc:
            self.logger.warning(f"Broadcast failed: location {location_id} not found")
            return
        
        if not loc.agents_present:
            self.logger.info(f"Broadcast to {location_id}: no agents present")
            return
        
        recipient_count = 0
        for agent_name in loc.agents_present:
            if agent_name != exclude_agent:
                if agent_name not in self.pending_events:
                    self.pending_events[agent_name] = []
                self.pending_events[agent_name].append(message)
                self.logger.info(f"Event queued for {agent_name}: {message}")
                recipient_count += 1
        
        if recipient_count == 0:
            self.logger.info(f"Broadcast to {location_id}: only sender present, no recipients")

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
            "events": "None",
            "inventory": "Empty"
        }
        
        if not loc:
            return data
        
        data["location_name"] = loc.name
        data["location_description"] = loc.description
        
        # People present (excluding self)
        people = [p for p in loc.agents_present if p != agent_name]
        data["people"] = ", ".join(people) if people else "No one else"
        
        if loc.objects:
            objects_info = []
            for obj_id in loc.objects:
                obj = self.get_object(obj_id)
                if obj:
                    # SimAgent sees name, id, STATE, and description
                    base_info = f"  - {obj.name} (id: {obj.id}, state: {obj.state})\n    {obj.description}"
                    objects_info.append(base_info)
                    
            if objects_info:
                data["objects"] = "\n".join(objects_info)
        
        # Connected locations
        data["connections"] = ", ".join(loc.connected_to) if loc.connected_to else "None"
        
        # Recent external events (from other agents)
        pending_events = self.get_pending_events(agent_name)
        data["pending_events"] = pending_events  # Return raw list for agent to merge
            
        # Agent Inventory
        inventory_items = self.get_agent_inventory(agent_name)
        data["inventory"] = ", ".join(inventory_items) if inventory_items else "Empty"
            
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
                    connected = self.get_connected_locations(current_location_id)
                    connected_str = ", ".join(connected) if connected else "None"
                    result["message"] = f"Failed to move to '{target}'. It is not connected to your current location. Connected locations: {connected_str}"
            else:
                result["message"] = "Move action requires a target."

        elif action_type == "talk":
            if content:
                self.logger.info(f"{agent_name} says: '{content}'")
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
                    # Check if object is in current location or agent's inventory
                    if obj.location_id != current_location_id and obj.location_id != agent_name:
                        result["message"] = f"Object '{target}' is not in your current location."
                        return result
                    
                    # Check if WorldEngine is available
                    if not self.world_engine:
                        result["message"] = "Cannot perform complex interactions without WorldEngine."
                        return result
                    
                    # Always use WorldEngine for richer interaction handling
                    action_desc = content or ""
                    loc = self.get_location(current_location_id)
                    witnesses = loc.agents_present if loc else []
                    
                    # Fetch inventory internally for WorldEngine
                    current_inventory = self.get_agent_inventory(agent_name)
                    
                    result = self.world_engine.resolve_interaction(
                        agent_name=agent_name,
                        target_object=obj,
                        action_description=action_desc,
                        location=loc,
                        witnesses=witnesses,
                        world=self,
                        inventory=current_inventory
                    )

                    
                    # Auto-broadcast successful interactions to others in the room
                    if result.get("success") and loc:
                        broadcast_msg = f"{agent_name}: {result.get('message')}"
                        self.broadcast_to_location(
                            current_location_id,
                            broadcast_msg,
                            exclude_agent=agent_name # The actor already knows the result
                        )
                else:
                    result["message"] = f"Object '{target}' not found."
            else:
                result["message"] = "Interact action requires a target object."
            
        else:
            result["message"] = f"Unknown action type: {action_type}"
            
        return result

    def advance_time(self) -> None:
        """Advance simulation time by one tick."""
        self.sim_time += timedelta(minutes=TICK_DURATION_MINUTES)
    
    def get_time_str(self) -> str:
        """Get formatted time string for agent context."""
        return self.sim_time.strftime("%A, %I:%M %p")

    def set_agent_lock(self, agent_name: str, duration_minutes: int, reason: str, 
                       completion_message: str = None, pending_effects: List[Dict] = None) -> None:
        """Set a lock on an agent for a duration with optional deferred effects."""
        until_time = self.sim_time + timedelta(minutes=duration_minutes)
        
        self.agent_locks[agent_name] = {
            "until_time": until_time,
            "reason": reason,
            "completion_message": completion_message or f"Finished {reason}.",
            "pending_effects": pending_effects or []
        }
        self.logger.info(f"Agent {agent_name} locked until {until_time.strftime('%I:%M %p')}: {reason}")

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
        elif effect_type == "UpdateObject":
            self.update_object(**args)


