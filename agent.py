from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError
import logging
import json
from utils import LLMClient
from config import TICK_DURATION_MINUTES, DEFAULT_AGENT_STATUS
from schemas import AgentDecision
from prompts import AGENT_SYSTEM_PROMPT, AGENT_USER_TEMPLATE


class AgentMemory(BaseModel):
    """Memory storage for SimAgent."""
    short_term: List[str] = Field(default_factory=list)
    long_term_summary: str = ""
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

    def get_recent_memories(self, limit: int = 5) -> str:
        """Get the most recent short-term memories as a formatted string."""
        return "\n".join(self.short_term[-limit:])

    def add_message(self, role: str, content: str) -> None:
        """Add a message to chat history."""
        self.chat_history.append({"role": role, "content": content})


class SimAgent:
    """A simulation agent capable of making decisions using LLM."""
    def __init__(self, 
                 name: str, 
                 age: int, 
                 occupation: str, 
                 personality: str, 
                 background: str,
                 llm_client: LLMClient,
                 initial_goal: str = "Explore the surroundings."):
        self.name = name
        self.age = age
        self.occupation = occupation
        self.personality = personality
        self.background = background
        self.current_goal = initial_goal
        self.memory = AgentMemory()
        self.llm = llm_client
        self.logger = logging.getLogger(f"Agentia.Agent.{self.name}")
        self.status: Dict[str, Any] = DEFAULT_AGENT_STATUS.copy()

    def get_system_prompt(self, tick_duration: int) -> str:
        return AGENT_SYSTEM_PROMPT.format(
            name=self.name,
            age=self.age,
            occupation=self.occupation,
            personality=self.personality,
            background=self.background,
            current_goal=self.current_goal,
            tick_duration=tick_duration
        )

    async def decide(self, world_context: Dict[str, Any]) -> Dict[str, Any]:
        """Async decision-making for the agent using JSON output."""
        system_prompt = self.get_system_prompt(TICK_DURATION_MINUTES)
        
        # Merge own short-term memory with external events from world
        own_memories_str = self.memory.get_recent_memories(limit=3)
        own_memories = own_memories_str.split("\n") if own_memories_str else []
        external_events = world_context.get("pending_events", [])
        
        # Combine: external events first, then own memories
        all_memories = []
        if external_events:
            all_memories.extend([f"[Event] {e}" for e in external_events])
        all_memories.extend([f"[Memory] {m}" for m in own_memories if m])
        
        memory_str = "\n".join(f"- {m}" for m in all_memories) if all_memories else "Nothing notable"
        
        # Build user message using template
        new_user_message = AGENT_USER_TEMPLATE.format(
            status=self.status,
            memory=memory_str,
            **world_context
        )
        
        # Build full context
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.memory.chat_history)
        messages.append({"role": "user", "content": new_user_message})

        # Request JSON response
        try:
            response = await self.llm.async_chat_completion(
                messages, 
                response_format={"type": "json_object"}
            )
        except Exception:
            response = await self.llm.async_chat_completion(messages)
        
        if not response or not response.content:
            self.logger.error(f"{self.name} failed to decide (empty response).")
            return {"action_type": "wait", "reasoning": "Decision failed"}
        
        # Strip markdown code blocks if present
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines[1:] if not l.strip().startswith("```")]
            content = "\n".join(lines)

        result = {"action_type": "wait", "reasoning": "Parsing failed", "target": None, "content": None}
        
        try:
            # Use AgentDecision for unified parsing and validation
            decision = AgentDecision.model_validate_json(content)
            
            result["reasoning"] = decision.reasoning
            result["action_type"] = decision.action_type
            
            # Helper to handle action params whether it's a dict or Pydantic model
            if isinstance(decision.action, BaseModel):
                action_params = decision.action.model_dump()
            else:
                action_params = decision.action
            
            # Extract common fields based on action type
            if decision.action_type == "move":
                result["target"] = action_params.get("location_id")
            elif decision.action_type == "talk":
                result["content"] = action_params.get("message")
                result["target"] = action_params.get("target_agent")
            elif decision.action_type == "interact":
                result["target"] = action_params.get("object_id")
                result["content"] = action_params.get("action")
            elif decision.action_type == "wait":
                result["content"] = action_params.get("reason")

            # Validate action params with typed model
            validated_action = decision.get_validated_action()
            if not validated_action:
                self.logger.warning(f"Could not validate action params for {decision.action_type}")

        except ValidationError as e:
            self.logger.error(f"Validation Error: {e}")
            result["reasoning"] = f"Validation Error: {str(e)[:100]}"
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON Parse Error: {e}")
            result["reasoning"] = f"JSON Error: {content[:100]}..."

        # Build detailed action log with parameters
        action_details = f"{result['action_type']}"
        if result.get("target"):
            action_details += f"(target={result['target']}"
            if result.get("content"):
                content_preview = result['content'][:50] if len(str(result.get('content', ''))) > 50 else result.get('content', '')
                action_details += f", content={content_preview}"
            action_details += ")"
        elif result.get("content"):
            content_preview = result['content'][:50] if len(str(result.get('content', ''))) > 50 else result.get('content', '')
            action_details += f"(content={content_preview})"
        
        self.logger.info(f"{self.name} decided: {action_details}")
        self.logger.info(f"{self.name} reasoning: {result['reasoning']}...")
        
        # Save both user context and assistant response to history
        self.memory.add_message("user", new_user_message)
        self.memory.add_message("assistant", content)
        
        return result

    def update_state(self, action_result: Dict[str, Any]) -> None:
        """Update agent's state and memory based on action result."""
        if "message" in action_result:
            self.memory.short_term.append(f"System: {action_result['message']}")
            self.logger.info(f"{self.name} memory updated: {action_result['message']}")
        
        # Update stress based on action success
        if not action_result.get("success", True):
            self.status["stress"] = "medium"

