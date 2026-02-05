from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
import logging
import json
from utils import LLMClient
from config import TICK_DURATION_MINUTES
from schemas import AgentDecision
from prompts import AGENT_SYSTEM_PROMPT, AGENT_USER_TEMPLATE


class AgentMemory(BaseModel):
    short_term: List[str] = Field(default_factory=list)
    long_term_summary: str = ""
    # History of simple dictionaries (user, assistant, system)
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

    def get_recent_memories(self, limit: int = 5) -> str:
        return "\n".join(self.short_term[-limit:])

    def add_interaction(self, user_content: str, assistant_content: str):
        # Deprecated: usage for string-only context
        self.chat_history.append({"role": "user", "content": user_content})
        self.chat_history.append({"role": "assistant", "content": assistant_content})

    
    def add_message(self, role: str, content: str):
        """Add a simplified message to history."""
        self.chat_history.append({"role": role, "content": content})


class SimAgent:
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
        self.busy_until = 0 # Timestamp or tick count
        
        self.memory = AgentMemory()
        self.llm = llm_client
        self.logger = logging.getLogger(f"Agentia.Agent.{self.name}")
        
        # self.inventory removed - stateless design
        self.status: Dict[str, Any] = {"fatigue": "low", "stress": "low"}

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

    async def decide(self, current_tick: int, world_context: Dict[str, Any]) -> Dict[str, Any]:
        """Async decision-making for the agent using JSON output."""
        if current_tick < self.busy_until:
             self.logger.info(f"{self.name} is busy.")
             return {"action_type": "wait", "reasoning": "Busy executing previous action"}
        
        system_prompt = self.get_system_prompt(TICK_DURATION_MINUTES)
        
        # Build user message using template
        new_user_message = AGENT_USER_TEMPLATE.format(
            status=self.status,
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

    def update_state(self, action_result: Dict[str, Any]):
        """
        Updates the agent's state and memory based on the result of an action.
        """
        if "message" in action_result:
            self.memory.short_term.append(f"System: {action_result['message']}")
            self.logger.info(f"{self.name} memory updated: {action_result['message']}")
        
        # No need for 'tool' messages anymore in JSON mode.
        # We can simulate the continuity by summarizing the outcome in the next user prompt
        # or appending a system message now.
        pass
        
        # Determine if stress/fatigue changed
        if action_result.get("success", True):
             # Placeholder for complex state update
             pass
        else:
             self.status["stress"] = "medium" # Failed actions cause stress

