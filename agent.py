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

    async def decide(self, world_context: Dict[str, Any]) -> AgentDecision:
        """Async decision-making for the agent. Returns a typed AgentDecision."""
        system_prompt = self.get_system_prompt(TICK_DURATION_MINUTES)
        
        # Get new memories since last decision (external events + own actions)
        external_events = world_context.get("pending_events", [])
        own_memories = self.memory.short_term.copy()
        
        # Combine: external events first, then own memories
        all_memories = []
        if external_events:
            all_memories.extend([f"[Event] {e}" for e in external_events])
        all_memories.extend([f"[Memory] {m}" for m in own_memories if m])
        
        memory_str = "\n".join(f"- {m}" for m in all_memories) if all_memories else "Nothing notable"
        
        # Clear short-term memory after using it (next time will only have new memories)
        self.memory.short_term.clear()
        
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
            return AgentDecision.fallback("Decision failed - empty LLM response")
        
        # Strip markdown code blocks if present
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines[1:] if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            decision = AgentDecision.model_validate_json(content)
            # Validate action params match action_type
            decision.get_validated_action()
        except ValidationError as e:
            self.logger.error(f"Validation Error: {e}")
            decision = AgentDecision.fallback(f"Validation Error: {str(e)}")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON Parse Error: {e}")
            decision = AgentDecision.fallback(f"JSON Error: {content}")

        # Log decision
        self.logger.info(f"{self.name} decided: {decision.action_type} | {decision.action}")
        self.logger.info(f"{self.name} reasoning: {decision.reasoning}")
        
        # Save both user context and assistant response to history
        self.memory.add_message("user", new_user_message)
        self.memory.add_message("assistant", content)
        
        return decision

    def update_state(self, action_result: Dict[str, Any]) -> None:
        """Update agent's state and memory based on action result."""
        if "message" in action_result:
            self.memory.short_term.append(f"System: {action_result['message']}")
            self.logger.info(f"{self.name} memory updated: {action_result['message']}")
