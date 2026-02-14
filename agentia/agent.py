from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError
import logging
import json
import json
from .schemas import AgentDecision, Task, Plan, get_update_plan_tool_schema
from .prompts import AGENT_SYSTEM_PROMPT, AGENT_USER_TEMPLATE
from .utils import LLMClient
from .config import MAX_HISTORY_LENGTH, DEFAULT_AGENT_STATUS, TICK_DURATION_MINUTES


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
        self.daily_plan: List[Task] = []

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
        
        # Prepare tools
        tools = [get_update_plan_tool_schema()]

        # Decision Loop (to handle tool calls)
        final_decision = None
        
        while not final_decision:
            # Format current plan for context
            if self.daily_plan:
                plan_str = "\n".join([f"- [{t.status}] {t.id}: {t.description}" for t in self.daily_plan])
            else:
                plan_str = "No plan yet. (Use `update_plan` to create one)"

            # Build user message using template
            new_user_message = AGENT_USER_TEMPLATE.format(
                status=self.status,
                memory=memory_str,
                current_plan=plan_str,
                **world_context
            )
            
            # Build full context
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.memory.chat_history)
            messages.append({"role": "user", "content": new_user_message})

            # Request response (allow tools)
            try:
                response = await self.llm.async_chat_completion(
                    messages, 
                    tools=tools,
                    response_format={"type": "json_object"}
                )
            except Exception as e:
                self.logger.error(f"LLM Error: {e}")
                return AgentDecision.fallback("LLM Error")
            
            if not response:
                return AgentDecision.fallback("Empty Response")

            # Check for tool calls
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call.function.name == "update_plan":
                        try:
                            args = json.loads(tool_call.function.arguments)
                            new_plan_data = Plan(**args)
                            self.daily_plan = new_plan_data.tasks
                            
                            # Log and add to memory/history
                            msg = f"Updated daily plan: {len(self.daily_plan)} tasks."
                            self.logger.info(f"{self.name} {msg}")
                            self.memory.short_term.append(f"[Planning] {msg}")
                            
                            # Add assistant's tool call to history so it knows it called it
                            # Note: For simplicity in this loop, we might just append the result 
                            # to short term memory and let the next loop iteration re-build the context 
                            # with the updated plan. Open AI API usually requires the tool call and result 
                            # messages to be appended. Here we are simplifying by updating state and 
                            # re-prompting with new state in `current_plan` field.
                            
                        except Exception as e:
                            self.logger.error(f"Plan Update Error: {e}")
                
                # After tool handling, continue loop to get final world action
                # We don't append tool messages to self.memory.chat_history here to keep it clean,
                # but we do rely on the re-generated prompt showing the NEW plan.
                continue

            # No tool calls -> This is the final content (World Action)
            content = response.content.strip()
            # Strip markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines[1:] if not l.strip().startswith("```")]
                content = "\n".join(lines)

            try:
                decision = AgentDecision.model_validate_json(content)
                decision.get_validated_action()
                final_decision = decision
            except (ValidationError, json.JSONDecodeError) as e:
                self.logger.error(f"Decision Parse Error: {e}")
                return AgentDecision.fallback(f"Parse Error: {e}")

        # Log decision
        self.logger.info(f"{self.name} decided: {final_decision.action_type} | {final_decision.action}")
        self.logger.info(f"{self.name} reasoning: {final_decision.reasoning}")
        
        # Save both user context and assistant response to history
        self.memory.add_message("user", new_user_message)
        self.memory.add_message("assistant", content)
        
        return final_decision

    def update_state(self, action_result: Dict[str, Any]) -> None:
        """Update agent's state and memory based on action result."""
        if "message" in action_result:
            self.memory.short_term.append(f"System: {action_result['message']}")
            self.logger.info(f"{self.name} memory updated: {action_result['message']}")
