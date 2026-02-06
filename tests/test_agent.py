"""Unit tests for SimAgent using mocked LLM with JSON output mode."""
import pytest
import json
from unittest.mock import MagicMock, AsyncMock
from agent import SimAgent, AgentMemory
from utils import LLMClient


class MockMessage:
    """Mock object for OpenAI chat completion response with JSON content."""
    def __init__(self, content: str):
        self.content = content


def make_world_context(**kwargs) -> dict:
    """Helper to create a valid world context dict."""
    defaults = {
        "time": "Monday, 08:00 AM",
        "location_name": "Test Room",
        "location_description": "A simple test room",
        "people": "No one else",
        "objects": "None",
        "connections": "['hallway']",
        "events": "None",
        "inventory": "Empty"
    }
    defaults.update(kwargs)
    return defaults


class TestAgentMemory:
    """Test AgentMemory functionality."""
    
    def test_initial_state(self):
        """Test memory initializes correctly."""
        memory = AgentMemory()
        assert memory.short_term == []
        assert memory.long_term_summary == ""
        assert memory.chat_history == []

    def test_add_short_term_memory(self):
        """Test adding to short-term memory."""
        memory = AgentMemory()
        memory.short_term.append("Event 1")
        memory.short_term.append("Event 2")
        assert len(memory.short_term) == 2

    def test_get_recent_memories(self):
        """Test getting recent memories with limit."""
        memory = AgentMemory()
        for i in range(10):
            memory.short_term.append(f"Event {i}")
        
        recent = memory.get_recent_memories(limit=3)
        assert "Event 7" in recent
        assert "Event 8" in recent
        assert "Event 9" in recent
        assert "Event 0" not in recent

    def test_add_message(self):
        """Test adding a single message to history."""
        memory = AgentMemory()
        memory.add_message("user", "Test message")
        
        assert len(memory.chat_history) == 1
        assert memory.chat_history[0]["role"] == "user"
        assert memory.chat_history[0]["content"] == "Test message"


class TestSimAgent:
    """Test SimAgent functionality with JSON output mode."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        return MagicMock(spec=LLMClient)
    
    @pytest.fixture
    def agent(self, mock_llm):
        """Create a test agent."""
        return SimAgent(
            name="TestAgent",
            age=30,
            occupation="Tester",
            personality="Analytical",
            background="Created in a lab environment in 2025. No prior work history.",
            llm_client=mock_llm,
            initial_goal="Test the system"
        )
    
    @pytest.fixture
    def world_context(self):
        """Create a standard world context for tests."""
        return make_world_context()

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.name == "TestAgent"
        assert agent.age == 30
        assert agent.occupation == "Tester"
        assert agent.current_goal == "Test the system"
        assert agent.background == "Created in a lab environment in 2025. No prior work history."
        assert agent.status["fatigue"] == "low"

    def test_get_system_prompt(self, agent):
        """Test system prompt generation includes agent details."""
        prompt = agent.get_system_prompt(tick_duration=10)
        
        assert "TestAgent" in prompt
        assert "30" in prompt
        assert "Tester" in prompt
        assert "Analytical" in prompt
        assert "Background: Created in a lab environment in 2025. No prior work history." in prompt
        assert "Test the system" in prompt

    @pytest.mark.asyncio
    async def test_decide_move_action(self, agent, mock_llm, world_context):
        """Test agent deciding to move using JSON output."""
        json_response = json.dumps({
            "reasoning": "I should explore the kitchen to find coffee.",
            "action_type": "move",
            "action": {"location_id": "kitchen_01"}
        })
        mock_llm.async_chat_completion = AsyncMock(return_value=MockMessage(json_response))
        
        decision = await agent.decide(world_context=world_context)
        
        assert decision["action_type"] == "move"
        assert decision["target"] == "kitchen_01"
        assert "kitchen" in decision["reasoning"].lower() or "coffee" in decision["reasoning"].lower()

    @pytest.mark.asyncio
    async def test_decide_talk_action(self, agent, mock_llm):
        """Test agent deciding to talk."""
        world_context = make_world_context(people="['Alice']")
        json_response = json.dumps({
            "reasoning": "I should greet Alice.",
            "action_type": "talk",
            "action": {"message": "Hello Alice!", "target_agent": "Alice"}
        })
        mock_llm.async_chat_completion = AsyncMock(return_value=MockMessage(json_response))
        
        decision = await agent.decide(world_context=world_context)
        
        assert decision["action_type"] == "talk"
        assert decision["content"] == "Hello Alice!"
        assert decision["target"] == "Alice"

    @pytest.mark.asyncio
    async def test_decide_interact_action(self, agent, mock_llm):
        """Test agent deciding to interact with object."""
        world_context = make_world_context(objects="  - Coffee Machine (id: coffee_machine, state: working)")
        json_response = json.dumps({
            "reasoning": "I need coffee, let me use the machine.",
            "action_type": "interact",
            "action": {"object_id": "coffee_machine", "action": "make coffee"}
        })
        mock_llm.async_chat_completion = AsyncMock(return_value=MockMessage(json_response))
        
        decision = await agent.decide(world_context=world_context)
        
        assert decision["action_type"] == "interact"
        assert decision["target"] == "coffee_machine"
        assert decision["content"] == "make coffee"

    @pytest.mark.asyncio
    async def test_decide_wait_action(self, agent, mock_llm, world_context):
        """Test agent deciding to wait."""
        json_response = json.dumps({
            "reasoning": "Nothing urgent, I'll observe.",
            "action_type": "wait",
            "action": {"reason": "observing surroundings"}
        })
        mock_llm.async_chat_completion = AsyncMock(return_value=MockMessage(json_response))
        
        decision = await agent.decide(world_context=world_context)
        
        assert decision["action_type"] == "wait"
        assert "observing" in decision["content"]



    @pytest.mark.asyncio
    async def test_decide_empty_context(self, agent, mock_llm):
        """Test agent handles minimal context gracefully."""
        json_response = json.dumps({
            "reasoning": "No context available, waiting.",
            "action_type": "wait",
            "action": {"reason": "no information"}
        })
        mock_llm.async_chat_completion = AsyncMock(return_value=MockMessage(json_response))
        
        minimal_context = make_world_context()
        decision = await agent.decide(world_context=minimal_context)
        
        assert decision["action_type"] == "wait"

    @pytest.mark.asyncio
    async def test_decide_llm_failure(self, agent, mock_llm, world_context):
        """Test graceful handling of LLM failure."""
        mock_llm.async_chat_completion = AsyncMock(return_value=None)
        
        decision = await agent.decide(world_context=world_context)
        
        assert decision["action_type"] == "wait"
        assert "failed" in decision["reasoning"].lower()

    @pytest.mark.asyncio
    async def test_decide_invalid_json(self, agent, mock_llm, world_context):
        """Test handling of invalid JSON response."""
        mock_llm.async_chat_completion = AsyncMock(return_value=MockMessage("not valid json"))
        
        decision = await agent.decide(world_context=world_context)
        
        assert decision["action_type"] == "wait"
        assert "error" in decision["reasoning"].lower() or "json" in decision["reasoning"].lower()

    @pytest.mark.asyncio
    async def test_decide_markdown_code_block_stripped(self, agent, mock_llm, world_context):
        """Test that markdown code blocks around JSON are properly stripped."""
        json_content = json.dumps({
            "reasoning": "Testing markdown stripping.",
            "action_type": "wait",
            "action": {"reason": "testing"}
        })
        # Wrap in markdown code block like some LLMs do
        markdown_response = f"```json\n{json_content}\n```"
        mock_llm.async_chat_completion = AsyncMock(return_value=MockMessage(markdown_response))
        
        decision = await agent.decide(world_context=world_context)
        
        assert decision["action_type"] == "wait"
        assert "markdown" in decision["reasoning"].lower() or "testing" in decision["reasoning"].lower()

    @pytest.mark.asyncio
    async def test_chat_history_updated(self, agent, mock_llm, world_context):
        """Test that chat history is updated after decision."""
        json_response = json.dumps({
            "reasoning": "Just waiting.",
            "action_type": "wait",
            "action": {"reason": "waiting"}
        })
        mock_llm.async_chat_completion = AsyncMock(return_value=MockMessage(json_response))
        
        assert len(agent.memory.chat_history) == 0
        
        await agent.decide(world_context=world_context)
        
        # Should have user message and assistant response
        assert len(agent.memory.chat_history) == 2
        assert agent.memory.chat_history[0]["role"] == "user"
        assert agent.memory.chat_history[1]["role"] == "assistant"

    def test_update_state_success(self, agent):
        """Test state update on successful action."""
        agent.update_state({"success": True, "message": "Action completed."})
        
        assert "Action completed." in agent.memory.short_term[-1]
        assert agent.status["stress"] == "low"

    def test_update_state_failure(self, agent):
        """Test state update on failed action increases stress."""
        agent.update_state({"success": False, "message": "Action failed."})
        
        assert agent.status["stress"] == "medium"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
