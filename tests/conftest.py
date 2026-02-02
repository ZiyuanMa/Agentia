"""
Shared pytest fixtures for Simworld tests.
"""
import pytest
import sys
import os
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from world import World
from schemas import WorldObject, Location
from agent import SimAgent
from env_agent import EnvironmentAgent
from utils import LLMClient


# =============================================================================
# Mock Classes
# =============================================================================

class MockToolCall:
    """Mock object for OpenAI tool call response."""
    def __init__(self, name: str, arguments: str):
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = arguments


class MockMessage:
    """Mock object for OpenAI chat completion response."""
    def __init__(self, content: str = None, tool_calls: list = None):
        self.content = content
        self.tool_calls = tool_calls


# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    return MagicMock(spec=LLMClient)


@pytest.fixture
def simple_world():
    """Create a simple world with two connected rooms."""
    config = {
        "locations": [
            {
                "id": "room_a",
                "name": "Room A",
                "description": "A test room",
                "connected_to": ["room_b"]
            },
            {
                "id": "room_b",
                "name": "Room B",
                "description": "Another test room",
                "connected_to": ["room_a"]
            }
        ],
        "objects": [
            {
                "id": "test_object",
                "name": "Test Object",
                "location": "room_a",
                "state": "normal",
                "description": "A test object",
                "properties": ["portable", "interactive"]
            }
        ]
    }
    return World(config)


@pytest.fixture
def test_agent(mock_llm):
    """Create a test agent with mocked LLM."""
    return SimAgent(
        name="TestBot",
        age=30,
        occupation="Tester",
        personality="Curious and methodical",
        llm_client=mock_llm,
        initial_goal="Test all the things"
    )


@pytest.fixture
def env_agent(mock_llm):
    """Create a test environment agent."""
    return EnvironmentAgent(mock_llm)
