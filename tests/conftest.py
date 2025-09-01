"""Pytest configuration and fixtures."""

import asyncio
import tempfile
import pytest
from unittest.mock import Mock, AsyncMock

from app.core.agent import ResearchAgent
from app.core.memory import MemoryStore
from app.llm.adapters import LLMAdapter
from app.tools.base import Tool, ToolResult
from app.utils.config import Settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db():
    """Temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        yield f"sqlite:///{f.name}"


@pytest.fixture
def test_settings(temp_db):
    """Test settings with overrides."""
    settings = Settings(
        DATABASE_URL=temp_db,
        SAFE_MODE=True,
        MAX_STEPS=3,
        TIMEOUT_SECONDS=10,
        DEBUG=True,
        LOG_LEVEL="DEBUG"
    )
    return settings


@pytest.fixture
def mock_llm():
    """Mock LLM adapter."""
    llm = Mock(spec=LLMAdapter)
    llm.generate = AsyncMock(return_value="Test response")
    return llm


@pytest.fixture
def mock_tool():
    """Mock tool for testing."""
    tool = Mock(spec=Tool)
    tool.name = "test_tool"
    tool.description = "Test tool for testing"
    tool.execute = AsyncMock(return_value=ToolResult(
        success=True,
        result="Test result"
    ))
    return tool


@pytest.fixture
def memory_store(test_settings):
    """Memory store with test database."""
    # Monkey patch settings
    import app.core.memory
    original_settings = app.core.memory.settings
    app.core.memory.settings = test_settings
    
    store = MemoryStore()
    
    yield store
    
    # Restore original settings
    app.core.memory.settings = original_settings


@pytest.fixture
def research_agent(mock_llm, mock_tool, memory_store):
    """Research agent with mocked dependencies."""
    agent = ResearchAgent()
    agent.memory = memory_store
    agent.llm = mock_llm
    agent.tools = {"test_tool": mock_tool}
    return agent
