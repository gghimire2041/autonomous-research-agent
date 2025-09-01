"""Tests for agent tools."""

import pytest
from unittest.mock import patch, Mock

from app.tools.web_search import WebSearchTool
from app.tools.calculator import CalculatorTool
from app.tools.base import ToolResult


@pytest.mark.asyncio
async def test_calculator_tool():
    """Test calculator tool functionality."""
    calculator = CalculatorTool()
    
    # Test simple calculation
    result = await calculator.execute(expression="2 + 3")
    
    assert result.success is True
    assert result.result == "5"
    
    # Test complex calculation
    result = await calculator.execute(expression="sqrt(16) * 2")
    
    assert result.success is True
    assert result.result == "8.0"
    
    # Test invalid expression
    result = await calculator.execute(expression="import os")
    
    assert result.success is False
    assert "Invalid expression" in result.error


@pytest.mark.asyncio
async def test_web_search_tool():
    """Test web search tool functionality."""
    search_tool = WebSearchTool()
    
    # Mock DuckDuckGo search results
    mock_results = [
        {
            'title': 'Test Result 1',
            'body': 'Test description 1',
            'href': 'https://example.com/1'
        },
        {
            'title': 'Test Result 2',
            'body': 'Test description 2',
            'href': 'https://example.com/2'
        }
    ]
    
    with patch.object(search_tool, '_search_sync', return_value=mock_results):
        result = await search_tool.execute(query="test query")
        
        assert result.success is True
        assert "Test Result 1" in result.result
        assert "Test Result 2" in result.result
        assert result.metadata["count"] == 2


@pytest.mark.asyncio
async def test_tool_validation():
    """Test tool input validation."""
    calculator = CalculatorTool()
    
    # Test with valid input
    validated = calculator.validate_input(expression="2 + 2")
    assert validated == {"expression": "2 + 2"}
    
    # Tool should handle validation gracefully
    result = await calculator.execute(expression="")
    assert result.success is False
