"""Tests for the research agent core functionality."""

import pytest
from datetime import datetime

from app.core.agent import ResearchAgent, TaskStatus, AgentStep, TaskResult
from app.core.planner import PlannerType


@pytest.mark.asyncio
async def test_start_task(research_agent):
    """Test starting a new task."""
    goal = "Test research goal"
    task_id = await research_agent.start_task(goal)
    
    assert task_id is not None
    assert len(task_id) > 0
    
    # Task should be in running_tasks
    assert task_id in research_agent.running_tasks


@pytest.mark.asyncio
async def test_get_task_status(research_agent):
    """Test getting task status."""
    # Create a test task result
    task_result = TaskResult(
        task_id="test-task-id",
        goal="Test goal",
        status=TaskStatus.COMPLETED,
        result="Test result",
        total_duration_ms=1000,
        created_at=datetime.utcnow()
    )
    
    await research_agent.memory.store_task_result(task_result)
    
    # Get task status
    status = await research_agent.get_task_status("test-task-id")
    
    assert status is not None
    assert status.task_id == "test-task-id"
    assert status.goal == "Test goal"
    assert status.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_cancel_task(research_agent):
    """Test cancelling a running task."""
    goal = "Test research goal"
    task_id = await research_agent.start_task(goal)
    
    # Cancel the task
    success = await research_agent.cancel_task(task_id)
    
    assert success is True
    assert task_id not in research_agent.running_tasks


@pytest.mark.asyncio
async def test_task_execution_flow(research_agent, mock_llm):
    """Test complete task execution flow."""
    # Mock LLM responses for planning
    mock_llm.generate.side_effect = [
        "Thought: I need to use a tool\nAction: Test action\nTool: test_tool\nInput: {\"query\": \"test\"}",
        "yes",  # Task completion check
        "Final result based on execution"  # Final result
    ]
    
    goal = "Simple test goal"
    task_id = await research_agent.start_task(goal, max_steps=1)
    
    # Wait a bit for execution to start
    import asyncio
    await asyncio.sleep(0.1)
    
    # Get final status
    status = await research_agent.get_task_status(task_id)
    
    # Verify task completed
    assert status is not None
    assert status.status in [TaskStatus.COMPLETED, TaskStatus.RUNNING]  # May still be running

