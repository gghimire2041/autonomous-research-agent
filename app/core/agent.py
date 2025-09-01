"""Main autonomous research agent implementation."""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel

from app.core.guardrails import Guardrails
from app.core.memory import MemoryStore
from app.core.planner import PlannerType, get_planner
from app.llm.adapters import get_llm_adapter
from app.tools.base import Tool, ToolResult
from app.tools.registry import get_available_tools
from app.utils.config import get_settings
from app.utils.observability import get_tracer
from app.utils.security import redact_pii

logger = structlog.get_logger()
tracer = get_tracer(__name__)
settings = get_settings()


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStep(BaseModel):
    """Single agent execution step."""
    step_id: str
    step_number: int
    thought: str
    action: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: datetime
    duration_ms: Optional[int] = None


class TaskResult(BaseModel):
    """Complete task execution result."""
    task_id: str
    goal: str
    status: TaskStatus
    result: Optional[str] = None
    error: Optional[str] = None
    steps: List[AgentStep] = []
    total_duration_ms: int
    created_at: datetime
    completed_at: Optional[datetime] = None


class ResearchAgent:
    """Main autonomous research agent."""
    
    def __init__(self):
        self.memory = MemoryStore()
        self.guardrails = Guardrails()
        self.llm = get_llm_adapter()
        self.tools = get_available_tools()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
    async def start_task(
        self, 
        goal: str, 
        planner_type: PlannerType = PlannerType.REACTIVE,
        max_steps: Optional[int] = None
    ) -> str:
        """Start a new research task."""
        task_id = str(uuid.uuid4())
        max_steps = max_steps or settings.MAX_STEPS
        
        logger.info("Starting new task", task_id=task_id, goal=goal)
        
        # Store initial task state
        task_result = TaskResult(
            task_id=task_id,
            goal=goal,
            status=TaskStatus.PENDING,
            total_duration_ms=0,
            created_at=datetime.utcnow()
        )
        await self.memory.store_task_result(task_result)
        
        # Start background execution
        task_coro = self._execute_task(task_id, goal, planner_type, max_steps)
        self.running_tasks[task_id] = asyncio.create_task(task_coro)
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get current task status."""
        return await self.memory.get_task_result(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.cancel()
            
            # Update status in memory
            task_result = await self.memory.get_task_result(task_id)
            if task_result:
                task_result.status = TaskStatus.CANCELLED
                await self.memory.store_task_result(task_result)
            
            del self.running_tasks[task_id]
            logger.info("Task cancelled", task_id=task_id)
            return True
        return False
    
    async def _execute_task(
        self, 
        task_id: str, 
        goal: str, 
        planner_type: PlannerType,
        max_steps: int
    ) -> None:
        """Execute a research task with sense-plan-act loop."""
        start_time = datetime.utcnow()
        steps = []
        
        try:
            with tracer.start_as_current_span("execute_task") as span:
                span.set_attribute("task_id", task_id)
                span.set_attribute("goal", goal)
                
                # Update status to running
                task_result = await self.memory.get_task_result(task_id)
                task_result.status = TaskStatus.RUNNING
                await self.memory.store_task_result(task_result)
                
                # Get planner instance
                planner = get_planner(planner_type, self.llm)
                
                # Initialize scratchpad with goal
                scratchpad = f"GOAL: {goal}\n\n"
                
                for step_num in range(1, max_steps + 1):
                    step_start = datetime.utcnow()
                    step_id = f"{task_id}-{step_num}"
                    
                    logger.info("Executing step", step_id=step_id, step_num=step_num)
                    
                    # Plan next action
                    action_plan = await planner.plan_next_action(
                        goal, scratchpad, self.tools
                    )
                    
                    step = AgentStep(
                        step_id=step_id,
                        step_number=step_num,
                        thought=action_plan.thought,
                        action=action_plan.action,
                        tool_name=action_plan.tool_name,
                        tool_input=action_plan.tool_input,
                        timestamp=step_start
                    )
                    
                    # Check guardrails
                    if not await self.guardrails.is_action_allowed(action_plan):
                        step.observation = "Action blocked by safety guardrails"
                        logger.warning("Action blocked", step_id=step_id, action=action_plan.action)
                    else:
                        # Execute tool if specified
                        if action_plan.tool_name and action_plan.tool_input:
                            tool_result = await self._execute_tool(
                                action_plan.tool_name, 
                                action_plan.tool_input
                            )
                            step.observation = tool_result.result
                            
                            if tool_result.error:
                                step.observation = f"Error: {tool_result.error}"
                    
                    # Calculate step duration
                    step_end = datetime.utcnow()
                    step.duration_ms = int((step_end - step_start).total_seconds() * 1000)
                    steps.append(step)
                    
                    # Update scratchpad (redact PII)
                    step_summary = f"Step {step_num}:\nThought: {step.thought}\nAction: {step.action}\nObservation: {step.observation}\n\n"
                    scratchpad += redact_pii(step_summary)
                    
                    # Check if task is complete
                    is_complete = await planner.is_task_complete(goal, scratchpad)
                    if is_complete:
                        break
                
                # Generate final result
                final_result = await planner.generate_final_result(goal, scratchpad)
                
                # Update task result
                end_time = datetime.utcnow()
                duration_ms = int((end_time - start_time).total_seconds() * 1000)
                
                task_result.status = TaskStatus.COMPLETED
                task_result.result = final_result
                task_result.steps = steps
                task_result.total_duration_ms = duration_ms
                task_result.completed_at = end_time
                
        except asyncio.CancelledError:
            task_result.status = TaskStatus.CANCELLED
            logger.info("Task execution cancelled", task_id=task_id)
        except Exception as e:
            task_result.status = TaskStatus.FAILED
            task_result.error = str(e)
            logger.error("Task execution failed", task_id=task_id, error=str(e))
        finally:
            await self.memory.store_task_result(task_result)
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> ToolResult:
        """Execute a tool with proper error handling and tracing."""
        with tracer.start_as_current_span("execute_tool") as span:
            span.set_attribute("tool_name", tool_name)
            span.set_attribute("tool_input", str(tool_input))
            
            if tool_name not in self.tools:
                return ToolResult(
                    success=False,
                    result="",
                    error=f"Unknown tool: {tool_name}"
                )
            
            tool = self.tools[tool_name]
            
            try:
                # Apply rate limiting
                await self.guardrails.check_rate_limit(tool_name)
                
                # Execute tool
                result = await tool.execute(**tool_input)
                
                logger.info(
                    "Tool executed successfully",
                    tool_name=tool_name,
                    success=result.success
                )
                
                return result
                
            except Exception as e:
                logger.error(
                    "Tool execution failed",
                    tool_name=tool_name,
                    error=str(e)
                )
                return ToolResult(
                    success=False,
                    result="",
                    error=str(e)
                )
