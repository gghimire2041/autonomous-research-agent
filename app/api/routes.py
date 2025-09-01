"""FastAPI routes for the research agent."""

from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.core.agent import ResearchAgent, TaskStatus
from app.core.planner import PlannerType
from app.utils.observability import task_counter, active_tasks

logger = structlog.get_logger()
router = APIRouter()

# Global agent instance
agent = ResearchAgent()


class TaskRequest(BaseModel):
    """Request to start a new task."""
    goal: str
    planner_type: PlannerType = PlannerType.REACTIVE
    max_steps: Optional[int] = None


class TaskResponse(BaseModel):
    """Response with task ID."""
    task_id: str
    status: str


@router.post("/task", response_model=TaskResponse)
async def start_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Start a new research task."""
    logger.info("Received task request", goal=request.goal, planner=request.planner_type)
    
    try:
        task_id = await agent.start_task(
            goal=request.goal,
            planner_type=request.planner_type,
            max_steps=request.max_steps
        )
        
        # Update metrics
        task_counter.labels(status='started').inc()
        active_tasks.inc()
        
        return TaskResponse(task_id=task_id, status="started")
        
    except Exception as e:
        logger.error("Failed to start task", error=str(e))
        task_counter.labels(status='failed_to_start').inc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get task status and results."""
    try:
        # Validate UUID format
        UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")
    
    result = await agent.get_task_status(task_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return result.dict()


@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task."""
    try:
        # Validate UUID format
        UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")
    
    success = await agent.cancel_task(task_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or already completed")
    
    # Update metrics
    task_counter.labels(status='cancelled').inc()
    active_tasks.dec()
    
    return {"status": "cancelled"}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_tasks": len(agent.running_tasks),
        "version": "0.1.0"
    }

