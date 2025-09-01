"""Memory store implementation using SQLite."""

import json
from datetime import datetime
from typing import Dict, List, Optional

import structlog
from sqlalchemy import Column, String, DateTime, Text, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.agent import TaskResult
from app.utils.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

Base = declarative_base()


class TaskRecord(Base):
    """SQLite table for storing task results."""
    __tablename__ = "tasks"
    
    task_id = Column(String, primary_key=True)
    goal = Column(Text, nullable=False)
    status = Column(String, nullable=False)
    result = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    steps = Column(Text, nullable=True)  # JSON serialized
    total_duration_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)


class MemoryStore:
    """SQLite-based memory store for agent state."""
    
    def __init__(self):
        self.engine = create_engine(settings.DATABASE_URL, echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    async def store_task_result(self, task_result: TaskResult) -> None:
        """Store or update a task result."""
        try:
            # Check if record exists
            existing = self.session.query(TaskRecord).filter(
                TaskRecord.task_id == task_result.task_id
            ).first()
            
            if existing:
                # Update existing record
                existing.status = task_result.status.value
                existing.result = task_result.result
                existing.error = task_result.error
                existing.steps = json.dumps([step.dict() for step in task_result.steps])
                existing.total_duration_ms = task_result.total_duration_ms
                existing.completed_at = task_result.completed_at
            else:
                # Create new record
                record = TaskRecord(
                    task_id=task_result.task_id,
                    goal=task_result.goal,
                    status=task_result.status.value,
                    result=task_result.result,
                    error=task_result.error,
                    steps=json.dumps([step.dict() for step in task_result.steps]),
                    total_duration_ms=task_result.total_duration_ms,
                    created_at=task_result.created_at,
                    completed_at=task_result.completed_at
                )
                self.session.add(record)
            
            self.session.commit()
            logger.debug("Task result stored", task_id=task_result.task_id)
            
        except Exception as e:
            self.session.rollback()
            logger.error("Failed to store task result", task_id=task_result.task_id, error=str(e))
            raise
    
    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Retrieve a task result by ID."""
        try:
            record = self.session.query(TaskRecord).filter(
                TaskRecord.task_id == task_id
            ).first()
            
            if not record:
                return None
            
            # Parse steps from JSON
            steps = []
            if record.steps:
                steps_data = json.loads(record.steps)
                from app.core.agent import AgentStep
                steps = [AgentStep(**step_data) for step_data in steps_data]
            
            from app.core.agent import TaskStatus
            return TaskResult(
                task_id=record.task_id,
                goal=record.goal,
                status=TaskStatus(record.status),
                result=record.result,
                error=record.error,
                steps=steps,
                total_duration_ms=record.total_duration_ms,
                created_at=record.created_at,
                completed_at=record.completed_at
            )
            
        except Exception as e:
            logger.error("Failed to get task result", task_id=task_id, error=str(e))
            raise
    
    async def get_recent_tasks(self, limit: int = 10) -> List[TaskResult]:
        """Get recent tasks."""
        try:
            records = self.session.query(TaskRecord).order_by(
                TaskRecord.created_at.desc()
            ).limit(limit).all()
            
            results = []
            for record in records:
                steps = []
                if record.steps:
                    steps_data = json.loads(record.steps)
                    from app.core.agent import AgentStep
                    steps = [AgentStep(**step_data) for step_data in steps_data]
                
                from app.core.agent import TaskStatus
                results.append(TaskResult(
                    task_id=record.task_id,
                    goal=record.goal,
                    status=TaskStatus(record.status),
                    result=record.result,
                    error=record.error,
                    steps=steps,
                    total_duration_ms=record.total_duration_ms,
                    created_at=record.created_at,
                    completed_at=record.completed_at
                ))
            
            return results
            
        except Exception as e:
            logger.error("Failed to get recent tasks", error=str(e))
            raise

