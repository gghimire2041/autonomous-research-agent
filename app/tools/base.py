"""Base tool interface and common functionality."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import structlog
from pydantic import BaseModel

logger = structlog.get_logger()


class ToolResult(BaseModel):
    """Result from tool execution."""
    success: bool
    result: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class Tool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def validate_input(self, **kwargs) -> Dict[str, Any]:
        """Validate and sanitize input parameters."""
        return kwargs
    
    async def _execute_with_logging(self, **kwargs) -> ToolResult:
        """Execute tool with proper logging."""
        logger.info(f"Executing tool {self.name}", **kwargs)
        
        try:
            validated_input = self.validate_input(**kwargs)
            result = await self.execute(**validated_input)
            
            logger.info(
                f"Tool {self.name} completed",
                success=result.success,
                result_length=len(result.result) if result.result else 0
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Tool {self.name} failed", error=str(e))
            return ToolResult(
                success=False,
                result="",
                error=str(e)
            )
