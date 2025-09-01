"""Task planning strategies for the research agent."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional

import structlog
from pydantic import BaseModel

from app.llm.adapters import LLMAdapter
from app.tools.base import Tool
from app.llm.prompts import (
    REACTIVE_PLANNING_PROMPT,
    DELIBERATIVE_PLANNING_PROMPT,
    COMPLETION_CHECK_PROMPT,
    FINAL_RESULT_PROMPT
)

logger = structlog.get_logger()


class PlannerType(str, Enum):
    """Available planner types."""
    REACTIVE = "reactive"
    DELIBERATIVE = "deliberative"


class ActionPlan(BaseModel):
    """Planned action with reasoning."""
    thought: str
    action: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None


class Planner(ABC):
    """Abstract base class for task planners."""
    
    def __init__(self, llm: LLMAdapter):
        self.llm = llm
    
    @abstractmethod
    async def plan_next_action(
        self, 
        goal: str, 
        scratchpad: str, 
        available_tools: Dict[str, Tool]
    ) -> ActionPlan:
        """Plan the next action to take."""
        pass
    
    async def is_task_complete(self, goal: str, scratchpad: str) -> bool:
        """Check if the task is complete based on current progress."""
        prompt = COMPLETION_CHECK_PROMPT.format(
            goal=goal,
            scratchpad=scratchpad
        )
        
        response = await self.llm.generate(prompt)
        return "yes" in response.lower()
    
    async def generate_final_result(self, goal: str, scratchpad: str) -> str:
        """Generate the final result summary."""
        prompt = FINAL_RESULT_PROMPT.format(
            goal=goal,
            scratchpad=scratchpad
        )
        
        return await self.llm.generate(prompt)


class ReactivePlanner(Planner):
    """Single-step reactive planner."""
    
    async def plan_next_action(
        self, 
        goal: str, 
        scratchpad: str, 
        available_tools: Dict[str, Tool]
    ) -> ActionPlan:
        """Plan next action based on immediate context."""
        tools_description = self._format_tools_description(available_tools)
        
        prompt = REACTIVE_PLANNING_PROMPT.format(
            goal=goal,
            scratchpad=scratchpad,
            tools=tools_description
        )
        
        response = await self.llm.generate(prompt)
        return self._parse_action_plan(response)
    
    def _format_tools_description(self, tools: Dict[str, Tool]) -> str:
        """Format available tools for the prompt."""
        descriptions = []
        for name, tool in tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
    
    def _parse_action_plan(self, response: str) -> ActionPlan:
        """Parse LLM response into structured action plan."""
        lines = response.strip().split('\n')
        
        thought = ""
        action = ""
        tool_name = None
        tool_input = None
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                current_section = "thought"
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                current_section = "action"
                action = line.replace("Action:", "").strip()
            elif line.startswith("Tool:"):
                tool_name = line.replace("Tool:", "").strip()
            elif line.startswith("Input:"):
                # Try to parse as JSON
                input_str = line.replace("Input:", "").strip()
                try:
                    import json
                    tool_input = json.loads(input_str)
                except:
                    tool_input = {"query": input_str}
            elif current_section == "thought" and line:
                thought += " " + line
            elif current_section == "action" and line:
                action += " " + line
        
        return ActionPlan(
            thought=thought,
            action=action,
            tool_name=tool_name,
            tool_input=tool_input
        )


class DeliberativePlanner(Planner):
    """Multi-step deliberative planner."""
    
    async def plan_next_action(
        self, 
        goal: str, 
        scratchpad: str, 
        available_tools: Dict[str, Tool]
    ) -> ActionPlan:
        """Plan next action with multi-step reasoning."""
        tools_description = self._format_tools_description(available_tools)
        
        prompt = DELIBERATIVE_PLANNING_PROMPT.format(
            goal=goal,
            scratchpad=scratchpad,
            tools=tools_description
        )
        
        response = await self.llm.generate(prompt)
        return self._parse_action_plan(response)
    
    def _format_tools_description(self, tools: Dict[str, Tool]) -> str:
        """Format available tools for the prompt."""
        descriptions = []
        for name, tool in tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
    
    def _parse_action_plan(self, response: str) -> ActionPlan:
        """Parse LLM response into structured action plan."""
        # Similar to reactive planner but with more sophisticated parsing
        # for multi-step plans
        lines = response.strip().split('\n')
        
        thought = ""
        action = ""
        tool_name = None
        tool_input = None
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                current_section = "thought"
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Plan:"):
                current_section = "thought"
                thought += " Plan: " + line.replace("Plan:", "").strip()
            elif line.startswith("Next Action:"):
                current_section = "action"
                action = line.replace("Next Action:", "").strip()
            elif line.startswith("Tool:"):
                tool_name = line.replace("Tool:", "").strip()
            elif line.startswith("Input:"):
                input_str = line.replace("Input:", "").strip()
                try:
                    import json
                    tool_input = json.loads(input_str)
                except:
                    tool_input = {"query": input_str}
            elif current_section == "thought" and line:
                thought += " " + line
            elif current_section == "action" and line:
                action += " " + line
        
        return ActionPlan(
            thought=thought,
            action=action,
            tool_name=tool_name,
            tool_input=tool_input
        )


def get_planner(planner_type: PlannerType, llm: LLMAdapter) -> Planner:
    """Factory function to get planner instance."""
    if planner_type == PlannerType.REACTIVE:
        return ReactivePlanner(llm)
    elif planner_type == PlannerType.DELIBERATIVE:
        return DeliberativePlanner(llm)
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")
