"""Demo script for the autonomous research agent."""

import asyncio
import time

from app.core.agent import ResearchAgent
from app.core.planner import PlannerType
from app.utils.config import get_settings

# Initialize logging
import structlog
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
)

logger = structlog.get_logger()


async def demo_basic_research():
    """Demo basic research functionality."""
    print("🤖 Autonomous Research Agent Demo")
    print("=" * 40)
    
    agent = ResearchAgent()
    
    # Demo task
    goal = "Find information about the latest developments in AI agents and summarize the key trends"
    
    print(f"📋 Research Goal: {goal}")
    print()
    
    # Start task
    print("🚀 Starting research task...")
    task_id = await agent.start_task(goal, PlannerType.REACTIVE)
    print(f"   Task ID: {task_id}")
    
    # Monitor progress
    print("📊 Monitoring progress...")
    while True:
        status = await agent.get_task_status(task_id)
        
        if status.status.value in ['completed', 'failed', 'cancelled']:
            break
        
        print(f"   Status: {status.status.value} | Steps: {len(status.steps)}")
        await asyncio.sleep(2)
    
    # Display results
    print("\n📋 Final Results:")
    print("-" * 20)
    print(f"Status: {status.status.value}")
    
    if status.result:
        print(f"Result: {status.result}")
    
    if status.error:
        print(f"Error: {status.error}")
    
    print(f"\nExecution Steps: {len(status.steps)}")
    for i, step in enumerate(status.steps, 1):
        print(f"  {i}. {step.action[:50]}{'...' if len(step.action) > 50 else ''}")
        if step.tool_name:
            print(f"     Tool: {step.tool_name}")
    
    print(f"\nTotal Duration: {status.total_duration_ms}ms")


async def demo_tool_usage():
    """Demo individual tool usage."""
    print("\n🔧 Tool Usage Demo")
    print("=" * 40)
    
    from app.tools.calculator import CalculatorTool
    from app.tools.web_search import WebSearchTool
    
    # Calculator demo
    calc = CalculatorTool()
    result = await calc.execute(expression="sqrt(144) + 5 * 2")
    print(f"Calculator: sqrt(144) + 5 * 2 = {result.result}")
    
    # Search demo (if not in safe mode)
    settings = get_settings()
    if not settings.SAFE_MODE:
        search = WebSearchTool()
        result = await search.execute(query="Python programming tips", max_results=3)
        print(f"Search Results: Found {result.metadata.get('count', 0)} results")
    else:
        print("Search demo skipped (SAFE_MODE enabled)")


if __name__ == "__main__":
    print("Starting Autonomous Research Agent Demo...")
    
    try:
        asyncio.run(demo_tool_usage())
        asyncio.run(demo_basic_research())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
