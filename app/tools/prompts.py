"""LLM prompt templates for the research agent."""

REACTIVE_PLANNING_PROMPT = """You are an autonomous research agent. Your goal is: {goal}

Current progress:
{scratchpad}

Available tools:
{tools}

Based on the goal and current progress, decide on the next action to take. Think step by step.

Format your response as:
Thought: [your reasoning about what to do next]
Action: [description of the action you want to take]
Tool: [tool name if using a tool, or "none"]
Input: [tool input as JSON if using a tool]

If you believe the goal has been achieved, set Action to "COMPLETE" and provide a summary.
"""

DELIBERATIVE_PLANNING_PROMPT = """You are an autonomous research agent with multi-step planning capabilities. Your goal is: {goal}

Current progress:
{scratchpad}

Available tools:
{tools}

Analyze the goal and create a plan. Consider what information you still need and how to obtain it efficiently.

Format your response as:
Thought: [your analysis of the current situation]
Plan: [your multi-step plan to achieve the goal]
Next Action: [the immediate next action to take]
Tool: [tool name if using a tool, or "none"]
Input: [tool input as JSON if using a tool]

If you believe the goal has been achieved, set Next Action to "COMPLETE".
"""

COMPLETION_CHECK_PROMPT = """Goal: {goal}

Progress so far:
{scratchpad}

Based on the goal and progress made, has the goal been sufficiently achieved? 
Answer with "Yes" if the goal is complete, or "No" if more work is needed.

Answer: """

FINAL_RESULT_PROMPT = """Goal: {goal}

Complete execution trace:
{scratchpad}

Based on all the work done, provide a comprehensive summary of the results. Include:
1. Key findings
2. Sources consulted
3. Any limitations or uncertainties

Summary: """

