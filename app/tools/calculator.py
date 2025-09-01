"""Mathematical calculation tool."""

import ast
import math
import operator
from typing import Union

import structlog

from app.tools.base import Tool, ToolResult

logger = structlog.get_logger()


class CalculatorTool(Tool):
    """Tool for performing mathematical calculations safely."""
    
    # Supported operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Supported functions
    FUNCTIONS = {
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'sqrt': math.sqrt,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'abs': abs,
        'round': round,
        'ceil': math.ceil,
        'floor': math.floor,
    }
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations. Input: {'expression': '2 + 3 * 4'}"
        )
    
    async def execute(self, expression: str) -> ToolResult:
        """Execute mathematical calculation."""
        try:
            # Parse the expression
            tree = ast.parse(expression, mode='eval')
            
            # Evaluate safely
            result = self._eval_node(tree.body)
            
            return ToolResult(
                success=True,
                result=str(result),
                metadata={"expression": expression}
            )
            
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            return ToolResult(
                success=False,
                result="",
                error=f"Calculation error: {str(e)}"
            )
        except Exception as e:
            logger.error("Calculator failed", expression=expression, error=str(e))
            return ToolResult(
                success=False,
                result="",
                error=f"Invalid expression: {str(e)}"
            )
    
    def _eval_node(self, node) -> Union[int, float]:
        """Safely evaluate AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise ValueError("Only numbers are allowed")
        
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type}")
            
            return self.OPERATORS[op_type](left, right)
        
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)
            
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type}")
            
            return self.OPERATORS[op_type](operand)
        
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            
            func_name = node.func.id
            if func_name not in self.FUNCTIONS:
                raise ValueError(f"Unsupported function: {func_name}")
            
            args = [self._eval_node(arg) for arg in node.args]
            return self.FUNCTIONS[func_name](*args)
        
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

