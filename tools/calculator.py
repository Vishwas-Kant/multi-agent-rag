"""
Calculator Tool — Safe mathematical expression evaluator.
"""

from __future__ import annotations

import ast
import math
import operator
import statistics
from typing import Any

from langchain_core.tools import tool

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "radians": math.radians,
    "degrees": math.degrees,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "mean": statistics.mean,
    "median": statistics.median,
    "stdev": statistics.stdev,
    "variance": statistics.variance,
    "pi": lambda: math.pi,
    "e": lambda: math.e,
    "tau": lambda: math.tau,
}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return op_func(left, right)

    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported.")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not allowed.")
        func = _SAFE_FUNCTIONS[func_name]
        args = [_safe_eval(a) for a in node.args]
        return func(*args) if args else func()

    if isinstance(node, ast.List):
        return [_safe_eval(el) for el in node.elts]

    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            func = _SAFE_FUNCTIONS[node.id]
            return func() if callable(func) else func
        raise ValueError(f"Unknown variable: '{node.id}'")

    raise ValueError(f"Unsupported expression type: {type(node).__name__}")

@tool
def calculate(expression: str) -> str:
    if not expression or not expression.strip():
        return "Error: No expression provided."

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)

        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                result = int(result)
            else:
                result = round(result, 10)

        return f"{expression.strip()} = {result}"

    except (ValueError, TypeError, SyntaxError, ZeroDivisionError) as e:
        return f"Error evaluating '{expression}': {str(e)}"
    except Exception as e:
        return f"Unexpected error evaluating '{expression}': {str(e)}"
