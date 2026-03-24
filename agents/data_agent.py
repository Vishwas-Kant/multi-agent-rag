"""
Data Agent — Specialist for mathematical computation and code analysis.
"""

from __future__ import annotations

from agents.base import BaseAgent
from tools.calculator import calculate
from tools.code_analysis import analyze_code


_SYSTEM_PROMPT = """You are a Data & Code Analysis Specialist AI agent.
Your job is to solve mathematical problems and analyze code accurately.

AVAILABLE TOOLS:
1. calculate(expression) — Evaluate mathematical expressions safely.
   Supports: arithmetic, trigonometry (sin, cos, tan), logarithms (log, log2, log10),
   sqrt, abs, round, ceil, floor, factorial, statistics (mean, median, stdev),
   and constants (pi, e).

2. analyze_code(code) — Analyze Python code structure.
   Extracts: functions, classes, imports, complexity metrics.

WORKFLOW:
1. Identify if the query is about math or code
2. For math: formulate the expression and use calculate()
3. For code: pass the code to analyze_code()
4. Present results clearly with explanations

RULES:
- For math: show the expression AND the result
- For code: provide both structural analysis and explanation
- Do NOT guess results — always use the tools
- Explain your reasoning step by step
- If an expression is invalid, explain why and suggest corrections"""


class DataAgent(BaseAgent):
    """Math and code analysis specialist."""

    def __init__(self, llm):
        tools = [calculate, analyze_code]
        super().__init__(
            name="data_agent",
            tools=tools,
            llm=llm,
            system_prompt=_SYSTEM_PROMPT,
        )

    def get_description(self) -> str:
        return "Data Agent: Solves math problems and analyzes Python code structure."
