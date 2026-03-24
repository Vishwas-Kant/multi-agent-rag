"""
Code Analysis Tool — Python code structure analysis and LLM explanation.
"""

from __future__ import annotations

import ast
import logging
from typing import Any, Dict, List

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _analyze_structure(source: str) -> Dict[str, Any]:
    tree = ast.parse(source)

    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    imports: List[str] = []
    global_vars: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            args = [a.arg for a in node.args.args]
            decorators = []
            for d in node.decorator_list:
                if isinstance(d, ast.Name):
                    decorators.append(d.id)
                elif isinstance(d, ast.Attribute):
                    decorators.append(f"{ast.dump(d)}")

            functions.append({
                "name": node.name,
                "args": args,
                "decorators": decorators,
                "line": node.lineno,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "docstring": ast.get_docstring(node) or "",
            })

        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(ast.dump(b))

            classes.append({
                "name": node.name,
                "bases": bases,
                "methods": methods,
                "line": node.lineno,
                "docstring": ast.get_docstring(node) or "",
            })

        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")

        elif isinstance(node, ast.Assign) and isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    global_vars.append(target.id)

    complexity = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                          ast.With, ast.Assert, ast.BoolOp))
    )

    return {
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "global_variables": global_vars,
        "total_lines": len(source.splitlines()),
        "cyclomatic_complexity": complexity,
    }


def _format_analysis(analysis: Dict[str, Any]) -> str:
    parts = [f"**Code Analysis** ({analysis['total_lines']} lines, complexity: {analysis['cyclomatic_complexity']})"]

    if analysis["imports"]:
        parts.append(f"\n**Imports** ({len(analysis['imports'])}):")
        for imp in analysis["imports"][:15]:
            parts.append(f"  • {imp}")
        if len(analysis["imports"]) > 15:
            parts.append(f"  ... and {len(analysis['imports']) - 15} more")

    if analysis["classes"]:
        parts.append(f"\n**Classes** ({len(analysis['classes'])}):")
        for cls in analysis["classes"]:
            bases = f"({', '.join(cls['bases'])})" if cls['bases'] else ""
            parts.append(f"  • **{cls['name']}{bases}** [line {cls['line']}]")
            parts.append(f"    Methods: {', '.join(cls['methods']) or 'none'}")
            if cls["docstring"]:
                parts.append(f"    Doc: {cls['docstring'][:100]}")

    if analysis["functions"]:
        parts.append(f"\n**Functions** ({len(analysis['functions'])}):")
        for fn in analysis["functions"]:
            async_prefix = "async " if fn["is_async"] else ""
            args_str = ", ".join(fn["args"])
            parts.append(f"  • **{async_prefix}{fn['name']}**({args_str}) [line {fn['line']}]")
            if fn["docstring"]:
                parts.append(f"    Doc: {fn['docstring'][:100]}")

    return "\n".join(parts)


@tool
def analyze_code(code: str) -> str:
    if not code or not code.strip():
        return "Error: No code provided for analysis."

    try:
        analysis = _analyze_structure(code)
        return _format_analysis(analysis)
    except SyntaxError as e:
        return f"Syntax error in code: {str(e)}"
    except Exception as e:
        logger.exception("Code analysis failed")
        return f"Error analyzing code: {str(e)}"
