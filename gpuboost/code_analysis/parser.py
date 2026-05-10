"""Python source parsing helpers for GPUBoost code analysis."""

from __future__ import annotations

import ast
from pathlib import Path

from gpuboost.schemas.code_analysis import CodeAnalysisResult, create_timestamp


def parse_python_source(
    source: str,
    filepath: str = "<string>",
) -> tuple[ast.AST | None, CodeAnalysisResult]:
    """Parse Python source text without raising parser exceptions."""

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        return None, _error_result(filepath, _format_syntax_error(exc))
    except Exception as exc:  # pragma: no cover - defensive guard for ast internals
        return None, _error_result(filepath, str(exc))

    return tree, CodeAnalysisResult(
        generated_at=create_timestamp(),
        filepath=filepath,
        status="ok",
    )


def parse_python_file(filepath: str) -> tuple[ast.AST | None, CodeAnalysisResult]:
    """Read and parse a Python file as UTF-8 without raising file errors."""

    try:
        source = Path(filepath).read_text(encoding="utf-8")
    except OSError as exc:
        return None, _error_result(filepath, f"Unable to read file: {exc}")
    except Exception as exc:  # pragma: no cover - defensive guard for path handling
        return None, _error_result(filepath, str(exc))

    return parse_python_source(source, filepath=filepath)


def _error_result(filepath: str, error: str) -> CodeAnalysisResult:
    return CodeAnalysisResult(
        generated_at=create_timestamp(),
        filepath=filepath,
        status="error",
        error=error,
    )


def _format_syntax_error(exc: SyntaxError) -> str:
    message = exc.msg or "Syntax error"
    if exc.lineno is None:
        return message

    return f"{message} at line {exc.lineno}"
