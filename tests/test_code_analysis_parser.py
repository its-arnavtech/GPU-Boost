"""Tests for Phase 4 code analysis parser helpers."""

import ast
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from gpuboost.code_analysis.parser import parse_python_file, parse_python_source


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    path = Path("code_analysis_parser_tmp")
    if path.exists():
        shutil.rmtree(path)

    path.mkdir()
    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(path)


def test_parse_python_source_success() -> None:
    tree, result = parse_python_source("x = 1\n", filepath="train.py")

    assert isinstance(tree, ast.Module)
    assert result.status == "ok"
    assert result.filepath == "train.py"
    assert result.findings == []
    assert result.warnings == []
    assert result.error is None


def test_parse_python_source_syntax_error_returns_error_status() -> None:
    tree, result = parse_python_source("def broken(:\n", filepath="broken.py")

    assert tree is None
    assert result.status == "error"
    assert result.filepath == "broken.py"
    assert result.findings == []
    assert result.warnings == []
    assert result.error is not None
    assert "line 1" in result.error


def test_parse_python_file_success(tmp_path) -> None:
    filepath = tmp_path / "train.py"
    filepath.write_text("import torch\nx = torch.ones(1)\n", encoding="utf-8")

    tree, result = parse_python_file(str(filepath))

    assert isinstance(tree, ast.Module)
    assert result.status == "ok"
    assert result.filepath == str(filepath)
    assert result.error is None


def test_parse_python_file_missing_file_returns_error_status(tmp_path) -> None:
    filepath = tmp_path / "missing.py"

    tree, result = parse_python_file(str(filepath))

    assert tree is None
    assert result.status == "error"
    assert result.filepath == str(filepath)
    assert result.findings == []
    assert result.warnings == []
    assert result.error is not None
    assert "Unable to read file" in result.error
