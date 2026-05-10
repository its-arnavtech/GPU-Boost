"""Tests for Phase 4 optimization static analysis findings."""

from __future__ import annotations

import ast

from gpuboost.code_analysis.optimizations import (
    analyze_optimization_source,
    is_autocast_context,
    is_cudnn_benchmark_true_assignment,
    is_inference_mode_context,
    is_no_grad_context,
    loop_contains_forward_like_call,
    loop_contains_training_step,
)


def test_emits_cudnn_benchmark_missing_when_absent() -> None:
    result = analyze_optimization_source("x = 1\n")

    assert _finding_ids(result) == ["cudnn_benchmark_missing"]
    finding = result.findings[0]
    assert finding.line is None
    assert finding.column is None
    assert finding.code_snippet == "torch.backends.cudnn.benchmark = True"


def test_does_not_emit_cudnn_finding_when_benchmark_true_exists() -> None:
    result = analyze_optimization_source(
        "import torch\n"
        "torch.backends.cudnn.benchmark = True\n"
    )

    assert _finding_ids(result) == []


def test_detects_eval_loop_missing_no_grad() -> None:
    result = analyze_optimization_source(
        "torch.backends.cudnn.benchmark = True\n"
        "for batch in loader:\n"
        "    outputs = model(batch)\n"
    )

    assert _finding_ids(result) == ["inference_missing_no_grad"]


def test_does_not_detect_eval_loop_wrapped_in_torch_no_grad() -> None:
    result = analyze_optimization_source(
        "torch.backends.cudnn.benchmark = True\n"
        "with torch.no_grad():\n"
        "    for batch in loader:\n"
        "        outputs = model(batch)\n"
    )

    assert _finding_ids(result) == []


def test_does_not_detect_eval_loop_wrapped_in_torch_inference_mode() -> None:
    result = analyze_optimization_source(
        "torch.backends.cudnn.benchmark = True\n"
        "with torch.inference_mode():\n"
        "    while step < max_steps:\n"
        "        outputs = model(inputs)\n"
        "        step += 1\n"
    )

    assert _finding_ids(result) == []


def test_detects_training_loop_missing_autocast() -> None:
    result = analyze_optimization_source(
        "torch.backends.cudnn.benchmark = True\n"
        "for inputs, targets in loader:\n"
        "    outputs = model(inputs)\n"
        "    loss = criterion(outputs, targets)\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
    )

    assert _finding_ids(result) == ["mixed_precision_autocast_missing"]


def test_does_not_detect_training_loop_with_torch_amp_autocast() -> None:
    result = analyze_optimization_source(
        "torch.backends.cudnn.benchmark = True\n"
        "for inputs, targets in loader:\n"
        "    with torch.amp.autocast('cuda'):\n"
        "        outputs = model(inputs)\n"
        "        loss = criterion(outputs, targets)\n"
        "    loss.backward()\n"
    )

    assert _finding_ids(result) == []


def test_does_not_detect_training_loop_with_torch_cuda_amp_autocast() -> None:
    result = analyze_optimization_source(
        "torch.backends.cudnn.benchmark = True\n"
        "with torch.cuda.amp.autocast():\n"
        "    for inputs, targets in loader:\n"
        "        outputs = model(inputs)\n"
        "        loss = criterion(outputs, targets)\n"
        "        scaler.step(optimizer)\n"
    )

    assert _finding_ids(result) == []


def test_does_not_classify_training_loop_as_inference_loop() -> None:
    result = analyze_optimization_source(
        "torch.backends.cudnn.benchmark = True\n"
        "for inputs, targets in loader:\n"
        "    outputs = model(inputs)\n"
        "    loss = criterion(outputs, targets)\n"
        "    loss.backward()\n"
    )

    assert "inference_missing_no_grad" not in _finding_ids(result)


def test_finding_includes_line_and_column_for_loop_based_findings() -> None:
    result = analyze_optimization_source(
        "torch.backends.cudnn.benchmark = True\n"
        "def evaluate(loader):\n"
        "    for batch in loader:\n"
        "        outputs = model(batch)\n",
        filepath="train.py",
    )

    finding = result.findings[0]
    assert finding.id == "inference_missing_no_grad"
    assert finding.filepath == "train.py"
    assert finding.line == 3
    assert finding.column == 4
    assert finding.end_line == 4
    assert finding.end_column is not None


def test_parse_error_returns_status_error() -> None:
    result = analyze_optimization_source("for batch in loader\n")

    assert result.status == "error"
    assert result.findings == []
    assert result.error is not None


def test_helper_checks_detect_expected_contexts_and_loop_patterns() -> None:
    module = ast.parse(
        "torch.backends.cudnn.benchmark = True\n"
        "with torch.no_grad():\n"
        "    pass\n"
        "with torch.inference_mode():\n"
        "    pass\n"
        "with torch.amp.autocast('cuda'):\n"
        "    pass\n"
        "for batch in loader:\n"
        "    outputs = self.model(batch)\n"
        "    loss.backward()\n"
    )
    assignment = module.body[0]
    no_grad = module.body[1]
    inference_mode = module.body[2]
    autocast = module.body[3]
    loop = module.body[4]

    assert isinstance(assignment, ast.Assign)
    assert is_cudnn_benchmark_true_assignment(assignment)
    assert isinstance(no_grad, ast.With)
    assert is_no_grad_context(no_grad.items[0].context_expr)
    assert isinstance(inference_mode, ast.With)
    assert is_inference_mode_context(inference_mode.items[0].context_expr)
    assert isinstance(autocast, ast.With)
    assert is_autocast_context(autocast.items[0].context_expr)
    assert loop_contains_forward_like_call(loop)
    assert loop_contains_training_step(loop)


def _finding_ids(result) -> list[str]:
    return [finding.id for finding in result.findings]
