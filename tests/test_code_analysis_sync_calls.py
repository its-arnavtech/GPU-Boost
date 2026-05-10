"""Tests for Phase 4 sync-call static analysis findings."""

from gpuboost.code_analysis.sync_calls import analyze_sync_calls_source


def test_detects_item_inside_for_loop() -> None:
    result = analyze_sync_calls_source(
        "for batch in loader:\n"
        "    loss = model(batch)\n"
        "    total += loss.item()\n"
    )

    assert _finding_ids(result) == ["sync_call_item_in_loop"]


def test_detects_cpu_inside_for_loop() -> None:
    result = analyze_sync_calls_source(
        "for batch in loader:\n"
        "    predictions.append(output.cpu())\n"
    )

    assert _finding_ids(result) == ["sync_call_cpu_in_loop"]


def test_detects_numpy_inside_for_loop() -> None:
    result = analyze_sync_calls_source(
        "for batch in loader:\n"
        "    values = output.numpy()\n"
    )

    assert _finding_ids(result) == ["sync_call_numpy_in_loop"]


def test_detects_calls_inside_while_loop() -> None:
    result = analyze_sync_calls_source(
        "while step < max_steps:\n"
        "    metric = loss.item()\n"
        "    step += 1\n"
    )

    assert _finding_ids(result) == ["sync_call_item_in_loop"]


def test_detects_calls_inside_async_for_loop() -> None:
    result = analyze_sync_calls_source(
        "async def consume(stream):\n"
        "    async for batch in stream:\n"
        "        batch.cpu()\n"
    )

    assert _finding_ids(result) == ["sync_call_cpu_in_loop"]


def test_does_not_detect_item_outside_loop() -> None:
    result = analyze_sync_calls_source("value = loss.item()\n")

    assert result.status == "ok"
    assert result.findings == []


def test_does_not_detect_cpu_outside_loop() -> None:
    result = analyze_sync_calls_source("value = tensor.cpu()\n")

    assert result.status == "ok"
    assert result.findings == []


def test_multiple_findings_in_one_loop() -> None:
    result = analyze_sync_calls_source(
        "for batch in loader:\n"
        "    scalar = loss.item()\n"
        "    host_tensor = output.cpu()\n"
        "    array = other.numpy()\n"
    )

    assert _finding_ids(result) == [
        "sync_call_item_in_loop",
        "sync_call_cpu_in_loop",
        "sync_call_numpy_in_loop",
    ]


def test_nested_loops_still_work() -> None:
    result = analyze_sync_calls_source(
        "for epoch in range(2):\n"
        "    for batch in loader:\n"
        "        scalar = loss.item()\n"
        "    host_tensor = output.cpu()\n"
    )

    assert _finding_ids(result) == [
        "sync_call_item_in_loop",
        "sync_call_cpu_in_loop",
    ]


def test_finding_includes_line_and_column_location() -> None:
    result = analyze_sync_calls_source(
        "for batch in loader:\n"
        "    loss = model(batch)\n"
        "    value = loss.item()\n",
        filepath="train.py",
    )

    finding = result.findings[0]
    assert finding.filepath == "train.py"
    assert finding.line == 3
    assert finding.column == 12
    assert finding.end_line == 3
    assert finding.end_column is not None


def test_parse_error_returns_status_error() -> None:
    result = analyze_sync_calls_source("for batch in loader\n")

    assert result.status == "error"
    assert result.findings == []
    assert result.error is not None


def _finding_ids(result) -> list[str]:
    return [finding.id for finding in result.findings]
