"""Patch planning and diff helpers for GPUBoost."""

from gpuboost.patching.diff import (
    apply_patch_edits_to_text,
    generate_patch_plan_diff,
    generate_unified_diff,
)
from gpuboost.patching.planner import (
    create_patch_plan_from_analysis,
    find_import_block,
    get_source_line,
    insert_kwarg_before_closing_paren,
    replace_on_line,
)

__all__ = [
    "apply_patch_edits_to_text",
    "create_patch_plan_from_analysis",
    "find_import_block",
    "generate_patch_plan_diff",
    "generate_unified_diff",
    "get_source_line",
    "insert_kwarg_before_closing_paren",
    "replace_on_line",
]
