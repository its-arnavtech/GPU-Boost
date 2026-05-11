"""Local-only GPUBoost run history helpers."""

from gpuboost.history.builder import (
    build_history_run_record,
    extract_advisor_summary,
    extract_benchmark_summary,
    extract_code_summary,
    extract_comparison_summary,
    extract_gpu_summary,
    extract_patch_summary,
    extract_trial_summary,
    hash_file_if_exists,
    hash_text,
)
from gpuboost.history.compare import compare_history_runs
from gpuboost.history.store import (
    default_history_db_path,
    default_history_dir,
    delete_history_run,
    initialize_history_store,
    insert_history_run,
    list_history_runs,
    load_history_run,
)

__all__ = [
    "build_history_run_record",
    "compare_history_runs",
    "default_history_db_path",
    "default_history_dir",
    "delete_history_run",
    "extract_advisor_summary",
    "extract_benchmark_summary",
    "extract_code_summary",
    "extract_comparison_summary",
    "extract_gpu_summary",
    "extract_patch_summary",
    "extract_trial_summary",
    "hash_file_if_exists",
    "hash_text",
    "initialize_history_store",
    "insert_history_run",
    "list_history_runs",
    "load_history_run",
]
