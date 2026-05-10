"""Code-aware static analysis framework for GPUBoost."""

from gpuboost.code_analysis.dataloader import (
    DataLoaderFindingVisitor,
    analyze_dataloader_source,
)
from gpuboost.code_analysis.optimizations import (
    OptimizationFindingVisitor,
    analyze_optimization_source,
)
from gpuboost.code_analysis.parser import parse_python_file, parse_python_source
from gpuboost.code_analysis.runner import (
    analyze_python_file,
    analyze_python_source,
    sort_findings,
)
from gpuboost.code_analysis.sync_calls import (
    SyncCallFindingVisitor,
    analyze_sync_calls_source,
)
from gpuboost.code_analysis.visitors import BaseFindingVisitor, run_visitors

__all__ = [
    "BaseFindingVisitor",
    "DataLoaderFindingVisitor",
    "OptimizationFindingVisitor",
    "SyncCallFindingVisitor",
    "analyze_dataloader_source",
    "analyze_optimization_source",
    "analyze_python_file",
    "analyze_python_source",
    "analyze_sync_calls_source",
    "parse_python_file",
    "parse_python_source",
    "run_visitors",
    "sort_findings",
]
