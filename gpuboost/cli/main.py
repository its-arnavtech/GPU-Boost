"""GPUBoost command-line interface.

The CLI exposes Phase 1 inspection and Phase 2 benchmark commands.
"""

from __future__ import annotations

import argparse
import json

from gpuboost.benchmarks.runner import run_full_benchmark, run_quick_benchmark
from gpuboost.inspector.profile import collect_profile
from gpuboost.utils.formatting import format_benchmark_suite, format_profile


def build_parser() -> argparse.ArgumentParser:
    """Build the GPUBoost argument parser."""

    parser = argparse.ArgumentParser(
        prog="gpuboost",
        description="Inspect NVIDIA GPU and system information.",
    )
    subparsers = parser.add_subparsers(dest="command")

    info_parser = subparsers.add_parser(
        "info",
        help="Show GPU and system information.",
    )
    info_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run Phase 2 synthetic GPU benchmarks.",
    )
    benchmark_parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the quick benchmark subset.",
    )
    benchmark_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    benchmark_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index to benchmark.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the GPUBoost CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        profile = collect_profile()
        if args.json:
            print(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
        else:
            try:
                from rich.console import Console
            except Exception:
                print(format_profile(profile))
            else:
                Console().print(format_profile(profile))
        return 0

    if args.command == "benchmark":
        suite = (
            run_quick_benchmark(args.device)
            if args.quick
            else run_full_benchmark(args.device)
        )
        if args.json:
            print(json.dumps(suite.to_dict(), indent=2, sort_keys=True))
        else:
            try:
                from rich.console import Console
            except Exception:
                print(format_benchmark_suite(suite))
            else:
                Console().print(format_benchmark_suite(suite))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
