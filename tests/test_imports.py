"""Import smoke tests for the Phase 1 package skeleton."""


def test_import_package() -> None:
    import gpuboost

    assert gpuboost.__version__


def test_import_cli_main() -> None:
    from gpuboost.cli.main import main

    assert callable(main)


def test_import_inspectors() -> None:
    from gpuboost.inspector.gpu import collect_gpu_profiles
    from gpuboost.inspector.profile import collect_profile
    from gpuboost.inspector.system import collect_system_profile
    from gpuboost.inspector.torch_env import collect_torch_environment

    assert callable(collect_gpu_profiles)
    assert callable(collect_system_profile)
    assert callable(collect_torch_environment)
    assert callable(collect_profile)


def test_import_benchmarks() -> None:
    from gpuboost.benchmarks.batch_sweep import run_batch_sweep_benchmark
    from gpuboost.benchmarks.dataloader import run_dataloader_benchmark
    from gpuboost.benchmarks.matmul import run_matmul_benchmark
    from gpuboost.benchmarks.mixed_precision import run_mixed_precision_benchmark
    from gpuboost.benchmarks.runner import run_full_benchmark, run_quick_benchmark

    assert callable(run_matmul_benchmark)
    assert callable(run_mixed_precision_benchmark)
    assert callable(run_batch_sweep_benchmark)
    assert callable(run_dataloader_benchmark)
    assert callable(run_quick_benchmark)
    assert callable(run_full_benchmark)
