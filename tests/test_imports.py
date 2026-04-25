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
