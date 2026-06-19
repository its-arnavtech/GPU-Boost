"""NVIDIA GPU inspection for GPUBoost Phase 1."""

from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import fields, replace
from io import StringIO
from typing import Any, Optional

from gpuboost.schemas.gpu_profile import GPUDeviceProfile

BYTES_PER_MIB = 1024 * 1024


def _decode(value: Any) -> Optional[str]:
    """Decode NVML byte strings and normalize empty values."""

    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text if text and text.upper() != "[N/A]" else None


def _safe_float(value: Any) -> Optional[float]:
    text = _decode(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_int(value: Any) -> Optional[int]:
    number = _safe_float(value)
    return int(round(number)) if number is not None else None


def _bytes_to_mb(value: int) -> int:
    return int(round(value / BYTES_PER_MIB))


def _tensor_cores_supported(compute_capability: Optional[str]) -> Optional[bool]:
    if compute_capability is None:
        return None
    try:
        major = int(compute_capability.split(".", maxsplit=1)[0])
    except (ValueError, IndexError):
        return None
    return major >= 7


def _tensor_cores_supported_by_architecture(
    architecture: Optional[str],
) -> Optional[bool]:
    if architecture is None:
        return None
    architectures_with_tensor_cores = {
        "Volta",
        "Turing",
        "Ampere",
        "Ada",
        "Hopper",
    }
    return architecture in architectures_with_tensor_cores


def _architecture_name(pynvml: Any, architecture: Any) -> Optional[str]:
    if architecture is None:
        return None

    known_architectures = {
        "Kepler": "NVML_DEVICE_ARCH_KEPLER",
        "Maxwell": "NVML_DEVICE_ARCH_MAXWELL",
        "Pascal": "NVML_DEVICE_ARCH_PASCAL",
        "Volta": "NVML_DEVICE_ARCH_VOLTA",
        "Turing": "NVML_DEVICE_ARCH_TURING",
        "Ampere": "NVML_DEVICE_ARCH_AMPERE",
        "Ada": "NVML_DEVICE_ARCH_ADA",
        "Hopper": "NVML_DEVICE_ARCH_HOPPER",
    }
    for name, constant in known_architectures.items():
        if getattr(pynvml, constant, object()) == architecture:
            return name
    return str(architecture)


def _torch_device_info(warnings: list[str]) -> dict[int, dict[str, object]]:
    """Collect per-device details available through torch.cuda."""

    try:
        import torch
    except Exception as exc:
        warnings.append(f"PyTorch GPU enrichment unavailable: {exc}")
        return {}

    try:
        device_count = int(torch.cuda.device_count())
    except Exception as exc:
        warnings.append(f"torch.cuda.device_count() failed during GPU inspection: {exc}")
        return {}

    devices: dict[int, dict[str, object]] = {}
    for index in range(device_count):
        info: dict[str, object] = {}
        try:
            info["name"] = torch.cuda.get_device_name(index)
        except Exception as exc:
            warnings.append(f"torch.cuda.get_device_name({index}) failed: {exc}")

        try:
            major, minor = torch.cuda.get_device_capability(index)
            compute_capability = f"{major}.{minor}"
            info["cuda_compute_capability"] = compute_capability
            info["tensor_cores_supported"] = _tensor_cores_supported(
                compute_capability
            )
        except Exception as exc:
            warnings.append(f"torch.cuda.get_device_capability({index}) failed: {exc}")

        devices[index] = info

    return devices


def _profiles_from_nvml(warnings: list[str]) -> list[GPUDeviceProfile]:
    """Collect GPU profiles with NVML when NVIDIA drivers are available."""

    try:
        import pynvml
    except Exception as exc:
        warnings.append(f"NVML inspection unavailable: {exc}")
        return []

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        warnings.append(f"NVML initialization failed: {exc}")
        return []

    profiles: list[GPUDeviceProfile] = []
    try:
        try:
            device_count = int(pynvml.nvmlDeviceGetCount())
        except Exception as exc:
            warnings.append(f"NVML device count failed: {exc}")
            return []

        for index in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except Exception as exc:
                warnings.append(f"NVML handle lookup failed for GPU {index}: {exc}")
                continue

            name = f"NVIDIA GPU {index}"
            uuid = None
            architecture = None
            total_vram_mb = None
            used_vram_mb = None
            free_vram_mb = None
            utilization_gpu_percent = None
            utilization_memory_percent = None
            temperature_c = None
            power_draw_w = None
            power_limit_w = None

            try:
                name = _decode(pynvml.nvmlDeviceGetName(handle)) or name
            except Exception as exc:
                warnings.append(f"NVML name lookup failed for GPU {index}: {exc}")

            try:
                uuid = _decode(pynvml.nvmlDeviceGetUUID(handle))
            except Exception as exc:
                warnings.append(f"NVML UUID lookup failed for GPU {index}: {exc}")

            try:
                raw_architecture = pynvml.nvmlDeviceGetArchitecture(handle)
                architecture = _architecture_name(pynvml, raw_architecture)
            except Exception as exc:
                warnings.append(
                    f"NVML architecture lookup failed for GPU {index}: {exc}"
                )
                architecture = None

            try:
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_vram_mb = _bytes_to_mb(int(memory.total))
                used_vram_mb = _bytes_to_mb(int(memory.used))
                free_vram_mb = _bytes_to_mb(int(memory.free))
            except Exception as exc:
                warnings.append(f"NVML memory lookup failed for GPU {index}: {exc}")

            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_gpu_percent = float(utilization.gpu)
                utilization_memory_percent = float(utilization.memory)
            except Exception as exc:
                warnings.append(
                    f"NVML utilization lookup failed for GPU {index}: {exc}"
                )

            try:
                temperature_c = float(
                    pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                )
            except Exception as exc:
                warnings.append(
                    f"NVML temperature lookup failed for GPU {index}: {exc}"
                )

            try:
                power_draw_w = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000
            except Exception as exc:
                warnings.append(f"NVML power draw lookup failed for GPU {index}: {exc}")
                power_draw_w = None

            try:
                power_limit_w = (
                    float(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)) / 1000
                )
            except Exception as exc:
                warnings.append(f"NVML power limit lookup failed for GPU {index}: {exc}")
                power_limit_w = None

            profiles.append(
                GPUDeviceProfile(
                    index=index,
                    name=name,
                    uuid=uuid,
                    architecture=architecture,
                    total_vram_mb=total_vram_mb,
                    used_vram_mb=used_vram_mb,
                    free_vram_mb=free_vram_mb,
                    utilization_gpu_percent=utilization_gpu_percent,
                    utilization_memory_percent=utilization_memory_percent,
                    temperature_c=temperature_c,
                    power_draw_w=power_draw_w,
                    power_limit_w=power_limit_w,
                )
            )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return profiles


def _query_nvidia_smi(warnings: list[str]) -> dict[int, dict[str, object]]:
    """Return per-GPU fields from nvidia-smi when the command is available."""

    if shutil.which("nvidia-smi") is None:
        warnings.append("nvidia-smi was not found on PATH.")
        return {}

    query_fields = [
        "index",
        "name",
        "uuid",
        "memory.total",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "utilization.memory",
        "temperature.gpu",
        "power.draw",
        "power.limit",
    ]
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        warnings.append(f"nvidia-smi query failed: {exc}")
        return {}

    rows = csv.reader(StringIO(completed.stdout.strip()))
    results: dict[int, dict[str, object]] = {}
    for row in rows:
        if not row or len(row) != len(query_fields):
            continue
        values = {field: value.strip() for field, value in zip(query_fields, row)}
        index = _safe_int(values["index"])
        if index is None:
            continue
        results[index] = {
            "name": _decode(values["name"]),
            "uuid": _decode(values["uuid"]),
            "total_vram_mb": _safe_int(values["memory.total"]),
            "used_vram_mb": _safe_int(values["memory.used"]),
            "free_vram_mb": _safe_int(values["memory.free"]),
            "utilization_gpu_percent": _safe_float(values["utilization.gpu"]),
            "utilization_memory_percent": _safe_float(
                values["utilization.memory"]
            ),
            "temperature_c": _safe_float(values["temperature.gpu"]),
            "power_draw_w": _safe_float(values["power.draw"]),
            "power_limit_w": _safe_float(values["power.limit"]),
        }

    return results


def _apply_info(
    profile: GPUDeviceProfile, info: dict[str, object]
) -> GPUDeviceProfile:
    """Return a profile with missing fields filled from another source.

    Builds a new instance via dataclasses.replace instead of mutating in place,
    so the function keeps working even if GPUDeviceProfile becomes frozen/slotted.
    """

    field_names = {f.name for f in fields(profile)}
    updates = {
        field_name: value
        for field_name, value in info.items()
        if field_name in field_names
        and value is not None
        and getattr(profile, field_name) is None
    }
    return replace(profile, **updates) if updates else profile


def collect_gpu_profiles(warnings: list[str] | None = None) -> list[GPUDeviceProfile]:
    """Collect NVIDIA GPU profiles using NVML, PyTorch, and nvidia-smi."""

    warning_sink = warnings if warnings is not None else []
    profiles = _profiles_from_nvml(warning_sink)
    torch_devices = _torch_device_info(warning_sink)
    smi_devices = _query_nvidia_smi(warning_sink)

    profiles_by_index = {profile.index: profile for profile in profiles}

    for index, info in smi_devices.items():
        profile = profiles_by_index.get(index)
        if profile is None:
            profile = GPUDeviceProfile(
                index=index,
                name=str(info.get("name") or f"NVIDIA GPU {index}"),
            )
            profiles_by_index[index] = profile
        profiles_by_index[index] = _apply_info(profile, info)

    for index, info in torch_devices.items():
        profile = profiles_by_index.get(index)
        if profile is None:
            profile = GPUDeviceProfile(
                index=index,
                name=str(info.get("name") or f"NVIDIA GPU {index}"),
            )
            profiles_by_index[index] = profile
        profiles_by_index[index] = _apply_info(profile, info)

    for profile in profiles_by_index.values():
        if profile.tensor_cores_supported is None:
            profile.tensor_cores_supported = _tensor_cores_supported(
                profile.cuda_compute_capability
            )
        if profile.tensor_cores_supported is None:
            profile.tensor_cores_supported = _tensor_cores_supported_by_architecture(
                profile.architecture
            )

    collected = [profiles_by_index[index] for index in sorted(profiles_by_index)]
    if not collected:
        warning_sink.append(
            "No NVIDIA GPU was detected through NVML, PyTorch CUDA, or nvidia-smi."
        )

    return collected


def inspect_gpu() -> dict[str, object | None]:
    """Backward-compatible dictionary view for the first detected GPU."""

    warnings: list[str] = []
    profiles = collect_gpu_profiles(warnings)
    return profiles[0].to_dict() if profiles else {}
