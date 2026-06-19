"""Lightweight leak detectors for safe JSON audit tests."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True, slots=True)
class JsonLeak:
    """A suspicious key or value discovered in JSON-safe data."""

    path: str
    reason: str


_FORBIDDEN_EXACT_KEYS = {
    "state_dict",
    "raw_model_weights",
    "raw_weights",
    "model_weights",
    "raw_source",
    "source_code",
    "raw_diff",
}
_SECRET_KEY_RE = re.compile(
    r"(^|_)(api_?key|access_?token|refresh_?token|secret|token|password|private_?key)($|_)",
    re.IGNORECASE,
)
_FORBIDDEN_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bstate_dict\b", re.IGNORECASE),
    re.compile(r"\braw model weights\b", re.IGNORECASE),
)


def find_json_leaks(value: Any, allow_raw_streams: bool = False) -> list[JsonLeak]:
    """Return suspicious keys or raw sensitive values in JSON-like data.

    Set ``allow_raw_streams=True`` when auditing output that the caller has
    explicitly opted into raw stdout/stderr for (e.g. ``--include-raw-artifacts``)
    so populated stream keys are not reported as leaks.
    """

    leaks: list[JsonLeak] = []
    _scan_json(value, "$", leaks, allow_raw_streams)
    return leaks


def _scan_json(
    value: Any,
    path: str,
    leaks: list[JsonLeak],
    allow_raw_streams: bool = False,
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}"
            _scan_key(key_text, item, item_path, leaks, allow_raw_streams)
            _scan_json(item, item_path, leaks, allow_raw_streams)
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _scan_json(item, f"{path}[{index}]", leaks, allow_raw_streams)
        return

    if isinstance(value, str):
        _scan_string(value, path, leaks)


def _scan_key(
    key: str,
    value: Any,
    path: str,
    leaks: list[JsonLeak],
    allow_raw_streams: bool = False,
) -> None:
    normalized = key.lower()
    if normalized in _FORBIDDEN_EXACT_KEYS:
        leaks.append(JsonLeak(path=path, reason=f"forbidden key {key!r}"))
        return

    if normalized in {"stdout", "stderr"}:
        if not allow_raw_streams and value not in (None, "", [], {}):
            leaks.append(JsonLeak(path=path, reason=f"raw stream key {key!r}"))
        return

    if normalized.endswith("_redacted"):
        return

    if _SECRET_KEY_RE.search(key):
        leaks.append(JsonLeak(path=path, reason=f"secret-looking key {key!r}"))


def _scan_string(value: str, path: str, leaks: list[JsonLeak]) -> None:
    for pattern in _FORBIDDEN_VALUE_PATTERNS:
        if pattern.search(value):
            leaks.append(JsonLeak(path=path, reason="forbidden sensitive value"))
            return
