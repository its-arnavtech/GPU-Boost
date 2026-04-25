"""Shell execution helpers for future inspection commands.

Later Phase 1 work can centralize safe calls to tools such as `nvidia-smi`
here, keeping subprocess handling consistent and testable.
"""

from __future__ import annotations


def command_available(command: str) -> bool:
    """Placeholder for checking whether an external command is available."""

    return bool(command)

