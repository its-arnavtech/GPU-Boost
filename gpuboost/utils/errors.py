"""Custom exceptions for GPUBoost."""


class GPUBoostError(Exception):
    """Base exception for GPUBoost errors."""


class InspectionError(GPUBoostError):
    """Raised when system or GPU inspection cannot be completed."""

