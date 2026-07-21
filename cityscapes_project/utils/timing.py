"""Runtime extrapolation helpers for planning full experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeEstimate:
    """Measured sample run and its linear full-dataset projection."""

    sample_count: int
    elapsed_seconds: float
    target_count: int

    @property
    def seconds_per_sample(self) -> float:
        return self.elapsed_seconds / self.sample_count

    @property
    def projected_seconds(self) -> float:
        return self.seconds_per_sample * self.target_count

    @property
    def projected_hours(self) -> float:
        return self.projected_seconds / 3600.0


def estimate_runtime(sample_count: int, elapsed_seconds: float, target_count: int) -> RuntimeEstimate:
    """Validate inputs and return a transparent linear runtime estimate."""

    if sample_count <= 0 or target_count <= 0 or elapsed_seconds < 0:
        raise ValueError("sample_count and target_count must be positive; elapsed_seconds cannot be negative")
    return RuntimeEstimate(sample_count, elapsed_seconds, target_count)
