"""Reference benchmark protocols for GSPBench datasets."""

from __future__ import annotations

from typing import Any

from ._common import BenchmarkResult
from ._gsp import run_compression, run_denoising, run_interpolation
from ._ml import run_anomaly_detection, run_next_day_forecasting, run_season_classification

__all__ = [
    "BenchmarkResult",
    "run_anomaly_detection",
    "run_benchmark",
    "run_compression",
    "run_denoising",
    "run_interpolation",
    "run_next_day_forecasting",
    "run_season_classification",
]

_RUNNERS = {
    "denoising": run_denoising,
    "interpolation": run_interpolation,
    "compression": run_compression,
    "season_classification": run_season_classification,
    "next_day_forecasting": run_next_day_forecasting,
    "anomaly_detection": run_anomaly_detection,
}


def run_benchmark(task: str, dataset: Any, **kwargs: Any) -> BenchmarkResult:
    """Run one named reference benchmark.

    ``dataset`` may be a packaged dataset identifier or an already loaded
    :class:`~gspbench.GraphSignalDataset`.
    """

    try:
        runner = _RUNNERS[task]
    except KeyError as exc:
        choices = ", ".join(sorted(_RUNNERS))
        raise ValueError(f"unknown benchmark {task!r}; choose one of: {choices}") from exc
    return runner(dataset, **kwargs)
