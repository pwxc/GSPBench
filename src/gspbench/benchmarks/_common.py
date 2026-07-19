"""Shared benchmark utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..datasets import load_dataset
from ..models import GraphSignalDataset


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """JSON-serializable output of a benchmark protocol."""

    task: str
    dataset_id: str
    config: dict[str, Any]
    records: list[dict[str, Any]]
    summary: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "dataset_id": self.dataset_id,
            "config": self.config,
            "records": self.records,
            "summary": self.summary,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def resolve_dataset(dataset: str | GraphSignalDataset) -> GraphSignalDataset:
    if isinstance(dataset, GraphSignalDataset):
        return dataset
    return load_dataset(dataset)


def unique_component_counts(n_nodes: int, ratios: Iterable[float]) -> list[int]:
    counts = {max(1, min(n_nodes, int(np.ceil(n_nodes * ratio)))) for ratio in ratios}
    return sorted(counts)


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - truth
    denominator = max(float(np.linalg.norm(truth)), np.finfo(float).eps)
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    relative_l2 = float(np.linalg.norm(error) / denominator)
    centered = truth - np.mean(truth)
    total = float(np.sum(np.square(centered)))
    r2 = float(1.0 - np.sum(np.square(error)) / total) if total > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "relative_l2": relative_l2, "r2": r2}


def signal_to_noise_ratio(truth: np.ndarray, error: np.ndarray) -> float:
    numerator = max(float(np.sum(np.square(truth))), np.finfo(float).eps)
    denominator = max(float(np.sum(np.square(error))), np.finfo(float).eps)
    return float(10.0 * np.log10(numerator / denominator))


def summarize_records(
    records: list[dict[str, Any]], *, group_keys: tuple[str, ...], metric_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(tuple(record[key] for key in group_keys), []).append(record)
    summary: list[dict[str, Any]] = []
    for key, rows in groups.items():
        item = dict(zip(group_keys, key, strict=True))
        item["n_trials"] = len(rows)
        for metric in metric_keys:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            item[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            item[f"{metric}_std"] = float(np.std(finite)) if finite.size else float("nan")
        summary.append(item)
    return summary


def impute_from_training(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Median-impute columns using only the supplied training rows."""

    train = np.asarray(train, dtype=float)
    medians = np.nanmedian(train, axis=0)
    global_median = float(np.nanmedian(train))
    medians = np.where(np.isfinite(medians), medians, global_median)

    def fill(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=float).copy()
        rows, columns = np.where(~np.isfinite(result))
        result[rows, columns] = medians[columns]
        return result

    return (fill(train), *(fill(values) for values in others))


def require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "This benchmark requires scikit-learn. Install it with "
            "`pip install gspbench[benchmarks]`."
        ) from exc
