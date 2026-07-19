"""Bandlimitedness and graph Fourier analysis."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from typing import Iterable

import numpy as np
from scipy.integrate import trapezoid

from .models import GraphSignalDataset


@dataclass(frozen=True, slots=True)
class BandlimitednessResult:
    """Spectral summary of one raw, non-centered graph signal."""

    eigenvalues: np.ndarray
    coefficients: np.ndarray
    energy: np.ndarray
    cumulative_energy: np.ndarray
    effective_bandwidth: dict[str, int]
    auc_energy_concentration: float
    knee: int
    total_variation: float
    normalized_total_variation: float
    zero_mode_energy_ratio: float

    def to_dict(self, include_arrays: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "effective_bandwidth": dict(self.effective_bandwidth),
            "auc_energy_concentration": self.auc_energy_concentration,
            "knee": self.knee,
            "total_variation": self.total_variation,
            "normalized_total_variation": self.normalized_total_variation,
            "zero_mode_energy_ratio": self.zero_mode_energy_ratio,
        }
        if include_arrays:
            result.update(
                eigenvalues=self.eigenvalues.tolist(),
                coefficients=self.coefficients.tolist(),
                energy=self.energy.tolist(),
                cumulative_energy=self.cumulative_energy.tolist(),
            )
        return result


def graph_fourier_basis(
    dataset: GraphSignalDataset, kind: str = "normalized"
) -> tuple[np.ndarray, np.ndarray]:
    """Return Laplacian eigenvalues and orthonormal eigenvectors in ascending order."""

    laplacian = dataset.laplacian(kind).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues[np.abs(eigenvalues) < 1e-12] = 0.0
    return eigenvalues, eigenvectors


def _resolve_signal(dataset: GraphSignalDataset, signal: str | np.ndarray) -> np.ndarray:
    if isinstance(signal, str):
        try:
            values = dataset.signals[signal]
        except KeyError as exc:
            choices = ", ".join(dataset.signals)
            raise ValueError(f"unknown signal {signal!r}; choose one of: {choices}") from exc
    else:
        values = np.asarray(signal, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape != (dataset.n_nodes,):
        raise ValueError(f"signal must have shape ({dataset.n_nodes},), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("bandlimitedness requires a finite signal")
    return values


def _knee(cumulative: np.ndarray) -> int:
    if cumulative.size <= 2:
        return 1
    x = np.linspace(0.0, 1.0, cumulative.size)
    chord = cumulative[0] + x * (cumulative[-1] - cumulative[0])
    return int(np.argmax(cumulative - chord) + 1)


def bandlimitedness(
    dataset: GraphSignalDataset,
    signal: str | np.ndarray,
    *,
    thresholds: Iterable[float] = (0.90, 0.95, 0.99),
    laplacian_kind: str = "normalized",
) -> BandlimitednessResult:
    """Analyze a signal without centering it or removing its zero-frequency mode."""

    values = _resolve_signal(dataset, signal)
    eigenvalues, eigenvectors = graph_fourier_basis(dataset, laplacian_kind)
    coefficients = eigenvectors.T @ values
    energy = np.square(np.abs(coefficients))
    total_energy = float(energy.sum())
    if total_energy <= 0:
        raise ValueError("bandlimitedness is undefined for a zero-energy signal")
    cumulative = np.cumsum(energy) / total_energy

    bandwidth: dict[str, int] = {}
    for threshold in thresholds:
        if not 0 < threshold <= 1:
            raise ValueError("energy thresholds must be in (0, 1]")
        key = f"k{int(round(100 * threshold))}"
        bandwidth[key] = int(np.searchsorted(cumulative, threshold, side="left") + 1)

    laplacian = dataset.laplacian(laplacian_kind)
    total_variation = float(values @ (laplacian @ values))
    auc = float(trapezoid(cumulative, dx=1.0 / max(1, values.size - 1)))
    return BandlimitednessResult(
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        energy=energy,
        cumulative_energy=cumulative,
        effective_bandwidth=bandwidth,
        auc_energy_concentration=auc,
        knee=_knee(cumulative),
        total_variation=total_variation,
        normalized_total_variation=total_variation / total_energy,
        zero_mode_energy_ratio=float(energy[0] / total_energy),
    )


def lowpass_reconstruct(
    signal: np.ndarray, eigenvectors: np.ndarray, n_components: int
) -> np.ndarray:
    """Reconstruct from the first K graph frequencies, including mode zero."""

    values = np.asarray(signal, dtype=float)
    if not 1 <= n_components <= eigenvectors.shape[1]:
        raise ValueError("n_components must be between 1 and the graph size")
    basis = eigenvectors[:, :n_components]
    return basis @ (basis.T @ values)


def load_reference_results(dataset: str | None = None) -> dict[str, object]:
    """Load the packaged 0.0.1 bandlimitedness reference results."""

    resource = resources.files("gspbench.data").joinpath("reference_bandlimitedness.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if dataset is None:
        return payload
    try:
        selected = payload["datasets"][dataset]
    except KeyError as exc:
        choices = ", ".join(sorted(payload["datasets"]))
        raise ValueError(f"unknown dataset {dataset!r}; choose one of: {choices}") from exc
    return {
        "schema_version": payload["schema_version"],
        "protocol": payload["protocol"],
        "datasets": {dataset: selected},
    }
