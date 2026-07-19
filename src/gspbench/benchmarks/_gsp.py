"""Core graph-signal benchmark implementations."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..analysis import graph_fourier_basis, lowpass_reconstruct
from ..models import GraphSignalDataset
from ._common import (
    BenchmarkResult,
    regression_metrics,
    resolve_dataset,
    signal_to_noise_ratio,
    summarize_records,
    unique_component_counts,
)

_SIGNAL_NAMES = (
    "winter_temperature_midrange_c",
    "summer_temperature_midrange_c",
)


def _noise_at_snr(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(size=signal.size)
    target_norm = np.linalg.norm(signal) / (10.0 ** (snr_db / 20.0))
    return noise * (target_norm / np.linalg.norm(noise))


def _spectral_filter(
    values: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    method: str,
    parameter: float | int,
) -> np.ndarray:
    coefficients = eigenvectors.T @ values
    if method == "tikhonov":
        gain = 1.0 / (1.0 + float(parameter) * eigenvalues)
    elif method == "heat":
        gain = np.exp(-float(parameter) * eigenvalues)
    elif method == "gft_lowpass":
        gain = np.zeros_like(eigenvalues)
        gain[: int(parameter)] = 1.0
    else:
        raise ValueError(f"unknown spectral filter {method!r}")
    return eigenvectors @ (gain * coefficients)


def run_denoising(
    dataset: str | GraphSignalDataset,
    *,
    snr_db: Iterable[float] = (5.0, 10.0, 20.0),
    validation_repeats: int = 20,
    test_repeats: int = 50,
    seed: int = 42,
    tikhonov_grid: Iterable[float] = tuple(np.logspace(-4, 4, 17)),
    heat_grid: Iterable[float] = tuple(np.logspace(-3, 3, 13)),
    component_ratios: Iterable[float] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0),
) -> BenchmarkResult:
    """Benchmark recovery from controlled additive Gaussian noise."""

    snr_db = tuple(snr_db)
    tikhonov_grid = tuple(tikhonov_grid)
    heat_grid = tuple(heat_grid)
    component_ratios = tuple(component_ratios)
    data = resolve_dataset(dataset)
    eigenvalues, eigenvectors = graph_fourier_basis(data)
    component_grid = unique_component_counts(data.n_nodes, component_ratios)
    grids: dict[str, list[float | int]] = {
        "tikhonov": list(tikhonov_grid),
        "heat": list(heat_grid),
        "gft_lowpass": component_grid,
    }
    records: list[dict[str, Any]] = []

    for signal_index, signal_name in enumerate(_SIGNAL_NAMES):
        truth = np.asarray(data.signals[signal_name], dtype=float)
        for snr_index, level in enumerate(snr_db):
            validation_samples = []
            for repeat in range(validation_repeats):
                rng = np.random.default_rng(
                    seed + signal_index * 100_000 + snr_index * 10_000 + repeat
                )
                noise = _noise_at_snr(truth, float(level), rng)
                validation_samples.append(truth + noise)

            selected: dict[str, float | int] = {}
            for method, grid in grids.items():
                scores = []
                for parameter in grid:
                    rmse = [
                        regression_metrics(
                            truth,
                            _spectral_filter(sample, eigenvalues, eigenvectors, method, parameter),
                        )["rmse"]
                        for sample in validation_samples
                    ]
                    scores.append(float(np.mean(rmse)))
                selected[method] = grid[int(np.argmin(scores))]

            for repeat in range(test_repeats):
                rng = np.random.default_rng(
                    seed + 1_000_000 + signal_index * 100_000 + snr_index * 10_000 + repeat
                )
                noise = _noise_at_snr(truth, float(level), rng)
                noisy = truth + noise
                predictions = {"identity": noisy}
                predictions.update(
                    {
                        method: _spectral_filter(
                            noisy, eigenvalues, eigenvectors, method, parameter
                        )
                        for method, parameter in selected.items()
                    }
                )
                input_snr = signal_to_noise_ratio(truth, noise)
                for method, prediction in predictions.items():
                    metrics = regression_metrics(truth, prediction)
                    output_snr = signal_to_noise_ratio(truth, prediction - truth)
                    records.append(
                        {
                            "signal": signal_name,
                            "snr_db": float(level),
                            "method": method,
                            "parameter": selected.get(method),
                            "trial": repeat,
                            **metrics,
                            "output_snr_db": output_snr,
                            "snr_improvement_db": output_snr - input_snr,
                        }
                    )

    summary = summarize_records(
        records,
        group_keys=("signal", "snr_db", "method", "parameter"),
        metric_keys=("rmse", "mae", "relative_l2", "output_snr_db", "snr_improvement_db"),
    )
    return BenchmarkResult(
        task="denoising",
        dataset_id=data.dataset_id,
        config={
            "snr_db": list(snr_db),
            "validation_repeats": validation_repeats,
            "test_repeats": test_repeats,
            "seed": seed,
            "zero_mode_retained": True,
        },
        records=records,
        summary=summary,
    )


def _pairwise_haversine(coordinates: np.ndarray) -> np.ndarray:
    radians = np.radians(coordinates)
    latitudes = radians[:, 0, None]
    longitudes = radians[:, 1, None]
    delta_latitude = latitudes.T - latitudes
    delta_longitude = longitudes.T - longitudes
    haversine = np.sin(delta_latitude / 2.0) ** 2 + (
        np.cos(latitudes) * np.cos(latitudes.T) * np.sin(delta_longitude / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def _interpolate(
    method: str,
    truth: np.ndarray,
    observed: np.ndarray,
    *,
    parameter: float | int | None,
    laplacian: sparse.csr_matrix,
    eigenvectors: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    prediction = truth.copy()
    missing = ~observed
    if method == "mean":
        prediction[missing] = np.mean(truth[observed])
    elif method == "nearest":
        observed_indices = np.flatnonzero(observed)
        nearest = observed_indices[np.argmin(distances[missing][:, observed], axis=1)]
        prediction[missing] = truth[nearest]
    elif method == "tikhonov":
        mask = sparse.diags(observed.astype(float), format="csr")
        system = mask + float(parameter) * laplacian + 1e-10 * sparse.eye(truth.size)
        prediction = np.asarray(spsolve(system.tocsc(), observed * truth), dtype=float)
    elif method == "bandlimited_ls":
        basis = eigenvectors[:, : int(parameter)]
        coefficients, *_ = np.linalg.lstsq(basis[observed], truth[observed], rcond=None)
        prediction = basis @ coefficients
    else:
        raise ValueError(f"unknown interpolation method {method!r}")
    return prediction


def run_interpolation(
    dataset: str | GraphSignalDataset,
    *,
    observed_ratios: Iterable[float] = (0.10, 0.30, 0.50),
    validation_repeats: int = 20,
    test_repeats: int = 50,
    seed: int = 42,
    tikhonov_grid: Iterable[float] = tuple(np.logspace(-4, 4, 17)),
    component_ratios: Iterable[float] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50),
) -> BenchmarkResult:
    """Benchmark recovery of values at unobserved graph nodes."""

    observed_ratios = tuple(observed_ratios)
    tikhonov_grid = tuple(tikhonov_grid)
    component_ratios = tuple(component_ratios)
    data = resolve_dataset(dataset)
    _, eigenvectors = graph_fourier_basis(data)
    laplacian = data.laplacian("normalized")
    distances = _pairwise_haversine(data.coordinates)
    records: list[dict[str, Any]] = []

    for signal_index, signal_name in enumerate(_SIGNAL_NAMES):
        truth = np.asarray(data.signals[signal_name], dtype=float)
        for ratio_index, ratio in enumerate(observed_ratios):
            n_observed = max(1, int(round(data.n_nodes * float(ratio))))
            component_grid = [
                value
                for value in unique_component_counts(data.n_nodes, component_ratios)
                if value <= n_observed
            ]
            if 1 not in component_grid:
                component_grid.insert(0, 1)
            grids: dict[str, list[float | int]] = {
                "tikhonov": list(tikhonov_grid),
                "bandlimited_ls": component_grid,
            }
            validation_masks = []
            for repeat in range(validation_repeats):
                rng = np.random.default_rng(
                    seed + signal_index * 100_000 + ratio_index * 10_000 + repeat
                )
                observed = np.zeros(data.n_nodes, dtype=bool)
                observed[rng.choice(data.n_nodes, n_observed, replace=False)] = True
                validation_masks.append(observed)

            selected: dict[str, float | int] = {}
            for method, grid in grids.items():
                scores = []
                for parameter in grid:
                    trial_scores = []
                    for observed in validation_masks:
                        prediction = _interpolate(
                            method,
                            truth,
                            observed,
                            parameter=parameter,
                            laplacian=laplacian,
                            eigenvectors=eigenvectors,
                            distances=distances,
                        )
                        trial_scores.append(
                            regression_metrics(truth[~observed], prediction[~observed])["rmse"]
                        )
                    scores.append(float(np.mean(trial_scores)))
                selected[method] = grid[int(np.argmin(scores))]

            for repeat in range(test_repeats):
                rng = np.random.default_rng(
                    seed + 1_000_000 + signal_index * 100_000 + ratio_index * 10_000 + repeat
                )
                observed = np.zeros(data.n_nodes, dtype=bool)
                observed[rng.choice(data.n_nodes, n_observed, replace=False)] = True
                methods: dict[str, float | int | None] = {
                    "mean": None,
                    "nearest": None,
                    **selected,
                }
                for method, parameter in methods.items():
                    prediction = _interpolate(
                        method,
                        truth,
                        observed,
                        parameter=parameter,
                        laplacian=laplacian,
                        eigenvectors=eigenvectors,
                        distances=distances,
                    )
                    metrics = regression_metrics(truth[~observed], prediction[~observed])
                    records.append(
                        {
                            "signal": signal_name,
                            "observed_ratio": float(ratio),
                            "method": method,
                            "parameter": parameter,
                            "trial": repeat,
                            **metrics,
                        }
                    )

    summary = summarize_records(
        records,
        group_keys=("signal", "observed_ratio", "method", "parameter"),
        metric_keys=("rmse", "mae", "relative_l2", "r2"),
    )
    return BenchmarkResult(
        task="interpolation",
        dataset_id=data.dataset_id,
        config={
            "observed_ratios": list(observed_ratios),
            "validation_repeats": validation_repeats,
            "test_repeats": test_repeats,
            "seed": seed,
            "zero_mode_retained": True,
        },
        records=records,
        summary=summary,
    )


def run_compression(
    dataset: str | GraphSignalDataset,
    *,
    component_ratios: Iterable[float] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0),
    seed: int = 42,
) -> BenchmarkResult:
    """Compare graph low-pass compression with equal-dimensional controls."""

    component_ratios = tuple(component_ratios)
    data = resolve_dataset(dataset)
    _, eigenvectors = graph_fourier_basis(data)
    counts = unique_component_counts(data.n_nodes, component_ratios)
    daily = np.asarray(data.temporal_signals["daily_temperature_midrange_c"], dtype=float)
    medians = np.nanmedian(daily, axis=0)
    rows, columns = np.where(~np.isfinite(daily))
    daily = daily.copy()
    daily[rows, columns] = medians[columns]
    pca_mean = np.mean(daily, axis=0)
    _, _, right_vectors = np.linalg.svd(daily - pca_mean, full_matrices=False)
    pca_basis = right_vectors.T
    random_basis, _ = np.linalg.qr(
        np.random.default_rng(seed).normal(size=(data.n_nodes, data.n_nodes))
    )
    records: list[dict[str, Any]] = []

    for signal_name in _SIGNAL_NAMES:
        truth = np.asarray(data.signals[signal_name], dtype=float)
        coefficients = eigenvectors.T @ truth
        for count in counts:
            gft = lowpass_reconstruct(truth, eigenvectors, count)
            oracle_indices = [0]
            if count > 1:
                ranked = np.argsort(np.square(coefficients[1:]))[::-1] + 1
                oracle_indices.extend(ranked[: count - 1].tolist())
            oracle_coefficients = np.zeros_like(coefficients)
            oracle_coefficients[oracle_indices] = coefficients[oracle_indices]
            predictions = {
                "gft_lowpass": gft,
                "gft_oracle_zero_retained": eigenvectors @ oracle_coefficients,
                "pca": pca_mean
                + pca_basis[:, :count] @ (pca_basis[:, :count].T @ (truth - pca_mean)),
                "random_projection": random_basis[:, :count] @ (random_basis[:, :count].T @ truth),
            }
            for method, prediction in predictions.items():
                metrics = regression_metrics(truth, prediction)
                squared_error = float(np.sum(np.square(prediction - truth)))
                energy = max(float(np.sum(np.square(truth))), np.finfo(float).eps)
                records.append(
                    {
                        "signal": signal_name,
                        "method": method,
                        "n_components": count,
                        "component_ratio": count / data.n_nodes,
                        "energy_retained": 1.0 - squared_error / energy,
                        **metrics,
                    }
                )

    return BenchmarkResult(
        task="compression",
        dataset_id=data.dataset_id,
        config={
            "component_ratios": list(component_ratios),
            "seed": seed,
            "zero_mode_retained": True,
            "pca_fit_samples": "all 2025 daily signals; missing values use station medians",
        },
        records=records,
        summary=list(records),
    )
