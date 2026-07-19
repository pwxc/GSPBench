"""Classical machine-learning benchmarks on graph signals."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import spearmanr

from ..analysis import graph_fourier_basis
from ..models import GraphSignalDataset
from ._common import (
    BenchmarkResult,
    impute_from_training,
    regression_metrics,
    require_sklearn,
    resolve_dataset,
    summarize_records,
    unique_component_counts,
)


def _local_season_labels(dataset_id: str, dates: np.ndarray) -> np.ndarray:
    northern = {
        12: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1,
        6: 2,
        7: 2,
        8: 2,
        9: 3,
        10: 3,
        11: 3,
    }
    southern = {
        12: 2,
        1: 2,
        2: 2,
        3: 3,
        4: 3,
        5: 3,
        6: 0,
        7: 0,
        8: 0,
        9: 1,
        10: 1,
        11: 1,
    }
    mapping = southern if dataset_id.startswith("australia") else northern
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    return np.asarray([mapping[int(month)] for month in months], dtype=int)


def _spectral_sample_statistics(
    samples: np.ndarray, eigenvectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = samples @ eigenvectors
    energy = np.square(coefficients)
    totals = np.maximum(energy.sum(axis=1, keepdims=True), np.finfo(float).eps)
    cumulative = np.cumsum(energy, axis=1) / totals
    k95 = np.argmax(cumulative >= 0.95, axis=1) + 1
    auc = trapezoid(cumulative, dx=1.0 / max(1, samples.shape[1] - 1), axis=1)
    return k95.astype(float), auc


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _classification_representation(
    name: str,
    n_components: int,
    train: np.ndarray,
    test: np.ndarray,
    eigenvectors: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if name == "raw":
        return train, test
    if name == "gft":
        basis = eigenvectors[:, :n_components]
        return train @ basis, test @ basis

    if name == "pca":
        from sklearn.decomposition import PCA

        transform = PCA(n_components=n_components, svd_solver="full", random_state=seed)
    elif name == "random_projection":
        from sklearn.random_projection import GaussianRandomProjection

        transform = GaussianRandomProjection(n_components=n_components, random_state=seed)
    else:
        raise ValueError(f"unknown representation {name!r}")
    return transform.fit_transform(train), transform.transform(test)


def _classification_models(seed: int, selected: Iterable[str]) -> dict[str, Any]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2_000, random_state=seed)
        ),
        "rbf_svm": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(SVC(kernel="rbf", random_state=seed), method="sigmoid", cv=3),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1
        ),
    }
    unknown = set(selected) - set(models)
    if unknown:
        raise ValueError(f"unknown classification models: {sorted(unknown)}")
    return {name: models[name] for name in selected}


def run_season_classification(
    dataset: str | GraphSignalDataset,
    *,
    component_ratios: Iterable[float] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0),
    representations: Iterable[str] = ("gft", "pca", "random_projection"),
    models: Iterable[str] = ("dummy", "logistic", "rbf_svm", "random_forest"),
    seed: int = 42,
) -> BenchmarkResult:
    """Classify each day into the four local meteorological seasons."""

    component_ratios = tuple(component_ratios)
    representations = tuple(representations)
    models = tuple(models)
    require_sklearn()
    from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

    data = resolve_dataset(dataset)
    _, eigenvectors = graph_fourier_basis(data)
    daily = np.asarray(data.temporal_signals["daily_temperature_midrange_c"], dtype=float)
    dates = np.asarray(data.dates)
    labels = _local_season_labels(data.dataset_id, dates)
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    test_month_folds = ((1, 4, 7, 10), (2, 5, 8, 11), (3, 6, 9, 12))
    counts = unique_component_counts(data.n_nodes, component_ratios)
    representation_specs = [("raw", data.n_nodes)] + [
        (name, count) for name in representations for count in counts
    ]
    records: list[dict[str, Any]] = []

    for fold, test_months in enumerate(test_month_folds):
        test_mask = np.isin(months, test_months)
        train, test = impute_from_training(daily[~test_mask], daily[test_mask])
        y_train, y_test = labels[~test_mask], labels[test_mask]
        k95, auc = _spectral_sample_statistics(test, eigenvectors)

        for representation, count in representation_specs:
            train_features, test_features = _classification_representation(
                representation, count, train, test, eigenvectors, seed + fold
            )
            for model_name, model in _classification_models(seed + fold, models).items():
                model.fit(train_features, y_train)
                prediction = model.predict(test_features)
                probabilities = model.predict_proba(test_features)
                class_columns = {int(label): index for index, label in enumerate(model.classes_)}
                aligned = np.full((y_test.size, 4), np.finfo(float).eps)
                for label, column in class_columns.items():
                    aligned[:, label] = probabilities[:, column]
                aligned /= aligned.sum(axis=1, keepdims=True)
                sample_loss = -np.log(
                    np.maximum(aligned[np.arange(y_test.size), y_test], np.finfo(float).eps)
                )
                records.append(
                    {
                        "fold": fold,
                        "representation": representation,
                        "n_components": count,
                        "component_ratio": count / data.n_nodes,
                        "model": model_name,
                        "macro_f1": float(f1_score(y_test, prediction, average="macro")),
                        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
                        "log_loss": float(log_loss(y_test, aligned, labels=np.arange(4))),
                        "spearman_k95_loss": _safe_spearman(k95, sample_loss),
                        "spearman_auc_loss": _safe_spearman(auc, sample_loss),
                    }
                )

    summary = summarize_records(
        records,
        group_keys=("representation", "n_components", "component_ratio", "model"),
        metric_keys=(
            "macro_f1",
            "balanced_accuracy",
            "log_loss",
            "spearman_k95_loss",
            "spearman_auc_loss",
        ),
    )
    return BenchmarkResult(
        task="season_classification",
        dataset_id=data.dataset_id,
        config={
            "labels": ["winter", "spring", "summer", "autumn"],
            "test_month_folds": [list(values) for values in test_month_folds],
            "component_ratios": list(component_ratios),
            "representations": ["raw", *representations],
            "models": list(models),
            "seed": seed,
            "zero_mode_retained": True,
        },
        records=records,
        summary=summary,
    )


def _causal_impute(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = values.copy()
    histories: list[list[float]] = [[] for _ in range(values.shape[1])]
    for day in range(values.shape[0]):
        observed_today = values[day, np.isfinite(values[day])]
        current_fallback = float(np.median(observed_today)) if observed_today.size else 0.0
        for node in range(values.shape[1]):
            if np.isfinite(values[day, node]):
                histories[node].append(float(values[day, node]))
            elif histories[node]:
                result[day, node] = float(np.median(histories[node]))
            else:
                result[day, node] = current_fallback
    return result


def _forecast_representation(
    name: str,
    n_components: int,
    train_windows: np.ndarray,
    test_windows: np.ndarray,
    eigenvectors: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if name == "raw":
        return train_windows.reshape(train_windows.shape[0], -1), test_windows.reshape(
            test_windows.shape[0], -1
        )
    if name == "gft":
        basis = eigenvectors[:, :n_components]
        train = train_windows @ basis
        test = test_windows @ basis
        return train.reshape(train.shape[0], -1), test.reshape(test.shape[0], -1)

    train_days = train_windows.reshape(-1, train_windows.shape[-1])
    test_days = test_windows.reshape(-1, test_windows.shape[-1])
    if name == "pca":
        from sklearn.decomposition import PCA

        transform = PCA(n_components=n_components, svd_solver="full", random_state=seed)
    elif name == "random_projection":
        from sklearn.random_projection import GaussianRandomProjection

        transform = GaussianRandomProjection(n_components=n_components, random_state=seed)
    else:
        raise ValueError(f"unknown representation {name!r}")
    train = transform.fit_transform(train_days).reshape(train_windows.shape[0], -1)
    test = transform.transform(test_days).reshape(test_windows.shape[0], -1)
    return train, test


def _graph_diffusion_features(windows: np.ndarray, laplacian: np.ndarray) -> np.ndarray:
    features = []
    for lag in range(windows.shape[1]):
        signal = windows[:, lag, :]
        first = signal @ laplacian
        second = first @ laplacian
        features.extend((signal, first, second))
    return np.concatenate(features, axis=1)


def _forecast_models(seed: int, selected: Iterable[str]) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "ridge_var": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_features="sqrt", random_state=seed, n_jobs=-1
        ),
        "graph_diffusion_ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }
    unknown = set(selected) - set(models)
    if unknown:
        raise ValueError(f"unknown forecast models: {sorted(unknown)}")
    return {name: models[name] for name in selected}


def run_next_day_forecasting(
    dataset: str | GraphSignalDataset,
    *,
    history: int = 7,
    n_splits: int = 5,
    gap: int = 7,
    component_ratios: Iterable[float] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0),
    representations: Iterable[str] = ("gft", "pca", "random_projection"),
    models: Iterable[str] = ("ridge_var", "random_forest", "graph_diffusion_ridge"),
    seed: int = 42,
) -> BenchmarkResult:
    """Forecast the next full graph signal from the previous seven days."""

    component_ratios = tuple(component_ratios)
    representations = tuple(representations)
    models = tuple(models)
    require_sklearn()
    from sklearn.model_selection import TimeSeriesSplit

    data = resolve_dataset(dataset)
    _, eigenvectors = graph_fourier_basis(data)
    raw_daily = np.asarray(data.temporal_signals["daily_temperature_midrange_c"], dtype=float)
    observed_mask = np.asarray(data.temporal_signals["observation_mask"], dtype=bool)
    daily = _causal_impute(raw_daily)
    windows = np.stack([daily[index - history : index] for index in range(history, len(daily))])
    targets = daily[history:]
    target_masks = observed_mask[history:]
    counts = unique_component_counts(data.n_nodes, component_ratios)
    representation_specs = [("raw", data.n_nodes)] + [
        (name, count) for name in representations for count in counts
    ]
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    laplacian = data.laplacian("normalized").toarray()
    records: list[dict[str, Any]] = []

    for fold, (train_indices, test_indices) in enumerate(splitter.split(windows)):
        y_train, y_test = targets[train_indices], targets[test_indices]
        y_mask = target_masks[test_indices]
        k95, auc = _spectral_sample_statistics(y_test, eigenvectors)

        persistence = windows[test_indices, -1]
        observed_truth = y_test[y_mask]
        persistence_metrics = regression_metrics(observed_truth, persistence[y_mask])
        per_day_rmse = np.asarray(
            [
                np.sqrt(
                    np.mean(np.square(persistence[row, y_mask[row]] - y_test[row, y_mask[row]]))
                )
                for row in range(len(test_indices))
            ]
        )
        records.append(
            {
                "fold": fold,
                "representation": "raw",
                "n_components": data.n_nodes,
                "component_ratio": 1.0,
                "model": "persistence",
                **persistence_metrics,
                "spearman_k95_loss": _safe_spearman(k95, per_day_rmse),
                "spearman_auc_loss": _safe_spearman(auc, per_day_rmse),
            }
        )

        for representation, count in representation_specs:
            train_features, test_features = _forecast_representation(
                representation,
                count,
                windows[train_indices],
                windows[test_indices],
                eigenvectors,
                seed + fold,
            )
            for model_name, model in _forecast_models(seed + fold, models).items():
                model_train_features = train_features
                model_test_features = test_features
                if model_name == "graph_diffusion_ridge":
                    if representation != "raw":
                        continue
                    model_train_features = _graph_diffusion_features(
                        windows[train_indices], laplacian
                    )
                    model_test_features = _graph_diffusion_features(
                        windows[test_indices], laplacian
                    )
                model.fit(model_train_features, y_train)
                prediction = np.asarray(model.predict(model_test_features), dtype=float)
                metrics = regression_metrics(observed_truth, prediction[y_mask])
                per_day_rmse = np.asarray(
                    [
                        np.sqrt(
                            np.mean(
                                np.square(prediction[row, y_mask[row]] - y_test[row, y_mask[row]])
                            )
                        )
                        for row in range(len(test_indices))
                    ]
                )
                record = {
                    "fold": fold,
                    "representation": representation,
                    "n_components": count,
                    "component_ratio": count / data.n_nodes,
                    "model": model_name,
                    **metrics,
                    "spearman_k95_loss": _safe_spearman(k95, per_day_rmse),
                    "spearman_auc_loss": _safe_spearman(auc, per_day_rmse),
                }
                quantiles = np.quantile(k95, (0.25, 0.50, 0.75))
                bins = np.digitize(k95, quantiles, right=True)
                for quartile in range(4):
                    values = per_day_rmse[bins == quartile]
                    record[f"rmse_k95_q{quartile + 1}"] = (
                        float(np.mean(values)) if values.size else float("nan")
                    )
                records.append(record)

    summary = summarize_records(
        records,
        group_keys=("representation", "n_components", "component_ratio", "model"),
        metric_keys=(
            "rmse",
            "mae",
            "relative_l2",
            "r2",
            "spearman_k95_loss",
            "spearman_auc_loss",
        ),
    )
    return BenchmarkResult(
        task="next_day_forecasting",
        dataset_id=data.dataset_id,
        config={
            "history": history,
            "n_splits": n_splits,
            "gap": gap,
            "component_ratios": list(component_ratios),
            "representations": ["raw", *representations],
            "models": ["persistence", *models],
            "seed": seed,
            "imputation": "causal expanding station median; current-day spatial median fallback",
            "zero_mode_retained": True,
        },
        records=records,
        summary=summary,
    )


def _best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return float("inf")
    f1 = (
        2.0
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], np.finfo(float).eps)
    )
    return float(thresholds[int(np.argmax(f1))])


def _anomaly_scores(
    corrupted: np.ndarray,
    method: str,
    *,
    basis: np.ndarray | None,
    center: np.ndarray | None,
    contamination: float,
    coordinates: np.ndarray,
    elevation: np.ndarray,
    seed: int,
) -> np.ndarray:
    if method == "robust_zscore":
        median = float(np.median(corrupted))
        mad = max(float(np.median(np.abs(corrupted - median))), np.finfo(float).eps)
        return np.abs(corrupted - median) / (1.4826 * mad)
    if method == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        static = np.column_stack((coordinates, elevation))
        static = (static - np.nanmedian(static, axis=0)) / np.maximum(
            np.nanstd(static, axis=0), np.finfo(float).eps
        )
        features = np.column_stack((corrupted, static))
        model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=100)
        model.fit(features)
        return -model.decision_function(features)
    if basis is None:
        raise ValueError("a reconstruction basis is required")
    if center is None:
        reconstruction = basis @ (basis.T @ corrupted)
    else:
        reconstruction = center + basis @ (basis.T @ (corrupted - center))
    return np.abs(corrupted - reconstruction)


def run_anomaly_detection(
    dataset: str | GraphSignalDataset,
    *,
    anomaly_ratios: Iterable[float] = (0.01, 0.05, 0.10),
    amplitudes_c: Iterable[float] = (5.0, 10.0),
    component_ratios: Iterable[float] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0),
    validation_repeats: int = 10,
    test_repeats: int = 30,
    seed: int = 42,
) -> BenchmarkResult:
    """Detect controlled synthetic node anomalies on real daily signals."""

    anomaly_ratios = tuple(anomaly_ratios)
    amplitudes_c = tuple(amplitudes_c)
    component_ratios = tuple(component_ratios)
    require_sklearn()
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    data = resolve_dataset(dataset)
    _, eigenvectors = graph_fourier_basis(data)
    daily = _causal_impute(data.temporal_signals["daily_temperature_midrange_c"])
    pca_center = np.mean(daily, axis=0)
    _, _, right_vectors = np.linalg.svd(daily - pca_center, full_matrices=False)
    pca_basis = right_vectors.T
    random_basis, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(data.n_nodes,) * 2))
    counts = unique_component_counts(data.n_nodes, component_ratios)
    specs: list[tuple[str, int | None, np.ndarray | None, np.ndarray | None]] = [
        ("robust_zscore", None, None, None),
        ("isolation_forest", None, None, None),
    ]
    for count in counts:
        specs.extend(
            (
                ("gft_residual", count, eigenvectors[:, :count], None),
                ("pca_residual", count, pca_basis[:, :count], pca_center),
                ("random_residual", count, random_basis[:, :count], None),
            )
        )
    records: list[dict[str, Any]] = []

    for ratio_index, ratio in enumerate(anomaly_ratios):
        n_anomalies = max(1, int(round(float(ratio) * data.n_nodes)))
        for amplitude_index, amplitude in enumerate(amplitudes_c):
            validation: dict[tuple[str, int | None], tuple[list[np.ndarray], list[np.ndarray]]] = {
                (method, count): ([], []) for method, count, _, _ in specs
            }
            for repeat in range(validation_repeats):
                trial_seed = seed + ratio_index * 100_000 + amplitude_index * 10_000 + repeat
                rng = np.random.default_rng(trial_seed)
                clean = daily[int(rng.integers(0, len(daily)))]
                labels = np.zeros(data.n_nodes, dtype=int)
                anomalous = rng.choice(data.n_nodes, n_anomalies, replace=False)
                labels[anomalous] = 1
                corrupted = clean.copy()
                corrupted[anomalous] += rng.choice((-1.0, 1.0), n_anomalies) * float(amplitude)
                for method, count, basis, center in specs:
                    scores = _anomaly_scores(
                        corrupted,
                        method,
                        basis=basis,
                        center=center,
                        contamination=float(ratio),
                        coordinates=data.coordinates,
                        elevation=data.elevation_m,
                        seed=trial_seed,
                    )
                    validation[(method, count)][0].append(labels)
                    validation[(method, count)][1].append(scores)
            thresholds = {
                key: _best_f1_threshold(np.concatenate(labels), np.concatenate(scores))
                for key, (labels, scores) in validation.items()
            }

            for repeat in range(test_repeats):
                trial_seed = (
                    seed + 1_000_000 + ratio_index * 100_000 + amplitude_index * 10_000 + repeat
                )
                rng = np.random.default_rng(trial_seed)
                clean = daily[int(rng.integers(0, len(daily)))]
                labels = np.zeros(data.n_nodes, dtype=int)
                anomalous = rng.choice(data.n_nodes, n_anomalies, replace=False)
                labels[anomalous] = 1
                corrupted = clean.copy()
                corrupted[anomalous] += rng.choice((-1.0, 1.0), n_anomalies) * float(amplitude)
                for method, count, basis, center in specs:
                    scores = _anomaly_scores(
                        corrupted,
                        method,
                        basis=basis,
                        center=center,
                        contamination=float(ratio),
                        coordinates=data.coordinates,
                        elevation=data.elevation_m,
                        seed=trial_seed,
                    )
                    threshold = thresholds[(method, count)]
                    prediction = scores >= threshold
                    records.append(
                        {
                            "anomaly_ratio": float(ratio),
                            "amplitude_c": float(amplitude),
                            "method": method,
                            "n_components": count,
                            "component_ratio": (count / data.n_nodes if count is not None else 1.0),
                            "trial": repeat,
                            "auroc": float(roc_auc_score(labels, scores)),
                            "average_precision": float(average_precision_score(labels, scores)),
                            "f1": float(f1_score(labels, prediction)),
                            "validation_threshold": threshold,
                        }
                    )

    summary = summarize_records(
        records,
        group_keys=(
            "anomaly_ratio",
            "amplitude_c",
            "method",
            "n_components",
            "component_ratio",
        ),
        metric_keys=("auroc", "average_precision", "f1"),
    )
    return BenchmarkResult(
        task="anomaly_detection",
        dataset_id=data.dataset_id,
        config={
            "anomaly_ratios": list(anomaly_ratios),
            "amplitudes_c": list(amplitudes_c),
            "component_ratios": list(component_ratios),
            "validation_repeats": validation_repeats,
            "test_repeats": test_repeats,
            "seed": seed,
            "label_provenance": "controlled synthetic anomalies on real 2025 daily signals",
            "zero_mode_retained": True,
        },
        records=records,
        summary=summary,
    )
