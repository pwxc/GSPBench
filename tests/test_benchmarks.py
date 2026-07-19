from __future__ import annotations

import pytest

from gspbench.benchmarks import run_benchmark


def test_core_benchmarks_smoke() -> None:
    denoising = run_benchmark(
        "denoising",
        "us_weather_2025",
        snr_db=(10,),
        validation_repeats=1,
        test_repeats=2,
        tikhonov_grid=(0.1,),
        heat_grid=(0.1,),
        component_ratios=(0.1,),
    )
    assert denoising.records
    assert all(
        record["parameter"] is not None
        for record in denoising.records
        if record["method"] != "identity"
    )

    interpolation = run_benchmark(
        "interpolation",
        "australia_weather_2025",
        observed_ratios=(0.3,),
        validation_repeats=1,
        test_repeats=1,
        tikhonov_grid=(0.1,),
        component_ratios=(0.1,),
    )
    assert {record["method"] for record in interpolation.records} == {
        "mean",
        "nearest",
        "tikhonov",
        "bandlimited_ls",
    }

    compression = run_benchmark("compression", "us_weather_2025", component_ratios=(0.1, 1.0))
    assert compression.records
    assert all(record["n_components"] >= 1 for record in compression.records)


def test_unknown_benchmark_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown benchmark"):
        run_benchmark("not-a-task", "us_weather_2025")


def test_ml_benchmarks_smoke() -> None:
    pytest.importorskip("sklearn")

    classification = run_benchmark(
        "season_classification",
        "us_weather_2025",
        component_ratios=(0.1,),
        representations=("gft",),
        models=("logistic",),
    )
    assert len(classification.records) == 6

    forecast = run_benchmark(
        "next_day_forecasting",
        "australia_weather_2025",
        n_splits=2,
        component_ratios=(0.1,),
        representations=("gft",),
        models=("ridge_var",),
    )
    assert forecast.records

    anomaly = run_benchmark(
        "anomaly_detection",
        "us_weather_2025",
        anomaly_ratios=(0.05,),
        amplitudes_c=(5.0,),
        component_ratios=(0.1,),
        validation_repeats=1,
        test_repeats=1,
    )
    assert anomaly.records
