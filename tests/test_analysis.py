from __future__ import annotations

import numpy as np

from gspbench import load_dataset
from gspbench.analysis import bandlimitedness, graph_fourier_basis, lowpass_reconstruct


def test_gft_retains_zero_mode_and_conserves_energy() -> None:
    dataset = load_dataset("us_weather_2025")
    signal = dataset.signals["winter_temperature_midrange_c"]
    _, eigenvectors = graph_fourier_basis(dataset)
    coefficients = eigenvectors.T @ signal
    assert not np.isclose(coefficients[0], 0.0)
    assert np.isclose(np.sum(np.square(coefficients)), np.sum(np.square(signal)))
    reconstructed = lowpass_reconstruct(signal, eigenvectors, 1)
    expected = eigenvectors[:, 0] * coefficients[0]
    assert np.allclose(reconstructed, expected)


def test_bandlimitedness_summary_uses_raw_signal() -> None:
    dataset = load_dataset("australia_weather_2025")
    signal = dataset.signals["summer_temperature_midrange_c"]
    result = bandlimitedness(dataset, signal)
    assert result.zero_mode_energy_ratio > 0
    assert result.effective_bandwidth["k90"] >= 1
    assert result.effective_bandwidth["k90"] <= result.effective_bandwidth["k95"]
    assert result.effective_bandwidth["k95"] <= result.effective_bandwidth["k99"]
    assert np.isclose(result.cumulative_energy[-1], 1.0)


def test_both_laplacians_are_symmetric() -> None:
    dataset = load_dataset("us_weather_2025")
    for kind in ("normalized", "combinatorial"):
        laplacian = dataset.laplacian(kind)
        assert np.allclose(laplacian.toarray(), laplacian.toarray().T)
        assert np.linalg.eigvalsh(laplacian.toarray()).min() >= -1e-10
