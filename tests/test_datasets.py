from __future__ import annotations

import hashlib
import json
from importlib import resources

import numpy as np
from scipy.sparse.csgraph import connected_components

from gspbench import available_datasets, load_dataset


EXPECTED_NODES = {
    "us_weather_2025": 144,
    "australia_weather_2025": 126,
}


def test_catalog_and_license_records_are_complete() -> None:
    licenses = json.loads(
        resources.files("gspbench.data").joinpath("DATA_LICENSES.json").read_text(encoding="utf-8")
    )
    assert set(available_datasets()) == set(EXPECTED_NODES)
    for dataset_id in available_datasets():
        record = licenses["datasets"][dataset_id]
        assert record["redistribution_status"] == "allowed"
        assert record["raw_sha256"]
        assert record["processed_sha256"]


def test_packaged_weather_graph_invariants() -> None:
    for dataset_id, expected_nodes in EXPECTED_NODES.items():
        dataset = load_dataset(dataset_id)
        adjacency = dataset.adjacency
        assert dataset.n_nodes == expected_nodes
        assert adjacency.shape == (expected_nodes, expected_nodes)
        assert adjacency.diagonal().sum() == 0
        assert (adjacency - adjacency.T).nnz == 0
        assert np.all(adjacency.data > 0)
        assert np.all(adjacency.data <= 1)
        assert np.all(np.diff(adjacency.indptr) >= 6)
        assert connected_components(adjacency, directed=False)[0] == 1

        rows, columns = adjacency.nonzero()
        distances = dataset.edge_distances_km[rows, columns].A1
        expected = np.exp(
            -np.square(distances)
            / (dataset.local_scales_km[rows] * dataset.local_scales_km[columns])
        )
        assert np.allclose(adjacency[rows, columns].A1, expected)


def test_signals_and_missingness_are_explicit() -> None:
    for dataset_id, expected_nodes in EXPECTED_NODES.items():
        dataset = load_dataset(dataset_id)
        daily = dataset.temporal_signals["daily_temperature_midrange_c"]
        observed = dataset.temporal_signals["observation_mask"]
        assert daily.shape == (365, expected_nodes)
        assert observed.shape == daily.shape
        assert np.array_equal(observed, np.isfinite(daily))
        assert np.all(np.isfinite(dataset.signals["winter_temperature_midrange_c"]))
        assert np.all(np.isfinite(dataset.signals["summer_temperature_midrange_c"]))
        assert dataset.observation_counts["winter_temperature_midrange_c"].min() >= 25
        assert dataset.observation_counts["summer_temperature_midrange_c"].min() >= 25
        assert dataset.observation_counts["annual_temperature_midrange_c"].min() >= 300


def test_processed_archive_checksum() -> None:
    licenses = json.loads(
        resources.files("gspbench.data").joinpath("DATA_LICENSES.json").read_text(encoding="utf-8")
    )
    for dataset_id in available_datasets():
        resource = resources.files("gspbench.data").joinpath(f"{dataset_id}.npz")
        with resources.as_file(resource) as path:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == licenses["datasets"][dataset_id]["processed_sha256"]
