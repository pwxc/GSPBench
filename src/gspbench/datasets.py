"""Load the small, processed datasets distributed with GSPBench."""

from __future__ import annotations

import json
from importlib import resources
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy import sparse

from .models import GraphSignalDataset

_DATA_PACKAGE = "gspbench.data"


def _read_json(filename: str) -> dict[str, Any]:
    resource = resources.files(_DATA_PACKAGE).joinpath(filename)
    return json.loads(resource.read_text(encoding="utf-8"))


def available_datasets() -> tuple[str, ...]:
    """Return the stable identifiers of all packaged datasets."""

    catalog = _read_json("catalog.json")
    return tuple(sorted(catalog["datasets"]))


def _csr_from_archive(archive: Any, prefix: str) -> sparse.csr_matrix:
    shape = tuple(int(value) for value in archive[f"{prefix}_shape"])
    return sparse.csr_matrix(
        (
            archive[f"{prefix}_data"],
            archive[f"{prefix}_indices"],
            archive[f"{prefix}_indptr"],
        ),
        shape=shape,
    )


def load_dataset(dataset: str) -> GraphSignalDataset:
    """Load a packaged graph signal dataset by identifier.

    Parameters
    ----------
    dataset:
        One of the values returned by :func:`available_datasets`.
    """

    catalog = _read_json("catalog.json")
    try:
        entry = catalog["datasets"][dataset]
    except KeyError as exc:
        choices = ", ".join(sorted(catalog["datasets"]))
        raise ValueError(f"unknown dataset {dataset!r}; choose one of: {choices}") from exc

    licenses = _read_json("DATA_LICENSES.json")
    license_entry = licenses["datasets"].get(dataset)
    if not license_entry or license_entry.get("redistribution_status") != "allowed":
        raise RuntimeError(f"dataset {dataset!r} is not cleared for redistribution")

    resource = resources.files(_DATA_PACKAGE).joinpath(entry["file"])
    with resources.as_file(resource) as path, np.load(path, allow_pickle=False) as archive:
        adjacency = _csr_from_archive(archive, "adjacency")
        edge_distances = _csr_from_archive(archive, "edge_distances")
        signals = {
            "winter_temperature_midrange_c": archive["winter_temperature_midrange_c"].copy(),
            "summer_temperature_midrange_c": archive["summer_temperature_midrange_c"].copy(),
        }
        temporal_signals = {
            "daily_temperature_midrange_c": archive["daily_temperature_midrange_c"].copy(),
            "observation_mask": archive["observation_mask"].astype(bool, copy=True),
        }
        observation_counts = {
            "winter_temperature_midrange_c": archive["winter_observation_counts"].copy(),
            "summer_temperature_midrange_c": archive["summer_observation_counts"].copy(),
            "annual_temperature_midrange_c": archive["annual_observation_counts"].copy(),
        }
        metadata = dict(entry)
        metadata["license"] = license_entry
        metadata["noaa_disclaimer"] = licenses["noaa_disclaimer"]
        result = GraphSignalDataset(
            dataset_id=dataset,
            adjacency=adjacency,
            edge_distances_km=edge_distances,
            local_scales_km=archive["local_scales_km"].copy(),
            coordinates=archive["coordinates"].copy(),
            elevation_m=archive["elevation_m"].copy(),
            station_ids=archive["station_ids"].copy(),
            station_names=archive["station_names"].copy(),
            dates=archive["dates"].astype("datetime64[D]"),
            signals=MappingProxyType(signals),
            temporal_signals=MappingProxyType(temporal_signals),
            observation_counts=MappingProxyType(observation_counts),
            metadata=MappingProxyType(metadata),
        )

    if result.n_nodes != int(entry["n_nodes"]):
        raise RuntimeError(f"packaged dataset {dataset!r} failed its node-count check")
    return result
