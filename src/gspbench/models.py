"""Core data structures used by GSPBench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy import sparse


@dataclass(frozen=True, slots=True)
class GraphSignalDataset:
    """A weighted graph, its node metadata, and signals defined on its nodes."""

    dataset_id: str
    adjacency: sparse.csr_matrix
    edge_distances_km: sparse.csr_matrix
    local_scales_km: np.ndarray
    coordinates: np.ndarray
    elevation_m: np.ndarray
    station_ids: np.ndarray
    station_names: np.ndarray
    dates: np.ndarray
    signals: Mapping[str, np.ndarray]
    temporal_signals: Mapping[str, np.ndarray]
    observation_counts: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    @property
    def n_nodes(self) -> int:
        """Number of graph nodes."""

        return int(self.adjacency.shape[0])

    @property
    def n_edges(self) -> int:
        """Number of undirected graph edges."""

        return int(self.adjacency.nnz // 2)

    def laplacian(self, kind: str = "normalized") -> sparse.csr_matrix:
        """Return the weighted normalized or combinatorial graph Laplacian."""

        degree = np.asarray(self.adjacency.sum(axis=1)).ravel()
        if kind == "combinatorial":
            return (sparse.diags(degree) - self.adjacency).tocsr()
        if kind != "normalized":
            raise ValueError("kind must be 'normalized' or 'combinatorial'")
        if np.any(degree <= 0):
            raise ValueError("normalized Laplacian is undefined for isolated nodes")
        inv_sqrt = sparse.diags(1.0 / np.sqrt(degree))
        laplacian = sparse.eye(self.n_nodes, format="csr") - (inv_sqrt @ self.adjacency @ inv_sqrt)
        laplacian.eliminate_zeros()
        return laplacian.tocsr()
