"""Spatial domain identification and identity-aligned niche construction.

Expression and embedding matrices use observations by features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _frame(values, name="values"):
    if hasattr(values, "obs_names"):
        matrix = values.X.toarray() if sparse.issparse(values.X) else values.X
        values = pd.DataFrame(matrix, index=values.obs_names, columns=values.var_names)
    elif not isinstance(values, pd.DataFrame):
        matrix = values.toarray() if sparse.issparse(values) else values
        values = pd.DataFrame(matrix)
    if values.empty or not values.index.is_unique or not values.columns.is_unique:
        raise ValueError(f"{name} must have nonempty unique observation and feature axes")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must be finite")
    return values.astype(float)


def _labels(values, index, name, *, allow_missing=False):
    if isinstance(values, pd.Series):
        if not values.index.is_unique or set(values.index) != set(index):
            raise ValueError(f"{name} observation identifiers differ")
        result = values.reindex(index)
    else:
        if len(values) != len(index):
            raise ValueError(f"{name} length differs from observations")
        result = pd.Series(values, index=index)
    if not allow_missing and result.isna().any():
        raise ValueError(f"{name} cannot contain missing labels")
    return result


def _aligned(values, index, name):
    frame = _frame(values, name)
    if set(frame.index) != set(index):
        raise ValueError(f"{name} observation identifiers differ")
    return frame.reindex(index)


def _knn_graph(values, n_neighbors, groups=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Neighbor features must be a finite two-dimensional matrix")
    if isinstance(n_neighbors, bool) or int(n_neighbors) != n_neighbors or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")
    groups = np.zeros(len(values), dtype=int) if groups is None else np.asarray(groups)
    if len(groups) != len(values):
        raise ValueError("Neighbor group labels must match observations")
    rows, cols, weights = [], [], []
    for group in pd.unique(groups):
        ids = np.flatnonzero(groups == group)
        k = min(int(n_neighbors), len(ids) - 1)
        if k < 1:
            continue
        distance, neighbors = NearestNeighbors(n_neighbors=k + 1).fit(values[ids]).kneighbors(values[ids])
        for local, (distances, indices) in enumerate(zip(distance, neighbors)):
            keep = indices != local
            indices, distances = indices[keep][:k], distances[keep][:k]
            rows.extend([ids[local]] * len(indices))
            cols.extend(ids[indices])
            weights.extend(1.0 / (1.0 + distances))
    return sparse.csr_matrix((weights, (rows, cols)), shape=(len(values), len(values)))


def identify_domains(embedding, *, coordinates=None, slices=None, n_neighbors=15,
                     spatial_weight=0.0, resolution=1.0, seed=42):
    """Louvain clustering of a latent kNN graph, optionally adding spatial edges.

    Spatial edges are restricted to each supplied slice. Latent neighbors may
    connect slices. Requires ``networkx>=3``; labels retain observation IDs.
    """
    import networkx as nx
    frame = _frame(embedding, "embedding")
    if len(frame) < 2 or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Domain clustering requires at least two spots and positive resolution")
    if not np.isfinite(spatial_weight) or spatial_weight < 0:
        raise ValueError("spatial_weight must be finite and nonnegative")
    adjacency = _knn_graph(frame, n_neighbors)
    adjacency = adjacency.maximum(adjacency.T)
    if spatial_weight:
        if coordinates is None:
            raise ValueError("coordinates are required when spatial_weight is positive")
        xy = _aligned(coordinates, frame.index, "coordinates")
        group = None if slices is None else _labels(slices, frame.index, "slices")
        spatial_graph = _knn_graph(xy, n_neighbors, group)
        adjacency = adjacency + spatial_weight * spatial_graph.maximum(spatial_graph.T)
    graph = nx.from_scipy_sparse_array(adjacency)
    communities = nx.community.louvain_communities(graph, resolution=resolution, seed=seed)
    labels = np.empty(len(frame), dtype=int)
    for number, members in enumerate(sorted(communities, key=lambda x: min(x))):
        labels[list(members)] = number
    return pd.Series(labels, index=frame.index, name="NicheBridge_domain")


def assemble_niches(*, phenotype=None, composition=None, embedding=None,
                    domains=None, reconstruction=None, standardize=True):
    """Combine Methods niche fields with explicit IDs and domain one-hot encoding.

    Available fields can be analyzed separately or passed together. Numeric
    columns are optionally z-scored; every block is divided by sqrt(width) so
    thousands of reconstructed genes do not dominate by dimensionality alone.
    """
    inputs = {"phenotype": phenotype, "composition": composition,
              "latent": embedding, "expression": reconstruction}
    inputs = {key: _frame(value, key) for key, value in inputs.items() if value is not None}
    if not inputs:
        raise ValueError("At least one numeric niche field is required")
    index = inputs["latent"].index if "latent" in inputs else next(iter(inputs.values())).index
    if domains is not None:
        labels = _labels(domains, index, "domains")
        inputs["domain"] = pd.get_dummies(labels.astype(str), dtype=float)
    parts = []
    for name, frame in inputs.items():
        frame = _aligned(frame, index, name)
        matrix = frame.to_numpy()
        if standardize:
            sd = matrix.std(axis=0)
            matrix = (matrix - matrix.mean(axis=0)) / np.where(sd > 0, sd, 1)
            matrix /= np.sqrt(matrix.shape[1])
        parts.append(pd.DataFrame(matrix, index=index, columns=[f"{name}:{column}" for column in frame.columns]))
    result = pd.concat(parts, axis=1)
    result.attrs["standardized"] = bool(standardize)
    result.attrs["blocks"] = list(inputs)
    return result


__all__ = ["identify_domains", "assemble_niches"]
