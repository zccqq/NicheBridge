"""Ordered expression inputs and reference-only candidate construction."""

import hashlib
from pathlib import Path

import nichebridge
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.decomposition import TruncatedSVD
from nichebridge._data import count_matrix


def backend_identity():
    root = Path(nichebridge.__file__).parent
    paths = [root / name for name in (
        "_config.py", "_bridge.py", "_data.py", "_mixtures.py", "_network.py",
        "joint/config.py", "joint/data.py", "joint/model.py", "joint/network.py")]
    return {"version": nichebridge.__version__, "path": str(root),
            "sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(paths)}}


def validate_axes(adata):
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Observation and gene identifiers must be unique")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Expression inputs must be nonempty")
    if not all(isinstance(x, str) for x in adata.obs_names):
        raise ValueError("Observation identifiers must be strings")


def raw_counts(adata, genes, layer):
    validate_axes(adata)
    return sparse.csr_matrix(count_matrix(adata, layer=layer, genes=genes), dtype=np.float32)


def bulk_profiles(adata, genes, layer, scale):
    validate_axes(adata)
    if scale not in ("tpm", "log1p_tpm"):
        raise ValueError("bulk_scale must explicitly be 'tpm' or 'log1p_tpm'")
    subset = adata[:, genes]
    x = subset.X if layer is None else subset.layers[layer]
    x = x.toarray() if sparse.issparse(x) else np.asarray(x)
    x = np.array(x, dtype=np.float64, copy=True)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Bulk expression must be finite and nonnegative")
    if scale == "log1p_tpm":
        with np.errstate(over="ignore"):
            x = np.expm1(x)
    totals = x.sum(axis=1, keepdims=True)
    if not np.isfinite(x).all() or (totals <= 0).any():
        raise ValueError("Bulk samples must have finite positive panel mass")
    return (x / totals).astype(np.float32)


def features(x, config):
    x = sparse.csr_matrix(x, dtype=np.float32)
    lib = np.asarray(x.sum(axis=1)).ravel()
    x = x.multiply((config.target_sum / np.maximum(lib, 1))[:, None]).tocsr()
    x.data = np.log1p(x.data) / config.log_scale
    return x


def normalized(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def make_projection(reference, informative, config):
    rank = min(config.relation_dim, reference.shape[1] - 1, max(1, int(informative.sum()) - 1))
    if rank < 1:
        raise ValueError("At least two shared genes are required")
    svd = TruncatedSVD(n_components=rank, random_state=config.seed)
    svd.fit(features(reference[informative], config))
    projection = svd.components_.T.astype(np.float32)
    return projection, normalized(features(reference, config) @ projection)


def make_spatial_projection(spatial, config):
    """Build a deterministic expression key space without a single-cell reference."""
    matrix = features(spatial, config)
    rank = min(config.relation_dim, matrix.shape[1] - 1, max(1, matrix.shape[0] - 1))
    if rank < 1:
        raise ValueError("At least two spatial rows and genes are required")
    svd = TruncatedSVD(n_components=rank, random_state=config.seed)
    keys = svd.fit_transform(matrix).astype(np.float32)
    projection = svd.components_.T.astype(np.float32)
    return projection, normalized(keys)


def candidates(query_keys, reference_keys, codes, informative, n_types, k):
    indices = np.zeros((len(query_keys), n_types, k), dtype=np.int64)
    valid = np.zeros_like(indices, dtype=bool)
    for t in range(n_types):
        pool = np.flatnonzero((codes == t) & informative)
        if len(pool) == 0:
            raise ValueError(f"Reference type {t} has no informative cells")
        width = min(k, len(pool))
        _, local = cKDTree(reference_keys[pool]).query(query_keys, k=width, workers=1)
        local = np.asarray(local).reshape(len(query_keys), width)
        indices[:, t, :width] = pool[local]
        indices[:, t, width:] = pool[0]
        valid[:, t, :width] = True
    return indices, valid


def spatial_edges(adata, informative, k, group_key=None):
    if "spatial" not in adata.obsm:
        return np.empty((0, 2), dtype=np.int64)
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2 or not np.isfinite(coords[:, :2]).all():
        raise ValueError("Spatial coordinates must have two finite columns")
    if group_key is not None:
        if adata.obs[group_key].isna().any():
            raise ValueError("Spatial section identifiers cannot be missing")
        groups = adata.obs[group_key].astype(str).to_numpy()
    else:
        groups = np.full(adata.n_obs, "section")
    edges = []
    for group in np.unique(groups):
        ids = np.flatnonzero((groups == group) & informative)
        if len(ids) < 2:
            continue
        width = min(k + 1, len(ids))
        _, near = cKDTree(coords[ids, :2]).query(coords[ids, :2], k=width)
        for i, row in enumerate(near):
            edges.extend((ids[i], ids[j]) for j in row if j != i)
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def fingerprint(x, names, genes):
    x = sparse.csr_matrix(x, dtype=np.float32)
    x.sort_indices()
    digest = hashlib.sha256()
    for item in (np.asarray(x.shape, dtype=np.int64), x.data,
                 x.indices.astype(np.int64), x.indptr.astype(np.int64)):
        digest.update(item.tobytes())
    for axis in (names, genes):
        for value in axis:
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()
