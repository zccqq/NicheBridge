"""Count validation, reference signatures, and labeled synthetic mixtures."""

import numpy as np
import pandas as pd
from scipy import sparse


def count_matrix(adata, layer=None, genes=None):
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Observation and gene identifiers must be unique")
    if genes is not None:
        missing = pd.Index(genes).difference(adata.var_names)
        if len(missing):
            raise ValueError(f"Missing required genes: {missing[:5].tolist()}")
        adata = adata[:, genes]
    values = adata.X if layer is None else adata.layers[layer]
    matrix = sparse.csr_matrix(values, dtype=np.float32)
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Count matrices cannot be empty")
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    if not np.allclose(matrix.data, np.rint(matrix.data), atol=1e-4, rtol=0):
        raise ValueError("The count likelihood requires raw integer counts")
    return matrix


def reference_profiles(counts, labels, types, mode="raw"):
    """Return type profiles and fixed reference-derived relative RNA yields.

    The size-factor mode scales type profiles by raw reference library totals.
    The decoder normalizes the final mixture rather than each stored profile.
    """
    labels = np.asarray(labels)
    profiles, yields = [], []
    for name in types:
        subset = counts[labels == name]
        if subset.shape[0] == 0:
            raise ValueError(f"No reference cells for type {name}")
        raw_mean = np.asarray(subset.mean(axis=0)).ravel().astype(np.float64)
        if mode == "size_factor":
            row_sums = np.asarray(subset.sum(axis=1)).ravel().astype(np.float64)
            normalized = subset.multiply(1.0 / np.maximum(row_sums, 1.0)[:, None])
            mean = np.asarray(normalized.mean(axis=0)).ravel().astype(np.float64)
            total = raw_mean.sum()
        else:
            mean = raw_mean
            total = mean.sum()
        if total <= 0:
            raise ValueError(f"No informative reference counts for type {name}")
        profiles.append((mean + 1e-8) / (total + 1e-8 * len(mean)))
        yields.append(total)
    yields = np.asarray(yields)
    yields /= np.exp(np.log(yields).mean())
    return np.asarray(profiles, dtype=np.float32), yields.astype(np.float32)


def make_pseudospots(
    counts,
    labels,
    types,
    n_spots,
    min_cells,
    max_cells,
    seed,
    strategy="random",
    active_max=5,
    single_fraction=0.3,
):
    """Aggregate raw reference cells and retain cell and measured RNA fractions.

    The balanced strategy mixes pure spots, sparse mixtures, and broader
    mixtures to reduce a mismatch between synthetic and observed spot support.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    by_type = [np.flatnonzero(labels == value) for value in types]
    n_types = len(types)
    row, col = [], []
    cell_fraction = np.zeros((n_spots, n_types), dtype=np.float32)
    rna_fraction = np.zeros_like(cell_fraction)
    libraries = np.asarray(counts.sum(axis=1)).ravel()
    for spot in range(n_spots):
        if strategy == "balanced":
            draw = rng.random()
            if draw < single_fraction:
                active = 1
            elif draw < 0.7:
                active = int(rng.integers(2, min(active_max, 4, n_types) + 1))
            else:
                active = (
                    int(rng.integers(min(5, n_types), min(active_max, n_types) + 1))
                    if n_types >= 5
                    else n_types
                )
            n = int(rng.integers(min_cells, max_cells + 1))
        else:
            n = int(rng.integers(min_cells, max_cells + 1))
            active = (
                1
                if rng.random() < 0.2
                else int(rng.integers(1, min(active_max, 5, n_types) + 1))
            )
        active = max(1, min(active, n_types))
        selected = rng.choice(n_types, active, replace=False)
        weights = rng.dirichlet(np.full(active, 0.5))
        chosen_types = rng.choice(selected, size=n, p=weights)
        chosen_cells = np.asarray([rng.choice(by_type[t]) for t in chosen_types])
        row.extend([spot] * n)
        col.extend(chosen_cells.tolist())
        cell_fraction[spot] = np.bincount(chosen_types, minlength=n_types) / n
        mass = np.bincount(
            chosen_types, weights=libraries[chosen_cells], minlength=n_types
        )
        rna_fraction[spot] = mass / max(mass.sum(), 1e-12)
    allocation = sparse.coo_matrix(
        (np.ones(len(row)), (row, col)), shape=(n_spots, counts.shape[0])
    ).tocsr()
    mixtures = (allocation @ counts).toarray().astype(np.float32)
    valid = mixtures.sum(axis=1) > 0
    if not valid.all():
        mixtures, cell_fraction, rna_fraction = (
            mixtures[valid],
            cell_fraction[valid],
            rna_fraction[valid],
        )
    if len(mixtures) == 0:
        raise ValueError(
            "Synthetic mixture construction produced no informative observations"
        )
    return mixtures, cell_fraction, rna_fraction
