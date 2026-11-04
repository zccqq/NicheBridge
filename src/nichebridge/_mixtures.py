"""Synthetic mixture generation with spatial-library matching."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def make_target_matched_pseudospots(
    counts,
    labels,
    types,
    target_library_sizes,
    n_spots,
    min_cells,
    max_cells,
    seed,
    strategy="balanced",
    active_max=8,
    single_fraction=0.3,
):
    """Generate labelled mixtures with count depth sampled from target spots.

    Reference cells determine the composition and expression profile. Counts are
    then sampled from that profile at an empirical spatial library size. This
    matches synthetic count depth to the observed spatial library sizes.

    Returns
    -------
    mixtures : ndarray
        Integer-like synthetic count matrix with shape ``(n_spots, n_genes)``.
    cell_fraction : ndarray
        Cell-count fractions for each synthetic spot and reference type.
    rna_fraction : ndarray
        Raw-RNA fractions before resampling to the target library depth.
    """
    if sparse.issparse(counts):
        counts = counts.tocsr()
    else:
        counts = sparse.csr_matrix(np.asarray(counts))
    labels = np.asarray(labels).astype(str)
    types = [str(x) for x in types]
    target_library_sizes = np.asarray(target_library_sizes, dtype=float).ravel()
    if counts.ndim != 2 or counts.shape[0] != len(labels):
        raise ValueError("counts and labels have incompatible shapes")
    if len(types) == 0 or n_spots < 1:
        raise ValueError("types and n_spots must be nonempty")
    if (
        len(target_library_sizes) == 0
        or not np.isfinite(target_library_sizes).all()
        or (target_library_sizes <= 0).any()
    ):
        raise ValueError("target_library_sizes must contain finite positive values")
    if min_cells < 1 or max_cells < min_cells:
        raise ValueError("Invalid cell-count range")
    rng = np.random.default_rng(seed)
    by_type = [np.flatnonzero(labels == value) for value in types]
    if any(len(indices) == 0 for indices in by_type):
        missing = [types[i] for i, indices in enumerate(by_type) if len(indices) == 0]
        raise ValueError(f"Missing reference cells for types: {missing}")

    n_types = len(types)
    libraries = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    mixtures = np.zeros((n_spots, counts.shape[1]), dtype=np.float32)
    cell_fraction = np.zeros((n_spots, n_types), dtype=np.float32)
    rna_fraction = np.zeros_like(cell_fraction)
    for spot in range(n_spots):
        if strategy == "balanced":
            draw = rng.random()
            if draw < single_fraction or min(active_max, n_types) == 1:
                active = 1
            elif draw < 0.7:
                active = int(rng.integers(2, min(active_max, 4, n_types) + 1))
            else:
                upper = min(active_max, n_types)
                lower = min(5, upper)
                active = (
                    lower if upper <= lower else int(rng.integers(lower, upper + 1))
                )
        else:
            active = (
                1
                if rng.random() < 0.2
                else int(rng.integers(1, min(active_max, n_types) + 1))
            )
        active = max(1, min(active, n_types))
        n_cells = int(rng.integers(min_cells, max_cells + 1))
        selected = rng.choice(n_types, active, replace=False)
        weights = rng.dirichlet(np.full(active, 0.5))
        chosen_types = rng.choice(selected, size=n_cells, p=weights)
        chosen_cells = np.asarray([rng.choice(by_type[t]) for t in chosen_types])
        cell_fraction[spot] = np.bincount(chosen_types, minlength=n_types) / n_cells
        mass = np.bincount(
            chosen_types, weights=libraries[chosen_cells], minlength=n_types
        )
        rna_fraction[spot] = mass / max(float(mass.sum()), 1e-12)
        profile = (
            np.asarray(counts[chosen_cells].sum(axis=0)).ravel().astype(np.float64)
        )
        profile_sum = float(profile.sum())
        if not np.isfinite(profile_sum) or profile_sum <= 0:
            raise ValueError("A sampled reference mixture has no counts")
        profile /= profile_sum
        library = int(max(1, round(float(rng.choice(target_library_sizes)))))
        mixtures[spot] = rng.multinomial(library, profile).astype(np.float32)
    return mixtures, cell_fraction, rna_fraction
