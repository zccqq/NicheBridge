"""Spatial composition visualization for NicheBridgeDeconvolver outputs."""

import math

import numpy as np
import pandas as pd


def spatial_proportions(
    adata,
    cell_types=None,
    fraction="cell",
    basis="spatial",
    size=80,
    ncols=3,
    cmap="viridis",
    vmin=0,
    vmax=1,
    invert_y=True,
    show=True,
    figsize=None,
):
    """Plot one spatial proportion map for each selected cell type.

    Coordinates are taken from ``obsm[basis]`` and use its first two columns.
    ``fraction`` selects ``cell``, ``reference_rna``, or ``rna`` fractions.
    ``size`` is the scatter marker area in points squared. All panels share
    the color limits ``vmin`` and ``vmax``; defaults represent proportions
    on their full zero-to-one scale. The coordinate axes have equal aspect.
    Return the matplotlib figure regardless of ``show``.
    """
    import matplotlib.pyplot as plt

    if fraction not in {"cell", "reference_rna", "rna"}:
        raise ValueError("fraction must be cell, reference_rna, or rna")
    key = f"nichebridge_{fraction}_fraction"
    if key not in adata.obsm:
        raise KeyError(f"Missing model output: obsm['{key}']")
    proportions = adata.obsm[key]
    if not isinstance(proportions, pd.DataFrame):
        raise TypeError("Composition outputs must be a DataFrame with named cell types")
    if (
        not proportions.index.equals(adata.obs_names)
        or not proportions.columns.is_unique
    ):
        raise ValueError(
            "Composition identifiers must align with observations and be unique"
        )
    if basis not in adata.obsm:
        raise KeyError(f"Missing coordinates: obsm['{basis}']")
    coordinates = np.asarray(adata.obsm[basis])
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] < 2
        or not np.isfinite(coordinates[:, :2]).all()
    ):
        raise ValueError("Coordinates must be finite with at least two columns")
    if isinstance(cell_types, str):
        cell_types = [cell_types]
    cell_types = list(proportions.columns if cell_types is None else cell_types)
    if not cell_types or len(set(cell_types)) != len(cell_types):
        raise ValueError("Select at least one unique cell type")
    missing = set(cell_types) - set(proportions.columns)
    if missing:
        raise KeyError(f"Unknown cell types: {sorted(missing)}")
    values = proportions.loc[:, cell_types].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1 + 1e-6).any():
        raise ValueError("Proportions must be finite values between zero and one")
    if isinstance(ncols, bool) or not isinstance(ncols, int) or ncols < 1:
        raise ValueError("ncols must be a positive integer")
    if not np.isscalar(size) or not np.isfinite(size) or size <= 0:
        raise ValueError("size must be a finite positive scalar")
    ncols = min(ncols, len(cell_types))
    nrows = math.ceil(len(cell_types) / ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        squeeze=False,
        figsize=figsize or (4 * ncols, 3.8 * nrows),
        constrained_layout=True,
    )
    for i, kind in enumerate(cell_types):
        axis = axes.flat[i]
        artist = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=values[:, i],
            s=size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        axis.set_title(str(kind))
        axis.set_aspect("equal")
        axis.set_axis_off()
        if invert_y:
            axis.invert_yaxis()
        figure.colorbar(artist, ax=axis, label="Proportion", shrink=0.8)
    for axis in list(axes.flat)[len(cell_types) :]:
        axis.set_visible(False)
    if show:
        plt.show()
    return figure


__all__ = ["spatial_proportions"]
