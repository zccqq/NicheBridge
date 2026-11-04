"""Fit NicheBridgeDeconvolver and attach named outputs to AnnData objects."""

from ._model import NicheBridgeDeconvolver


def deconvolve(
    reference,
    spatial,
    config=None,
    cell_type_key="cell_type",
    counts_layer="counts",
    copy=False,
    verbose=True,
    **model_kwargs
):
    """Fit the model and write compositions and latent means to spatial data.

    Prepare aligned raw counts with ``pp.prepare`` first, or set
    ``counts_layer=None`` to read raw counts directly from ``X``. Supply a
    ``NicheBridgeConfig`` object or model keyword settings, but not both. Optional
    keyword settings override the model's default configuration.

    The output ``obsm`` keys are ``nichebridge_cell_fraction``,
    ``nichebridge_reference_rna_fraction``, ``nichebridge_rna_fraction``, and
    ``X_nichebridge``. Composition matrices are spot-by-type DataFrames with
    explicit column labels. ``obs['nichebridge_low_information']`` identifies
    zero-count observations, which are retained with uniform cell fractions.
    Model metadata are stored in ``uns['nichebridge']``.

    With ``copy=False``, modify ``spatial`` and return the fitted model.
    With ``copy=True``, return ``(spatial_copy, fitted_model)`` without
    modifying either input. The model provides ``save``, ``predict``, and
    ``transform`` methods for reuse.
    """
    if not isinstance(copy, bool):
        raise TypeError("copy must be boolean")
    if spatial.is_view and not copy:
        raise ValueError("Use copy=True to attach predictions to an AnnData view")
    target = spatial.copy() if copy else spatial
    model = NicheBridgeDeconvolver(config=config, **model_kwargs)
    model.fit(
        reference,
        target,
        cell_type_key=cell_type_key,
        counts_layer=counts_layer,
        verbose=verbose,
    )
    model.transform(target, counts_layer=counts_layer)
    target.uns["nichebridge"]["config"] = model.config.to_dict()
    return (target, model) if copy else model


__all__ = ["deconvolve"]
