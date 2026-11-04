"""Prepare count-preserving, gene-aligned inputs for NicheBridgeDeconvolver."""

from __future__ import annotations

from dataclasses import asdict

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ._gene_selection import (
    MarkerThresholds,
    rank_reference_genes,
    select_reference_panel,
)


def _validate_object(adata, name):
    if not isinstance(adata, ad.AnnData):
        raise TypeError(f"{name} must be an AnnData object")
    if adata.isbacked:
        raise ValueError(f"Load {name} into memory before preparing inputs")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"{name} must contain observations and genes")
    for index, label in ((adata.obs_names, "observation"), (adata.var_names, "gene")):
        if (
            not index.is_unique
            or index.hasnans
            or any(not str(value).strip() for value in index)
        ):
            raise ValueError(f"{name} {label} identifiers must be unique and nonempty")


def _validated_counts(adata, layer, name):
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Missing {name} count layer: {layer}")
    values = adata.X if layer is None else adata.layers[layer]
    if values is None:
        raise ValueError(f"{name} contains no count matrix")
    matrix = sparse.csr_matrix(values, copy=True)
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError(f"{name} counts must be finite and nonnegative")
    if not np.allclose(matrix.data, np.rint(matrix.data), atol=1e-4, rtol=0):
        raise ValueError(
            f"{name} requires raw integer counts, not normalized expression"
        )
    return matrix


def _balanced_panel(counts, genes, labels, budget):
    """Allocate type-specific genes evenly and fill by reference log variance."""
    if len(genes) <= budget:
        return pd.DataFrame({"gene": genes, "selection": "all_shared"})
    libraries = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    normalized = sparse.diags(1.0 / np.maximum(libraries, 1.0)) @ counts
    types = sorted(set(labels))
    signatures = np.stack(
        [np.asarray(normalized[labels == kind].mean(axis=0)).ravel() for kind in types]
    )
    other = (signatures.sum(axis=0, keepdims=True) - signatures) / max(
        1, len(types) - 1
    )
    specificity = np.log2((signatures + 1e-8) / (other + 1e-8))
    specificity[signatures <= 0] = -np.inf
    quota = max(1, min(100, budget // (2 * len(types))))
    ranks = [
        np.lexsort((genes, -signatures[k], -specificity[k])) for k in range(len(types))
    ]
    pointers = np.zeros(len(types), dtype=int)
    selected = {}
    for _ in range(quota):
        for k, kind in enumerate(types):
            while pointers[k] < len(genes) and int(ranks[k][pointers[k]]) in selected:
                pointers[k] += 1
            if pointers[k] == len(genes):
                continue
            index = int(ranks[k][pointers[k]])
            pointers[k] += 1
            if not np.isfinite(specificity[k, index]) or len(selected) >= budget:
                continue
            selected[index] = f"specificity:{kind}"
    logged = normalized.copy().tocsr()
    logged.data = np.log1p(logged.data * 10000.0)
    mean = np.asarray(logged.mean(axis=0)).ravel()
    variance = np.asarray(logged.multiply(logged).mean(axis=0)).ravel() - mean**2
    for index in np.lexsort((genes, -variance)):
        if len(selected) >= budget:
            break
        if int(index) not in selected:
            selected[int(index)] = "reference_log_variance"
    return pd.DataFrame(
        {"gene": [genes[i] for i in selected], "selection": list(selected.values())}
    )


def _variable_panel(counts, genes, budget):
    """Rank reference genes by mean-adjusted dispersion after log normalization."""
    import scanpy as sc

    reference = ad.AnnData(counts.astype(np.float64), var=pd.DataFrame(index=genes))
    sc.pp.normalize_total(reference, target_sum=10000)
    sc.pp.log1p(reference)
    sc.pp.highly_variable_genes(reference, flavor="seurat", n_top_genes=None)
    scores = reference.var["dispersions_norm"].fillna(-np.inf).to_numpy()
    chosen = np.lexsort((genes, -scores))[:budget]
    return pd.DataFrame({"gene": genes[chosen], "selection": "reference_variable"})


def prepare(
    reference,
    spatial,
    cell_type_key="cell_type",
    counts_layer=None,
    genes=None,
    gene_selection="all",
    n_genes=2000,
    copy=True,
    marker_thresholds=None,
):
    """Align shared genes and preserve raw counts in ``layers['counts']``.

    Parameters
    ----------
    reference, spatial : anndata.AnnData
        Annotated reference cells and spatial observations, each by genes.
        Observation identifiers and all observations are retained. Coordinates,
        when present, must be finite with at least two columns in ``obsm['spatial']``.
    cell_type_key : str
        Reference observation column containing a label for every cell.
        Spatial observation annotations are never read for feature selection.
    counts_layer : str or None
        Raw count source in both inputs. ``None`` reads ``X``. Choose a count
        layer explicitly when ``X`` already contains normalized expression.
    genes : sequence of str or None
        Exact ordered gene panel. Every listed gene must occur in both inputs.
        This option requires ``gene_selection='all'``.
    gene_selection : {'all', 'balanced', 'marker', 'hvg', 'marker_hvg'}
        ``all`` retains shared genes. ``balanced`` combines type specificity
        and reference log variance. ``marker`` selects reference differential
        expression markers with balanced allocation across types. ``hvg`` uses
        reference normalized dispersion. ``marker_hvg`` reserves half the
        budget for markers and fills remaining slots with variable genes.
        Marker selection can yield fewer than ``n_genes`` eligible genes.
    n_genes : int
        Maximum gene budget for optional selection, ignored for ``all``.
    copy : bool
        If true, return copies without changing either input. Otherwise align
        both inputs in place. ``X`` retains its supplied expression values;
        validated raw counts are stored independently in ``layers['counts']``.
    marker_thresholds : MarkerThresholds or None
        Eligibility thresholds for optional differential expression markers.

    Returns
    -------
    tuple of anndata.AnnData
        Prepared reference and spatial objects. Genes follow the spatial
        input order unless an explicit ``genes`` sequence is supplied.
        Selection provenance is recorded in ``uns['nichebridge_preparation']``
        and ``var['nichebridge_selection']``.
    """
    _validate_object(reference, "Reference")
    _validate_object(spatial, "Spatial")
    if not isinstance(copy, bool):
        raise TypeError("copy must be boolean")
    if not copy and (reference.is_view or spatial.is_view):
        raise ValueError("Use copy=True to prepare AnnData views")
    if cell_type_key not in reference.obs:
        raise KeyError(f"Missing reference cell-type annotation: {cell_type_key}")
    labels = reference.obs[cell_type_key]
    if labels.isna().any() or labels.astype(str).str.strip().eq("").any():
        raise ValueError("Every reference cell requires a nonempty cell-type label")
    if gene_selection not in {"all", "balanced", "marker", "hvg", "marker_hvg"}:
        raise ValueError(f"Unknown gene selection strategy: {gene_selection}")
    if (
        isinstance(n_genes, bool)
        or not isinstance(n_genes, (int, np.integer))
        or n_genes < 2
    ):
        raise ValueError("n_genes must be an integer of at least two")
    if "spatial" in spatial.obsm:
        coordinates = np.asarray(spatial.obsm["spatial"])
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] < 2
            or not np.isfinite(coordinates).all()
        ):
            raise ValueError(
                "Spatial coordinates must be finite with at least two columns"
            )
    if genes is None:
        candidates = np.asarray(
            [gene for gene in spatial.var_names if gene in reference.var_names],
            dtype=str,
        )
    else:
        if isinstance(genes, str):
            raise TypeError("genes must be a sequence, not a single string")
        if gene_selection != "all":
            raise ValueError("An explicit gene panel requires gene_selection='all'")
        candidates = np.asarray(list(genes), dtype=str)
        if candidates.ndim != 1 or len(set(candidates)) != len(candidates):
            raise ValueError(
                "The explicit gene panel must contain unique gene identifiers"
            )
        missing = set(candidates) - (set(reference.var_names) & set(spatial.var_names))
        if missing:
            raise ValueError(
                f"Requested genes missing from an input: {sorted(missing)[:5]}"
            )
    if len(candidates) < 2:
        raise ValueError("At least two shared genes are required")
    ref_counts = _validated_counts(reference, counts_layer, "Reference")
    spatial_counts = _validated_counts(spatial, counts_layer, "Spatial")
    ref_indices = reference.var_names.get_indexer(candidates)
    candidate_counts = ref_counts[:, ref_indices]
    thresholds = marker_thresholds or MarkerThresholds()
    if not isinstance(thresholds, MarkerThresholds):
        raise TypeError("marker_thresholds must be a MarkerThresholds object")
    if gene_selection == "all":
        table = pd.DataFrame(
            {
                "gene": candidates,
                "selection": "explicit" if genes is not None else "all_shared",
            }
        )
    elif gene_selection == "balanced":
        table = _balanced_panel(
            candidate_counts, candidates, labels.astype(str).to_numpy(), n_genes
        )
    elif gene_selection == "hvg":
        table = _variable_panel(candidate_counts, candidates, n_genes)
    else:
        ranked, hvg, _ = rank_reference_genes(
            candidate_counts, candidates, labels.astype(str).to_numpy(), thresholds
        )
        _, table = select_reference_panel(
            candidates, ranked, hvg, gene_selection, n_genes
        )
    chosen = set(table.gene)
    selected = [gene for gene in candidates if gene in chosen]
    if len(selected) < 2:
        raise ValueError("Selection yielded fewer than two genes")
    provenance = {
        "gene_selection": gene_selection,
        "requested_n_genes": int(n_genes),
        "n_shared_candidates": len(candidates),
        "n_selected_genes": len(selected),
        "source_counts": "X" if counts_layer is None else counts_layer,
        "counts_layer": "counts",
        "cell_type_key": cell_type_key,
        "gene_order": "explicit" if genes is not None else "spatial_input",
        "observation_filtering": False,
    }
    if gene_selection in {"marker", "marker_hvg"}:
        provenance["marker_thresholds"] = asdict(thresholds)
    output = []
    selected_table = table.set_index("gene").loc[selected]
    for original, matrix in ((reference, ref_counts), (spatial, spatial_counts)):
        selected_counts = matrix[:, original.var_names.get_indexer(selected)].copy()
        if copy:
            prepared = original[:, selected].copy()
        else:
            original._inplace_subset_var(selected)
            prepared = original
        prepared.layers["counts"] = selected_counts
        prepared.var["nichebridge_selection"] = selected_table["selection"].to_numpy()
        prepared.uns["nichebridge_preparation"] = dict(provenance)
        output.append(prepared)
    return tuple(output)


def gene_activity(atac, peak_to_gene, *, layer=None, weight_key="weight"):
    """Aggregate ATAC peaks using a user-supplied genomic annotation table.

    ``peak_to_gene`` has ``peak`` and ``gene`` columns and optional nonnegative
    weights. Rows describe promoter/body/other associations selected by the
    caller; no genome build or genomic overlap is guessed. Repeated edges are
    rejected. Unmapped peaks are omitted and observations remain unchanged.
    """
    _validate_object(atac, "ATAC")
    counts = _validated_counts(atac, layer, "ATAC")
    if not isinstance(peak_to_gene, pd.DataFrame):
        raise TypeError("peak_to_gene must be a DataFrame")
    if not {"peak", "gene"}.issubset(peak_to_gene.columns):
        raise ValueError("peak_to_gene requires peak and gene columns")
    mapping = peak_to_gene.copy()
    if mapping.empty or mapping[["peak", "gene"]].isna().any().any():
        raise ValueError("Mapping must contain nonmissing peak and gene identifiers")
    if mapping.duplicated(["peak", "gene"]).any():
        raise ValueError("Duplicate peak-to-gene edges")
    if not mapping["peak"].isin(atac.var_names).all():
        raise ValueError("Mapping contains peak identifiers absent from ATAC")
    weights = (mapping[weight_key].to_numpy(dtype=float) if weight_key in mapping
               else np.ones(len(mapping)))
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Mapping weights must be finite and nonnegative")
    genes = pd.Index(pd.unique(mapping["gene"]), dtype=str)
    association = sparse.csr_matrix((weights, (
        atac.var_names.get_indexer(mapping["peak"]),
        genes.get_indexer(mapping["gene"]))), shape=(atac.n_vars, len(genes)))
    result = ad.AnnData(X=(counts @ association).tocsr(), obs=atac.obs.copy(),
                       var=pd.DataFrame(index=genes))
    for key in ("spatial",):
        if key in atac.obsm:
            result.obsm[key] = atac.obsm[key].copy()
    result.uns["nichebridge_gene_activity"] = {
        "n_input_peaks": int(atac.n_vars), "n_mapped_peaks": int(mapping.peak.nunique()),
        "n_edges": int(len(mapping)), "weighted": bool(weight_key in mapping),
        "definition": "User-supplied peak-to-gene annotation and weights"}
    return result


def spatial_enhance(adata, *, layer=None, output_layer="nichebridge_enhanced",
                    n_neighbors=6, strength=0.2, slice_key=None, copy=True):
    """Coordinate-neighbor enhancement for trajectory input, within each slice.

    The output is ``(1-strength)*X + strength*W@X`` with row-normalized,
    inverse-distance kNN weights and no self edges. This is an explicit
    configurable preprocessing choice; the manuscript's supplemental kernel
    is not supplied. Enhanced values are continuous and must not be passed to
    a raw-count likelihood. The original count matrix remains available.
    """
    from sklearn.neighbors import NearestNeighbors
    _validate_object(adata, "spatial")
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    if "spatial" not in adata.obsm:
        raise ValueError("obsm['spatial'] coordinates are required")
    coordinates = np.asarray(adata.obsm["spatial"], dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2 or not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be finite with at least two dimensions")
    if output_layer == layer:
        raise ValueError("Use a distinct output layer to preserve the input matrix")
    values = sparse.csr_matrix(adata.X if layer is None else adata.layers[layer], dtype=float)
    if not np.isfinite(values.data).all() or (values.data < 0).any():
        raise ValueError("Expression must be finite and nonnegative")
    if slice_key is None:
        groups = np.zeros(adata.n_obs, dtype=int)
    else:
        if slice_key not in adata.obs or adata.obs[slice_key].isna().any():
            raise ValueError("Every observation requires a valid slice label")
        groups = adata.obs[slice_key].astype(str).to_numpy()
    rows, cols, weights = [], [], []
    for group in pd.unique(groups):
        ids = np.flatnonzero(groups == group)
        k = min(n_neighbors, len(ids) - 1)
        if k == 0:
            rows.append(int(ids[0])); cols.append(int(ids[0])); weights.append(1.0)
            continue
        distance, indices = NearestNeighbors(n_neighbors=k).fit(coordinates[ids]).kneighbors()
        local_weights = 1.0 / np.maximum(distance, 1e-8)
        local_weights /= local_weights.sum(axis=1, keepdims=True)
        rows.extend(np.repeat(ids, k)); cols.extend(ids[indices.ravel()]); weights.extend(local_weights.ravel())
    graph = sparse.csr_matrix((weights, (rows, cols)), shape=(adata.n_obs, adata.n_obs))
    target = adata.copy() if copy else adata
    target.layers[output_layer] = ((1 - strength) * values + strength * graph @ values).tocsr()
    target.uns["nichebridge_spatial_enhancement"] = {
        "n_neighbors": n_neighbors, "strength": float(strength),
        "slice_key": "" if slice_key is None else slice_key,
        "input_layer": "X" if layer is None else layer, "output_layer": output_layer,
        "kernel": "inverse_distance_knn_row_normalized"}
    return target


__all__ = ["prepare", "gene_activity", "spatial_enhance", "MarkerThresholds"]
