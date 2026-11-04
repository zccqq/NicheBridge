"""Reference-only marker and variable-gene panel construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class MarkerThresholds:
    """Eligibility thresholds for one-versus-rest reference markers."""

    min_logfoldchange: float = 0.5
    min_detection_fraction: float = 0.1
    min_detection_difference: float = 0.05
    max_adjusted_pvalue: float = 0.05

    def __post_init__(self):
        if not math.isfinite(self.min_logfoldchange) or self.min_logfoldchange < 0:
            raise ValueError("min_logfoldchange must be finite and nonnegative")
        for name in (
            "min_detection_fraction",
            "min_detection_difference",
            "max_adjusted_pvalue",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")


def rank_reference_genes(counts, genes, labels, thresholds=None):
    """Rank genes using reference counts and annotations without spatial outcomes.

    Libraries are normalized over the shared candidate genes to 10,000 counts
    and log transformed. Markers use tie-corrected rank tests against all
    other reference cells. Variable genes use normalized expression dispersions.
    """
    import scanpy as sc

    thresholds = thresholds or MarkerThresholds()
    genes = np.asarray(genes, dtype=str)
    if pd.isna(np.asarray(labels)).any():
        raise ValueError("Every reference cell requires a nonmissing label")
    labels = np.asarray(labels, dtype=str)
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    if matrix.shape != (len(labels), len(genes)):
        raise ValueError("Counts, genes, and labels have inconsistent dimensions")
    if len(set(genes)) != len(genes) or np.any(genes == ""):
        raise ValueError("Gene identifiers must be unique and nonempty")
    if (
        np.any(labels == "")
        or not np.isfinite(matrix.data).all()
        or np.any(matrix.data < 0)
    ):
        raise ValueError("Reference labels and nonnegative finite counts are required")
    sizes = pd.Series(labels).value_counts()
    if len(sizes) < 2 or sizes.min() < 2:
        raise ValueError(
            "Wilcoxon markers require at least two cells in each of two types"
        )
    reference = ad.AnnData(
        matrix,
        obs=pd.DataFrame(
            {"cell_type": pd.Categorical(labels, categories=sorted(sizes.index))},
            index=[f"cell_{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=genes),
    )
    sc.pp.normalize_total(reference, target_sum=10000)
    sc.pp.log1p(reference)
    sc.pp.highly_variable_genes(reference, flavor="seurat", n_top_genes=None)
    hvg = reference.var[["means", "dispersions", "dispersions_norm"]].copy()
    hvg.insert(0, "gene", genes)
    hvg["ranking_dispersion"] = hvg.dispersions_norm.fillna(-np.inf)
    hvg = hvg.sort_values(
        ["ranking_dispersion", "gene"], ascending=[False, True]
    ).reset_index(drop=True)
    hvg["hvg_rank"] = np.arange(1, len(hvg) + 1)
    hvg = hvg.drop(columns="ranking_dispersion")
    sc.tl.rank_genes_groups(
        reference,
        groupby="cell_type",
        method="wilcoxon",
        reference="rest",
        n_genes=len(genes),
        use_raw=False,
        tie_correct=True,
        pts=True,
        corr_method="benjamini-hochberg",
    )
    ranking = reference.uns["rank_genes_groups"]
    tables = []
    for kind in sorted(sizes.index):
        names = ranking["names"][kind].astype(str)
        table = pd.DataFrame(
            {
                "gene": names,
                "cell_type": kind,
                "score": ranking["scores"][kind],
                "logfoldchange": ranking["logfoldchanges"][kind],
                "adjusted_pvalue": ranking["pvals_adj"][kind],
                "detection_fraction": ranking["pts"].loc[names, kind].to_numpy(),
                "rest_detection_fraction": ranking["pts_rest"]
                .loc[names, kind]
                .to_numpy(),
            }
        )
        table["detection_difference"] = (
            table.detection_fraction - table.rest_detection_fraction
        )
        table["eligible"] = (
            np.isfinite(table.score)
            & np.isfinite(table.logfoldchange)
            & (table.logfoldchange >= thresholds.min_logfoldchange)
            & (table.detection_fraction >= thresholds.min_detection_fraction)
            & (table.detection_difference >= thresholds.min_detection_difference)
            & (table.adjusted_pvalue <= thresholds.max_adjusted_pvalue)
        )
        table = table.sort_values(
            ["score", "logfoldchange", "detection_difference", "gene"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        table["type_rank"] = np.arange(1, len(table) + 1)
        tables.append(table)
    return pd.concat(tables, ignore_index=True), hvg, asdict(thresholds)


def select_reference_panel(genes, marker_ranking, hvg_ranking, panel, budget):
    """Select unique genes with balanced marker allocation and lexical ordering."""
    if panel not in {"hvg", "marker", "marker_hvg"}:
        raise ValueError(f"Unknown panel: {panel}")
    if budget < 1:
        raise ValueError("Gene budget must be positive")
    genes = np.asarray(genes, dtype=str)
    if len(set(genes)) != len(genes):
        raise ValueError("Gene identifiers must be unique")
    budget = min(budget, len(genes))
    available = set(genes)
    if not set(marker_ranking.gene) <= available or set(hvg_ranking.gene) != available:
        raise ValueError(
            "Rankings must refer to the complete candidate gene vocabulary"
        )
    selected = {}
    marker_budget = (
        budget if panel == "marker" else budget // 2 if panel == "marker_hvg" else 0
    )
    eligible = marker_ranking.loc[marker_ranking.eligible].copy()
    queues = {
        kind: frame.sort_values("type_rank").gene.tolist()
        for kind, frame in eligible.groupby("cell_type", sort=True)
    }
    pointers = dict.fromkeys(queues, 0)
    while len(selected) < marker_budget:
        progress = False
        for kind, queue in queues.items():
            while pointers[kind] < len(queue) and queue[pointers[kind]] in selected:
                pointers[kind] += 1
            if pointers[kind] == len(queue):
                continue
            gene = queue[pointers[kind]]
            pointers[kind] += 1
            selected[gene] = {
                "gene": gene,
                "selection": "wilcoxon_marker",
                "source_type": kind,
            }
            progress = True
            if len(selected) == marker_budget:
                break
        if not progress:
            break
    if panel != "marker":
        for gene in hvg_ranking.sort_values("hvg_rank").gene:
            if len(selected) >= budget:
                break
            if gene not in selected:
                selected[gene] = {
                    "gene": gene,
                    "selection": "reference_seurat_hvg",
                    "source_type": "",
                }
    if len(selected) < 2:
        raise ValueError(
            "Selection yielded fewer than two genes; review reference coverage or thresholds"
        )
    table = pd.DataFrame([selected[gene] for gene in sorted(selected)])
    table = table.merge(hvg_ranking, on="gene", how="left", validate="one_to_one")
    indices = pd.Index(genes).get_indexer(table.gene)
    return indices, table
