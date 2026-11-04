"""Reproducible Python analyses for the NicheBridge kidney tutorial.

These helpers belong to the tutorial, not the installable modeling package.
Supplied annotations, latent coordinates and reconstruction remain explicitly
marked as supplied data. No function in this module fits NicheBridge.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import platform
import re
import time

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import scanpy as sc


AGES = ["W3", "W12", "W52", "W92"]
SEXES = ["F", "M"]
SEX_COLORS = {"F": "#e36146", "M": "#377eb8"}
AGE_COLORS = {"W3": "#8dd3c7", "W12": "#e7c85c", "W52": "#aaa4cc", "W92": "#ef8d80"}
DOMAIN_COLORS = ["#eeb7d3", "#f4f19c", "#70CDBE", "#7AC3DF", "#AC99D2", "#F6948D", "#DB6968", "#C1E0DB"]
MARKER_GENES = [
    "9530026P05Rik", "Prlr", "Hao2", "C030029H02Rik", "Snhg11", "Rdh16f2", "Myo9a", "Alas1",
    "Mylip", "Prim2", "Slc22a29", "Kcnc3", "Kyat3", "Slc22a19", "Slc7a12", "Npnt", "Ptprm",
    "Clec2d", "Svil", "Igf1r", "Ano4", "Serpine2", "Abca13", "Dgkg", "Cadm1", "Tenm3", "Magi1",
    "Immp2l", "Acsm3", "Slc22a30", "Slc22a28", "Nat8f6", "Kcnk5", "Larp4b", "Cyp7b1", "Fmo2",
    "Tasor", "Resf1",
]
HALLMARK_TERMS = ["Oxidative Phosphorylation", "Fatty Acid Metabolism", "Xenobiotic Metabolism",
                  "Interferon Gamma Response", "Complement"]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _output(folder):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _save_figure(fig, folder, stem):
    folder = _output(folder)
    paths = {}
    for extension in ("png", "pdf"):
        path = folder / f"NicheBridge_{stem}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        paths[extension] = str(path)
    plt.close(fig)
    return paths


def _save_table(table, folder, stem, *, index=False):
    path = _output(folder) / f"NicheBridge_{stem}.csv"
    table.to_csv(path, index=index)
    return str(path)


def _result(section, folder, *, figures=None, tables=None, summary=None):
    result = {"section": section, "data_status": "supplied_processed_input",
              "figures": figures or {}, "tables": tables or {}, "summary": summary or {}}
    path = _output(folder) / f"NicheBridge_{section}_manifest.json"
    _json(path, result)
    result["manifest"] = str(path)
    return result


def show_figures(result):
    """Display figures produced by the current execution in a notebook."""
    from IPython.display import Image, display
    for item in result.get("figures", {}).values():
        display(Image(filename=item["png"]))


def _dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _categories(values):
    return sorted(pd.unique(values.astype(str)), key=lambda value: (0, int(value)) if value.isdigit() else (1, value))


def _subset(adata, *, domain=None, ages=None, sexes=None):
    mask = np.ones(adata.n_obs, dtype=bool)
    if domain is not None:
        domains = [str(domain)] if isinstance(domain, (str, int)) else list(map(str, domain))
        mask &= adata.obs["supplied_domain"].astype(str).isin(domains).to_numpy()
    if ages is not None:
        mask &= adata.obs["age"].astype(str).isin(ages).to_numpy()
    if sexes is not None:
        mask &= adata.obs["sex"].astype(str).isin(sexes).to_numpy()
    # Build an expression-only object rather than copying the reconstruction.
    result = ad.AnnData(X=adata.X[mask].copy(), obs=adata.obs.loc[mask].copy(), var=adata.var.copy())
    result.uns["nichebridge_tutorial"] = dict(adata.uns["nichebridge_tutorial"])
    if adata.uns["nichebridge_tutorial"]["analysis_scale"] == "log1p":
        result.uns["log1p"] = {"base": None}
    return result, mask


def load_supplied_dataset(path, output_dir, *, seed=42, analysis_layer="log1p", analysis_scale="log1p"):
    """Load explicit supplied input fields without normalization or log guessing.

    The default uses the existing log1p layer unchanged. Set analysis_layer=None
    and analysis_scale='supplied' to replay X unchanged instead. This alternate
    convention is explicit because the upstream scale of X is not documented.
    """
    started = time.perf_counter()
    path, folder = Path(path).resolve(), _output(output_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    if analysis_scale not in {"log1p", "supplied"}:
        raise ValueError("analysis_scale must be log1p or supplied")
    np.random.seed(seed)
    source = sc.read_h5ad(path)
    if not source.obs_names.is_unique or not source.var_names.is_unique:
        raise ValueError("Input observation and gene identifiers must be unique")
    for key in ("domain", "age", "sex", "section"):
        if key not in source.obs or source.obs[key].isna().any():
            raise ValueError(f"Required complete observation column missing: {key}")
    for key in ("spatial", "qz2", "X_umap"):
        if key not in source.obsm:
            raise KeyError(f"Supplied input is missing obsm[{key!r}]")
    if "x4" not in source.layers:
        raise KeyError("Supplied input is missing the archived reconstruction layer x4")
    if analysis_layer is not None and analysis_layer not in source.layers:
        raise KeyError(f"Supplied input is missing analysis layer {analysis_layer!r}")
    matrix = source.X if analysis_layer is None else source.layers[analysis_layer]
    data = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError("Analysis expression must be finite and nonnegative")
    columns = ["domain", "age", "sex", "section"] + (["in_tissue"] if "in_tissue" in source.obs else [])
    obs = source.obs[columns].copy().rename(columns={"domain": "supplied_domain"})
    for column in ("supplied_domain", "age", "sex", "section"):
        obs[column] = obs[column].astype(str)
    result = ad.AnnData(X=matrix, obs=obs, var=source.var.copy())
    result.layers["supplied_reconstruction"] = source.layers["x4"]
    result.obsm["spatial"] = np.asarray(source.obsm["spatial"]).copy()
    result.obsm["X_supplied_latent"] = np.asarray(source.obsm["qz2"]).copy()
    result.obsm["X_supplied_umap"] = np.asarray(source.obsm["X_umap"]).copy()
    for key in result.obsm:
        if not np.isfinite(result.obsm[key]).all():
            raise ValueError(f"Nonfinite supplied coordinates in {key}")
    versions = {"python": platform.python_version()}
    for package in ("nichebridge", "numpy", "pandas", "scipy", "scanpy", "anndata", "matplotlib", "gseapy", "statsmodels"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    manifest = {
        "tutorial": "NicheBridge kidney tutorial", "status": "loaded_supplied_input", "seed": int(seed),
        "input_path": str(path), "input_sha256": _sha256(path), "input_bytes": path.stat().st_size,
        "n_spots": result.n_obs, "n_genes": result.n_vars, "analysis_layer": analysis_layer or "X",
        "analysis_scale": analysis_scale, "preprocessing": "none; explicit supplied matrix is used unchanged",
        "source_mapping": {"supplied_domain": "obs/domain", "X_supplied_latent": "obsm/qz2",
                           "X_supplied_umap": "obsm/X_umap", "supplied_reconstruction": "layers/x4"},
        "representation_status": "Precomputed supplied fields; not a fresh NicheBridge model fit",
        "marker_panel_present": [gene for gene in MARKER_GENES if gene in result.var_names],
        "marker_panel_missing": [gene for gene in MARKER_GENES if gene not in result.var_names],
        "versions": versions, "load_seconds": time.perf_counter() - started,
    }
    result.uns["nichebridge_tutorial"] = manifest
    if analysis_scale == "log1p":
        result.uns["log1p"] = {"base": None}
    _json(folder / "NicheBridge_input_manifest.json", manifest)
    groups = result.obs.groupby(["section", "sex", "age", "supplied_domain"], observed=True).size().rename("n_spots").reset_index()
    _save_table(groups, folder, "input_group_sizes")
    marker_table = pd.DataFrame({"gene": MARKER_GENES, "available": [gene in result.var_names for gene in MARKER_GENES]})
    _save_table(marker_table, folder, "marker_availability")
    return result


def _orient(ax, age):
    # Preserve the supplied figure's section orientation convention.
    ax.invert_yaxis()
    if age in ("W52", "W92"):
        ax.invert_yaxis()
    if age == "W92":
        ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_domain_maps(adata, output_dir):
    """Eight age/sex tissue panels showing the supplied domain annotations."""
    domains = _categories(adata.obs.supplied_domain)
    palette = {domain: DOMAIN_COLORS[i % len(DOMAIN_COLORS)] for i, domain in enumerate(domains)}
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), gridspec_kw={"wspace": 0.02, "hspace": 0.05})
    for row, sex in enumerate(SEXES):
        for col, age in enumerate(AGES):
            ax = axes[row, col]
            mask = (adata.obs.sex == sex) & (adata.obs.age == age)
            xy = adata.obsm["spatial"][mask.to_numpy()]
            labels = adata.obs.loc[mask, "supplied_domain"]
            ax.scatter(xy[:, 0], xy[:, 1], c=[palette[x] for x in labels], s=4, linewidths=0, rasterized=True)
            _orient(ax, age)
            if row == 0:
                ax.set_title(age, fontsize=15, fontweight="bold")
            if col == 0:
                ax.set_ylabel("Female" if sex == "F" else "Male", fontsize=15)
    handles = [mpl.lines.Line2D([], [], marker="o", linestyle="", color=palette[domain], label=f"Domain {domain}") for domain in domains]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(domains), 8), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Kidney spatial domains · supplied annotations", fontsize=16, y=0.99)
    fig.subplots_adjust(bottom=0.09, top=0.92)
    figure = _save_figure(fig, output_dir, "supplied_spatial_domains")
    return _result("spatial_domains", output_dir, figures={"domains": figure}, summary={"n_spots": adata.n_obs, "domains": domains})


def _rank_age(subset, *, younger="W12", older="W92"):
    subset.obs["age"] = pd.Categorical(subset.obs.age, categories=[younger, older])
    sc.tl.rank_genes_groups(subset, groupby="age", groups=[older], reference=younger,
                           method="wilcoxon", use_raw=False, n_genes=subset.n_vars, pts=False)
    return sc.get.rank_genes_groups_df(subset, group=older)


def age_de_by_domain(adata, output_dir, *, younger="W12", older="W92", min_spots=5,
                     logfc_threshold=0.1, adjusted_p_threshold=1e-20):
    """Sex-stratified W92/W12 Wilcoxon comparisons across supplied domains."""
    domains = _categories(adata.obs.supplied_domain)
    rows, gene_tables = [], []
    for domain in domains:
        for sex in SEXES:
            subset, _ = _subset(adata, domain=domain, ages=[younger, older], sexes=[sex])
            n_young, n_old = int((subset.obs.age == younger).sum()), int((subset.obs.age == older).sum())
            row = {"domain": domain, "sex": sex, "n_younger": n_young, "n_older": n_old,
                   "up": np.nan, "down": np.nan, "status": "insufficient_spots"}
            if min(n_young, n_old) >= min_spots:
                ranked = _rank_age(subset, younger=younger, older=older)
                ranked.insert(0, "domain", domain)
                ranked.insert(1, "sex", sex)
                gene_tables.append(ranked)
                significant = ranked.pvals_adj.le(adjusted_p_threshold) & np.isfinite(ranked.logfoldchanges)
                row.update(up=int((significant & ranked.logfoldchanges.ge(logfc_threshold)).sum()),
                           down=int((significant & ranked.logfoldchanges.le(-logfc_threshold)).sum()), status="completed")
            rows.append(row)
    counts = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.7))
    x = np.arange(len(domains))
    for sex, shift in [("F", -0.18), ("M", 0.18)]:
        table = counts[counts.sex == sex].set_index("domain").reindex(domains)
        ax.bar(x + shift, table.up, 0.35, color=SEX_COLORS[sex], label="Female" if sex == "F" else "Male")
        ax.bar(x + shift, -table.down, 0.35, color=SEX_COLORS[sex])
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(x, labels=domains)
    ax.set_xlabel("Supplied spatial domain")
    ax.set_ylabel(f"Upregulated / downregulated genes ({older} vs {younger})")
    ax.set_title("Age-associated differential expression within each sex")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _result("age_differential_expression", output_dir,
                   figures={"counts": _save_figure(fig, output_dir, "age_sex_DE_counts")},
                   tables={"counts": _save_table(counts, output_dir, "age_sex_DE_counts"),
                           "genes": _save_table(pd.concat(gene_tables, ignore_index=True) if gene_tables else pd.DataFrame(), output_dir, "age_sex_DE_genes")},
                   summary={"younger": younger, "older": older, "min_spots": min_spots, "logfc_threshold": logfc_threshold,
                            "adjusted_p_threshold": adjusted_p_threshold, "analysis_layer": adata.uns["nichebridge_tutorial"]["analysis_layer"]})


def plot_proximal_latent(adata, output_dir):
    """Supplied PT(S1/S2)/PT(S3) UMAP views and latent centroid trajectories."""
    labels = {"1": "PT(S1/S2)", "6": "PT(S3)"}
    palette = {"PT(S1/S2)": "#d88caf", "PT(S3)": "#e48075"}
    keep = adata.obs.supplied_domain.isin(labels)
    obs = adata.obs.loc[keep].copy()
    obs["cell_type"] = obs.supplied_domain.map(labels)
    umap = adata.obsm["X_supplied_umap"][keep.to_numpy()]
    latent = adata.obsm["X_supplied_latent"][keep.to_numpy()]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    for ax, (key, colors, title) in zip(axes, [("cell_type", palette, "Domain"), ("sex", SEX_COLORS, "Sex"), ("age", AGE_COLORS, "Age")]):
        for category, color in colors.items():
            mask = obs[key].eq(category).to_numpy()
            ax.scatter(umap[mask, 0], umap[mask, 1], s=2, alpha=0.55, color=color, linewidths=0, rasterized=True, label=category)
        ax.set_title(title)
        ax.set_xlabel("Supplied UMAP 1")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
        ax.legend(frameon=False, markerscale=4, loc="upper right", fontsize=8)
    axes[0].set_ylabel("Supplied UMAP 2")
    fig.suptitle("Proximal-tubule domains · supplied latent coordinates")
    fig.tight_layout()
    figures = {"umap": _save_figure(fig, output_dir, "supplied_proximal_umap")}
    records = []
    for domain, cell_type in labels.items():
        for age in AGES:
            masks = {sex: (obs.supplied_domain.eq(domain) & obs.age.eq(age) & obs.sex.eq(sex)).to_numpy() for sex in SEXES}
            distance = float(np.linalg.norm(latent[masks["F"]].mean(0) - latent[masks["M"]].mean(0))) if all(mask.any() for mask in masks.values()) else np.nan
            records.append({"domain": domain, "cell_type": cell_type, "age": age, "distance": distance,
                            "n_female": int(masks["F"].sum()), "n_male": int(masks["M"].sum())})
    table = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(5, 3.7))
    for cell_type, color in palette.items():
        selected = table[table.cell_type == cell_type].set_index("age").reindex(AGES)
        ax.plot(AGES, selected.distance, marker="o", color=color, label=cell_type)
    ax.set(xlabel="Age", ylabel="Female–male latent centroid distance", title="Supplied latent representation")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    figures["centroids"] = _save_figure(fig, output_dir, "supplied_centroid_distances")
    return _result("proximal_latent", output_dir, figures=figures,
                   tables={"centroids": _save_table(table, output_dir, "supplied_centroid_distances")},
                   summary={"embedding_status": "precomputed supplied latent and UMAP", "n_spots": int(keep.sum()), "latent_dimension": latent.shape[1]})


def _normalise_term(term):
    term = re.sub(r"^HALLMARK[\s_-]+", "", str(term).strip(), flags=re.I)
    return re.sub(r"[\s_-]+", " ", term).casefold()


def _fdr_star(value):
    if not np.isfinite(value):
        return ""
    return "***" if value < 0.01 else "**" if value < 0.05 else "*" if value < 0.25 else ""


def run_hallmark(adata, output_dir, gene_sets, *, domain="6", permutations=1000, seed=42):
    """Sex-stratified Hallmark GSEA using a local GMT and explicit input X."""
    import gseapy as gp
    gene_sets = Path(gene_sets).resolve()
    if not gene_sets.is_file():
        raise FileNotFoundError(f"Local Hallmark GMT not found: {gene_sets}")
    tables, rankings, paths = [], [], {}
    for sex in SEXES:
        subset, _ = _subset(adata, domain=domain, ages=["W12", "W92"], sexes=[sex])
        if min(int((subset.obs.age == age).sum()) for age in ["W12", "W92"]) < 2:
            raise ValueError(f"Domain {domain}, sex {sex} needs two spots at each age")
        ranked = _rank_age(subset)[["names", "scores"]].copy()
        ranked = ranked[np.isfinite(ranked.scores)].drop_duplicates("names")
        original_names = ranked.names.copy()
        ranked["names"] = ranked.names.astype(str).str.upper()
        if ranked.names.duplicated().any():
            raise ValueError("Uppercase gene-symbol matching produced duplicate symbols; resolve them explicitly")
        rankings.append(ranked.assign(source_gene=original_names, sex=sex))
        result = gp.prerank(rnk=ranked, gene_sets=str(gene_sets), permutation_num=permutations,
                            min_size=10, max_size=500, seed=seed, threads=1, outdir=None, no_plot=True, verbose=False)
        table = result.res2d.copy()
        table["sex"] = sex
        table["term_key"] = table.Term.map(_normalise_term)
        table["NES"] = pd.to_numeric(table.NES, errors="coerce")
        table["FDR q-val"] = pd.to_numeric(table["FDR q-val"], errors="coerce")
        tables.append(table)
    enriched = pd.concat(tables, ignore_index=True)
    selected = enriched[enriched.term_key.isin([_normalise_term(term) for term in HALLMARK_TERMS])].copy()
    paths["all_pathways"] = _save_table(enriched, output_dir, "PT_S3_Hallmark_all")
    paths["selected_pathways"] = _save_table(selected, output_dir, "PT_S3_Hallmark_selected")
    paths["rankings"] = _save_table(pd.concat(rankings, ignore_index=True), output_dir, "PT_S3_Hallmark_rankings")
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.2), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.6, 1], "wspace": 0.04})
    max_abs = max(2.0, float(np.abs(selected.NES).max())) if len(selected) else 2.0
    missing = []
    for sex, ax in [("F", axes[0]), ("M", axes[2])]:
        table = selected[selected.sex == sex].set_index("term_key")
        for row, term in enumerate(HALLMARK_TERMS):
            key = _normalise_term(term)
            if key not in table.index:
                ax.text(0.1, row, "Not testable", va="center", fontsize=9)
                missing.append(f"{sex}: {term}")
                continue
            value, fdr = float(table.loc[key, "NES"]), float(table.loc[key, "FDR q-val"])
            ax.barh(row, abs(value), height=0.55, color="#D48982" if value >= 0 else "#4C78A8")
            ax.text(abs(value) + 0.05, row, f"{value:+.2f}{_fdr_star(fdr)}", va="center",
                    ha="right" if sex == "F" else "left", fontsize=9)
        ax.set_xlim(max_abs + 1, 0) if sex == "F" else ax.set_xlim(0, max_abs + 1)
        ax.set_title("Female" if sex == "F" else "Male")
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlabel("|NES|")
    axes[1].axis("off")
    for row, term in enumerate(HALLMARK_TERMS):
        axes[1].text(0.5, row, term.replace("Interferon Gamma", "Interferon-γ"), ha="center", va="center", fontsize=10)
    axes[0].set_ylim(len(HALLMARK_TERMS) - 0.5, -0.5)
    fig.suptitle("Domain 6 PT(S3) · W92 versus W12", y=0.99)
    fig.legend(handles=[mpl.patches.Patch(color="#D48982", label="W92 enriched"), mpl.patches.Patch(color="#4C78A8", label="W12 enriched")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.01, "* FDR < 0.25   ** FDR < 0.05   *** FDR < 0.01", ha="center", fontsize=8)
    fig.subplots_adjust(bottom=0.19, top=0.86)
    return _result("hallmark", output_dir, figures={"hallmark": _save_figure(fig, output_dir, "PT_S3_Hallmark")}, tables=paths,
                   summary={"gene_sets_path": str(gene_sets), "gene_sets_sha256": _sha256(gene_sets), "permutations": permutations,
                            "min_size": 10, "max_size": 500, "seed": seed, "missing_selected_terms": missing,
                            "symbol_matching": "uppercase symbols; duplicate-case collisions rejected; no orthology mapping inferred",
                            "n_tested_pathways": {sex: int((enriched.sex == sex).sum()) for sex in SEXES}})


def plot_marker_profiles(adata, output_dir, *, domain="6", marker_genes=MARKER_GENES):
    """Six-group marker heatmap, age/sex OLS sidebars and marker trajectories."""
    import statsmodels.api as sm
    from matplotlib.gridspec import GridSpec
    ages = ["W12", "W52", "W92"]
    subset, _ = _subset(adata, domain=domain, ages=ages, sexes=SEXES)
    genes = [gene for gene in marker_genes if gene in subset.var_names]
    missing = [gene for gene in marker_genes if gene not in subset.var_names]
    if not genes:
        raise ValueError("No requested marker genes occur in the supplied gene panel")
    if subset.n_obs < 4:
        raise ValueError("Too few spots for marker profiles")
    expression = pd.DataFrame(_dense(subset[:, genes].X), index=subset.obs_names, columns=genes)
    metadata = subset.obs[["sex", "age"]]
    means = expression.groupby([metadata.sex, metadata.age], observed=True).mean()
    means = means.reindex(pd.MultiIndex.from_product([SEXES, ages], names=["sex", "age"]))
    matrix = means.to_numpy().T
    center, scale = np.nanmean(matrix, axis=1, keepdims=True), np.nanstd(matrix, axis=1, keepdims=True)
    zscore = (matrix - center) / np.where(scale > 0, scale, 1)
    design = pd.DataFrame({"age_num": subset.obs.age.map({"W12": 12, "W52": 52, "W92": 92}).astype(float),
                           "sex_num": subset.obs.sex.map({"F": 0, "M": 1}).astype(float)}, index=subset.obs_names)
    design = sm.add_constant(design, has_constant="add")
    records = []
    for gene in genes:
        y = expression[gene].to_numpy()
        record = {"gene": gene, "n_spots": len(y), "age_coefficient": np.nan, "sex_coefficient": np.nan,
                  "p_age": np.nan, "p_sex": np.nan, "p_model": np.nan, "status": "constant_expression"}
        if np.isfinite(y).all() and np.std(y) > 1e-12 and np.linalg.matrix_rank(design) == design.shape[1]:
            fitted = sm.OLS(y, design).fit()
            record.update(age_coefficient=float(fitted.params.age_num), sex_coefficient=float(fitted.params.sex_num),
                          p_age=float(fitted.pvalues.age_num), p_sex=float(fitted.pvalues.sex_num),
                          p_model=float(fitted.f_pvalue), status="completed")
        records.append(record)
    regression = pd.DataFrame(records).set_index("gene")
    tree = linkage(np.nan_to_num(zscore), method="average", metric="euclidean") if len(genes) > 1 else None
    order = leaves_list(tree) if tree is not None else np.arange(len(genes))
    ordered = [genes[i] for i in order]
    zscore = zscore[order]
    p = regression.reindex(ordered)
    age_p = -np.log10(np.clip(p.p_age.to_numpy(), 1e-300, 1))
    sex_p = -np.log10(np.clip(p.p_sex.to_numpy(), 1e-300, 1))
    bars = np.clip((age_p + sex_p) / 2, 0, 280)
    fig = plt.figure(figsize=(10, max(6, 0.30 * len(genes) + 2)))
    grid = GridSpec(1, 6, figure=fig, width_ratios=[0.7, 0.32, 0.32, 1.0, 4.2, 1.6], wspace=0.06)
    axes = [fig.add_subplot(grid[0, i]) for i in range(6)]
    if tree is not None:
        dendrogram(tree, orientation="left", no_labels=True, color_threshold=0, above_threshold_color="black", ax=axes[0])
        axes[0].invert_yaxis()
    axes[0].axis("off")
    for ax, values, title in [(axes[1], age_p, "Age"), (axes[2], sex_p, "Sex")]:
        ax.imshow(values[:, None], aspect="auto", cmap="Purples")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    axes[3].barh(np.arange(len(genes)), bars, height=0.85, color="#999999")
    axes[3].set_ylim(len(genes) - 0.5, -0.5)
    axes[3].set_xlim(0, 300)
    axes[3].set_xticks([0, 150, 300])
    axes[3].tick_params(axis="x", labelsize=8)
    axes[3].set_yticks([])
    axes[3].set_title("Mean −log10(P)", fontsize=9)
    axes[3].spines[["top", "right", "left"]].set_visible(False)
    heat = axes[4].imshow(zscore, aspect="auto", cmap=plt.get_cmap("RdBu_r").with_extremes(bad="#dddddd"), vmin=-2, vmax=2)
    axes[4].set_xticks(range(6), labels=[f"{sex}\n{age}" for sex in SEXES for age in ages])
    axes[4].tick_params(axis="x", labelsize=10, length=0)
    axes[4].set_yticks([])
    axes[4].set_title("Female                         Male", fontsize=11)
    axes[4].axvline(2.5, color="white", linewidth=2)
    axes[5].set_xlim(0, 1)
    axes[5].set_ylim(len(genes) - 0.5, -0.5)
    axes[5].axis("off")
    for row, gene in enumerate(ordered):
        axes[5].text(0, row, gene, va="center", fontsize=10, fontstyle="italic")
    fig.suptitle(f"Domain {domain} PT(S3) · age and sex marker profiles", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.89, bottom=0.13, left=0.04, right=0.99)
    color_ax = fig.add_axes([0.48, 0.045, 0.22, 0.015])
    fig.colorbar(heat, cax=color_ax, orientation="horizontal", label="Gene-wise Z score of group means")
    figures = {"heatmap": _save_figure(fig, output_dir, "PT_S3_marker_heatmap")}
    requested_lines = [gene for gene in ["Prlr", "Acsm3", "Cyp7b1"] if gene in genes]
    if requested_lines:
        fig, axes = plt.subplots(1, len(requested_lines), figsize=(4 * len(requested_lines), 3.5), squeeze=False)
        for ax, gene in zip(axes[0], requested_lines):
            for sex in SEXES:
                ax.plot(ages, means.loc[sex, gene].reindex(ages), marker="o", color=SEX_COLORS[sex], label=sex)
            ax.set_title(gene, fontstyle="italic")
            ax.set_ylabel("Mean supplied analysis expression")
            ax.set_xlabel("Age")
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False)
        fig.suptitle("Domain 6 marker trajectories", y=1.02)
        fig.tight_layout()
        figures["trajectories"] = _save_figure(fig, output_dir, "PT_S3_marker_trajectories")
    zframe = pd.DataFrame(zscore, index=pd.Index(ordered, name="gene"), columns=[f"{sex}_{age}" for sex in SEXES for age in ages])
    return _result("marker_profiles", output_dir, figures=figures,
                   tables={"group_means": _save_table(means.reset_index(), output_dir, "PT_S3_marker_group_means"),
                           "regression": _save_table(regression, output_dir, "PT_S3_marker_regression", index=True),
                           "z_scores": _save_table(zframe, output_dir, "PT_S3_marker_z_scores", index=True)},
                   summary={"n_spots": subset.n_obs, "present_genes": genes, "missing_genes": missing,
                            "regression": "expression ~ intercept + numeric age + binary sex; unadjusted spot-level P values",
                            "grey_bar": "arithmetic mean of age and sex -log10(P), capped at 280", "analysis_layer": adata.uns["nichebridge_tutorial"]["analysis_layer"]})


def plot_gene_maps(adata, output_dir, *, genes=("Cyp7b1",), high_fractions=None, spot_radius=112.5):
    """Age/sex distributions and six tissue maps of supplied reconstruction."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle
    high_fractions = {"F": 0.0041, "M": 0.1} if high_fractions is None else dict(high_fractions)
    if set(high_fractions) != set(SEXES) or any(not 0 < value <= 1 for value in high_fractions.values()):
        raise ValueError("high_fractions must give one fraction in (0,1] for each sex")
    ages = ["W12", "W52", "W92"]
    keep = adata.obs.age.isin(ages) & adata.obs.sex.isin(SEXES)
    if "in_tissue" in adata.obs:
        keep &= pd.to_numeric(adata.obs.in_tissue, errors="coerce").eq(1)
    obs = adata.obs.loc[keep].copy()
    xy = adata.obsm["spatial"][keep.to_numpy(), :2]
    if obs.empty:
        raise ValueError("No matching tissue spots for the selected age/sex groups")
    for sex in SEXES:
        for age in ages:
            sections = obs.loc[(obs.sex == sex) & (obs.age == age), "section"].unique()
            if len(sections) != 1:
                raise ValueError(f"Expected one section for sex={sex}, age={age}; found {list(sections)}")
    figures, summaries, correlations = {}, [], []
    present = [gene for gene in genes if gene in adata.var_names]
    missing = [gene for gene in genes if gene not in adata.var_names]
    if not present:
        raise ValueError("No requested spatial marker occurs in the supplied panel")
    for gene in present:
        values = _dense(adata.layers["supplied_reconstruction"][:, adata.var_names.get_loc(gene)]).ravel()[keep.to_numpy()]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite reconstruction values for {gene}")
        frame = obs.assign(expression=values, age_num=obs.age.map({"W12": 12, "W52": 52, "W92": 92}))
        vmin, vmax = np.quantile(values, [0.02, 0.98])
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = Normalize(vmin=vmin, vmax=vmax)
        fig, axes = plt.subplots(2, 4, figsize=(14, 8), gridspec_kw={"width_ratios": [1.2, 1, 1, 1], "wspace": 0.16, "hspace": 0.18})
        for row, sex in enumerate(SEXES):
            sub = frame[frame.sex == sex]
            distributions = [sub.loc[sub.age == age, "expression"].to_numpy() for age in ages]
            ax = axes[row, 0]
            box = ax.boxplot(distributions, positions=np.arange(3), patch_artist=True, showfliers=False,
                             medianprops={"color": "#d8902f", "linewidth": 1.1})
            for patch in box["boxes"]:
                patch.set_facecolor(SEX_COLORS[sex])
                patch.set_alpha(0.35)
            means = [float(group.mean()) if len(group) else np.nan for group in distributions]
            ax.plot(range(3), means, "o-", color=SEX_COLORS[sex])
            threshold = float(sub.expression.quantile(1 - high_fractions[sex]))
            high = sub[sub.expression >= threshold]
            coefficient, pvalue = np.nan, np.nan
            if len(high) >= 10 and high.age_num.nunique() >= 2 and high.expression.std() > 0:
                coefficient, pvalue = spearmanr(high.age_num, high.expression)
            correlations.append({"gene": gene, "sex": sex, "top_fraction": high_fractions[sex],
                                 "expression_threshold": threshold, "n_spots": len(sub), "n_selected": len(high),
                                 "spearman_r": coefficient, "p_value": pvalue})
            rlabel = f"{coefficient:.2f}" if np.isfinite(coefficient) else "NA"
            plabel = f"{pvalue:.2g}" if np.isfinite(pvalue) else "NA"
            ax.text(0.05, 0.97, f"High subset: {100 * high_fractions[sex]:g}%\nρ = {rlabel}; P = {plabel}",
                    transform=ax.transAxes, va="top", fontsize=9)
            ax.set_ylim(0, max(float(values.max()) * 1.25, 1e-6))
            ax.set_xticks(range(3), labels=ages)
            ax.set_xlabel("Age")
            ax.set_ylabel(("Female" if sex == "F" else "Male") + " · supplied reconstruction")
            ax.spines[["top", "right"]].set_visible(False)
            for col, age in enumerate(ages, start=1):
                selected = frame.sex.eq(sex) & frame.age.eq(age)
                coordinates = xy[selected.to_numpy()]
                signal = frame.loc[selected, "expression"].to_numpy()
                order = np.argsort(signal, kind="stable")
                ax = axes[row, col]
                collection = PatchCollection([Circle(point, radius=spot_radius) for point in coordinates[order]],
                                             cmap="Spectral_r", norm=norm, edgecolors="none", linewidths=0, rasterized=True)
                collection.set_array(signal[order])
                ax.add_collection(collection)
                ax.set_xlim(coordinates[:, 0].min() - spot_radius, coordinates[:, 0].max() + spot_radius)
                ax.set_ylim(coordinates[:, 1].min() - spot_radius, coordinates[:, 1].max() + spot_radius)
                _orient(ax, age)
                if row == 0:
                    ax.set_title(age, fontsize=13)
                summaries.append({"gene": gene, "sex": sex, "age": age, "section": frame.loc[selected, "section"].iloc[0],
                                  "n_spots": len(signal), "mean_expression": float(signal.mean()),
                                  "median_expression": float(np.median(signal))})
        fig.suptitle(f"{gene} · supplied precomputed reconstruction", fontsize=16, y=0.99)
        fig.subplots_adjust(bottom=0.12, top=0.91)
        cax = fig.add_axes([0.43, 0.045, 0.4, 0.02])
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="Spectral_r"), cax=cax, orientation="horizontal", label="Supplied expression · shared 2nd–98th percentile color limits")
        figures[gene] = _save_figure(fig, output_dir, f"supplied_{gene}_age_sex_maps")
    return _result("gene_maps", output_dir, figures=figures,
                   tables={"groups": _save_table(pd.DataFrame(summaries), output_dir, "supplied_gene_map_groups"),
                           "correlations": _save_table(pd.DataFrame(correlations), output_dir, "supplied_gene_map_correlations")},
                   summary={"expression_source": "supplied_reconstruction (precomputed input)", "all_domains_retained": True,
                            "present_genes": present, "missing_genes": missing, "high_fractions": high_fractions,
                            "color_quantiles": [0.02, 0.98], "n_spots": len(obs)})


def run_reproduction(input_path, output_dir, gene_sets, *, seed=42, analysis_layer="log1p", analysis_scale="log1p", permutations=1000):
    """Execute all six supplied-data analyses and record a complete manifest."""
    started = time.perf_counter()
    adata = load_supplied_dataset(input_path, output_dir, seed=seed, analysis_layer=analysis_layer, analysis_scale=analysis_scale)
    results = {}
    for name, function in [("domains", plot_domain_maps), ("de", age_de_by_domain), ("latent", plot_proximal_latent)]:
        results[name] = function(adata, output_dir)
    results["hallmark"] = run_hallmark(adata, output_dir, gene_sets, permutations=permutations, seed=seed)
    results["markers"] = plot_marker_profiles(adata, output_dir)
    results["gene_maps"] = plot_gene_maps(adata, output_dir)
    manifest = {"status": "completed", "kind": "analysis_of_supplied_processed_input", "seed": seed,
                "input_sha256": adata.uns["nichebridge_tutorial"]["input_sha256"],
                "runtime_seconds": time.perf_counter() - started, "sections": results}
    _json(_output(output_dir) / "NicheBridge_reproduction_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the NicheBridge kidney supplied-data tutorial")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", type=Path, default=root / "adata_all_rec.h5ad")
    parser.add_argument("--output", type=Path, default=root / "results" / "kidney_tutorial" / "reproduction")
    parser.add_argument("--gmt", type=Path, default=root / "tutorials" / "resources" / "MSigDB_Hallmark_2020.gmt")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_reproduction(args.input, args.output, args.gmt, seed=args.seed, permutations=args.permutations)
    print(json.dumps({"status": result["status"], "runtime_seconds": result["runtime_seconds"], "output": str(args.output)}, indent=2))
