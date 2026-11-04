# NicheBridge

**Version 0.0.1**

NicheBridge integrates reference-guided spatial deconvolution, joint single-cell/spatial/bulk modeling, and diffusion trajectory transport in one Python package. It provides spatial cell-type and phenotype maps, latent representations, molecular reconstruction, and multi-slice or cross-modal integration.

## Installation

Python 3.10 or later is required. From the repository root, install the package in a dedicated environment:

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install --upgrade pip
python -m pip install .
```

For the notebook and all six experiments:

```bash
python -m pip install ".[tutorial,reproduction]"
```

For a CUDA installation, install the PyTorch build appropriate for your system before installing NicheBridge. CPU execution is supported. The import name is `nichebridge`.

## Repository layout

```text
.
├── src/nichebridge/             # Installable modeling package
├── tutorials/
│   ├── notebook_kidney.ipynb    # Kidney analysis and fresh model fitting
│   ├── fit_kidney.py
│   ├── kidney_workflow.py
│   └── resources/              # Cached Hallmark gene sets and provenance
├── ablation study/             # Three ablation entry points
├── sensitivity analysis/       # Three sensitivity entry points
├── reproducibility/            # Shared experiment helpers and protocol
├── docs/                       # API and Methods mapping
├── data/README.md              # Required input files and data conventions
├── pyproject.toml
└── MANIFEST.in
```

Datasets, fitted models and generated results are not bundled. Outputs are written to `results/` by default. Evaluation metrics are implemented in the experiment helpers; downstream statistical analyses are in the tutorial. Neither is installed as part of the core modeling package. No R analysis is included.

## Select a model

| Mode | Inputs | Main outputs |
| --- | --- | --- |
| `pair` | Single-cell reference and spatial raw counts | Cell-type proportions, latent representation, expected counts |
| `joint` | Single-cell reference, spatial raw counts and bulk profiles | Cell-type proportions, spot-to-cell and spot-to-bulk relations, phenotype maps |
| `trajectory` | Named molecular datasets, with optional slice IDs or paired observation IDs | Reverse diffusion reconstruction, shared representations and transport relations |

Two-input deconvolution:

```python
import nichebridge as nb

reference, spatial = nb.pp.prepare(
    reference, spatial, cell_type_key="cell_type", gene_selection="hvg", n_genes=3000
)
model = nb.NicheBridge(mode="pair")
model.fit(reference, spatial, cell_type_key="cell_type", counts_layer="counts")
result = model.predict(spatial)
proportions = result["cell_fraction"]
```

Three-input deconvolution and phenotype mapping:

```python
model = nb.NicheBridge(mode="joint")
model.fit(reference, spatial, bulk, cell_type_key="cell_type",
          counts_layer="counts", bulk_layer="TPM", bulk_scale="tpm")
result = model.predict(spatial)
phenotype = model.map_phenotypes(spatial, bulk.obs[["stage_numeric"]])
```

Multi-slice generative transport:

```python
model = nb.NicheBridge(mode="trajectory", seed=42)
model.fit({"spatial": spatial}, layers={"spatial": "counts"}, slice_key="section")
latent = model.transform("spatial")
reconstruction = model.reconstruct("spatial")
model.save("results/kidney_model")
```

These examples assume loaded AnnData objects with unique observation and feature IDs. Trajectory inputs must be nonnegative and unlogged. Reconstruction is on the selected, library-normalized log1p feature scale. See [the API](docs/API.md) for paired RNA/ATAC inputs, preprocessing, mappings and checkpoint loading.

## Kidney tutorial

Place the supplied `adata_all_rec.h5ad` at `data/kidney/adata_all_rec.h5ad`, then start Jupyter from the repository root:

```bash
jupyter lab tutorials/notebook_kidney.ipynb
```

Run all cells in order. An external file can be selected with the `NICHEBRIDGE_KIDNEY_DATA` environment variable or by editing `INPUT_H5AD` in the parameter cell. `NICHEBRIDGE_TUTORIAL_OUTPUT` optionally changes the output folder.

The notebook includes domain maps, age-associated differential expression, proximal-tubule representations, Hallmark enrichment, marker trajectories and reconstructed gene maps. These six sections use explicitly identified supplied annotations and representations. A separate fresh NicheBridge fit uses recovered integer counts from 32 spots per section and 256 genes, with 20 pretraining and 40 joint updates. It saves a new checkpoint, representation and reconstruction. The notebook distinguishes these outputs from the supplied full-dataset results. See [input requirements](data/README.md).

## Ablation and sensitivity analyses

The six entry points are grouped by experiment type. Paths containing spaces must be quoted:

```bash
python "ablation study/01_luad_ablation.py" --data-root /path/to/datasets
python "ablation study/02_mouse_brain_ablation.py" --data-root /path/to/datasets
python "ablation study/03_kidney_ablation.py" --input data/kidney/adata_all_rec.h5ad
python "sensitivity analysis/01_luad_sensitivity.py" --data-root /path/to/datasets
python "sensitivity analysis/02_mouse_brain_sensitivity.py" --data-root /path/to/datasets
python "sensitivity analysis/03_kidney_sensitivity.py" --input data/kidney/adata_all_rec.h5ad
```

Use `--help` for options. [The experiment protocol](reproducibility/README.md) specifies ablations, parameter meanings, evaluation and sampling. The kidney experiments explicitly default to 128 spots per section and 512 genes because trajectory relations are dense. These runs produce new results for this implementation; no archived scores are substituted.

## Methods and scope

[Methods documentation](docs/METHODS.md) maps modeling components to their APIs and records the numerical choices. The implementation uses Brownian bridges, differentiable reverse Heun trajectories and trajectory coupling. Its symmetric, size-normalized objectives and additional regularizers are explicit implementation choices. The code does not establish exact equality to unspecified manuscript settings or reproduction of reported cohort-level performance. Full-dataset convergence and scientific evaluation require the stated input data and experiment settings.
