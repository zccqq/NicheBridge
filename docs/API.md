# NicheBridge API

## Preparation

    reference, spatial = nichebridge.pp.prepare(
        reference,
        spatial,
        cell_type_key="cell_type",
        counts_layer=None,
        genes=None,
        gene_selection="all",
        n_genes=2000,
        copy=True,
        marker_thresholds=None,
    )

The function validates raw counts, aligns shared genes, stores them in layers["counts"], and records the ordered panel in uns["nichebridge_preparation"]. It retains all observations. A marker panel is computed from the reference cell labels and reference expression only.

## Estimator

    config = nichebridge.NicheBridgeConfig(
        latent_dim=16,
        hidden_dim=128,
        epochs=1000,
        pretrain_epochs=200,
        pseudospots=5000,
        learning_rate=0.0005,
        kl_weight=0.001,
        bridge_weight=0.0,
        mask_fraction=0.05,
        mask_weight=0.1,
        pseudo_weight=2.0,
        batch_correction=False,
        spatial_weight=0.01,
        marker_weight=0.1,
    )
    model = nichebridge.NicheBridgeDeconvolver(config)
    model.fit(reference, spatial, cell_type_key="cell_type", counts_layer="counts")
    model.transform(spatial)

Alternatively call nichebridge.tl.deconvolve with the same configuration fields as keyword arguments. The fitted model provides fit, predict, transform, save, and load.

## Prediction

model.predict(spatial) returns a dictionary with cell_fraction, rna_fraction, reference_rna_fraction, latent, expected_counts, and low_information. Gene identifiers are aligned to the fitted order and missing genes raise an error. The output DataFrames preserve spatial observation identifiers and cell-type column order.

## Batch covariate

batch_correction=True enables the reference-anchored gene offset. The model records the correction status, offset RMS, and two-level batch vocabulary in metadata. It does not infer or use spatial annotation labels.

## Plotting

    nichebridge.pl.spatial_proportions(
        spatial,
        cell_types=None,
        fraction="cell",
        basis="spatial",
        size=80,
    )

The plotting helper reads the named proportion matrix and spatial coordinates, draws Matplotlib scatter panels, and returns a Matplotlib Figure.

## Configuration defaults

| Parameter | Default |
| --- | --- |
| `latent_dim` | `16` |
| `hidden_dim` | `128` |
| `dropout` | `0.05` |
| `epochs` | `1000` |
| `pretrain_epochs` | `200` |
| `batch_size` | `128` |
| `pseudospots` | `5000` |
| `pseudo_min_cells` | `15` |
| `pseudo_max_cells` | `30` |
| `learning_rate` | `0.0005` |
| `weight_decay` | `1e-05` |
| `kl_weight` | `0.001` |
| `bridge_weight` | `0.0` |
| `bridge_warmup` | `100` |
| `mask_fraction` | `0.05` |
| `mask_weight` | `0.1` |
| `pseudo_weight` | `2.0` |
| `batch_correction` | `False` |
| `platform_penalty` | `0.0` |
| `max_log_platform` | `2.0` |
| `target_sum` | `10000.0` |
| `log_scale` | `4.0` |
| `seed` | `42` |
| `device` | `auto` |
| `log_every` | `50` |
| `spatial_weight` | `0.01` |
| `spatial_k` | `6` |
| `marker_weight` | `0.1` |
| `pseudo_strategy` | `balanced` |
| `pseudo_active_max` | `8` |
| `pseudo_single_fraction` | `0.0` |
| `bridge_time_mode` | `uniform` |
| `composition_temperature` | `0.5` |
| `entropy_weight` | `0.005` |
| `profile_mode` | `size_factor` |
| `pseudo_library_match` | `True` |
| `likelihood` | `poisson` |
| `reference_anchor_weight` | `0.25` |

## Unified entry point

`NicheBridge(mode="pair"|"joint"|"trajectory", config=None, **settings)` delegates to the selected estimator. `fit` returns the facade; `save` and `NicheBridge.load` retain the selected mode. The explicit estimators remain available as `NicheBridgeDeconvolver`, `NicheBridgeJoint`, and `NicheBridgeTrajectory`.

## Joint reference, spatial and bulk analysis

```python
model = nichebridge.NicheBridge(mode="joint", epochs=300, pretrain_epochs=50)
model.fit(reference, spatial, bulk, cell_type_key="cell_type", counts_layer="counts",
          bulk_layer="TPM", bulk_scale="tpm", spatial_group_key="section")
result = model.predict(spatial)
stage = model.map_phenotypes(spatial, bulk.obs[["stage_numeric"]])
```

The joint result contains `cell_fraction`, sparse `spot_cell`, labeled `spot_bulk`, `latent`, `expected_counts`, `low_information`, and ordered `cell_ids`/`genes`. Calling `transform` writes `nichebridge_cell_fraction`, `nichebridge_spot_cell`, `nichebridge_spot_bulk`, and `X_nichebridge` to `obsm`, with metadata in `uns["nichebridge"]`. The mapping function aligns phenotype rows by bulk ID and renormalizes over nonmissing values per trait.

## Generative trajectories

```python
from nichebridge import NicheBridgeTrajectory, TrajectoryConfig

model = NicheBridgeTrajectory(TrajectoryConfig(pretrain_epochs=50, epochs=200))
model.fit({"spatial": spatial, "reference": reference, "bulk": bulk},
          layers={"spatial": "counts", "reference": "counts", "bulk": "TPM"},
          slice_key="section")
latent = model.transform("spatial")
reconstructed = model.reconstruct("spatial")
spot_cell = model.transport("spatial", "reference")
proportions = model.map_cell_types(reference.obs["cell_type"])
phenotype = model.map_phenotypes(bulk.obs[["stage_numeric"]])
```

Input layers must be unlogged, nonnegative profiles. Default preprocessing uses library normalization, log1p, and Scanpy Seurat-dispersion HVG selection (up to 3,000 features). Set `normalize_library=False` for log1p alone or `hvg_method="variance"` for variance-ranked panels. All scale branches share one ordered gene intersection; modalities retain distinct feature spaces. Coordinate enhancement is an optional explicit preprocessing step below.

With diffusion enabled and default library normalization, `reconstruct` and `generate` return **normalized log1p expression on the selected feature panel**, not raw integer counts. With `normalize_library=False`, they return log1p values on the supplied unlogged scale without an additional library normalization. `transform` returns an observation-by-latent-dimension NumPy array; `transport`, `reconstruct`, and `generate` return labeled DataFrames. `transport(target, source)` has target rows and source columns, with row sums equal to one. `map_phenotypes` requires complete numeric data or an indexed categorical Series; resolve missing clinical values explicitly before projection.

```python
omics = NicheBridgeTrajectory(TrajectoryConfig())
omics.fit({"rna": rna, "atac": atac}, paired=True)
rna_from_atac = omics.generate("atac", "rna")
atac_from_rna = omics.generate("rna", "atac")
```

`paired=True` uses identical observation IDs, even when input order differs. `paired=False` learns an unpaired relation. Relations and cross-modal generation are transductive and refer to fitted observations; they are not a held-out missing-modality accuracy estimate. New observations can be encoded with `transform(modality, adata=new_data)` if fitted features and slice identities are available. Supply the same unlogged scale and feature universe used during fitting: library totals are computed before selecting the fitted feature panel, so cropping the input to that panel changes normalization.

Use `max_relation_entries`, `max_matrix_entries`, and `max_trajectory_entries` to control the explicit dense-memory limits. `reconstruct(return_trajectory=True)` additionally returns numerical integration times and states. Ordinary reconstruction does not retain the entire history. Brownian-bridge mathematics, regularizers, and numerical choices are recorded in `METHODS.md`.

Molecular transport penalties act on differentiable reverse ODE states and directly update the score networks. `coupling_steps=16` controls the training rollout; `trajectory_times=2` samples its shared evaluation times. `ode_steps=64` controls inference. Analytic forward samples are used for score matching only.

## Spatial and ATAC preprocessing

```python
activity = nichebridge.pp.gene_activity(atac, peak_to_gene)
spatial = nichebridge.pp.spatial_enhance(spatial, layer="counts", slice_key="section")
```

`peak_to_gene` is a DataFrame with `peak`, `gene`, and optional nonnegative `weight` columns. The user supplies the genome-specific annotation. `spatial_enhance` writes continuous values to `layers["nichebridge_enhanced"]`; use this layer with the trajectory estimator, while retaining integer counts for deconvolution. It never links spots in different supplied slices. The enhancement kernel is an explicit implementation choice because the referenced supplementary formula is absent from the supplied manuscript.

## Spatial domains and niche construction

`nichebridge.analysis.identify_domains` uses Louvain clustering of a latent-neighbor graph with optional spatial edges. `nichebridge.analysis.assemble_niches` combines phenotype fields, cell composition, latent state, domain labels and reconstructed expression into a common matrix. Use labeled DataFrames/Series for identifier-safe alignment.

```python
import pandas as pd
embedding = pd.DataFrame(model.transform("spatial"), index=spatial.obs_names)
domains = nichebridge.analysis.identify_domains(embedding)
niches = nichebridge.analysis.assemble_niches(
    embedding=embedding, domains=domains, composition=proportions,
    reconstruction=model.reconstruct("spatial"))
```

## Trajectory ablation controls

`TrajectoryConfig(use_vae=False)` removes stochastic latent sampling, the variance head and KL terms, retaining diffusion. `TrajectoryConfig(use_diffusion=False)` retains the variational encoder by default but removes score matching and reverse integration. `reconstruct` then returns the endpoint decoder evaluated at the posterior mean; `return_trajectory=True` is unavailable. `endpoint_weight` must be positive in this variant. Both flags can also be disabled together for a deterministic autoencoder.

The complete experiment definitions and parameter sweeps are in `reproducibility/README.md`. They are separate from the installable API.
