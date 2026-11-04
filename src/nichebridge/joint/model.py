"""Train and export joint expression mappings without clinical supervision."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.nn import functional as F
from nichebridge._bridge import VPBridge
from nichebridge._data import reference_profiles
from nichebridge._mixtures import make_target_matched_pseudospots
from nichebridge._network import poisson_loss, negative_binomial_loss

from .config import NicheBridgeJointConfig
from .data import (backend_identity, raw_counts, bulk_profiles, validate_axes,
                   make_projection, make_spatial_projection, features, normalized, candidates,
                   spatial_edges, fingerprint)
from .network import JointNetwork


class NicheBridgeJoint:
    """Joint spot-cell and spot-bulk mappings using the current installed backend.

    Epoch settings count sampled optimization updates, as in upstream NicheBridgeDeconvolver.
    Bulk expression is continuous panel-normalized TPM, never count likelihood data.
    """

    def __init__(self, config=None, **kwargs):
        if config is not None and kwargs:
            raise ValueError("Supply either a configuration or keyword settings")
        self.config = config or NicheBridgeJointConfig(**kwargs)
        if not isinstance(self.config, NicheBridgeJointConfig):
            raise TypeError("config must be NicheBridgeJointConfig")

    def _device(self):
        name = self.config.device
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(name)

    def _tensor(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.device_)

    def _dictionary(self, ids):
        shape = ids.shape
        flat = ids.ravel()
        x = self.reference_[flat].toarray()
        alpha = self.config.dictionary_pseudocount
        x = (x + alpha * self.signatures_[self.codes_[flat]]) / (self.libraries_[flat, None] + alpha)
        return self._tensor(x.reshape(*shape, -1))

    def _candidate_arrays(self, x):
        keys = normalized(features(x, self.config) @ self.projection_)
        return candidates(keys, self.reference_keys_, self.codes_, self.informative_reference_,
                          len(self.types_), self.config.candidates_per_type)

    def _count_loss(self, x, expected):
        if self.config.likelihood == "nb":
            return negative_binomial_loss(x, expected, self.network_.core.log_dispersion,
                                         self.network_.core.gene_weights)
        return poisson_loss(x, expected, self.network_.core.gene_weights)

    def _set_seed(self):
        c = self.config
        np.random.seed(c.seed)
        torch.manual_seed(c.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(c.seed)

    def _fit_spatial_bulk(self, spatial, bulk, *, counts_layer=None, bulk_layer="TPM",
                          bulk_scale="tpm", spatial_group_key=None, verbose=True):
        """Fit the no-scRNA ablation, retaining only spatial-to-bulk relations."""
        started = time.perf_counter()
        c = self.config
        self._set_seed()
        rng = np.random.default_rng(c.seed)
        self.device_ = self._device()
        validate_axes(spatial)
        validate_axes(bulk)
        shared = set(spatial.var_names) & set(bulk.var_names)
        self.genes_ = [str(g) for g in spatial.var_names if g in shared]
        if len(self.genes_) < 2:
            raise ValueError("At least two genes must be shared by spatial and bulk inputs")
        st = raw_counts(spatial, self.genes_, counts_layer)
        self.bulk_ = bulk_profiles(bulk, self.genes_, bulk_layer, bulk_scale)
        st_lib = np.asarray(st.sum(axis=1)).ravel()
        informative_st = st_lib > 0
        if not informative_st.any():
            raise ValueError("At least one spatial observation must contain counts")
        # A generic spatial profile keeps the decoder well-defined without scRNA.
        signature = np.asarray(st[informative_st].mean(axis=0)).ravel().astype(np.float32)
        signature = signature / max(float(signature.sum()), 1e-12)
        self.signatures_ = signature[None, :]
        self.yields_ = np.ones(1, dtype=np.float32)
        self.projection_, spatial_keys = make_spatial_projection(st, c)
        self.reference_keys_ = spatial_keys[:1]
        self.network_ = JointNetwork(self.signatures_, self.yields_, self.projection_,
                                     self.reference_keys_, c).to(self.device_)
        bridge = VPBridge(time_mode=c.bridge_time_mode)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=c.learning_rate,
                                      weight_decay=c.weight_decay)
        st_pool = np.flatnonzero(informative_st)
        edges = spatial_edges(spatial, informative_st, c.spatial_k, spatial_group_key)
        bulk_x = self._tensor(self.bulk_ * c.target_sum)
        bulk_domain = torch.full((len(bulk_x),), 2, dtype=torch.long, device=self.device_)
        self.history_ = []
        self.initial_parameter_digest_ = self._parameter_digest()
        total_steps = c.pretrain_epochs + c.epochs
        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            self.network_.train()
            picked = rng.choice(st_pool, min(c.batch_size, len(st_pool)), replace=False)
            sx = self._tensor(st[picked].toarray())
            domain = torch.ones(len(sx), dtype=torch.long, device=self.device_)
            sp, sz, sm, sv, _ = self.network_.encode(sx, domain, sample=True)
            bp, bz, bm, bv, _ = self.network_.encode(bulk_x, bulk_domain, sample=True)
            w, profile = self.network_.bulk_relation(sx, domain, sm, bulk_x, bm)
            count = self._count_loss(sx, profile * sx.sum(dim=1, keepdim=True))
            bulk_loss = F.mse_loss(self.network_.profile_features(profile), self.network_.core.features(sx))
            kl = sx.new_zeros(())
            if c.use_vae:
                kl = -0.5 * (1 + sv - sm.square() - sv.exp()).mean()
                kl = kl - 0.5 * (1 + bv - bm.square() - bv.exp()).mean()
            bridge_loss = sx.new_zeros(())
            joint_step = max(0, step - c.pretrain_epochs + 1)
            ramp = min(1.0, joint_step / max(c.bridge_warmup, 1))
            spatial_loss = sx.new_zeros(())
            if c.spatial_weight > 0 and len(edges):
                edge = edges[rng.integers(len(edges), size=min(c.batch_size, len(edges)))]
                vertices, inverse = np.unique(edge, return_inverse=True)
                ex = self._tensor(st[vertices].toarray())
                ed = torch.ones(len(ex), dtype=torch.long, device=self.device_)
                em = self.network_.encode(ex, ed)[2]
                ew = self.network_.bulk_relation(ex, ed, em, bulk_x, bm)[0]
                pairs = self._tensor(inverse.reshape(-1, 2), torch.long)
                spatial_loss = F.mse_loss(ew[pairs[:, 0]], ew[pairs[:, 1]])
            if c.use_diffusion and c.bridge_weight > 0 and joint_step > 0:
                bridge_loss = bridge.loss(self.network_, self.network_.core.features(sx),
                                          self.network_.profile_features(profile), sz, domain)
                bridge_loss = bridge_loss + bridge.loss(self.network_, self.network_.core.features(bulk_x),
                                                        self.network_.profile_features(bulk_x),
                                                        bz, bulk_domain)
            penalty = self.network_.platform_logs()[1:].square().mean()
            loss = (count + c.bulk_weight * bulk_loss + c.kl_weight * kl
                    + c.platform_penalty * penalty + c.spatial_weight * spatial_loss
                    + c.bridge_weight * ramp * bridge_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite objective at update {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network_.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            row = {"step": step + 1, "phase": "joint", "loss": float(loss.detach().cpu()),
                   "count": float(count.detach().cpu()), "pseudo": 0.0,
                   "bulk": float(bulk_loss.detach().cpu()), "bulk_reference": 0.0,
                   "consistency": 0.0, "bridge": float(bridge_loss.detach().cpu()),
                   "spatial": float(spatial_loss.detach().cpu()), "kl": float(kl.detach().cpu()),
                   "platform": float(penalty.detach().cpu())}
            self.history_.append(row)
            if verbose and (step == 0 or (step + 1) % c.log_every == 0 or step + 1 == total_steps):
                print(f"joint {step + 1}/{total_steps}: loss={row['loss']:.5f}, bridge={row['bridge']:.5f}", flush=True)
        self.history_ = pd.DataFrame(self.history_)
        self.counts_layer_ = counts_layer
        self.cell_ids_ = []
        self.bulk_ids_ = list(bulk.obs_names)
        self.types_ = []
        self.codes_ = np.empty(0, dtype=np.int64)
        self.reference_ = sparse.csr_matrix((0, len(self.genes_)), dtype=np.float32)
        self.libraries_ = np.empty(0, dtype=np.float32)
        self.informative_reference_ = np.empty(0, dtype=bool)
        self.metadata_ = {
            "format": "nichebridge-joint-v2", "backend": backend_identity(),
            "config": c.to_dict(), "mode": "no_single_cell",
            "n_cells": 0, "n_spots": spatial.n_obs, "n_bulk": len(self.bulk_ids_),
            "n_genes": len(self.genes_), "types": [], "pseudospots_actual": 0,
            "zero_reference_cells_excluded_from_candidates": 0,
            "zero_spots_excluded_from_training": int((~informative_st).sum()),
            "input_expression_sha256": {
                "spatial": fingerprint(st, spatial.obs_names, self.genes_),
                "bulk_panel_profile": fingerprint(self.bulk_, self.bulk_ids_, self.genes_)},
            "spatial_group_key": spatial_group_key, "spatial_edges": len(edges),
            "bulk_input_layer": bulk_layer, "bulk_input_scale": bulk_scale,
            "training_seconds": time.perf_counter() - started, "device": str(self.device_),
            "initial_parameter_digest": self.initial_parameter_digest_,
            "final_parameter_digest": self._parameter_digest(),
            "interpretation": "No-scRNA ablation; only spatial-to-bulk expression compatibility is identified."}
        self.network_.eval()
        return self

    def fit(self, reference, spatial, bulk, *, cell_type_key="cell_type", counts_layer=None,
            bulk_layer="TPM", bulk_scale="tpm", spatial_group_key=None, verbose=True):
        if not self.config.use_single_cell:
            return self._fit_spatial_bulk(
                spatial, bulk, counts_layer=counts_layer, bulk_layer=bulk_layer,
                bulk_scale=bulk_scale, spatial_group_key=spatial_group_key, verbose=verbose)
        started = time.perf_counter()
        c = self.config
        self._set_seed()
        rng = np.random.default_rng(c.seed)
        self.device_ = self._device()
        for item in (reference, spatial, bulk):
            validate_axes(item)
        shared = set(reference.var_names) & set(bulk.var_names)
        self.genes_ = [str(g) for g in spatial.var_names if g in shared]
        if len(self.genes_) < 2:
            raise ValueError("At least two genes must be shared by all three inputs")
        labels = reference.obs[cell_type_key]
        if labels.isna().any():
            raise ValueError("Reference cell-type labels cannot be missing")
        labels = labels.astype(str).to_numpy()
        self.types_ = sorted(set(labels))
        self.codes_ = pd.Categorical(labels, categories=self.types_).codes.astype(np.int64)
        self.cell_ids_ = list(reference.obs_names)
        self.bulk_ids_ = list(bulk.obs_names)
        self.reference_ = raw_counts(reference, self.genes_, counts_layer)
        st = raw_counts(spatial, self.genes_, counts_layer)
        self.bulk_ = bulk_profiles(bulk, self.genes_, bulk_layer, bulk_scale)
        self.libraries_ = np.asarray(self.reference_.sum(axis=1)).ravel()
        self.informative_reference_ = self.libraries_ > 0
        st_lib = np.asarray(st.sum(axis=1)).ravel()
        informative_st = st_lib > 0
        if not informative_st.any():
            raise ValueError("At least one spatial observation must contain counts")
        self.signatures_, self.yields_ = reference_profiles(
            self.reference_[self.informative_reference_], labels[self.informative_reference_],
            self.types_, mode=c.profile_mode)
        self.projection_, self.reference_keys_ = make_projection(
            self.reference_, self.informative_reference_, c)
        self.network_ = JointNetwork(self.signatures_, self.yields_, self.projection_,
                                     self.reference_keys_, c).to(self.device_)
        bridge = VPBridge(time_mode=c.bridge_time_mode)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=c.learning_rate,
                                      weight_decay=c.weight_decay)
        # Only raw expression, reference labels and optional coordinates enter training.
        n_pseudo = min(c.pseudospots, 10 * int(informative_st.sum()))
        pseudo, fractions, rna = make_target_matched_pseudospots(
            self.reference_[self.informative_reference_], labels[self.informative_reference_],
            self.types_, st_lib[informative_st], n_pseudo, c.pseudo_min_cells,
            c.pseudo_max_cells, c.seed, strategy=c.pseudo_strategy,
            active_max=c.pseudo_active_max, single_fraction=c.pseudo_single_fraction)
        ids, valid = self._candidate_arrays(st)
        edges = spatial_edges(spatial, informative_st, c.spatial_k, spatial_group_key)
        st_pool = np.flatnonzero(informative_st)
        ref_pool = np.flatnonzero(self.informative_reference_)
        bulk_x = self._tensor(self.bulk_ * c.target_sum)
        bulk_domain = torch.full((len(bulk_x),), 2, dtype=torch.long, device=self.device_)
        self.history_ = []
        self.initial_parameter_digest_ = self._parameter_digest()
        for step in range(c.pretrain_epochs + c.epochs):
            pretrain = step < c.pretrain_epochs
            optimizer.zero_grad(set_to_none=True)
            self.network_.train()
            chosen = rng.integers(n_pseudo, size=min(c.batch_size, n_pseudo))
            px, target, target_rna = self._tensor(pseudo[chosen]), self._tensor(fractions[chosen]), self._tensor(rna[chosen])
            zero = torch.zeros(len(px), dtype=torch.long, device=self.device_)
            pp, pz, pm, pv, observed = self.network_.encode(px, zero, mask=True, sample=True)
            pq = self.network_.decode_types(pp, zero)
            pseudo_ce = -(target * pp.clamp_min(1e-8).log()).sum(dim=1).mean()
            predicted_rna = pp * self.network_.core.rna_yield
            predicted_rna = predicted_rna / predicted_rna.sum(dim=1, keepdim=True)
            pseudo_ce = pseudo_ce - 0.5 * (target_rna * predicted_rna.clamp_min(1e-8).log()).sum(dim=1).mean()
            count = self._count_loss(px, pq * px.sum(dim=1, keepdim=True))
            kl = (-0.5 * (1 + pv - pm.square() - pv.exp()).mean()
                  if c.use_vae else px.new_zeros(()))
            mask_loss = ((1 - observed) * (self.network_.profile_features(pq) - self.network_.core.features(px)).square()).mean()
            bridge_loss = px.new_zeros(())
            joint_step = max(0, step - c.pretrain_epochs + 1)
            ramp = min(1.0, joint_step / max(c.bridge_warmup, 1)) if not pretrain else 0.0
            bulk_loss = px.new_zeros(())
            bulk_ref = px.new_zeros(())
            consistency = px.new_zeros(())
            spatial_loss = px.new_zeros(())
            entropy = px.new_zeros(())
            if not pretrain:
                picked = rng.choice(st_pool, min(c.batch_size, len(st_pool)), replace=False)
                sx = self._tensor(st[picked].toarray())
                domain = torch.ones(len(sx), dtype=torch.long, device=self.device_)
                sp, sz, sm, sv, _ = self.network_.encode(sx, domain, sample=True)
                _, cell_profile = self.network_.cell_relation(
                    sx, domain, sp, sm, self._tensor(ids[picked], torch.long),
                    self._tensor(valid[picked], torch.bool), self._dictionary(ids[picked]))
                count = count + self._count_loss(sx, cell_profile * sx.sum(dim=1, keepdim=True))
                if c.use_vae:
                    kl = kl - 0.5 * (1 + sv - sm.square() - sv.exp()).mean()
                bp, bz, bm, bv, _ = self.network_.encode(bulk_x, bulk_domain, sample=True)
                w, bulk_profile = self.network_.bulk_relation(sx, domain, sm, bulk_x, bm)
                bulk_loss = F.mse_loss(self.network_.profile_features(bulk_profile), self.network_.core.features(sx))
                decoded_bulk = self.network_.decode_types(bp, bulk_domain)
                bulk_ref = F.mse_loss(self.network_.profile_features(decoded_bulk), self.network_.core.features(bulk_x))
                consistency = F.mse_loss(sp, w @ bp)
                if c.use_vae:
                    kl = kl - 0.5 * (1 + bv - bm.square() - bv.exp()).mean()
                entropy = -(sp * sp.clamp_min(1e-8).log()).sum(dim=1).mean()
                # Reference cells anchor the native composition encoder to named cell types.
                ref_idx = rng.choice(ref_pool, min(c.batch_size, len(ref_pool)), replace=False)
                rx = self._tensor(self.reference_[ref_idx].toarray())
                rd = torch.zeros(len(rx), dtype=torch.long, device=self.device_)
                rp = self.network_.encode(rx, rd)[0]
                pseudo_ce = pseudo_ce + F.nll_loss(rp.clamp_min(1e-8).log(), self._tensor(self.codes_[ref_idx], torch.long))
                if c.spatial_weight > 0 and len(edges):
                    edge = edges[rng.integers(len(edges), size=min(c.batch_size, len(edges)))]
                    vertices, inverse = np.unique(edge, return_inverse=True)
                    ex = self._tensor(st[vertices].toarray())
                    ed = torch.ones(len(ex), dtype=torch.long, device=self.device_)
                    ep = self.network_.encode(ex, ed)[0]
                    pairs = self._tensor(inverse.reshape(-1, 2), torch.long)
                    spatial_loss = F.mse_loss(ep[pairs[:, 0]], ep[pairs[:, 1]])
                if c.use_diffusion and c.bridge_weight > 0:
                    bridge_loss = bridge.loss(self.network_, self.network_.core.features(sx),
                        self.network_.profile_features(cell_profile), sz, domain)
                    bridge_loss = bridge_loss + bridge.loss(self.network_, self.network_.core.features(bulk_x),
                        self.network_.profile_features(decoded_bulk), bz, bulk_domain)
            penalty = self.network_.platform_logs()[1:].square().mean()
            loss = (count + c.pseudo_weight * pseudo_ce + c.kl_weight * kl + c.mask_weight * mask_loss
                    + c.bulk_weight * bulk_loss + c.bulk_reference_weight * bulk_ref
                    + c.consistency_weight * consistency + c.spatial_weight * spatial_loss
                    + c.entropy_weight * entropy + c.platform_penalty * penalty
                    + c.bridge_weight * ramp * bridge_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite objective at update {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network_.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            row = {"step": step + 1, "phase": "pretrain" if pretrain else "joint"}
            for name, value in (("loss", loss), ("count", count), ("pseudo", pseudo_ce),
                                ("bulk", bulk_loss), ("bulk_reference", bulk_ref),
                                ("consistency", consistency), ("bridge", bridge_loss),
                                ("spatial", spatial_loss), ("kl", kl), ("platform", penalty)):
                row[name] = float(value.detach().cpu())
            self.history_.append(row)
            if verbose and (step == 0 or (step + 1) % c.log_every == 0 or step + 1 == c.pretrain_epochs + c.epochs):
                print(f"{row['phase']} {step + 1}/{c.pretrain_epochs + c.epochs}: loss={row['loss']:.5f}, bridge={row['bridge']:.5f}", flush=True)
        self.history_ = pd.DataFrame(self.history_)
        self.counts_layer_ = counts_layer
        self.metadata_ = {
            "format": "nichebridge-joint-v2", "backend": backend_identity(), "config": c.to_dict(),
            "mode": "full",
            "n_cells": len(self.cell_ids_), "n_spots": spatial.n_obs, "n_bulk": len(self.bulk_ids_),
            "n_genes": len(self.genes_), "types": self.types_, "pseudospots_actual": n_pseudo,
            "zero_reference_cells_excluded_from_candidates": int((~self.informative_reference_).sum()),
            "zero_spots_excluded_from_training": int((~informative_st).sum()),
            "input_expression_sha256": {"reference": fingerprint(self.reference_, self.cell_ids_, self.genes_),
                "spatial": fingerprint(st, spatial.obs_names, self.genes_),
                "bulk_panel_profile": fingerprint(self.bulk_, self.bulk_ids_, self.genes_)},
            "reference_labels_sha256": hashlib.sha256(json.dumps(labels.tolist()).encode()).hexdigest(),
            "spatial_group_key": spatial_group_key, "spatial_edges": len(edges),
            "bulk_input_layer": bulk_layer, "bulk_input_scale": bulk_scale,
            "training_seconds": time.perf_counter() - started, "device": str(self.device_),
            "initial_parameter_digest": self.initial_parameter_digest_,
            "final_parameter_digest": self._parameter_digest(),
            "interpretation": "Expression compatibility mappings; individual-cell locations and clinical probabilities are not identified."}
        self.network_.eval()
        return self

    def _parameter_digest(self):
        h = hashlib.sha256()
        for p in self.network_.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def _score_bulk_profiles(self, spatial, bulk_profile, bulk_ids, *, return_reconstruction=True):
        """Score arbitrary bulk profiles against fitted spatial observations."""
        layer = self.counts_layer_
        st = raw_counts(spatial, self.genes_, layer)
        low = np.asarray(st.sum(axis=1)).ravel() == 0
        c = self.config
        matrices, latent, expected = [], [], []
        self.network_.eval()
        with torch.no_grad():
            bx = self._tensor(bulk_profile * c.target_sum)
            bd = torch.full((len(bx),), 2, dtype=torch.long, device=self.device_)
            bm = self.network_.encode(bx, bd)[2]
            for start in range(0, spatial.n_obs, c.batch_size):
                stop = min(spatial.n_obs, start + c.batch_size)
                sx = self._tensor(st[start:stop].toarray())
                domain = torch.ones(len(sx), dtype=torch.long, device=self.device_)
                _, _, mean, _, _ = self.network_.encode(sx, domain)
                w, profile = self.network_.bulk_relation(sx, domain, mean, bx, bm)
                local_low = low[start:stop]
                if local_low.any():
                    w[local_low] = 1 / len(bulk_ids)
                    mean[local_low] = 0
                matrices.append(w.cpu().numpy())
                if return_reconstruction:
                    expected.append((profile * sx.sum(dim=1, keepdim=True)).cpu().numpy())
                latent.append(mean.cpu().numpy())
        return {
            "spot_bulk": pd.DataFrame(np.vstack(matrices), index=spatial.obs_names,
                                      columns=list(bulk_ids)),
            "latent": np.vstack(latent),
            "expected_counts": np.vstack(expected) if return_reconstruction else None,
            "low_information": low,
        }

    def predict(self, spatial, *, counts_layer=None):
        if not hasattr(self, "network_"):
            raise RuntimeError("Fit or load the model first")
        layer = self.counts_layer_ if counts_layer is None else counts_layer
        if not self.config.use_single_cell:
            original = self.counts_layer_
            self.counts_layer_ = layer
            try:
                out = self._score_bulk_profiles(spatial, self.bulk_, self.bulk_ids_)
            finally:
                self.counts_layer_ = original
            out.update({"spot_cell": None, "cell_fraction": None,
                        "cell_ids": [], "genes": self.genes_.copy()})
            return out
        st = raw_counts(spatial, self.genes_, layer)
        low = np.asarray(st.sum(axis=1)).ravel() == 0
        ids, valid = self._candidate_arrays(st)
        c = self.config
        matrices, bulk_weights, latent, expected, type_rows = [], [], [], [], []
        self.network_.eval()
        with torch.no_grad():
            bx = self._tensor(self.bulk_ * c.target_sum)
            bd = torch.full((len(bx),), 2, dtype=torch.long, device=self.device_)
            bm = self.network_.encode(bx, bd)[2]
            for start in range(0, spatial.n_obs, c.batch_size):
                stop = min(spatial.n_obs, start + c.batch_size)
                sx = self._tensor(st[start:stop].toarray())
                domain = torch.ones(len(sx), dtype=torch.long, device=self.device_)
                p, _, mean, _, _ = self.network_.encode(sx, domain)
                weights, profile = self.network_.cell_relation(
                    sx, domain, p, mean, self._tensor(ids[start:stop], torch.long),
                    self._tensor(valid[start:stop], torch.bool), self._dictionary(ids[start:stop]))
                w, _ = self.network_.bulk_relation(sx, domain, mean, bx, bm)
                rows = np.repeat(np.arange(len(sx)), ids.shape[1] * ids.shape[2])
                a = sparse.csr_matrix((weights.cpu().numpy().ravel(),
                    (rows, ids[start:stop].ravel())), shape=(len(sx), len(self.cell_ids_)))
                a.eliminate_zeros()
                local_low = low[start:stop]
                if local_low.any():
                    # A no-information spot has a uniform prior over all individual cells.
                    a = a.tolil()
                    for row in np.flatnonzero(local_low):
                        a[row, :] = np.full(len(self.cell_ids_), 1 / len(self.cell_ids_))
                    a = a.tocsr()
                    w[local_low] = 1 / len(self.bulk_ids_)
                    mean[local_low] = 0
                aggregation = sparse.csr_matrix((np.ones(len(self.codes_), dtype=np.float32),
                    (np.arange(len(self.codes_)), self.codes_)), shape=(len(self.codes_), len(self.types_)))
                type_rows.append((a @ aggregation).toarray())
                matrices.append(a)
                bulk_weights.append(w.cpu().numpy())
                latent.append(mean.cpu().numpy())
                expected.append((profile * sx.sum(dim=1, keepdim=True)).cpu().numpy())
        return {"spot_cell": sparse.vstack(matrices, format="csr"),
                "spot_bulk": pd.DataFrame(np.vstack(bulk_weights), index=spatial.obs_names, columns=self.bulk_ids_),
                "cell_fraction": pd.DataFrame(np.vstack(type_rows), index=spatial.obs_names, columns=self.types_),
                "latent": np.vstack(latent), "expected_counts": np.vstack(expected),
                "low_information": low, "cell_ids": self.cell_ids_.copy(), "genes": self.genes_.copy()}

    def score_bulk(self, spatial, bulk, *, bulk_layer="TPM", bulk_scale="tpm",
                   return_reconstruction=True):
        """Return spot weights for external bulk samples without refitting."""
        if not hasattr(self, "network_"):
            raise RuntimeError("Fit or load the model first")
        profiles = bulk_profiles(bulk, self.genes_, bulk_layer, bulk_scale)
        return self._score_bulk_profiles(spatial, profiles, bulk.obs_names,
                                         return_reconstruction=return_reconstruction)

    def transform(self, spatial, *, counts_layer=None, copy=False):
        target = spatial.copy() if copy else spatial
        out = self.predict(target, counts_layer=counts_layer)
        target.obsm["nichebridge_spot_bulk"] = out["spot_bulk"]
        target.obsm["X_nichebridge"] = out["latent"]
        target.obs["nichebridge_low_information"] = out["low_information"]
        metadata = {"format": "nichebridge-joint-v2", "mode": self.metadata_.get("mode", "full"),
            "cell_ids": self.cell_ids_, "cell_type_codes": self.codes_, "types": self.types_,
            "bulk_ids": self.bulk_ids_, "genes": self.genes_}
        if self.config.use_single_cell:
            target.obsm["nichebridge_spot_cell"] = out["spot_cell"]
            target.obsm["nichebridge_cell_fraction"] = out["cell_fraction"]
        target.uns["nichebridge"] = metadata
        return target

    def map_phenotypes(self, spatial, phenotypes, *, counts_layer=None):
        """Project numeric bulk phenotypes after fitting, aligned by sample ID.

        For each phenotype, missing bulk values are excluded and the remaining
        weights are renormalized. A phenotype without any observed value raises
        an error. The expression model is never retrained using these labels.
        """
        if isinstance(phenotypes, pd.Series):
            phenotypes = phenotypes.to_frame()
        if not isinstance(phenotypes, pd.DataFrame) or not phenotypes.index.is_unique:
            raise ValueError("phenotypes must have unique labeled sample rows")
        missing = pd.Index(self.bulk_ids_).difference(phenotypes.index)
        if len(missing):
            raise ValueError(f"Missing bulk phenotype rows: {missing[:5].tolist()}")
        values = phenotypes.loc[self.bulk_ids_].to_numpy(dtype=float)
        if np.isinf(values).any() or values.shape[1] == 0:
            raise ValueError("Phenotypes must have numeric finite or missing values")
        observed = ~np.isnan(values)
        if not observed.any(axis=0).all():
            raise ValueError("Every phenotype needs at least one observed value")
        weights = self.predict(spatial, counts_layer=counts_layer)["spot_bulk"].to_numpy()
        scores = (weights @ np.nan_to_num(values)) / (weights @ observed).clip(1e-12)
        return pd.DataFrame(scores, index=spatial.obs_names, columns=phenotypes.columns)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        state = {"format": "nichebridge-joint-v2", "config": self.config.to_dict(),
                 "metadata": self.metadata_, "counts_layer": self.counts_layer_,
                 "genes": self.genes_, "types": self.types_, "cell_ids": self.cell_ids_,
                 "bulk_ids": self.bulk_ids_, "state_dict": {k: v.detach().cpu() for k, v in self.network_.state_dict().items()}}
        torch.save(state, folder / "model.pt")
        sparse.save_npz(folder / "reference_counts.npz", self.reference_)
        np.savez_compressed(folder / "dictionary.npz", codes=self.codes_, bulk=self.bulk_,
                            signatures=self.signatures_, yields=self.yields_, projection=self.projection_,
                            reference_keys=self.reference_keys_)
        self.history_.to_csv(folder / "history.csv", index=False)
        (folder / "metadata.json").write_text(json.dumps(self.metadata_, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, folder, *, device="auto"):
        folder = Path(folder)
        state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        if state.get("format") != "nichebridge-joint-v2":
            raise ValueError("Unsupported checkpoint; old triomics checkpoints cannot be loaded")
        identity = backend_identity()
        if identity["sha256"] != state["metadata"]["backend"]["sha256"]:
            raise ValueError("The installed NicheBridgeDeconvolver source differs from the fitted backend")
        self = cls(replace(NicheBridgeJointConfig(**state["config"]), device=device))
        self.device_ = self._device()
        self.metadata_ = state["metadata"]
        self.counts_layer_ = state["counts_layer"]
        for name in ("genes", "types", "cell_ids", "bulk_ids"):
            setattr(self, name + "_", state[name])
        self.reference_ = sparse.load_npz(folder / "reference_counts.npz")
        with np.load(folder / "dictionary.npz", allow_pickle=False) as data:
            for name in ("codes", "bulk", "signatures", "yields", "projection", "reference_keys"):
                setattr(self, name + "_", data[name])
        self.libraries_ = np.asarray(self.reference_.sum(axis=1)).ravel()
        self.informative_reference_ = self.libraries_ > 0
        self.network_ = JointNetwork(self.signatures_, self.yields_, self.projection_, self.reference_keys_, self.config).to(self.device_)
        self.network_.load_state_dict(state["state_dict"], strict=True)
        self.network_.eval()
        self.history_ = pd.read_csv(folder / "history.csv")
        return self
