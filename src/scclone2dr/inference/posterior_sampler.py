"""Posterior sampler for scClone2DR.

:class:`PosteriorSampler` takes a **fitted** :class:`~scclone2dr.model.scClone2DR`
model, a guide, and learned parameters, then draws Monte-Carlo samples from
the posterior to accumulate statistics of interest (survival probabilities,
local importances, log-odds ratios, etc.) and persist them to HDF5.

It has no knowledge of optimisation or data loading.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Callable

import h5py
import numpy as np
import pyro
import torch
from tqdm import tqdm

from ..utils import masked_softmax

logger = logging.getLogger(__name__)

_POSTERIOR_VARS_DATA = (
    "n0_c", "n0_r", "n_rna", "frac_r", "frac_c",
    "frac_mean_r", "frac_mean_c", "log_score"
)

_POSTERIOR_VARS_PARAMS = (
    "pi", "nu_healthy_drug", "nu_healthy_control",
)


class PosteriorSampler:
    """Monte-Carlo posterior sampler for the scClone2DR generative model.

    Parameters
    ----------
    model : scClone2DR
        A fitted model instance.  Only its ``compute_survival_probas_*``,
        ``get_survival_probas``, and ``sampling`` methods are called.
    guide : callable
        The fitted Pyro guide (e.g. ``AutoMultivariateNormal``).  Used to
        draw latent samples via ``guide.sample_latent()``.  When the guide
        does not expose ``sample_latent``, *guide_distribution* is used as
        a fallback.
    guide_distribution : torch.distributions.Distribution, optional
        Fallback distribution to sample from when *guide* does not support
        ``sample_latent`` (e.g. when parameters have been loaded from disk).

    Examples
    --------
    >>> sampler = PosteriorSampler(model=fitted_model, guide=trainer.guide)
    >>> results = sampler.sample(data, params, nb_ites=200)
    >>> sampler.save_results(results, dir_save="./outputs/", data=data,
    ...                      sample_names=sample_names)
    """

    def __init__(
        self,
        model,
        guide: Callable | None = None,
        guide_distribution=None,
    ) -> None:
        self._model              = model
        self._guide              = guide
        self._guide_distribution = guide_distribution

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        data: dict,
        params: dict,
        idxs_sample: list | None = None,
        *,
        nb_ites: int = 100,
    ) -> dict:
        """Draw *nb_ites* posterior samples and return accumulated statistics.

        Parameters
        ----------
        data:
            Observed data dictionary.
        params:
            Learned parameter dictionary (output of ``Trainer.train``).
        idxs_sample:
            Optional list of sample indices to include in the posterior sampling.
            If ``None``, all samples are included.
        nb_ites:
            Number of Monte-Carlo posterior samples.

        Returns
        -------
        dict
            Accumulated posterior means for all quantities in
            :data:`_POSTERIOR_VARS`, plus ``"PI"``, ``"LOR"``, ``"LRR"``,
            ``"ME"``, ``"subclone_features"``.
        """
        N, D, R, Kmax  = data["N"], data["D"], data["R"], data["Kmax"]
        if idxs_sample is None:
            idxs_sample = np.arange(N)
            assert params['proportions'].shape[0] == N, "If inference is made on all samples, the first dimension of 'proportions' in params must match N."
        else:
            assert len(idxs_sample) == N, "Length of idxs_sample must match number of samples in data."
        latent_dim     = params["beta"].shape[1]
        n_clonelabels  = self._model.n_clonelabels
        clonelabels    = self._model.clonelabels

        if self._guide is None and self._guide_distribution is None:
            nb_ites = 1
            using_clone_features = True
            dim = None
            samp_gamma = None
        else:
            dim = len(self._draw_latent(params)) // n_clonelabels
            using_clone_features = False

        # Zero-initialise accumulators
        PI               = torch.zeros((D, Kmax, N))
        LOR              = torch.zeros((D, Kmax, N, latent_dim))
        LRR              = torch.zeros((D, Kmax, N, latent_dim))
        ME               = torch.zeros((D, Kmax, N, latent_dim))
        subclone_features = torch.zeros((Kmax, N, latent_dim))
        accumulator      = self._init_accumulator(data, params, dim, idxs_sample=idxs_sample, using_clone_features=using_clone_features)

        with torch.no_grad():
            for _ in tqdm(range(nb_ites)):
                if not(using_clone_features):
                    samp_gamma = self._draw_latent(params)
                    self._assign_gamma(params, samp_gamma, dim, clonelabels)
                    pi    = self._model.compute_survival_probas_single_cell_features(
                        data, params
                    )
                    self._accumulate_subclone_features(
                        subclone_features, data, params, nb_ites, clonelabels
                    )
                else:
                    pi = self._model.compute_survival_probas_subclone_features(data, params)
                ghost_data, ghost_params = self._model.sampling(data, params, idxs_sample=idxs_sample)

                PI += pi.detach() / nb_ites

                for var in _POSTERIOR_VARS_DATA:
                    val = ghost_data.get(var)
                    if val is not None:
                        accumulator['data'][var] = accumulator['data'].get(var, torch.zeros_like(val))
                        accumulator['data'][var] = accumulator['data'][var] + val / nb_ites
                for var in _POSTERIOR_VARS_PARAMS:
                    val = ghost_params.get(var)
                    if val is not None:
                        accumulator['params'][var] = accumulator['params'].get(var, torch.zeros_like(val))
                        accumulator['params'][var] = accumulator['params'][var] + val / nb_ites

                for j in range(latent_dim):
                    pi_j = self._compute_pi_ablated(
                        data, params, samp_gamma, dim, j, clonelabels, using_clone_features=using_clone_features
                    )
                    LOR[:, :, :, j] += (
                        torch.log(pi / (1 - pi)) - torch.log(pi_j / (1 - pi_j))
                    ) / nb_ites
                    LRR[:, :, :, j] += torch.log(pi / pi_j) / nb_ites

                ME += (
                    (ghost_params["pi"] * (1 - ghost_params["pi"]))[:, :, :, None]
                    * params["beta"][:, None, None, :]
                ) / nb_ites

        for sample_dependent_var in ['proportions', 'theta_fd']:
            if sample_dependent_var in accumulator['params']:
                val = accumulator['params'][sample_dependent_var]
                accumulator['params'][sample_dependent_var] = val[idxs_sample, ...]
        accumulator["params"].update({
            "PI": PI,
            "LOR": LOR,
            "LRR": LRR,
            "ME": ME,
        })
        if not(using_clone_features):
            accumulator["params"]["subclone_features"] = subclone_features
        return accumulator

    def save_results(
        self,
        results: dict,
        *,
        dir_save: str,
        data: dict,
        sample_names: list | None = None,
        model_name: str = "",
    ) -> None:
        """Persist posterior statistics to HDF5 files.

        Parameters
        ----------
        results:
            Output of :meth:`sample`.
        dir_save:
            Directory where HDF5 files are written.
        data:
            Original data dictionary (supplies ``Kmax``, ``D``, masks, …).
        sample_names:
            Names for the sample axis.  Defaults to ``[0, 1, …, N-1]``.
        model_name:
            Prefix for output file names.
        """
        if sample_names is None:
            sample_names = list(range(data["N"]))

        mask        = self._get_output_mask(data)
        latent_dim  = results["ME"].shape[-1]
        sf          = results["subclone_features"].copy()

        for j in range(latent_dim):
            sf[:, :, j][~mask] = float("nan")

        drug_names    = self._model.FD.selected_drugs
        feature_names = self._model.feature_names
        kmax_range    = list(range(data["Kmax"]))

        # Local importance
        all_li = results["PI"].numpy()[None, :, :, :] * sf[None, :, :, :]  # placeholder
        all_li = params["beta"][:, None, None, :] * sf[None, :, :, :]  # noqa: F821
        self._write_h5(
            os.path.join(dir_save, f"{model_name}local_importance.h5"),
            "local_importance_mean",
            results["PI"].numpy()[..., None] * sf[None, :, :, :],  # D × Kmax × N × dim
            {"dim1_drugs": drug_names, "dim2_subclones": kmax_range,
             "dim3_samples": sample_names, "dim4_dimensions": feature_names},
        )

        # Subclone features
        self._write_h5(
            os.path.join(dir_save, f"{model_name}subclone_features.h5"),
            "subclone_features_posterior_mean", sf,
            {"dim1_subclones": kmax_range, "dim2_samples": sample_names,
             "dim3_dimensions": feature_names},
        )

        # Survival probabilities
        postmean_pi = results["PI"].numpy()
        for d in range(data["D"]):
            postmean_pi[d, :, :][~mask] = float("nan")
        self._write_h5(
            os.path.join(dir_save, f"{model_name}survival_probabilities.h5"),
            "survival_probabilities_posterior_mean", postmean_pi,
            {"dim1_drugs": drug_names, "dim2_subclones": kmax_range,
             "dim3_samples": sample_names},
        )

        # Effect statistics
        for stat in ("ME", "LOR", "LRR"):
            arr = results[stat].detach().numpy() if hasattr(results[stat], "detach") else results[stat]
            for d in range(data["D"]):
                arr[d, :, :][~mask] = float("nan")
            self._write_h5(
                os.path.join(dir_save, f"{model_name}_{stat}.h5"),
                f"{stat}_posterior_mean", arr,
                {"dim1_drugs": drug_names, "dim2_subclones": kmax_range,
                 "dim3_samples": sample_names, "dim4_dimensions": feature_names},
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_latent(self, params: dict) -> torch.Tensor:
        """Draw one sample from the fitted guide."""
        if self._guide is not None and hasattr(self._guide, "sample_latent"):
            return self._guide.sample_latent()
        if self._guide_distribution is not None:
            return self._guide_distribution.sample()
        raise RuntimeError(
            "Cannot draw latent samples: guide has no `sample_latent` and "
            "no `guide_distribution` was provided."
        )

    def _init_accumulator(
        self, data: dict, params: dict, dim: int, idxs_sample: list, using_clone_features: bool = False
    ) -> dict:
        if not(using_clone_features):
            samp = self._draw_latent(params)
            self._assign_gamma(params, samp, dim, self._model.clonelabels)
        ghost_data, ghost_params = self._model.sampling(data, params, idxs_sample=idxs_sample)
        acc   = {'data': {}, 'params': {}}#deepcopy(data)
        for key, val in params.items():
            acc['params'][key] = torch.tensor(val) if isinstance(val, np.ndarray) else val
        for key, val in ghost_data.items():
            acc['data'][key] = val
        for key, val in ghost_params.items():
            acc['params'][key] = val
        for var in _POSTERIOR_VARS_DATA:
            if ghost_data.get(var) is not None:
                acc['data'][var] = torch.zeros_like(ghost_data[var])
        for var in _POSTERIOR_VARS_PARAMS:
            if ghost_params.get(var) is not None:
                acc['params'][var] = torch.zeros_like(ghost_params[var])
        return acc

    @staticmethod
    def _assign_gamma(
        state: dict,
        samp_gamma: torch.Tensor,
        dim: int,
        clonelabels: list,
    ) -> None:
        for i, clonelabel in enumerate(clonelabels):
            state[f"gamma_{clonelabel}"] = samp_gamma[dim * i: dim * (i + 1)]

    def _accumulate_subclone_features(
        self,
        acc: torch.Tensor,
        data: dict,
        params: dict,
        nb_ites: int,
        clonelabels: list,
    ) -> None:
        for clonelabel in clonelabels:
            idxs  = self._model.clonelabel2clusters[clonelabel]
            gamma = params[f"gamma_{clonelabel}"]
            acc[idxs, :, :] += torch.sum(
                data["X"][idxs, :, :, :] * (
                    masked_softmax(
                        torch.matmul(data["X"][idxs, :, :, :], gamma),
                        data["masks"]["SingleCell"][idxs, :, :],
                        dim=2,
                    )
                ).unsqueeze(3),
                dim=2,
            ) / nb_ites

    def _compute_pi_ablated(
        self,
        data: dict,
        params: dict,
        samp_gamma: torch.Tensor,
        dim: int,
        j: int,
        clonelabels: list,
        using_clone_features: bool = False,
    ) -> torch.Tensor:
        """Survival probabilities with feature dimension *j* zeroed out."""
        params_pi = {}
        for l, clonelabel in enumerate(clonelabels):
            if not(using_clone_features):
                params_pi[f"gamma_{clonelabel}"]  = samp_gamma[dim * l: dim * (l + 1)]
            params_pi[f"offset_{clonelabel}"] = torch.as_tensor(params[f"offset_{clonelabel}"])
        beta_j       = deepcopy(params["beta"])
        beta_j[:, j] = 0
        params_pi["beta"] = torch.as_tensor(beta_j)
        if using_clone_features:
            return self._model.compute_survival_probas_subclone_features(data, params_pi)
        else:
            return self._model.compute_survival_probas_single_cell_features(data, params_pi)

    @staticmethod
    def _get_output_mask(data: dict) -> torch.Tensor:
        mask = data["masks"]["RNA"]
        if np.sum(mask.detach().numpy()) == 0:
            mask = torch.sum(data["masks"]["SingleCell"], dim=2) > 0.5
        return mask

    @staticmethod
    def _write_h5(path: str, dataset_name: str, array: np.ndarray, attrs: dict) -> None:
        with h5py.File(path, "w") as f:
            dset = f.create_dataset(dataset_name, data=array)
            for k, v in attrs.items():
                dset.attrs[k] = v

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        guide_info = type(self._guide).__name__ if self._guide is not None else "None"
        return f"{type(self).__name__}(guide={guide_info})"
