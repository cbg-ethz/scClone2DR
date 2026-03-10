"""Generative model for scClone2DR.

This module contains **only** the probabilistic model: the prior / likelihood
(``prior``), the survival-probability algebra, and the mean-fraction helpers
derived from it.  It has no knowledge of optimisation, data loading, posterior
sampling, or IO.

The clear identity of this class is:
    ``scClone2DR`` *is* a generative model — it is not a trainer, a sampler,
    or a data manager.
"""

from __future__ import annotations

import logging
from enum import Enum

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from torch.distributions import constraints
import numpy as np

from .data import BaseDataset
from .utils import masked_softmax, merge_data_params, sigmoid
from .types import NuMode, ThetaMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class scClone2DR:
    """Generative model for single-cell clone-to-drug-response.

    This class encodes the joint prior and likelihood only.  It holds no
    reference to a trainer, a data loader, or any IO layer.  All of those
    concerns belong to collaborating objects that *use* this model.

    Parameters
    ----------
    mode_nu:
        How the pre-assay effect is modelled.
        ``"fixed"`` — no correction; ``"noise_correction"`` — GAM-based.
    mode_theta:
        How overdispersion is parameterised across samples and modalities.

    Notes
    -----
    The cluster-label mappings (``clonelabels``, ``clonelabel2clusters``,
    ``cat2clusters``, …) and the ``feature_names`` / ``FD`` attributes are
    expected to be set on the instance before ``prior`` is called.  In
    practice this is done by the pipeline object that assembles the model
    together with a data source.
    """

    # ------------------------------------------------------------------
    # Numerical constants
    # ------------------------------------------------------------------
    EPSILON             = 1e-6
    DEFAULT_TOTAL_COUNT = 100_000
    DEFAULT_THETA_RNA   = 40.0
    DEFAULT_THETA_FD    = 1.0
    DEFAULT_VAR_F       = 0.1
    _CONFIG_FIELDS = (
        "cluster2clonelabel",
        "clonelabel2cat",
        "cat2clusters",
        "cat2clonelabels",
        "clonelabel2clusters",
        "clonelabels",
        "n_cat",
        "n_clonelabels",
        "sample_names",
        "drugs",
    )

    def __init__(
        self,
        dataset: BaseDataset | None = None, 
        mode_nu: NuMode | str       = NuMode.NOISE_CORRECTION,
        mode_theta: ThetaMode | str = ThetaMode.NOT_SHARED_DECOUPLED,
    ) -> None:
        self.mode_nu      = NuMode(mode_nu)
        self.mode_theta   = ThetaMode(mode_theta)
        # Populated by the pipeline after fit() or load_params(); never set internally.
        self.fitted_params: dict | None = None
        self.is_configured = False
        if dataset is not None:
            self.configure(dataset)


    def configure(self, dataset: BaseDataset) -> None:
        """Bind data-derived structural metadata to the model.
        
        Must be called (by the pipeline) after the dataset is ready
        and before prior() or sampling() are called.
        
        Parameters
        ----------
        dataset:
            Any BaseDataset instance on which init_cat_clonelabel()
            has already been called.
        """
        if dataset.cluster2clonelabel is None or dataset.clonelabel2cat is None:
            raise RuntimeError(
                "Dataset topology is not initialised. "
                "Call dataset.init_topology(...) or a data-loading routine first."
            )

        self.cluster2clonelabel  = list(dataset.cluster2clonelabel)
        self.clonelabel2cat      = dict(dataset.clonelabel2cat)
        self.cat2clusters        = {
            k: list(v) for k, v in (dataset.cat2clusters or {}).items()
        }
        self.cat2clonelabels     = {
            k: list(v) for k, v in (dataset.cat2clonelabels or {}).items()
        }
        self.clonelabel2clusters = {
            k: list(v) for k, v in (dataset.clonelabel2clusters or {}).items()
        }

        # Derive canonical counts and ordering once.
        self.clonelabels   = list(np.unique(self.cluster2clonelabel))
        self.n_cat         = len(self.cat2clusters)
        self.n_clonelabels = len(self.clonelabels)
        self.sample_names  = list(dataset.sample_names) if dataset.sample_names is not None else None
        self.drugs = list(dataset.drugs) if dataset.drugs is not None else None
        self.is_configured = True

    def export_configuration(self) -> dict:
        """Return a serializable snapshot of the model structural configuration."""
        if not self.is_configured:
            return {}

        return {
            field: getattr(self, field, None)
            for field in self._CONFIG_FIELDS
        }

    def load_configuration(self, config: dict) -> None:
        """Load structural configuration previously produced by export_configuration."""
        if not config:
            return

        for field in self._CONFIG_FIELDS:
            if field in config:
                setattr(self, field, config[field])
        self.is_configured = True

    def _require_fitted_params(self) -> dict:
        """Return ``fitted_params``, raising clearly when the model is unfitted."""
        if self.fitted_params is None:
            raise RuntimeError(
                "This model has not been fitted yet.  "
                "Call pipeline.fit() or pipeline.load_params() first, "
                "or pass params explicitly."
            )
        return self.fitted_params

    # ------------------------------------------------------------------
    # Prior / likelihood  (Pyro model)
    # ------------------------------------------------------------------

    def prior(self, data: dict, fixed_proportions: bool = False, theta_rna=None) -> None:
        """Joint prior and likelihood — the Pyro model callable.

        Parameters
        ----------
        data:
            Training data dictionary.  Required keys: ``"Kmax"``, ``"n_r"``,
            ``"n_c"``, ``"masks"``, ``"X"``, ``"single_cell_features"``,
            and conditional keys depending on ``mode_nu`` / ``mode_theta``.
        fixed_proportions:
            When ``True`` clone proportions are derived directly from RNA
            counts rather than treated as free parameters.
        theta_rna:
            Optional fixed override for the RNA overdispersion parameter.
        """
        Kmax        = data["Kmax"]
        R, D, Ndrug = data["n_r"].shape
        C, N        = data["n_c"].shape
        masks       = data["masks"]
        dim         = data["X"].shape[-1]

        beta      = pyro.param("beta", torch.zeros((D, dim)))
        params_pi = {"beta": beta}
        for clonelabel in self.clonelabels:
            params_pi[f"offset_{clonelabel}"] = pyro.param(
                f"offset_{clonelabel}", torch.zeros(D)
            )

        if data["single_cell_features"]:
            var_fs = {}
            for clonelabel in self.clonelabels:
                var_fs[clonelabel] = pyro.param(
                    "f_{0}".format(clonelabel),
                    torch.tensor(self.DEFAULT_VAR_F),
                    constraints.positive
                )
                params_pi['gamma_{0}'.format(clonelabel)] = pyro.sample(
                    "gamma_{0}".format(clonelabel),
                    dist.Normal(torch.zeros(dim), var_fs[clonelabel] * torch.ones(dim)).to_event(1)
                )
            pi  = self.compute_survival_probas_single_cell_features(data, params_pi)
        else:
            pi = self.compute_survival_probas_subclone_features(data, params_pi)

        if theta_rna is None:
            theta_rna = pyro.param(
                "theta_rna",
                torch.tensor(self.DEFAULT_THETA_RNA),
                constraints.positive,
            )

        if self.mode_nu is NuMode.NOISE_CORRECTION:
            dim_c = data["X_nu_control"].shape[2]
            pyro.param("beta_control", torch.zeros(dim_c))

        weights    = data.get("weights",    torch.ones(N))
        weights_ND = data.get("weights_ND", torch.ones((D, N)))  # noqa: F841 (reserved)

        with pyro.plate("samples", N):
            with poutine.scale(scale=weights):
                proportions = self._get_proportions(data, fixed_proportions, Kmax, N)
                theta_fd    = self._get_theta_fd(theta_rna, N)

                self._observe_rna(data, proportions, theta_rna, theta_fd)
                self._observe_control_wells(data, proportions, theta_fd, masks, C)

        with pyro.plate("samples_drug", Ndrug):
            with pyro.plate("drugs", D):
                with pyro.plate("replicates", R), poutine.mask(mask=masks["R"]):
                    self._observe_drug_wells(data, proportions, pi, theta_fd, Ndrug, R, D)


    # ------------------------------------------------------------------
    # Sampling from the generative model
    # ------------------------------------------------------------------

    def sampling(self, dic: dict, params: dict | None = None, idxs_sample: list | None = None) -> dict:
        """Sample new data from the generative model using learned parameters.

        Runs a forward pass through the model — control wells, drug wells, and
        (optionally) RNA counts — and returns all sampled quantities together
        with derived statistics (per-well fractions and log-scores).

        Parameters
        ----------
        dic:
            Data dictionary in the same format as passed to ``prior``.
        params:
            Parameter dictionary.  When ``None``, :attr:`fitted_params` is
            used.  If single-cell features are active, must additionally
            contain ``"gamma_{clonelabel}"`` for every clone label.

        Returns
        -------
        dict
            Keys: ``n0_c``, ``n0_r``, ``n_rna``, ``frac_r``, ``frac_c``,
            ``frac_mean_r``, ``frac_mean_c``, ``log_score``, ``pi``,
            ``nu_healthy_drug``, ``nu_healthy_control``,
            ``N``, ``Kmax``, ``R``, ``C``, ``D``.
        """
        params    = params if params is not None else self._require_fitted_params()
        data        = merge_data_params(dic, params)
        N, Kmax     = data["N"], data["Kmax"]
        R, D, Ndrug = data["n_r"].shape
        C           = data["C"]
        masks       = data["masks"]
        if idxs_sample is None:
            proportions_loaded = data["proportions"]
            theta_fd_loaded    = data["theta_fd"]
        else:
            proportions_loaded = data["proportions"][idxs_sample, :]
            theta_fd_loaded    = data["theta_fd"][idxs_sample]

        pi          = self.get_survival_probas(data, params)
        proportions = (
            proportions_loaded / torch.sum(proportions_loaded, dim=1).unsqueeze(1)
        ).T[:, -N:]

        theta_rna = data["theta_rna"]
        theta_fd  = (
            theta_rna
            if self.mode_theta is ThetaMode.EQUAL
            else theta_fd_loaded
        )
        beta_control = (
            data["beta_control"]
            if self.mode_nu is NuMode.NOISE_CORRECTION
            else None
        )

        # ---- RNA counts ------------------------------------------------
        n_rna = self._sample_rna(data, proportions, theta_rna, theta_fd, N, Kmax)

        # ---- Control and drug wells ------------------------------------
        with pyro.plate("samples", N):
            nu_healthy_c, n0_c, frac_c = self._sample_control_wells(
                data, proportions, theta_fd, C, beta_control
            )

        with pyro.plate("samples", Ndrug):
            with pyro.plate("drugs", D):
                with pyro.plate("replicates", R), poutine.mask(mask=masks["R"]):
                    nu_healthy_drug, n0_r, frac_r = self._sample_drug_wells(
                        data, proportions, pi, theta_fd, Ndrug, R, D, beta_control
                    )

                frac_mean_r = (
                    torch.sum(masks["R"] * torch.nan_to_num(frac_r), dim=0)
                    / torch.sum(masks["R"], dim=0)
                )
                frac_mean_c = (
                    torch.sum(masks["C"] * torch.nan_to_num(frac_c), dim=0)
                    / torch.sum(masks["C"], dim=0)
                )
                log_score = torch.log(frac_mean_r) - torch.log(frac_mean_c)[:Ndrug]

        return {
            "n0_c":             n0_c,
            "n0_r":             n0_r,
            "n_rna":            n_rna,
            "frac_r":           frac_r,
            "frac_c":           frac_c,
            "frac_mean_r":      frac_mean_r,
            "frac_mean_c":      frac_mean_c,
            "log_score":        log_score,
            "N": N, "Kmax": Kmax, "R": R, "C": C, "D": D,
        }, {"pi":               pi,
            "nu_healthy_drug":  nu_healthy_drug,
            "nu_healthy_control": nu_healthy_c
        }
    
    def _sample_rna(self, data, proportions, theta_rna, theta_fd, N, Kmax):
        n_rna = torch.zeros((Kmax, N))
        if data['n_rna'] is not None:
            for i in range(N):
                idxs_notnull = np.where(proportions[:, i] > 0)[0]
                if self.mode_theta in ['equal', 'shared', 'not shared decoupled']:
                    n_rna[idxs_notnull, i] = dist.DirichletMultinomial(
                        theta_rna * proportions[idxs_notnull, i],
                        torch.sum(data['n_rna'][idxs_notnull, i])
                    ).sample()
                elif self.mode_theta == 'not shared coupled':
                    n_rna[idxs_notnull, i] = dist.DirichletMultinomial(
                        theta_rna * theta_fd[i] * proportions[idxs_notnull, i],
                        torch.sum(data['n_rna'][idxs_notnull, i])
                    ).sample()
        else:
            n_rna = None
        return n_rna

    def _sample_control_wells(self, data, proportions, theta_fd, C, beta_control):
        if self.mode_nu == NuMode.FIXED:
            nu_tumor_over_nu_healthy = torch.tensor(1)
            nu_healthy_c = 1. / (1 + nu_tumor_over_nu_healthy)
        elif self.mode_nu == NuMode.NOISE_CORRECTION:
            nu_tumor_over_nu_healthy = torch.exp(data['X_nu_control'] @ beta_control)
            nu_healthy_c = 1. / (1 + nu_tumor_over_nu_healthy)

        n0_c = pyro.sample(
            'n0_c',
            dist.BetaBinomial(
                (theta_fd * torch.sum(proportions[self.cat2clusters['healthy'], :], dim=0)).unsqueeze(0).repeat(C, 1) * nu_healthy_c,
                (theta_fd * torch.sum(proportions[self.cat2clusters['tumor'], :], dim=0)).unsqueeze(0).repeat(C, 1) * (1 - nu_healthy_c),
                data['n_c']
            )
        )

        frac_c = 1. - n0_c / data['n_c']
        return nu_healthy_c, n0_c, frac_c
    
    def _sample_drug_wells(self, data, proportions, pi, theta_fd, Ndrug, R, D, beta_control):
        if self.mode_nu == NuMode.FIXED:
            nu_tumor_over_nu_healthy = torch.tensor(1)
            nu_healthy_drug = 1. / (1 + nu_tumor_over_nu_healthy)
        elif self.mode_nu == NuMode.NOISE_CORRECTION:
            nu_tumor_over_nu_healthy = torch.exp(data['X_nu_drug'] @ beta_control)
            nu_healthy_drug = 1. / (1 + nu_tumor_over_nu_healthy)

        if 'not' in self.mode_theta:
            theta_fd_mode = theta_fd[:Ndrug]
        else:
            theta_fd_mode = theta_fd
        n0_r = pyro.sample(
            'n0_r',
            dist.BetaBinomial(
                (theta_fd_mode * torch.sum(
                    proportions[self.cat2clusters['healthy'], :Ndrug].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters['healthy'], :],
                    dim=1
                )).unsqueeze(0).repeat(R, 1, 1) * nu_healthy_drug,
                (theta_fd_mode * torch.sum(
                    proportions[self.cat2clusters['tumor'], :Ndrug].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters['tumor'], :],
                    dim=1
                )).unsqueeze(0).repeat(R, 1, 1) * (1 - nu_healthy_drug),
                data['n_r']
            )
        )
        frac_r = 1. - n0_r / data['n_r']
        return nu_healthy_drug, n0_r, frac_r
    # ------------------------------------------------------------------
    # Survival probability computation
    # ------------------------------------------------------------------

    def compute_survival_probas_single_cell_features(
        self, data: dict, params: dict | None = None
    ) -> torch.Tensor:
        """Survival probability via attention over single-cell features.

        Parameters
        ----------
        data:
            Must contain ``"X"`` (cell features) and
            ``"masks"["SingleCell"]``.
        params:
            Must contain ``"beta"``, ``"offset_{clonelabel}"``, and
            ``"gamma_{clonelabel}"`` for each clone label.
            Defaults to :attr:`fitted_params` when ``None``.

        Returns
        -------
        torch.Tensor
            Shape ``(D, Kmax, N)``.
        """
        params = params if params is not None else self._require_fitted_params()
        all_pis = []
        for clonelabel in self.clonelabels:
            idxs   = self.clonelabel2clusters[clonelabel]
            offset = params[f"offset_{clonelabel}"]
            gamma  = params[f"gamma_{clonelabel}"]

            attended = torch.sum(
                data["X"][idxs, :, :, :] * (
                    masked_softmax(
                        torch.matmul(data["X"][idxs, :, :, :], gamma),
                        data["masks"]["SingleCell"][idxs, :, :],
                        dim=2,
                    )
                ).unsqueeze(3),
                dim=2,
            )
            logit = (
                torch.matmul(attended, params["beta"].T)
                + offset.unsqueeze(0).unsqueeze(0)
            )
            all_pis.append(sigmoid(logit).permute(2, 0, 1))  # D × |idxs| × N

        return torch.cat(all_pis, dim=1)

    def compute_survival_probas_subclone_features(
        self, data: dict, params: dict | None = None
    ) -> torch.Tensor:
        """Survival probability from subclone-level features.

        Parameters
        ----------
        data:
            Must contain ``"X"`` (subclone features).
        params:
            Must contain ``"beta"`` and ``"offset_{clonelabel}"`` for each
            clone label.  Defaults to :attr:`fitted_params` when ``None``.

        Returns
        -------
        torch.Tensor
            Shape ``(D, Kmax, N)``.
        """
        params = params if params is not None else self._require_fitted_params()
        all_pis = []
        for clonelabel in self.clonelabels:
            cat    = self.clonelabel2cat[clonelabel]
            idxs   = self.cat2clusters[cat]
            offset = params[f"offset_{clonelabel}"]
            logit  = (
                torch.matmul(data["X"][idxs, :, :], params["beta"].T)
                + offset.unsqueeze(0).unsqueeze(0)
            )
            all_pis.append(sigmoid(logit).permute(2, 0, 1))  # D × |idxs| × N

        return torch.cat(all_pis, dim=1)

    def get_survival_probas(self, data: dict, params: dict | None = None) -> torch.Tensor:
        """Dispatch to the correct survival-probability method.

        Parameters
        ----------
        data:
            Must contain ``"single_cell_features"`` (bool).
        params:
            Must contain ``"beta"``, ``"offset_{clonelabel}"``, and
            (when single-cell) ``"gamma_{clonelabel}"`` for each clone label.
            Defaults to :attr:`fitted_params` when ``None``.
        """
        params    = params if params is not None else self._require_fitted_params()
        params_pi = {"beta": params["beta"]}
        for clonelabel in self.clonelabels:
            params_pi[f"offset_{clonelabel}"] = params[f"offset_{clonelabel}"]

        if data["single_cell_features"]:
            for clonelabel in self.clonelabels:
                params_pi[f"gamma_{clonelabel}"] = params[f"gamma_{clonelabel}"]
            return self.compute_survival_probas_single_cell_features(data, params_pi)

        return self.compute_survival_probas_subclone_features(data, params_pi)

    # ------------------------------------------------------------------
    # Mean-fraction statistics  (derived from the generative model)
    # ------------------------------------------------------------------

    def get_mean_logscore(
        self,
        proportions: torch.Tensor,
        D: int,
        pi: torch.Tensor,
        nu_healthy: torch.Tensor,
    ) -> torch.Tensor:
        """Log-ratio of treated vs. control tumour fractions."""
        proportions = torch.as_tensor(proportions)
        pi          = torch.as_tensor(pi)
        control_0   = torch.sum(proportions[self.cat2clusters["healthy"], :], dim=0).unsqueeze(0).repeat(D, 1) * nu_healthy
        control_t   = torch.sum(proportions[self.cat2clusters["tumor"], :].unsqueeze(0).repeat(D, 1, 1), dim=1) * (1 - nu_healthy)
        drug_0      = torch.sum(proportions[self.cat2clusters["healthy"], :].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters["healthy"], :], dim=1) * nu_healthy
        drug_t      = torch.sum(proportions[self.cat2clusters["tumor"], :].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters["tumor"], :], dim=1) * (1 - nu_healthy)
        return torch.log(control_t * (drug_0 + drug_t)) - torch.log((control_0 + control_t) * drug_t)

    def get_mean_fracMEL_control(
        self,
        proportions: torch.Tensor,
        C: int,
        nu_healthy: torch.Tensor,
    ) -> torch.Tensor:
        """Expected tumour-cell fraction in control wells — shape ``(C, N)``."""
        proportions = torch.as_tensor(proportions)
        control_0   = torch.sum(proportions[self.cat2clusters["healthy"], :], dim=0).unsqueeze(0).repeat(C, 1) * nu_healthy
        control_t   = torch.sum(proportions[self.cat2clusters["tumor"], :].unsqueeze(0).repeat(C, 1, 1), dim=1) * (1 - nu_healthy)
        return control_t / (control_0 + control_t)

    def get_mean_fracMEL_treated(
        self,
        proportions: torch.Tensor,
        D: int,
        pi: torch.Tensor,
        nu_healthy: torch.Tensor,
    ) -> torch.Tensor:
        """Expected tumour-cell fraction in treated wells — shape ``(D, N)``."""
        proportions = torch.as_tensor(proportions)
        pi          = torch.as_tensor(pi)
        drug_0      = torch.sum(proportions[self.cat2clusters["healthy"], :].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters["healthy"], :], dim=1) * nu_healthy
        drug_t      = torch.sum(proportions[self.cat2clusters["tumor"], :].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters["tumor"], :], dim=1) * (1 - nu_healthy)
        return drug_t / (drug_0 + drug_t)

    # ------------------------------------------------------------------
    # Private: prior helpers
    # ------------------------------------------------------------------

    def _get_proportions(
        self, data: dict, fixed_proportions: bool, Kmax: int, N: int
    ) -> torch.Tensor:
        if fixed_proportions:
            return data["n_rna"] / torch.tile(
                torch.sum(data["n_rna"], dim=0).reshape(1, N), (Kmax, 1)
            )
        return pyro.param(
            "proportions", data["ini_proportions"].clone(), constraints.simplex
        ).T

    def _get_theta_fd(self, theta_rna: torch.Tensor, N: int) -> torch.Tensor:
        if self.mode_theta is ThetaMode.EQUAL:
            return theta_rna
        if self.mode_theta is ThetaMode.SHARED:
            return pyro.param(
                "theta_fd", torch.tensor(self.DEFAULT_THETA_FD), constraints.positive
            )
        if self.mode_theta in (ThetaMode.NOT_SHARED_COUPLED, ThetaMode.NOT_SHARED_DECOUPLED):
            return pyro.param(
                "theta_fd", self.DEFAULT_THETA_RNA * torch.ones(N), constraints.positive
            )
        return theta_rna

    def _observe_rna(
        self,
        data: dict,
        proportions: torch.Tensor,
        theta_rna: torch.Tensor,
        theta_fd: torch.Tensor,
    ) -> None:
        if data["n_rna"] is None:
            return
        if self.mode_theta is ThetaMode.NO_OVERDISPERSION:
            pyro.sample(
                "n_rna",
                dist.Multinomial(self.DEFAULT_TOTAL_COUNT, proportions.T),
                obs=data["n_rna"].T,
            )
        elif self.mode_theta in (ThetaMode.EQUAL, ThetaMode.SHARED, ThetaMode.NOT_SHARED_DECOUPLED):
            pyro.sample(
                "n_rna",
                dist.DirichletMultinomial(
                    (theta_rna * proportions).T, torch.sum(data["n_rna"], dim=0)
                ),
                obs=data["n_rna"].T,
            )
        else:  # NOT_SHARED_COUPLED
            pyro.sample(
                "n_rna",
                dist.DirichletMultinomial(
                    (theta_rna * theta_fd * proportions).T,
                    torch.sum(data["n_rna"], dim=0),
                ),
                obs=data["n_rna"].T,
            )

    def _observe_control_wells(
        self,
        data: dict,
        proportions: torch.Tensor,
        theta_fd: torch.Tensor,
        masks: dict,
        C: int,
    ) -> None:
        beta_control = (
            pyro.param("beta_control")
            if self.mode_nu is NuMode.NOISE_CORRECTION
            else None
        )
        with pyro.plate("controls", C), poutine.mask(mask=masks["C"]):
            nu_healthy = self._compute_nu_healthy(data, beta_control, mode="control")
            if self.mode_theta is ThetaMode.NO_OVERDISPERSION:
                pyro.sample(
                    "n0_c",
                    dist.Binomial(
                        self.DEFAULT_TOTAL_COUNT,
                        proportions[0, :].unsqueeze(0).repeat(C, 1) * nu_healthy,
                    ),
                    obs=data["n0_c"],
                )
            else:
                pyro.sample(
                    "n0_c",
                    dist.BetaBinomial(
                        (theta_fd * torch.sum(proportions[self.cat2clusters["healthy"], :], dim=0)).unsqueeze(0).repeat(C, 1) * nu_healthy,
                        (theta_fd * torch.sum(proportions[self.cat2clusters["tumor"],   :], dim=0)).unsqueeze(0).repeat(C, 1) * (1 - nu_healthy),
                        data["n_c"],
                    ),
                    obs=data["n0_c"],
                )

    def _observe_drug_wells(
        self,
        data: dict,
        proportions: torch.Tensor,
        pi: torch.Tensor,
        theta_fd: torch.Tensor,
        Ndrug: int,
        R: int,
        D: int,
    ) -> None:
        beta_control = (
            pyro.param("beta_control")
            if self.mode_nu is NuMode.NOISE_CORRECTION
            else None
        )
        nu_healthy = self._compute_nu_healthy(data, beta_control, mode="drug")
        theta_fd_m = theta_fd[:Ndrug] if "not" in self.mode_theta.value else theta_fd

        if self.mode_theta is ThetaMode.NO_OVERDISPERSION:
            pyro.sample(
                "n0_r",
                dist.Binomial(
                    self.DEFAULT_TOTAL_COUNT,
                    torch.sum(
                        proportions[self.cat2clusters["healthy"], :Ndrug].unsqueeze(0).repeat(D, 1, 1)
                        * pi[:, self.cat2clusters["healthy"], :],
                        dim=1,
                    ).unsqueeze(0).repeat(R, 1, 1) * nu_healthy,
                ),
                obs=data["n0_r"],
            )
        else:
            pyro.sample(
                "n0_r",
                dist.BetaBinomial(
                    (theta_fd_m * torch.sum(
                        proportions[self.cat2clusters["healthy"], :Ndrug].unsqueeze(0).repeat(D, 1, 1)
                        * pi[:, self.cat2clusters["healthy"], :],
                        dim=1,
                    )).unsqueeze(0).repeat(R, 1, 1) * nu_healthy,
                    (theta_fd_m * torch.sum(
                        proportions[self.cat2clusters["tumor"], :Ndrug].unsqueeze(0).repeat(D, 1, 1)
                        * pi[:, self.cat2clusters["tumor"], :],
                        dim=1,
                    )).unsqueeze(0).repeat(R, 1, 1) * (1 - nu_healthy),
                    data["n_r"],
                ),
                obs=data["n0_r"],
            )

    def _compute_nu_healthy(
        self, data: dict, beta_control, *, mode: str
    ) -> torch.Tensor:
        """Compute ``nu_healthy`` for control or drug wells."""
        if self.mode_nu is NuMode.FIXED or beta_control is None:
            return torch.tensor(0.5)
        key = "X_nu_control" if mode == "control" else "X_nu_drug"
        return 1.0 / (1.0 + torch.exp(data[key] @ beta_control))

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"mode_nu={self.mode_nu.value!r}, "
            f"mode_theta={self.mode_theta.value!r})"
        )