"""Model evaluation and simulation diagnostics for scClone2DR.

Two classes are provided:

* :class:`ModelEvaluator` — scores *fitted* models on observed data:
  log-likelihood, guide distribution reconstruction, posterior means.

* :class:`Evaluator` — compares fitted parameters against ground
  truth in simulation studies: KL divergences, Spearman correlations, beta
  errors, fold-change analysis.  All methods **return** their results rather
  than accumulating them in mutable state.

Design notes
------------
* Neither class inherits from the other or from the model.
* ``Evaluator`` takes the model as a constructor argument so it can
  call ``get_mean_fracMEL_*`` without redefining that algebra.
* The ``results`` dict anti-pattern (mutable accumulator on ``self``) has been
  replaced: every method returns a typed value, and
  :meth:`Evaluator.compute_all` returns a single
  :class:`Results` dataclass.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pyro.distributions as dist
import torch
from scipy import stats as scipy_stats
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal

from ..types import NuMode, ThetaMode

from ..model import scClone2DR
from ..trainer import GuideType
from ..utils import *

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Reserved key written by Trainer into the params dict after fitting.
_GUIDE_TYPE_KEY = "__guide_type__"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kl_divergence(a: torch.Tensor, b: torch.Tensor, *, mode: str = "survival") -> torch.Tensor:
    """KL divergence D_KL(a ‖ b) with NaN-safe arithmetic.

    Parameters
    ----------
    a, b:
        Probability tensors in ``[0, 1]``.
    mode:
        ``"survival"`` uses the Bernoulli KL ``a log(a/b) + (1-a) log((1-a)/(1-b))``.
        Any other value uses the one-sided form ``a log(a/b)``.
    """
    assert torch.all(a >= 0) and torch.all(a <= 1), "a must be in [0, 1]"
    assert torch.all(b >= 0) and torch.all(b <= 1), "b must be in [0, 1]"

    if mode == "survival":
        raw = a * torch.log(a / b) + (1 - a) * torch.log((1 - a) / (1 - b))
    else:
        raw = a * torch.log(a / b)

    return torch.sum(torch.nan_to_num(raw))





# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class Results:
    """Container for all metrics.

    Every field is ``None`` until the corresponding metric has been computed.
    """
    # Proportions
    kl_proportions: float | None                = None
    proportions: torch.Tensor | None            = None
    true_proportions: torch.Tensor | None       = None

    # Survival probabilities
    kl_survival_probas: float | None            = None
    survival_probas: torch.Tensor | None        = None
    true_survival_probas: torch.Tensor | None   = None

    # Drug scores (Spearman over drugs)
    spearman_drugs_avg: float | None            = None
    drug_scores: list = field(default_factory=list)
    true_drug_scores: list = field(default_factory=list)

    # Subclone Spearman
    spearman_subclones_avg: float | None        = None

    # Beta reconstruction
    mse_beta: float | None                      = None

    # Drug effects
    drug_effects: torch.Tensor | None           = None
    true_drug_effects: torch.Tensor | None      = None
    l1err_drug_effects: float | None            = None

    # Overall survival
    l1err_overall_survival: float | None        = None

    # Fold changes
    fold_change_pred: list = field(default_factory=list)
    fold_change_data: list = field(default_factory=list)
    fold_change_true: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------

class BaseModelEvaluator:
    """Score a fitted scClone2DR model against observed data.

    Responsibilities
    ----------------
    * Compute the normalised log-likelihood of held-out or training data.
    * Reconstruct the fitted variational distribution from the parameter
      dictionary (works after reloading parameters from disk, because
      ``Trainer.train`` embeds the guide type under ``"__guide_type__"``).
    * Estimate the posterior mean of the latent ``gamma`` variables.

    Parameters
    ----------
    model : scClone2DR
        Generative model instance (only its mode flags and cluster mappings
        are accessed — no live guide reference required).

    Examples
    --------
    >>> evaluator = ModelEvaluator(model)
    >>> ll         = evaluator.log_likelihood(data, params)
    >>> guide_dist = evaluator.build_guide_distribution(params)
    >>> post_means = evaluator.posterior_mean_latent(params, nsamples=200)
    """

    def __init__(self, model: scClone2DR) -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------

    def log_likelihood(self, data: dict, params: dict) -> torch.Tensor:
        """Normalised log-likelihood of *data* under the fitted model.

        Parameters
        ----------
        data:
            Observed data dictionary.
        params:
            Learned parameters (output of ``Trainer.train``).

        Returns
        -------
        torch.Tensor
            Scalar: sum of log-probabilities divided by the number of
            observed data points.
        """
        merged = merge_data_params(data, params)
        R, D, Ndrug = merged["n_r"].shape
        C, N        = merged["n_c"].shape
        masks       = merged["masks"]
        pi          = self._model.get_survival_probas(merged)

        proportions = (
            merged["proportions"] / torch.sum(merged["proportions"], dim=1).unsqueeze(1)
        ).T[:, -N:]

        beta_control = (
            merged.get("beta_control")
            if self._model.mode_nu is NuMode.NOISE_CORRECTION
            else None
        )
        theta_fd  = (
            merged["theta_rna"]
            if self._model.mode_theta is ThetaMode.EQUAL
            else merged["theta_fd"]
        )
        theta_rna = merged["theta_rna"]

        log_lik, count = torch.tensor(0.0), 0
        log_lik, count = self._loglik_rna(data, proportions, theta_rna, theta_fd, log_lik, count)
        log_lik, count = self._loglik_controls(data, merged, proportions, theta_fd, masks, C, N, beta_control, log_lik, count)
        log_lik, count = self._loglik_drug_wells(data, merged, proportions, pi, theta_fd, masks, R, D, Ndrug, beta_control, log_lik, count)

        return log_lik / count

    # ------------------------------------------------------------------
    # Guide distribution reconstruction
    # ------------------------------------------------------------------

    def build_guide_distribution(self, params: dict, guide_type: GuideType):
        """Reconstruct the variational distribution from *params*.

        Parameters
        ----------
        params:
            Parameter dictionary returned by ``Trainer.train``.

        Returns
        -------
        torch.distributions.Distribution

        Raises
        ------
        KeyError
            When *params* has no ``"__guide_type__"`` key (MAP/MLE path).
        ValueError
            For unrecognised guide-type values.
        """
        # if _GUIDE_TYPE_KEY not in params:
        #     raise KeyError(
        #         f"'{_GUIDE_TYPE_KEY}' not found in params. "
        #         "The model was fitted on the MAP/MLE path — no guide available."
        #     )

        # guide_type = GuideType(params[_GUIDE_TYPE_KEY])

        if guide_type is GuideType.FULL_MVN:
            return MultivariateNormal(
                loc=torch.as_tensor(params["AutoMultivariateNormal.loc"]),
                scale_tril=torch.as_tensor(params["AutoMultivariateNormal.scale_tril"]),
            )
        if guide_type is GuideType.LOWRANK_MVN:
            return LowRankMultivariateNormal(
                loc=torch.as_tensor(params["AutoLowRankMultivariateNormal.loc"]),
                cov_factor=torch.as_tensor(params["AutoLowRankMultivariateNormal.cov_factor"]),
                cov_diag=torch.as_tensor(params["AutoLowRankMultivariateNormal.scale"]),
            )
        if guide_type is GuideType.DIAGONAL:
            scale = torch.as_tensor(params["AutoDiagonalNormal.scale"])
            return MultivariateNormal(
                loc=torch.as_tensor(params["AutoDiagonalNormal.loc"]),
                covariance_matrix=torch.diag(scale ** 2),
            )
        if guide_type is GuideType.NONE:
            return None
        else:
            raise ValueError(f"Unrecognised guide type: {guide_type!r}")

    # ------------------------------------------------------------------
    # Posterior mean of latent variables
    # ------------------------------------------------------------------

    def posterior_mean_latent(
        self, params: dict, nsamples: int = 100
    ) -> dict[str, torch.Tensor]:
        """Monte-Carlo estimate of the posterior mean of each ``gamma``.

        Parameters
        ----------
        params:
            Parameter dictionary returned by ``Trainer.train``.
        nsamples:
            Number of Monte-Carlo draws.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"gamma_{clonelabel}": tensor, …}`` for each clone label.
        """
        guide_dist  = self.build_guide_distribution(params)
        dim         = guide_dist.sample().shape[0] // self._model.n_clonelabels
        postmeans: dict[str, torch.Tensor] = {}

        for _ in range(nsamples):
            samp = guide_dist.sample()
            for i, clonelabel in enumerate(self._model.clonelabels):
                chunk = samp[dim * i: dim * (i + 1)] / nsamples
                key   = f"gamma_{clonelabel}"
                postmeans[key] = postmeans.get(key, torch.zeros_like(chunk)) + chunk

        return postmeans

    # ------------------------------------------------------------------
    # Private: log-likelihood helpers
    # ------------------------------------------------------------------

    def _loglik_rna(self, data, proportions, theta_rna, theta_fd, log_lik, count):
        if data.get("n_rna") is None:
            return log_lik, count
        valid = (proportions > 0).nonzero(as_tuple=True)
        mode  = self._model.mode_theta
        if mode in (ThetaMode.EQUAL, ThetaMode.SHARED, ThetaMode.NOT_SHARED_DECOUPLED):
            concentration = theta_rna * proportions[valid]
        elif mode is ThetaMode.NOT_SHARED_COUPLED:
            concentration = theta_rna * theta_fd[valid[1]] * proportions[valid]
        else:
            return log_lik, count
        log_lik += dist.DirichletMultinomial(
            concentration, torch.sum(data["n_rna"][valid], dim=0)
        ).log_prob(data["n_rna"][valid]).sum()
        count += len(valid[0])
        return log_lik, count

    def _loglik_controls(self, data, merged, proportions, theta_fd, masks, C, N, beta_control, log_lik, count):
        nu_h       = self._nu_healthy_or_default(merged, beta_control, mode="control", shape=(C, N))
        h_term, t_term = self._betabinomial_params_controls(proportions, theta_fd, nu_h, C, N)
        idx        = masks["C"].nonzero(as_tuple=True)
        log_lik   += dist.BetaBinomial(h_term[idx], t_term[idx], total_count=data["n_c"][idx]).log_prob(data["n0_c"][idx]).sum()
        count     += len(idx[0])
        return log_lik, count

    def _loglik_drug_wells(self, data, merged, proportions, pi, theta_fd, masks, R, D, Ndrug, beta_control, log_lik, count):
        nu_h       = self._nu_healthy_or_default(merged, beta_control, mode="drug", shape=(R, D, Ndrug))
        h_term, t_term = self._betabinomial_params_drugs(proportions, pi, theta_fd, nu_h, R, D, Ndrug)
        idx        = masks["R"].nonzero(as_tuple=True)
        log_lik   += dist.BetaBinomial(h_term[idx], t_term[idx], total_count=data["n_r"][idx]).log_prob(data["n0_r"][idx]).sum()
        count     += len(idx[0])
        return log_lik, count

    # ------------------------------------------------------------------
    # Private: BetaBinomial parameter builders
    # ------------------------------------------------------------------

    def _nu_healthy_or_default(self, merged, beta_control, *, mode: str, shape: tuple) -> torch.Tensor:
        if beta_control is not None:
            return self._model._compute_nu_healthy(merged, beta_control, mode=mode)
        return torch.full(shape, 0.5)

    def _betabinomial_params_controls(self, proportions, theta_fd, nu_h, C, N):
        eps  = self._model.EPSILON
        h_cl = self._model.cat2clusters["healthy"]
        t_cl = self._model.cat2clusters["tumor"]
        h_term = (
            torch.clamp(theta_fd.view(1, -1) * torch.sum(proportions[h_cl, :], dim=0)[None, :] * nu_h, min=eps)
            if h_cl else torch.zeros(C, N)
        )
        t_term = (
            torch.clamp(theta_fd.view(1, -1) * torch.sum(proportions[t_cl, :], dim=0)[None, :] * (1 - nu_h), min=eps)
            if t_cl else torch.zeros(C, N)
        )
        return h_term, t_term

    def _betabinomial_params_drugs(self, proportions, pi, theta_fd, nu_h, R, D, Ndrug):
        eps       = self._model.EPSILON
        h_cl      = self._model.cat2clusters["healthy"]
        t_cl      = self._model.cat2clusters["tumor"]
        theta_fd_m = theta_fd[:Ndrug] if "not" in self._model.mode_theta.value else theta_fd
        h_term = (
            torch.clamp(theta_fd_m.view(1, 1, -1) * torch.sum(proportions[None, h_cl, :] * pi[:, h_cl, :], dim=1)[None, :, :] * nu_h, min=eps)
            if h_cl else torch.zeros(R, D, Ndrug)
        )
        t_term = (
            torch.clamp(theta_fd_m.view(1, 1, -1) * torch.sum(proportions[None, t_cl, :] * pi[:, t_cl, :], dim=1)[None, :, :] * (1 - nu_h), min=eps)
            if t_cl else torch.zeros(R, D, Ndrug)
        )
        return h_term, t_term

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r})"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator(BaseModelEvaluator):
    """
    Compute metrics from fitted parameters, observed data, and (for simulation studies) ground-truth parameters.

    Parameters
    ----------
    model : scClone2DR
        Generative model instance.  Used to call ``get_mean_fracMEL_*`` and
        to access cluster mappings.

    Examples
    --------
    >>> sim_eval = Evaluator(model)
    >>> results  = sim_eval.compute_all(true_params, data, fitted_params)
    >>> print(results.spearman_drugs_avg)
    """
    def __init__(self, model: scClone2DR) -> None:
        super().__init__(model)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def compute_all(
        self,
        data: dict,
        params: dict,
        true_params: dict = None,
        fold_change: nool = True
    ) -> Results:
        """Compute every available metric and return a :class:`Results`.

        Parameters
        ----------
        true_params:
            Ground-truth parameter dictionary.
        data:
            Observed data dictionary.
        params:
            Fitted parameter dictionary (output of ``Trainer.train``).
        """
        # check that params in _model.fitted_params are the same as in the argument params, and log a warning if not
        if self._model.fitted_params is not None:
            for key in self._model.fitted_params:
                if key in params and key not in ['proportions', 'theta_fd']:
    
                    v1 = self._model.fitted_params[key]
                    v2 = params[key]
    
                    # only compare tensor-like values
                    if isinstance(v1, (torch.Tensor, np.ndarray)) and isinstance(v2, (torch.Tensor, np.ndarray)):
                        if not torch.equal(torch.as_tensor(v1), torch.as_tensor(v2)):
                            logger.warning(
                                f"Value for key '{key}' differs between model.fitted_params and the argument params."
                            )
        p = merge_data_params(data, params)  # numpy → tensor
        r = Results()
        if true_params is not None:
            r.kl_proportions, r.proportions, r.true_proportions = (
                self.kl_proportions(true_params, p)
            )
            r.kl_survival_probas, r.survival_probas, r.true_survival_probas = (
                self.kl_survival_probas(true_params, p)
            )
            r.spearman_drugs_avg, r.drug_scores, r.true_drug_scores = (
                self.spearman_drug(true_params, data, params=p)
            )
            r.spearman_subclones_avg = self.spearman_subclone(true_params, data, params=p)
            if 'beta' in p.keys():
                r.mse_beta               = self.beta_mse(true_params, p)
            r.l1err_overall_survival = self.overall_survival_error(true_params, p)

        r.drug_effects, r.true_drug_effects, r.l1err_drug_effects = (
            self.drug_effects(p, true_params=true_params)
        )
        if fold_change:
            fc = self.fold_change(data, p, true_params=true_params)
            r.fold_change_pred = fc["pred"]
            r.fold_change_data = fc["not pred"]
            r.fold_change_true = fc.get("true", [])

        return r

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def kl_proportions(
        self, true_params: dict, params: dict
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Per-sample KL divergence between estimated and true clone proportions.

        Returns
        -------
        tuple
            ``(kl_value, estimated_proportions, true_proportions)``
        """
        kl = _kl_divergence(
            params["proportions"], true_params["proportions"], mode="proportions"
        ) / true_params["proportions"].shape[0]
        return float(kl), params["proportions"], true_params["proportions"]

    def kl_survival_probas(
        self, true_params: dict, params: dict
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Scale-corrected KL divergence between estimated and true survival probabilities.

        Returns
        -------
        tuple
            ``(kl_value, rescaled_estimated_pi, true_pi)``
        """
        D, Kmax, N = params["pi"].shape
        pi = deepcopy(params["pi"])
        for d in range(D):
            for i in range(N):
                coeff      = torch.mean(true_params["pi"][d, :, i] / params["pi"][d, :, i])
                scale      = min(float(coeff), float(1.0 / torch.max(params["pi"][d, :, i])))
                pi[d, :, i] = pi[d, :, i] * scale
        kl = _kl_divergence(pi, true_params["pi"], mode="survival") / (D * N)
        return float(kl), pi, true_params["pi"]

    def spearman_drug(
        self,
        true_params: dict,
        data: dict,
        params: dict | None = None,
    ) -> tuple[float, list, list]:
        """Average per-patient Spearman correlation of drug scores over drugs.

        Returns
        -------
        tuple
            ``(mean_spearman, fitted_scores, true_scores)``
        """
        merged = merge_data_params(data, params)
        h_cl   = self._model.cat2clusters["healthy"]
        t_cl   = self._model.cat2clusters["tumor"]

        spearman_avg = 0.0
        hat_scores, true_scores = [], []

        for i in range(merged["N"]):
            hat  = torch.min(merged["pi"][:, h_cl, i], dim=1)[0] / torch.max(merged["pi"][:, t_cl, i], dim=1)[0]
            true = torch.min(true_params["pi"][:, h_cl, i], dim=1)[0] / torch.max(true_params["pi"][:, t_cl, i], dim=1)[0]
            spearman_avg += scipy_stats.spearmanr(hat, true)[0] / merged["N"]
            hat_scores  += hat.tolist()
            true_scores += true.tolist()

        return spearman_avg, hat_scores, true_scores

    def spearman_subclone(
        self,
        true_params: dict,
        data: dict,
        params: dict | None = None,
    ) -> float:
        """Average per-patient-per-drug Spearman correlation at the subclone level.

        Returns
        -------
        float
            Mean Spearman correlation across all (patient, drug) pairs.
        """
        merged = merge_data_params(data, params)
        h_cl   = self._model.cat2clusters["healthy"]
        t_cl   = self._model.cat2clusters["tumor"]

        total = 0.0
        for i in range(merged["N"]):
            for d in range(merged["D"]):
                hat  = torch.mean(merged["pi"][d, h_cl, i]) / merged["pi"][d, t_cl, i]
                true = torch.mean(true_params["pi"][d, h_cl, i]) / true_params["pi"][d, t_cl, i]
                total += scipy_stats.spearmanr(hat, true)[0] / (merged["N"] * merged["D"])
        return total

    def beta_mse(self, true_params: dict, params: dict) -> float:
        """Normalised MSE between the estimated and true ``beta`` matrices.

        Returns
        -------
        float
        """
        return float(
            torch.norm(params["beta"] - true_params["beta"]) ** 2
            / np.prod(true_params["beta"].shape)
        )

    def drug_effects(
        self,
        params: dict,
        true_params: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, float | None]:
        """Tumour-weighted drug-effect scores.

        Returns
        -------
        tuple
            ``(fitted_scores, true_scores_or_None, l1_error_or_None)``
        """
        def _score(p: dict) -> torch.Tensor:
            t_cl = self._model.cat2clusters["tumor"]
            prop = p["proportions"].T
            s    = torch.sum(p["pi"][:, t_cl, :] * prop[None, t_cl, :], dim=1)
            s   /= torch.sum(p["pi"] * prop[None, :, :], dim=1)
            s   /= 1 - torch.mean(prop[None, self._model.cat2clusters["healthy"], :], dim=1)
            return s.reshape(-1)

        scores      = _score(params)
        true_scores = _score(true_params) if true_params is not None else None
        l1_err      = float(torch.mean(torch.abs(true_scores - scores))) if true_scores is not None else None
        return scores, true_scores, l1_err

    def overall_survival_error(self, true_params: dict, params: dict) -> float:
        """Mean absolute error of the overall (population-averaged) survival probability.

        Returns
        -------
        float
        """
        D, Kmax, N = params["pi"].shape
        pi = deepcopy(params["pi"])
        for d in range(D):
            for i in range(N):
                coeff       = torch.mean(true_params["pi"][d, :, i] / params["pi"][d, :, i])
                scale       = min(float(coeff), float(1.0 / torch.max(params["pi"][d, :, i])))
                pi[d, :, i] = pi[d, :, i] * scale

        proba_hat  = np.array([[float(torch.sum(pi[d, :, i] * params["proportions"])) for i in range(N)] for d in range(D)])
        proba_true = np.array([[float(torch.sum(true_params["pi"][d, :, i] * true_params["proportions"])) for i in range(N)] for d in range(D)])
        return float(np.mean(np.abs(proba_hat - proba_true)))

    def fold_change(
        self,
        data: dict,
        params: dict,
        *,
        true_params: dict | None = None,
        sample_indices: list[int] | None = None,
        drug_indices: np.ndarray | None = None,
        stat_test: str = "t_test",
    ) -> dict[str, list]:
        """Compute per-(patient, drug) fold changes and p-values.

        Parameters
        ----------
        data:
            Observed data dictionary.
        params:
            Fitted parameter dictionary.
        true_params:
            When provided, also computes fold changes under the true model.
        sample_indices:
            Subset of patient indices to evaluate.  Defaults to all.
        drug_indices:
            Subset of drug indices to evaluate.  Defaults to all.
        stat_test:
            One of ``"mannwhitneyu"``, ``"t_test"``, ``"Welch"``.

        Returns
        -------
        dict[str, list]
            Keys: ``"pred"`` (fitted), ``"not pred"`` (raw data),
            and optionally ``"true"`` (ground truth).
        """
        merged = merge_data_params(data, params)
        N, D   = merged["N"], merged["D"]

        if sample_indices is None:
            sample_indices = list(range(N))
        if drug_indices is None:
            drug_indices = np.arange(D)

        _, DIC_sample        = self.over_sample(merged, multi_C=1, multi_R=1)
        DIC_sample           = load_from_sampling(DIC_sample, data)  # noqa: F821  (external util)

        preds_c      = 1.0 - self._model.get_mean_fracMEL_control(merged["proportions"].T, merged["C"], DIC_sample["nu_healthy_control"])
        preds_drugs  = 1.0 - self._model.get_mean_fracMEL_treated(merged["proportions"].T, D, DIC_sample["pi"], DIC_sample["nu_healthy_drug"])
        ones_pi      = torch.ones_like(DIC_sample["pi"])
        preds_c4drugs = 1.0 - self._model.get_mean_fracMEL_treated(merged["proportions"].T, D, ones_pi, DIC_sample["nu_healthy_drug"])

        modes = ["not pred", "pred"]
        preds_drugs_true = preds_c4drugs_true = preds_c_true = None

        if true_params is not None:
            data_true   = merge_data_params(data, true_params)
            _, DIC_true = self.over_sample(data_true, multi_C=1, multi_R=1)
            DIC_true    = load_from_sampling(DIC_true, data)  # noqa: F821
            preds_drugs_true  = 1.0 - self._model.get_mean_fracMEL_treated(data_true["proportions"].T, D, DIC_true["pi"], DIC_true["nu_healthy_drug"])
            preds_c4drugs_true = 1.0 - self._model.get_mean_fracMEL_treated(data_true["proportions"].T, D, torch.ones_like(DIC_true["pi"]), DIC_true["nu_healthy_drug"])
            preds_c_true      = 1.0 - self._model.get_mean_fracMEL_control(data_true["proportions"].T, data_true["C"], DIC_true["nu_healthy_control"])
            modes.append("true")

        _stat_fn = {
            "mannwhitneyu": lambda a, b: scipy_stats.mannwhitneyu(np.log(a), np.log(b)),
            "t_test":        lambda a, b: scipy_stats.ttest_ind(np.log(a), np.log(b)),
            "Welch":         lambda a, b: scipy_stats.ttest_ind(np.log(a), np.log(b), equal_var=False),
        }
        if stat_test not in _stat_fn:
            raise ValueError(f"Unknown stat_test {stat_test!r}. Choose from {list(_stat_fn)}")
        _test = _stat_fn[stat_test]

        all_fold_changes: dict[str, list] = {m: [] for m in modes}

        for mode in modes:
            for patient_id in sample_indices:
                for drug_id in drug_indices:
                    nb_r = int(torch.sum(merged["masks"]["R"][:, drug_id, patient_id]))
                    nb_c = int(torch.sum(merged["masks"]["C"][:, patient_id]))

                    ctrl_raw  = (merged["n0_c"] / merged["n_c"])[:nb_c, patient_id]
                    drug_raw  = (merged["n0_r"] / merged["n_r"])[:nb_r, drug_id, patient_id]

                    if mode == "not pred":
                        ctrl_stat = np.sort(ctrl_raw.numpy())
                        drug_stat = drug_raw.numpy()
                        ctrl_fc   = torch.log(ctrl_raw)
                        drug_fc   = torch.log(drug_raw)
                    elif mode == "pred":
                        ctrl_stat = preds_c4drugs[:nb_c, drug_id, patient_id].numpy()
                        drug_stat = preds_drugs[:nb_r, drug_id, patient_id].numpy()
                        ctrl_fc   = torch.log(preds_c[:nb_c, patient_id])
                        drug_fc   = torch.log(preds_drugs[:nb_r, drug_id, patient_id])
                    else:  # "true"
                        ctrl_stat = preds_c4drugs_true[:nb_c, drug_id, patient_id].numpy()
                        drug_stat = preds_drugs_true[:nb_r, drug_id, patient_id].numpy()
                        ctrl_fc   = torch.log(preds_c_true[:nb_c, patient_id])
                        drug_fc   = torch.log(preds_drugs_true[:nb_r, drug_id, patient_id])

                    _test(ctrl_stat, drug_stat)  # p-value computed but not stored (extend if needed)
                    fold_change = ctrl_fc.mean() - drug_fc.mean()
                    all_fold_changes[mode].append(fold_change)

        return all_fold_changes
    

    def over_sample(self, dic, multi_C = 4, multi_R = 4):
        DIC = deepcopy(dic)
        DIC['R'] = multi_R * dic['R']
        DIC['masks']['R'] =  torch.repeat_interleave(dic['masks']['R'], multi_R, axis=0)
        DIC['n_r'] = torch.repeat_interleave(dic['n_r'], multi_R, axis=0)
        DIC['X_nu_drug'] = torch.repeat_interleave(dic['X_nu_drug'], multi_R, axis=0)
        DIC['C'] = multi_C * dic['C']
        DIC['masks']['C'] =  torch.repeat_interleave(dic['masks']['C'], multi_C, axis=0)
        DIC['n_c'] = torch.repeat_interleave(dic['n_c'], multi_C, axis=0)
        DIC['X_nu_control'] = torch.repeat_interleave(dic['X_nu_control'], multi_C, axis=0)
        DIC_sample_data, DIC_sample_params = self._model.sampling(DIC, DIC)
        DIC = load_from_sampling(DIC, DIC_sample_data)
        DIC_sample = merge_data_params(DIC_sample_data, DIC_sample_params)
        return DIC, DIC_sample

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r})"