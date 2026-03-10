"""scClone2DR pipeline — high-level entry point.

:class:`scClone2DRPipeline` is the **only** class that knows about all
collaborators.  It wires them together and exposes a simple, stable API for
the end user.  Internally it delegates every concern to the appropriate
specialist:

* data loading → ``DataSource`` (``RealData`` or ``SimulatedData``)
* model definition → :class:`~scclone2dr.model.scClone2DR`
* optimisation → :class:`~scclone2dr.trainers.Trainer`
* posterior sampling → :class:`~scclone2dr.posterior_sampler.PosteriorSampler`
* evaluation / scoring → :class:`~scclone2dr.model_evaluator.ModelEvaluator`

The original ``scClone2DR(Trainer, SimulatedData, RealData, ComputeStatistics)``
class has been completely dissolved into these focused components.

Typical usage
-------------
>>> from scclone2dr.datasets import RealData
>>> from scclone2dr.trainers import Trainer, GuideType
>>> from scclone2dr.resultanalysis import ComputeStatistics

>>> pipeline = scClone2DRPipeline(
...     data_source=RealData(path_fastdrug=..., path_rna=...),
...     trainer=Trainer(guide_type=GuideType.FULL_MVN),
...     mode_nu="noise_correction",
...     mode_theta="not shared decoupled",
... )
>>> params = pipeline.fit(n_steps=2000, penalty_l2=1e-3)
>>> ll     = pipeline.log_likelihood(data, params)
>>> pipeline.sample_posterior(data, params, dir_save="./out/")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from copy import deepcopy
import torch

from .types import NuMode, ThetaMode
from .model import scClone2DR
from .inference.model_evaluator import ModelEvaluator, _GUIDE_TYPE_KEY, Results
from .inference.posterior_sampler import PosteriorSampler
from .trainer import GuideType, Trainer
from .data.basedataset import BaseDataset

logger = logging.getLogger(__name__)
_MODEL_CONFIG_KEY = "__model_configuration__"
_PIPELINE_CONFIG_KEY = "__pipeline_configuration__"

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class scClone2DRPipeline:
    """High-level orchestrator for the scClone2DR workflow.

    Parameters
    ----------
    data_source:
        Object satisfying :class:`DataSource` (``RealData`` or
        ``SimulatedData``).
    trainer:
        :class:`~scclone2dr.trainers.Trainer` instance.
    mode_nu:
        Pre-assay effect mode — ``"fixed"`` or ``"noise_correction"``.
    mode_theta:
        Overdispersion mode — one of ``"no_overdispersion"``, ``"equal"``,
        ``"shared"``, ``"not shared coupled"``, ``"not shared decoupled"``.
    """

    def __init__(
        self,
        data_source: BaseDataset,
        trainer: Trainer | None = None,
        *,
        mode_nu: str    = NuMode.NOISE_CORRECTION,
        mode_theta: str = ThetaMode.NOT_SHARED_DECOUPLED,
    ) -> None:
        self._data_source = data_source
        self._trainer     = trainer if trainer is not None else Trainer()

        # The model is a pure generative object — it takes no collaborators.
        self.model = scClone2DR(mode_nu=mode_nu, mode_theta=mode_theta)

        # Evaluator only needs the model; guide type is read from params at call time.
        self.evaluator = ModelEvaluator(self.model)

        # Sampler is (re-)created after fitting when the guide is available.
        self._sampler: PosteriorSampler | None = None

        self._params: dict[str, np.ndarray] | None = None
        self._try_configure_model_from_data_source()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def guide(self):
        """The fitted variational guide (``None`` before :meth:`fit`)."""
        return self._trainer.guide

    @property
    def params(self) -> dict[str, np.ndarray] | None:
        """Learned parameters (``None`` before :meth:`fit`)."""
        return self._params

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: dict | None = None,
        *,
        penalty_l1: float | None = None,
        penalty_l2: float | None = None,
        lr: float = 0.01,
        n_steps: int = 2000,
        **data_kwargs,
    ) -> dict[str, np.ndarray]:
        """Fit the model and cache the learned parameters.

        Parameters
        ----------
        data:
            Pre-built data dictionary.  When ``None``, the data source is
            called with *data_kwargs*.
        penalty_l1, penalty_l2:
            Regularisation coefficients on ``beta``.
        lr:
            Adam learning rate.
        n_steps:
            Number of gradient steps.
        **data_kwargs:
            Forwarded to ``data_source.prepare_training_data()`` when
            *data* is ``None``.

        Returns
        -------
        dict[str, np.ndarray]
            Learned parameters from the Pyro param store.
        """
        self._try_configure_model_from_data_source()
        model_fn = lambda d: self.model.prior(d, fixed_proportions=False)
        self._params = self._trainer.train(
            model_fn,
            data,
            penalty_l1=penalty_l1,
            penalty_l2=penalty_l2,
            lr=lr,
            n_steps=n_steps,
        )
        # Make fitted parameters available on the model directly so callers
        # can use model.get_survival_probas(data) without passing params.
        self.model.fitted_params = self._params

        # Build the sampler now that the live guide is available.
        self._sampler = self._build_sampler_from_params(self._params)

        return self._params

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def log_likelihood(self, data: dict, params: dict | None = None) -> torch.Tensor:
        """Normalised log-likelihood of *data* under the fitted model.

        Parameters
        ----------
        data:
            Observed data dictionary.
        params:
            Learned parameters.  Defaults to :attr:`params` from the last
            :meth:`fit` call.
        """
        params = params or self._require_params()
        return self.evaluator.log_likelihood(data, params)

    def posterior_mean_latent(
        self, params: dict | None = None, nsamples: int = 100
    ) -> dict[str, torch.Tensor]:
        """Monte-Carlo posterior mean of the latent ``gamma`` variables."""
        params = params or self._require_params()
        return self.evaluator.posterior_mean_latent(params, nsamples=nsamples)

    # ------------------------------------------------------------------
    # Posterior sampling
    # ------------------------------------------------------------------

    def sample_posterior(
        self,
        data: dict,
        idxs_sample_eval: list,
        params: dict | None = None,
        *,
        nb_ites: int = 100,
        dir_save: str | None = None,
        sample_names: list | None = None,
        model_name: str = "",
    ) -> dict:
        """Run posterior sampling and optionally persist results to HDF5.

        Parameters
        ----------
        data:
            Observed data dictionary.
        idxs_sample_eval:
            Indices of samples to include in the evaluation.
        params:
            Learned parameters.  Defaults to :attr:`params` from last fit.
        nb_ites:
            Number of Monte-Carlo iterations.
        dir_save:
            When provided, HDF5 results are written here.
        sample_names:
            Names for the sample axis.
        model_name:
            File-name prefix for HDF5 output.

        Returns
        -------
        dict
            Accumulated posterior statistics.
        """
        params  = params or self._require_params()
        sampler = self._require_sampler()

        params_eval = {}
        for key, val in params.items():
            if torch.is_tensor(val):
                params_eval[key] = val.clone().detach()
            else:
                params_eval[key] = val

        data_eval = deepcopy(data)

        results = sampler.sample(data_eval, params_eval, idxs_sample=idxs_sample_eval, nb_ites=nb_ites)

        if dir_save is not None:
            sampler.save_results(
                results,
                dir_save=dir_save,
                data=data,
                sample_names=sample_names,
                model_name=model_name,
            )

        return results

    def evaluate(self, data: dict, params: dict, true_params: dict = None) -> Results:
        """Evaluate posterior sampling results using the model evaluator."""
        if true_params is not None and "pi" not in true_params:
            if data['single_cell_features']:
                true_params["pi"] = self.model.compute_survival_probas_single_cell_features(data, true_params)
            else:
                true_params["pi"] = self.model.compute_survival_probas_subclone_features(data, true_params)
        return self.evaluator.compute_all(data, params, true_params=true_params)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _require_params(self) -> dict[str, np.ndarray]:
        if self._params is None:
            raise RuntimeError("Call fit() before accessing results.")
        return self._params

    def _require_sampler(self) -> PosteriorSampler:
        if self._sampler is None:
            raise RuntimeError("Call fit() before sampling from the posterior.")
        return self._sampler

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | os.PathLike) -> None:
        """Persist the learned parameters to a ``.npz`` file."""

        params = self._require_params()

        arrays = {}
        metadata = {}

        for k, v in params.items():
            if torch.is_tensor(v):
                arrays[k] = v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray):
                arrays[k] = v
            else:
                metadata[k] = v

        model_config = self.model.export_configuration()
        if model_config:
            metadata[_MODEL_CONFIG_KEY] = model_config

        metadata[_PIPELINE_CONFIG_KEY] = {
            "mode_nu": self.model.mode_nu.value,
            "mode_theta": self.model.mode_theta.value,
            "guide_type": self._trainer.guide_type.value,
            "rank": self._trainer.rank,
        }

        if metadata:
            arrays["_metadata"] = np.array(metadata, dtype=object)

        np.savez(path, **arrays)
        logger.info("Parameters saved to %s", Path(path).with_suffix(".npz"))


    def load_params(self, path: str | os.PathLike) -> dict:
        """Load parameters from a ``.npz`` file produced by :meth:`save`."""

        resolved = Path(path)
        if not resolved.exists():
            resolved = resolved.with_suffix(".npz")
        if not resolved.exists():
            raise FileNotFoundError(f"Parameter file not found: {path!r}")

        archive = np.load(resolved, allow_pickle=True)

        params = {}

        for k in archive.files:
            if k == "_metadata":
                continue
            arr = archive[k]

            # convert numpy arrays back to torch tensors
            if isinstance(arr, np.ndarray):
                params[k] = torch.from_numpy(arr)
            else:
                params[k] = arr

        if "_metadata" in archive.files:
            metadata = archive["_metadata"].item()

            pipeline_config = metadata.pop(_PIPELINE_CONFIG_KEY, None)
            if pipeline_config is not None:
                self.model.mode_nu = NuMode(pipeline_config["mode_nu"])
                self.model.mode_theta = ThetaMode(pipeline_config["mode_theta"])
                self._trainer.guide_type = GuideType(pipeline_config["guide_type"])
                self._trainer.rank = int(pipeline_config["rank"])
                params.setdefault(_GUIDE_TYPE_KEY, pipeline_config["guide_type"])

            model_config = metadata.pop(_MODEL_CONFIG_KEY, None)
            self.model.load_configuration(model_config)
            params.update(metadata)

        self._params = params
        self.model.fitted_params = params

        logger.info("Parameters loaded from %s", resolved)

        # rebuild posterior sampler
        self._sampler = self._build_sampler_from_params(params)

        return params

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike,
        data_source: BaseDataset,
    ) -> "scClone2DRPipeline":
        """Construct a pipeline and immediately load parameters from *path*.

        This is the preferred way to restore a previously fitted pipeline
        from disk without re-running :meth:`fit`.

        Parameters
        ----------
        path:
            Path to a ``.npz`` file produced by :meth:`save`.
        data_source:
            Data source for any subsequent operations that need data (e.g.
            :meth:`sample_posterior`).

        Returns
        -------
        scClone2DRPipeline
            A fully initialised pipeline with :attr:`params` populated and
            the posterior sampler ready.

        Examples
        --------
        >>> pipeline = scClone2DRPipeline.from_file(
        ...     "checkpoints/run_01.npz",
        ...     data_source=RealData(...),
        ... )
        >>> ll = pipeline.log_likelihood(data)
        """
        instance = cls(
            data_source=data_source,
        )
        instance.load_params(path)
        return instance

    def _try_configure_model_from_data_source(self) -> None:
        """Configure model from data source when structural metadata is available."""
        if self.model.is_configured:
            return

        ds = self._data_source
        if ds.cluster2clonelabel is None or ds.clonelabel2cat is None:
            return
        self.model.configure(ds)

    # ------------------------------------------------------------------
    # Private: sampler construction from saved params
    # ------------------------------------------------------------------

    def _build_sampler_from_params(self, params: dict) -> PosteriorSampler:
        """Reconstruct a :class:`PosteriorSampler` from a parameter dictionary.

        When a live guide is available on the trainer (i.e. right after
        ``fit``), it is used directly.  When parameters are loaded from disk
        (no live guide), the guide distribution is reconstructed from the
        ``"__guide_type__"`` metadata embedded in *params* by
        :meth:`~scclone2dr.trainers.Trainer._serialize_guide`.

        On the MAP/MLE path (no ``"__guide_type__"`` key) the sampler is
        constructed without a guide distribution — posterior sampling will
        raise a clear error if attempted.
        """
        live_guide = self._trainer.guide


        if live_guide is not None:
            return PosteriorSampler(model=self.model, guide=live_guide)

        if _GUIDE_TYPE_KEY in params:
            guide_dist = self.evaluator.build_guide_distribution(params)
            return PosteriorSampler(model=self.model, guide_distribution=guide_dist)

        # MAP/MLE path: no guide available.  Sampler is created without one;
        # it will raise an informative error only if posterior sampling is called.
        return PosteriorSampler(model=self.model)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"model={self.model!r}, "
            f"trainer={self._trainer!r})"
        )