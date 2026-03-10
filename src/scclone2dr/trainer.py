"""Stochastic Variational Inference training engine.

:class:`Trainer` is a self-contained optimisation engine.  It has no
knowledge of any specific model's internals and is never meant to be
subclassed.  All model behaviour is injected at call time.

Typical usage
-------------
>>> trainer = Trainer(guide_type=GuideType.FULL_MVN)
>>> params  = trainer.train(model_fn, data, penalty_l2=1e-3, n_steps=2000)
>>> guide   = trainer.guide   # fitted guide, available for posterior work
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable

import numpy as np
import pyro
import torch
from tqdm import tqdm
from pyro import poutine
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import (
    AutoDiagonalNormal,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    init_to_mean,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MASK_KEYS: tuple[str, ...] = ("C", "R", "RNA")
_DEFAULT_RANK: int = 20
_LOG_INTERVAL: int = 100


# ---------------------------------------------------------------------------
# Public enumerations
# ---------------------------------------------------------------------------

class GuideType(str, Enum):
    """Supported variational guide families."""

    NONE        = "none"
    FULL_MVN    = "full_MVN"
    LOWRANK_MVN = "lowrank_MVN"
    DIAGONAL    = "diagonal"


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def l2_regularizer(parameter: torch.Tensor) -> torch.Tensor:
    """Mean squared L2 norm: ``sum(p²) / numel(p)``."""
    return parameter.pow(2.0).sum() / parameter.numel()


def _count_observed(masks: dict[str, torch.Tensor]) -> int:
    """Total number of ``True`` entries across all mask tensors."""
    return int(sum(masks[k].sum() for k in _MASK_KEYS))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Self-contained SVI / MAP training engine for Pyro models.

    The engine owns the optimisation loop and variational guide construction.
    It is decoupled from every model-specific concern: priors, likelihood
    terms, and parameter names are never referenced directly.

    Parameters
    ----------
    guide_type:
        Variational family used in SVI mode.  Accepts :class:`GuideType`
        members or their string equivalents (``"full_MVN"``,
        ``"lowrank_MVN"``, ``"diagonal"``).
    rank:
        Rank of the low-rank MVN factor matrix.  Ignored unless
        *guide_type* is ``GuideType.LOWRANK_MVN``.  Defaults to
        ``20``.

    Attributes
    ----------
    guide : callable or None
        The fitted guide after :meth:`train` has been called; ``None``
        before the first training run.

    Notes
    -----
    ``Trainer`` is **not** a base class.  Downstream model classes should
    hold it as a composed attribute and delegate to it explicitly:

    .. code-block:: python

        class MyModel:
            def __init__(self):
                self._trainer = Trainer(guide_type="full_MVN")

            def fit(self, data, **kw):
                return self._trainer.train(
                    model=lambda d: self._prior(d),
                    data=data,
                    **kw,
                )

            @property
            def guide(self):
                return self._trainer.guide
    """

    def __init__(
        self,
        guide_type: GuideType | str = GuideType.FULL_MVN,
        rank: int | None = None,
    ) -> None:
        self.guide_type: GuideType = GuideType(guide_type)
        self.rank: int = rank if rank is not None else _DEFAULT_RANK
        self.guide: Any | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_guide(self, model: Callable) -> Any:
        """Construct and cache a variational guide for *model*.

        Parameters
        ----------
        model:
            Pyro model callable.

        Returns
        -------
        callable
            A Pyro ``AutoGuide`` instance.

        Raises
        ------
        ValueError
            For unrecognised *guide_type* values.
        """
        match self.guide_type:
            case GuideType.FULL_MVN:
                guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)
            case GuideType.LOWRANK_MVN:
                guide = AutoLowRankMultivariateNormal(model, rank=self.rank)
            case GuideType.DIAGONAL:
                guide = AutoDiagonalNormal(model)
            case GuideType.NONE:
                guide = self._null_guide
            case _:
                raise ValueError(f"Unknown guide type: {self.guide_type!r}")

        self.guide = guide
        return guide

    def train(
        self,
        model: Callable,
        data: dict,
        *,
        guide: Callable | None = None,
        penalty_l1: float | None = None,
        penalty_l2: float | None = None,
        lr: float = 0.01,
        n_steps: int = 2000,
    ) -> dict[str, np.ndarray]:
        """Fit *model* to *data* and return learned parameters.

        Runs full SVI when ``data["single_cell_features"]`` is truthy,
        otherwise runs MAP/MLE via a no-op (delta) guide.

        Parameters
        ----------
        model:
            Pyro model callable with signature ``(data: dict) -> None``.
        data:
            Training data.  Must contain ``"single_cell_features"`` (bool)
            and ``"masks"`` with keys ``"C"``, ``"R"``, ``"RNA"``.
        guide:
            Pre-built guide.  When provided, overrides :attr:`guide_type`
            and skips guide construction entirely.
        penalty_l1:
            L1 coefficient on ``pyro.param("beta")``.
        penalty_l2:
            L2 coefficient on ``pyro.param("beta")``.
        lr:
            Adam learning rate.
        n_steps:
            Number of gradient steps.

        Returns
        -------
        dict[str, np.ndarray]
            Detached numpy copy of every parameter in the Pyro param store.
        """
        if guide is not None:
            self.guide = guide
        elif data.get("single_cell_features"):
            assert self.guide_type is not GuideType.NONE, "Guide type cannot be NONE when single-cell features are present."
        else:
            assert self.guide_type is GuideType.NONE, "Guide type must be NONE when no single-cell features are present."
        self.build_guide(model)

        return self._run_svi(
            model=model,
            guide=self.guide,
            data=data,
            penalty_l1=penalty_l1,
            penalty_l2=penalty_l2,
            lr=lr,
            n_steps=n_steps,
        )

    # ------------------------------------------------------------------
    # Private implementation
    # ------------------------------------------------------------------

    @staticmethod
    def _null_guide(_data: dict) -> None:
        """No-op guide for MAP / MLE training."""

    def _run_svi(
        self,
        *,
        model: Callable,
        guide: Callable,
        data: dict,
        penalty_l1: float | None,
        penalty_l2: float | None,
        lr: float,
        n_steps: int,
    ) -> dict[str, np.ndarray]:
        """Core gradient-descent loop (internal)."""
        pyro.clear_param_store()

        normaliser = _count_observed(data["masks"])

        def loss_fn() -> torch.Tensor:
            return Trace_ELBO().differentiable_loss(model, guide, data)

        # Dry run — collect all unconstrained tensors before entering the loop.
        with poutine.trace(param_only=True) as param_capture:
            loss_fn()
        trainable = {
            site["value"].unconstrained()
            for site in param_capture.trace.nodes.values()
        }
        optimizer = torch.optim.Adam(trainable, lr=lr, betas=(0.90, 0.999))

        pbar = tqdm(range(n_steps), desc="Training")
        for step in pbar:
            loss = loss_fn() / normaliser + self._penalty(penalty_l1, penalty_l2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % _LOG_INTERVAL == 0:
                logger.info("[iter %d]  loss: %.4f", step, loss.item())
                pbar.set_postfix({"loss": loss.item()})

        return {
            key: pyro.param(key)
            for key, _ in pyro.get_param_store().named_parameters()
        }

    @staticmethod
    def _penalty(
        penalty_l1: float | None,
        penalty_l2: float | None,
    ) -> torch.Tensor:
        """Combined L1 + L2 regularisation term (zero when both are ``None``)."""
        reg = torch.tensor(0.0)
        if penalty_l1 is not None:
            beta = pyro.param("beta")
            reg = reg + penalty_l1 * torch.nn.L1Loss()(beta, torch.zeros_like(beta))
        if penalty_l2 is not None:
            reg = reg + penalty_l2 * l2_regularizer(pyro.param("beta"))
        return reg

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        extra = f", rank={self.rank}" if self.guide_type is GuideType.LOWRANK_MVN else ""
        return f"{type(self).__name__}(guide_type={self.guide_type.value!r}{extra})"
