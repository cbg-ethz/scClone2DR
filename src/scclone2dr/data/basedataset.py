"""Base dataset class for data handling."""

from __future__ import annotations

import numpy as np
import torch


class BaseDataset:
    """Base class for dataset handling and preprocessing.

    Subclasses must set ``cluster2clonelabel`` and ``clonelabel2cat``
    before calling ``init_cat_clonelabel()``.  The expected way to do
    this is via ``init_topology()``, which sets both and then calls
    ``init_cat_clonelabel()`` in one step.
    """

    def __init__(self) -> None:
        self.cluster2clonelabel: list[str] | None = None
        self.clonelabel2cat:     dict[str, str] | None = None

        # Populated by init_cat_clonelabel()
        self.clonelabels:         list[str] | None = None
        self.n_clonelabels:       int | None = None
        self.n_cat:               int | None = None
        self.cat2clonelabels:     dict[str, list[str]] | None = None
        self.cat2clusters:        dict[str, list[int]] | None = None
        self.cluster2cat:         np.ndarray | None = None
        self.clonelabel2clusters: dict[str, list[int]] | None = None
        self.drugs:                list[str] | None = None
        self.sample_names:         list[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_topology(
        self,
        cluster2clonelabel: list[str],
        clonelabel2cat: dict[str, str],
    ) -> None:
        """Set clone-label / category mappings and derive all downstream attributes.

        This is the single entry point for structural configuration.
        Call this (via the pipeline or dataset subclass) before any
        method that relies on clone/category mappings.

        Parameters
        ----------
        cluster2clonelabel:
            Length-Kmax list mapping each cluster index to a clone label,
            e.g. ``["healthy", "tumor", "tumor"]``.
        clonelabel2cat:
            Maps each clone label to a category, e.g.
            ``{"healthy": "healthy", "tumor": "tumor"}``.
        """
        self.cluster2clonelabel = cluster2clonelabel
        self.clonelabel2cat     = clonelabel2cat
        self._init_cat_clonelabel()

    # ------------------------------------------------------------------
    # Private: derived mappings
    # ------------------------------------------------------------------

    def _init_cat_clonelabel(self) -> None:
        """Derive all clone/category mappings from the two primary attributes."""
        self._require_topology()

        cats = list(np.unique(list(self.clonelabel2cat.values())))
        assert set(cats) == {"healthy", "tumor"}, (
            f"Expected exactly the categories 'healthy' and 'tumor', got {cats}"
        )
        self.n_cat           = len(cats)
        self.clonelabels     = list(np.unique(self.cluster2clonelabel))
        self.n_clonelabels   = len(self.clonelabels)

        self.cat2clonelabels = {
            cat: [cl for cl in self.clonelabels if self.clonelabel2cat[cl] == cat]
            for cat in cats
        }
        self.cat2clusters = {
            cat: [
                i for i, cl in enumerate(self.cluster2clonelabel)
                if self.clonelabel2cat[cl] == cat
            ]
            for cat in cats
        }
        self.cluster2cat = np.array([
            self.clonelabel2cat[cl] for cl in self.cluster2clonelabel
        ])
        self.clonelabel2clusters = {
            cl: [i for i, c in enumerate(self.cluster2clonelabel) if c == cl]
            for cl in self.clonelabels
        }

    def _require_topology(self) -> None:
        """Raise clearly if ``init_topology()`` has not been called."""
        if self.cluster2clonelabel is None or self.clonelabel2cat is None:
            raise RuntimeError(
                "Dataset topology is not initialised. "
                "Call dataset.init_topology(cluster2clonelabel, clonelabel2cat) first."
            )

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    def get_fold_change_obs(self, DIC: dict) -> np.ndarray:
        """Calculate observed fold changes from data.

        Parameters
        ----------
        DIC:
            Dictionary with keys ``n_r``, ``n0_r``, ``n_c``, ``n0_c``,
            and ``masks`` (containing ``"R"`` and ``"C"``).

        Returns
        -------
        np.ndarray
            Shape ``(D, Ndrug)``.
        """
        D     = DIC["n_r"].shape[1]
        Ndrug = DIC["n_r"].shape[2]
        fold_change_obs = np.zeros((D, Ndrug))

        log_survival_r = torch.log(DIC["n0_r"] / DIC["n_r"])
        log_survival_c = torch.log(DIC["n0_c"] / DIC["n_c"])

        for patient_id in range(Ndrug):
            for drug_id in range(D):
                nb_r = int(torch.sum(DIC["masks"]["R"][:, drug_id, patient_id]))
                nb_c = int(torch.sum(DIC["masks"]["C"][:, patient_id]))

                mean_drug    = log_survival_r[:nb_r, drug_id, patient_id].mean()
                mean_control = log_survival_c[:nb_c, patient_id].mean()

                fold_change_obs[drug_id, patient_id] = mean_control - mean_drug

        return fold_change_obs