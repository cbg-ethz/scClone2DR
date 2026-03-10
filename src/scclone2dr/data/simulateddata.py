"""Simulated dataset for scClone2DR."""

import copy
import pickle

import numpy as np
import pyro
import pyro.distributions as dist
import torch

from ..utils import load_from_sampling
from .basedataset import BaseDataset
from ..types import NuMode, ThetaMode

# ---------------------------------------------------------------------------
# Fixed simulation topology
# ---------------------------------------------------------------------------

def _make_simple_topology(Kmax: int) -> tuple[list[str], dict[str, str]]:
    """Return (cluster2clonelabel, clonelabel2cat) for the standard healthy+tumor setup."""
    cluster2clonelabel = ["healthy"] + ["tumor"] * (Kmax - 1)
    clonelabel2cat     = {"healthy": "healthy", "tumor": "tumor"}
    return cluster2clonelabel, clonelabel2cat


def _make_biclone_topology() -> tuple[list[str], dict[str, str]]:
    """Return topology for collapsed 2-clone (healthy / tumor) representations."""
    return _make_simple_topology(Kmax=2)


class SimulatedData(BaseDataset):
    """Generates and manipulates simulated training/test data."""

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Simulated training data
    # ------------------------------------------------------------------

    def get_simulated_training_data(
        self,
        data_train: dict | None = None,
        neg_bin_n: float = 2,
        mode_nu: NuMode = NuMode.NOISE_CORRECTION, 
        mode_theta: ThetaMode = ThetaMode.NOT_SHARED_DECOUPLED,
    ) -> tuple[dict, dict]:
        """Generate a full simulated training dataset by sampling from ``model``.

        Parameters
        ----------
        model:
            A configured ``scClone2DR`` instance (must have ``configure``
            already called, or will be configured here via ``init_topology``).
        data_train:
            Optional dict of simulation dimensions and settings.  When
            ``None`` a default EASY setting is used.
        neg_bin_n:
            Controls overdispersion of ``theta_fd`` via a negative-binomial draw.

        Returns
        -------
        data_train, params
        """
        from ..model import scClone2DR # avoid circular import
        if data_train is None:
            settings   = {"HARD": {"disp": 20.0, "etheta": 3.0},
                          "EASY": {"disp": 100.0, "etheta": 100.0}}
            setting    = settings["EASY"]
            C, R, N, Kmax, D = 24, 10, 30, 7, 30
            data_train = {
                "C": C, "R": R, "N": N, "D": D, "Kmax": Kmax,
                "single_cell_features": False,
                "dispersion_fd": setting["disp"],
                "etheta_fd":     setting["etheta"],
            }
            var_preassay = 0.03
        else:
            N, Kmax = data_train["N"], data_train["Kmax"]
            R, C, D = data_train["R"], data_train["C"], data_train["D"]
            data_train["single_cell_features"] = False
            var_preassay = data_train.get("var_preassay", 0.03)

        self.sample_names = [f"sample_{i}" for i in range(N)]
        self.drugs = [f"drug_{d}" for d in range(D)]

        # Structural setup — single call, on self
        cluster2clonelabel, clonelabel2cat = _make_simple_topology(Kmax)
        self.init_topology(cluster2clonelabel, clonelabel2cat)
        model = scClone2DR(mode_nu=mode_nu, mode_theta=mode_theta)
        model.configure(self)

        # ---- masks -------------------------------------------------------
        masks = {
            "RNA": torch.ones((Kmax, N), dtype=torch.bool),
            "C":   torch.ones((C,    N), dtype=torch.bool),
            "R":   torch.ones((R, D, N), dtype=torch.bool),
        }
        data_train["masks"] = masks

        # ---- proportions -------------------------------------------------
        props = torch.zeros((Kmax, N))
        for i in range(N):
            props[:, i] = torch.distributions.Dirichlet(
                torch.tensor([4.0] + (Kmax - 1) * [1.0])
            ).sample()
        params = {"proportions": props.T}
        data_train["proportions"] = params["proportions"]

        # ---- features / parameters ---------------------------------------
        dim_all = 20
        self.feature_names = [f"dim_{i}" for i in range(dim_all)]
        data_train["X"] = torch.tensor(
            np.abs(np.random.normal(0, 0.3, (Kmax, N, dim_all))),
            dtype=torch.float32,
        )
        for k in range(Kmax):
            data_train["X"][k] *= 1 - k / Kmax

        params["beta"] = torch.zeros((D, dim_all))
        for d in range(D):
            params["beta"][d, :dim_all] = (
                torch.tensor(np.abs(np.random.normal(0, 1, dim_all)), dtype=torch.float32)
                / np.sqrt(dim_all)
            )

        params["offset_healthy"] = torch.zeros(D)
        params["offset_tumor"]   = torch.zeros(D)
        params["beta_control"]   = torch.ones(1, dtype=torch.float32)

        if neg_bin_n >= 1:
            theta_fd_np = np.random.negative_binomial(neg_bin_n, 0.001, N)
        else:
            theta_fd_np = np.full(N, 1000.0 * neg_bin_n)
        params["theta_fd"] = torch.tensor(theta_fd_np[:N], dtype=torch.float32)
        params["theta_rna"] = 40.0

        # ---- pre-assay covariates ----------------------------------------
        log_ratio = np.log(0.6 / 0.9)
        data_train["X_nu_control"] = torch.tensor(
            np.random.normal(log_ratio, var_preassay, (C, N, 1)), dtype=torch.float32
        )
        data_train["X_nu_drug"] = torch.tensor(
            np.random.normal(log_ratio, var_preassay, (R, D, N, 1)), dtype=torch.float32
        )

        # ---- cell / well counts ------------------------------------------
        data_train["n_rna"] = (5000 * torch.ones((Kmax, N))).int()
        data_train["n_r"]   = (1000 * torch.ones((R, D, N))).int()
        data_train["n_c"]   = (1000 * torch.ones((C, N))).int()

        # ---- sample from model -------------------------------------------
        pyro.clear_param_store()
        data_samp, _  = model.sampling(data_train, params)
        data_train = load_from_sampling(data_train, data_samp)

        return data_train, params

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------

    def get_data_split(
        self, data: dict, idxs_train: list[int], idxs_test: list[int]
    ) -> tuple[dict, dict]:
        """Split a simulated dataset into train and test subsets.

        Parameters
        ----------
        data:
            Full dataset dict as returned by ``get_simulated_training_data``.
        idxs_train, idxs_test:
            Sample indices for each split.

        Returns
        -------
        data_train, data_test
        """
        Ntrain, Ntest = len(idxs_train), len(idxs_test)
        Ntot          = Ntrain + Ntest
        Kmax, R, C, D = data["Kmax"], data["R"], data["C"], data["D"]

        # ---- masks -------------------------------------------------------
        masks_train = self._build_split_masks(data, idxs_train, idxs_test, Kmax, R, C, D, Ntrain, Ntot)
        masks_test  = self._build_subset_masks(data, idxs_test, Kmax, R, C, D)

        # ---- train dict --------------------------------------------------
        tr = idxs_train
        data_train = {
            "X":       data["X"][:, tr, :],
            "D": D, "R": R, "C": C, "Kmax": Kmax, "N": Ntot,
            "single_cell_features": False,
            "simulated_data": True,
            "masks": masks_train,
        }
        data_train["X_nu_drug"]    = data["X_nu_drug"][:, :, tr, :]
        data_train["X_nu_control"] = torch.zeros(data["X_nu_control"].shape)
        data_train["X_nu_control"][:, :Ntrain, :] = data["X_nu_control"][:, tr, :]
        data_train["X_nu_control"][:, Ntrain:, :] = data["X_nu_control"][:, idxs_test, :]

        for key in ("n0_c", "n_c"):
            buf = torch.zeros((C, Ntot))
            buf[:, :Ntrain] = data[key][:, tr]
            buf[:, Ntrain:] = data[key][:, idxs_test]
            data_train[key] = buf

        n_rna_buf = torch.zeros((Kmax, Ntot))
        n_rna_buf[:, :Ntrain] = torch.as_tensor(data["n_rna"][:, tr])
        n_rna_buf[:, Ntrain:] = torch.as_tensor(data["n_rna"][:, idxs_test])
        data_train["n_rna"] = n_rna_buf

        data_train["n0_r"] = data["n0_r"][:, :, tr]
        data_train["n_r"]  = data["n_r"][:, :, tr]

        props_buf = torch.zeros((Ntot, Kmax))
        props_buf[:Ntrain] = data["proportions"][tr]
        props_buf[Ntrain:] = data["proportions"][idxs_test]
        data_train["proportions"] = props_buf
        data_train["ini_proportions"] = _ini_proportions(data_train["n_rna"], Kmax, Ntot)

        data_train = _add_frac_stats(data_train, masks_train)

        # ---- test dict ---------------------------------------------------
        te = idxs_test
        data_test = {
            "R": R, "N": Ntest, "D": D, "C": C, "Kmax": Kmax,
            "single_cell_features": False,
            "simulated_data": True,
            "masks": masks_test,
        }
        data_test["X"]             = data["X"][:, te, :]
        data_test["X_nu_drug"]     = data["X_nu_drug"][:, :, te, :]
        data_test["X_nu_control"]  = data["X_nu_control"][:, te, :]
        data_test["n_rna"]         = torch.as_tensor(data["n_rna"][:, te])
        data_test["n0_c"]          = data["n0_c"][:, te]
        data_test["n_c"]           = data["n_c"][:, te]
        data_test["n0_r"]          = data["n0_r"][:, :, te]
        data_test["n_r"]           = data["n_r"][:, :, te]
        data_test["proportions"]   = data["proportions"][te]
        data_test["ini_proportions"] = _ini_proportions(data_test["n_rna"], Kmax, Ntest)
        data_test = _add_frac_stats(data_test, masks_test)

        return data_train, data_test

    def get_params_split(self, params: dict, idxs_train: list[int], idxs_test: list[int]) -> tuple[dict, dict]:
        """Split a dict of simulation parameters into train and test subsets.

        Parameters
        ----------
        params:
            Full dict of simulation parameters as returned by ``get_simulated_training_data``.
        idxs_train, idxs_test:
            Sample indices for each split.
        Returns
        -------
        params_train, params_test
        """
        params_train = copy.deepcopy(params)
        params_test  = copy.deepcopy(params)
        for key in ("proportions", "theta_fd"):
            params_train[key]  = params[key][idxs_train,...]
            params_test[key]  = params[key][idxs_test,...]
        return params_train, params_test

    # ------------------------------------------------------------------
    # Data-representation transforms  (pure — do not mutate self)
    # ------------------------------------------------------------------

    def get_base_from_data(self, dic: dict) -> tuple[dict, "SimulatedData"]:
        """Return a copy of ``dic`` with zeroed features, plus a configured dataset."""
        data = copy.deepcopy(dic)
        data["X"] = torch.zeros_like(data["X"])
        dataset = self._clone_with_topology(data["Kmax"])
        return data, dataset

    def get_bulk_from_data(self, dic: dict) -> tuple[dict, "SimulatedData"]:
        """Collapse to a 2-clone (healthy / tumor) bulk representation."""
        data    = copy.deepcopy(dic)
        dataset = self._clone_with_topology(2)  # binary topology

        healthy = dataset.cat2clusters["healthy"]
        tumor   = dataset.cat2clusters["tumor"]
        Ndrug   = data["X"].shape[1]

        weights = data["proportions"].T[:, :Ndrug]
        weights = weights / weights.sum(dim=0, keepdim=True)
        Z = torch.zeros((2, data["X"].shape[1], data["X"].shape[2]))
        Z[0] = (torch.nan_to_num(data["X"]) * weights[:, :, None]).sum(dim=0)
        Z[1] = Z[0].clone()
        data["X"] = Z

        props2 = torch.zeros((data["n_c"].shape[1], 2))
        props2[:, 0] = torch.mean(
            (data["n0_c"] / data["n_c"])[healthy, :], dim=0
        )
        props2[:, 1] = 1 - props2[:, 0]
        data["ini_proportions"] = props2

        data["Kmax"] = 2
        p = torch.zeros((2, data["n_c"].shape[1]))
        p[0] = data["proportions"][:, healthy].sum(dim=1)
        p[1] = 1 - p[0]
        data["proportions"] = p.T

        data["n_rna"]          = None
        data["masks"]["RNA"]   = torch.tensor([0])

        return data, dataset

    def get_bimodal_from_data(self, dic: dict) -> tuple[dict, "SimulatedData"]:
        """Collapse to a 2-clone representation preserving healthy/tumor RNA sums."""
        data    = copy.deepcopy(dic)
        dataset = self._clone_with_topology(2)

        healthy = dataset.cat2clusters["healthy"]
        tumor   = dataset.cat2clusters["tumor"]
        Ndrug   = data["X"].shape[1]

        Z = torch.zeros((2, data["X"].shape[1], data["X"].shape[2]))
        for out_idx, clust_idxs in enumerate([healthy, tumor]):
            w = data["proportions"].T[clust_idxs, :Ndrug]
            w = w / w.sum(dim=0, keepdim=True)
            Z[out_idx] = (torch.nan_to_num(data["X"][clust_idxs]) * w[:, :, None]).sum(dim=0)
        data["X"] = Z

        rna = torch.nan_to_num(data["n_rna"])
        rna_total = rna.sum(dim=0)
        props2 = torch.zeros((data["n_rna"].shape[1], 2))
        props2[:, 0] = rna[healthy].sum(dim=0) / rna_total
        props2[:, 1] = 1 - props2[:, 0]
        data["ini_proportions"] = props2

        n_rna2 = torch.zeros((2, data["n_rna"].shape[1]))
        n_rna2[0] = rna[healthy].sum(dim=0)
        n_rna2[1] = rna[tumor].sum(dim=0)
        data["n_rna"]        = n_rna2
        data["masks"]["RNA"] = torch.full((2, data["N"]), True)
        data["Kmax"]         = 2

        p = torch.zeros((2, data["n_c"].shape[1]))
        p[0] = data["proportions"][:, healthy].sum(dim=1)
        p[1] = 1 - p[0]
        data["proportions"] = p.T

        return data, dataset

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_data(self, data: dict, path: str, name_dataset: str) -> None:
        with open(path + name_dataset + ".pkl", "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clone_with_topology(self, Kmax: int) -> "SimulatedData":
        """Return a *new* SimulatedData instance configured for ``Kmax`` clones."""
        ds = SimulatedData()
        ds.init_topology(Kmax)
        return ds

    @staticmethod
    def _build_split_masks(data, idxs_train, idxs_test, Kmax, R, C, D, Ntrain, Ntot):
        masks = {
            "RNA": torch.ones((Kmax, Ntot), dtype=torch.bool),
            "C":   torch.ones((C,    Ntot), dtype=torch.bool),
            "R":   torch.ones((R, D, Ntrain), dtype=torch.bool),
        }
        for split, base in [(idxs_train, 0), (idxs_test, Ntrain)]:
            for j, i in enumerate(split):
                masks["RNA"][:, base + j] = data["masks"]["RNA"][:, i]
                masks["C"][:,   base + j] = data["masks"]["C"][:, i]
        for d in range(D):
            for j, i in enumerate(idxs_train):
                masks["R"][:, d, j] = data["masks"]["R"][:, d, i]
        return masks

    @staticmethod
    def _build_subset_masks(data, idxs, Kmax, R, C, D):
        N = len(idxs)
        masks = {
            "RNA": torch.ones((Kmax, N), dtype=torch.bool),
            "C":   torch.ones((C,    N), dtype=torch.bool),
            "R":   torch.ones((R, D, N), dtype=torch.bool),
        }
        for j, i in enumerate(idxs):
            masks["RNA"][:, j] = data["masks"]["RNA"][:, i]
            masks["C"][:,   j] = data["masks"]["C"][:, i]
            for d in range(D):
                masks["R"][:, d, j] = data["masks"]["R"][:, d, i]
        return masks


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _ini_proportions(n_rna: torch.Tensor, Kmax: int, N: int) -> torch.Tensor:
    return (n_rna / n_rna.sum(dim=0).reshape(1, N).expand(Kmax, -1)).T


def _add_frac_stats(data: dict, masks: dict) -> dict:
    frac_r = torch.nan_to_num(1.0 - data["n0_r"] / data["n_r"])
    frac_c = torch.nan_to_num(1.0 - data["n0_c"] / data["n_c"])
    data["frac_r"] = frac_r
    data["frac_c"] = frac_c
    data["frac_mean_r"] = (
        (masks["R"] * frac_r).sum(dim=0) / masks["R"].sum(dim=0)
    )
    data["frac_mean_c"] = (
        (masks["C"] * frac_c).sum(dim=0) / masks["C"].sum(dim=0)
    )
    return data