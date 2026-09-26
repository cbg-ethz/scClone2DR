"""Misspecified simulated dataset for scClone2DR.

This module mirrors the structure of ``SimulatedData`` but generates data
under a *misspecified* data-generating process.  Three families of
misspecification are supported and can be combined:

1. **Alternative drug-response function** – the true survival probability
   follows a Hill (sigmoidal) curve instead of the log-linear model assumed
   by scClone2DR.
2. **Different noise structure** – observation noise is heteroscedastic
   (variance scales with the mean) and/or heavy-tailed (Student-t), rather
   than the homoscedastic negative-binomial assumed by the model.
3. **Errors in clonal composition** – ground-truth proportions are corrupted
   by systematic label-swap noise.
"""

import copy
import pickle
from enum import Enum, auto

import numpy as np
import pyro
import torch
import os
import pandas as pd
import random

from ..utils import load_from_sampling
from .basedataset import BaseDataset
from ..types import NuMode, ThetaMode

# ---------------------------------------------------------------------------
# Misspecification options
# ---------------------------------------------------------------------------

class ResponseMisspec(Enum):
    """Which drug-response function to use for the *true* DGP."""
    NONE = auto()          # identical to the well-specified model
    HILL = auto()          # sigmoidal Hill curve
    SATURATING = auto()    # saturating exponential (1 − exp)
    THRESHOLD = auto()     # step-function with soft threshold


class NoiseMisspec(Enum):
    """Which observation-noise distribution to use."""
    NONE = auto()          # standard negative-binomial (well-specified)
    HETEROSCEDASTIC = auto()  # variance ∝ mean²  (quasi-Poisson style)
    HEAVY_TAIL = auto()    # Student-t observation noise (df = 3)
    ZERO_INFLATED = auto() # zero-inflated negative-binomial



# ---------------------------------------------------------------------------
# Fixed simulation topology  (re-used from the well-specified module)
# ---------------------------------------------------------------------------

def _make_simple_topology(Kmax: int) -> tuple[list[str], dict[str, str]]:
    cluster2clonelabel = ["healthy"] + ["tumor"] * (Kmax - 1)
    clonelabel2cat     = {"healthy": "healthy", "tumor": "tumor"}
    return cluster2clonelabel, clonelabel2cat


def _make_biclone_topology() -> tuple[list[str], dict[str, str]]:
    return _make_simple_topology(Kmax=2)


# ---------------------------------------------------------------------------
# Alternative drug-response functions
# ---------------------------------------------------------------------------

def _hill_response(
    x: torch.Tensor, beta: torch.Tensor, ec50: float = 1.5, hill_n: float = 2.0
) -> torch.Tensor:
    """Sigmoidal Hill survival: π = 1 / (1 + (activity / EC50)^n).

    ``activity`` is the linear predictor x @ beta (same inputs the model
    uses), but the *true* mapping to survival is non-linear.
    """
    activity = (x @ beta.T).clamp(min=-10, max=10)
    return 1.0 / (1.0 + (activity.abs() / ec50) ** hill_n)


def _saturating_response(
    x: torch.Tensor, beta: torch.Tensor, rate: float = 2.0, minval: float = -1.0
) -> torch.Tensor:
    """Saturating exponential: π = exp(−rate · |activity|)."""
    activity = (x @ beta.T).clamp(min=minval, max=10)
    return torch.exp(-rate * (activity-minval))


def _threshold_response(
    x: torch.Tensor, beta: torch.Tensor, threshold: float = 0.3, steepness: float = 5.0
) -> torch.Tensor:
    """Soft step-function: π = σ(−steepness · (|activity| − threshold))."""
    activity = (x @ beta.T).clamp(min=-10, max=10)
    return torch.sigmoid(-steepness * (activity - threshold))


_RESPONSE_FN = {
    ResponseMisspec.HILL:       _hill_response,
    ResponseMisspec.SATURATING: _saturating_response,
    ResponseMisspec.THRESHOLD:  _threshold_response,
}


# ---------------------------------------------------------------------------
# Noise helpers
# ---------------------------------------------------------------------------

def _apply_noise(
    counts: torch.Tensor,
    kind: NoiseMisspec,
    dispersion: float = 5.0,
    zi_prob: float = 0.15,
    t_df: float = 3.0,
) -> torch.Tensor:
    """Re-draw observation counts under the chosen noise model.

    Parameters
    ----------
    counts : torch.Tensor
        Expected counts (λ), strictly positive.
    kind : NoiseMisspec
        Which noise family to apply.
    dispersion : float
        Base dispersion (used differently per family).
    zi_prob : float
        Zero-inflation probability (only for ZERO_INFLATED).
    t_df : float
        Degrees of freedom for Student-t noise.
    """
    lam = counts.float().clamp(min=1e-6)

    if kind == NoiseMisspec.HETEROSCEDASTIC:
        # variance = dispersion · mean²  →  σ = sqrt(dispersion) · mean
        sigma = lam * np.sqrt(1/dispersion)
        noisy = torch.normal(lam, sigma).clamp(min=0).round()
        return noisy.to(counts.dtype)

    if kind == NoiseMisspec.HEAVY_TAIL:
        # Student-t centred at λ, scaled by sqrt(λ)
        t_samples = torch.distributions.StudentT(df=t_df).sample(lam.shape)
        noisy = (lam + t_samples * lam.sqrt()).clamp(min=0).round()
        return noisy.to(counts.dtype)

    if kind == NoiseMisspec.ZERO_INFLATED:
        # Standard NB draw, then zero-inflate
        p = dispersion / (dispersion + lam)
        nb = torch.distributions.NegativeBinomial(
            total_count=dispersion, probs=1 - p
        ).sample()
        zero_mask = torch.bernoulli(torch.full_like(lam, zi_prob)).bool()
        nb[zero_mask] = 0.0
        return nb.to(counts.dtype)

    # NONE – pass through
    return counts



# ===================================================================
# Main class
# ===================================================================

class MisspecifiedSimulatedData(BaseDataset):
    """Generates simulated data under a *misspecified* data-generating process.

    The public API mirrors ``SimulatedData`` so the rest of the pipeline
    (splitting, collapsing, saving) works unchanged.
    """

    def __init__(
        self,
        response_misspec: ResponseMisspec = ResponseMisspec.HILL,
        noise_misspec: NoiseMisspec = NoiseMisspec.NONE,
        *,
        hill_ec50: float = 0.5,
        hill_n: float = 2.0,
        saturating_rate: float = 2.0,
        threshold_val: float = 0.3,
        noise_dispersion: float = 5.0,
        zi_prob: float = 0.15,
        t_df: float = 3.0,
        swap_frac: float = 0.,
    ) -> None:
        super().__init__()

        # ---- misspecification knobs --------------------------------------
        self.response_misspec = response_misspec
        self.noise_misspec    = noise_misspec

        # response-function params
        self.hill_ec50       = hill_ec50
        self.hill_n          = hill_n
        self.saturating_rate = saturating_rate
        self.threshold_val   = threshold_val

        # noise params
        self.noise_dispersion = noise_dispersion
        self.zi_prob          = zi_prob
        self.t_df             = t_df

        # clonal params
        self.swap_frac    = swap_frac

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
        """Generate a full simulated training dataset under misspecification.

        The call signature and return shapes are identical to
        ``SimulatedData.get_simulated_training_data`` so that downstream
        code (training, evaluation) is unaware of the misspecification.

        Returns
        -------
        data_train, params
        """
        from ..model import scClone2DR

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

        # Structural setup
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
                torch.tensor(
                    np.abs(np.random.normal(0, 1, dim_all)),
                    dtype=torch.float32,
                )
                / np.sqrt(dim_all)
            )

        params["offset_healthy"] = torch.zeros(D)
        params["offset_tumor"]   = torch.zeros(D)
        params["beta_control"]   = torch.ones(1, dtype=torch.float32)

        if neg_bin_n >= 1:
            theta_fd_np = np.random.negative_binomial(neg_bin_n, 0.001, N)
        else:
            theta_fd_np = np.full(N, 1000.0 * neg_bin_n)
        params["theta_fd"]  = torch.tensor(theta_fd_np[:N], dtype=torch.float32)
        params["theta_rna"] = 40.0

        # ---- pre-assay covariates ----------------------------------------
        log_ratio = np.log(0.6 / 0.9)
        data_train["X_nu_control"] = torch.tensor(
            np.random.normal(log_ratio, var_preassay, (C, N, 1)),
            dtype=torch.float32,
        )
        data_train["X_nu_drug"] = torch.tensor(
            np.random.normal(log_ratio, var_preassay, (R, D, N, 1)),
            dtype=torch.float32,
        )

        # ---- cell / well counts ------------------------------------------
        data_train["n_rna"] = (5000 * torch.ones((Kmax, N))).int()
        data_train["n_r"]   = (1000 * torch.ones((R, D, N))).int()
        data_train["n_c"]   = (1000 * torch.ones((C, N))).int()

        # ---- sample from the well-specified model first ------------------
        pyro.clear_param_store()
        data_samp, _ = model.sampling(data_train, params)
        data_train   = load_from_sampling(data_train, data_samp)

        # ==================================================================
        # >>> MISSPEC 1: replace survival probabilities with an alternative
        #     drug-response function
        # ==================================================================
        if self.response_misspec != ResponseMisspec.NONE:
            response_fn = _RESPONSE_FN[self.response_misspec]

            # build kwargs specific to the chosen function
            fn_kwargs: dict = {}
            if self.response_misspec == ResponseMisspec.HILL:
                fn_kwargs = {"ec50": self.hill_ec50, "hill_n": self.hill_n}
            elif self.response_misspec == ResponseMisspec.SATURATING:
                fn_kwargs = {"rate": self.saturating_rate}
            elif self.response_misspec == ResponseMisspec.THRESHOLD:
                fn_kwargs = {"threshold": self.threshold_val}

            # compute misspecified survival for every (clone, drug, sample)
            pi_misspec = torch.zeros((Kmax, D, N))
            for k in range(Kmax):
                x_k = data_train["X"][k]           # (N, dim_all)
                pi_misspec[k] = response_fn(
                    x_k, params["beta"], **fn_kwargs
                ).T                                 # (D, N)

            # re-draw drug-response counts from misspecified π
            for r_idx in range(R):
                for d_idx in range(D):
                    for n_idx in range(N):
                        n_total = data_train["n_r"][r_idx, d_idx, n_idx].item()
                        # weighted survival across clones
                        pi_weighted = (
                            pi_misspec[:, d_idx, n_idx]
                            * params["proportions"][n_idx]
                        ).sum().clamp(0, 1)
                        survivors = torch.distributions.Binomial(
                            total_count=n_total,
                            probs=pi_weighted,
                        ).sample()
                        data_train["n0_r"][r_idx, d_idx, n_idx] = survivors.int()

            params["pi"] = pi_misspec.permute(1, 0, 2) # (D, Kmax, N) for consistency with the model

        # ==================================================================
        # >>> MISSPEC 2: re-draw counts under alternative noise
        # ==================================================================
        if self.noise_misspec != NoiseMisspec.NONE:
            # Drug-response counts
            data_train["n0_r"] = _apply_noise(
                data_train["n0_r"].float(),
                self.noise_misspec,
                dispersion=self.noise_dispersion,
                zi_prob=self.zi_prob,
                t_df=self.t_df,
            ).int()
            # Clamp so survivors ≤ seeded
            data_train["n0_r"] = torch.minimum(
                data_train["n0_r"], data_train["n_r"]
            )

            # Control-well counts
            data_train["n0_c"] = _apply_noise(
                data_train["n0_c"].float(),
                self.noise_misspec,
                dispersion=self.noise_dispersion,
                zi_prob=self.zi_prob,
                t_df=self.t_df,
            ).int()
            data_train["n0_c"] = torch.minimum(
                data_train["n0_c"], data_train["n_c"]
            )

        # ==================================================================
        # >>> Compute ground-truth survival (may use well-specified or
        #     misspecified π depending on response_misspec above)
        # ==================================================================
        if self.response_misspec == ResponseMisspec.NONE:
            if data_train["single_cell_features"]:
                params["pi"] = model.compute_survival_probas_single_cell_features(
                    data_train, params
                )
            else:
                params["pi"] = model.compute_survival_probas_subclone_features(
                    data_train, params
                )
        params['nu_healthy_drug'] = model._get_nu_healthy_drug(data_train, params['beta_control'])
        params['nu_healthy_control'] = model._get_nu_healthy_control(data_train, params['beta_control'])

        return data_train, params


    def _save_clone_infos(self, data_ref, params, save_dir="./"):
        Kmax = data_ref['Kmax']
        N = data_ref['N']
        dic = {"cloneID": np.arange(0,Kmax,1),
               "clonelabel": ["healthy"]+["tumor" for i in range(Kmax-1)],
               "clonecategory": ["healthy"]+["tumor" for i in range(Kmax-1)]             
              }
        for i in range(N):
            dic['clonetype_{0}'.format(f'sample_{i}')] = [random.choice(['T cells', 'B cells'])] + ['Melanoma' for k in range(Kmax-1)]
        
        clone_infos = pd.DataFrame(dic)
        clone_infos.to_csv(os.path.join(save_dir, 'clone_infos.csv'))
        clone_infos = clone_infos.set_index("cloneID")
        return clone_infos

    def _save_RNA_sc(self, data_ref, params, clone_infos, save_dir="./"):
        L = data_ref['X'].shape[2]
        Kmax = data_ref['Kmax']
        N = data_ref['N']
        for sample in range(N):
        
            columns = (
                ['cell_id']
                + [f'dim_{k+1}' for k in range(L)]
                + [
                    'celltype',
                    'cellcategory',
                    'initial_cloneID',
                    'clonetype',
                    'clonelabel',
                    'clonecategory',
                    'cloneID'
                ]
            )
        
            total_cells = np.random.randint(20, 100)
            props = params['proportions'][sample, :]
            cloneID2nb_cells = np.random.multinomial(total_cells, props)
        
            cells = []
        
            for cloneID in range(Kmax):
                clonecells = []                
                for cell_id in range(cloneID2nb_cells[cloneID]):

                    if self.swap_frac<np.random.rand():
                        cloneID_sc = cloneID
                    else:
                        cloneID_sc = np.random.randint(0, Kmax)
                    clone_info_sc = clone_infos.iloc[cloneID_sc]
                    clone_cat_sc = clone_info_sc['clonecategory']
                    clonetype_sample_sc = clone_info_sc[f'clonetype_sample_{sample}']
                    clone_lab_sc = clone_info_sc['clonelabel']
                    clone_cat_sc = clone_info_sc['clonecategory']

                    feature = (
                        [f'cell_id_{cell_id}']
                        + [el.item() for el in data_ref['X'][cloneID, sample, :]]
                        + [
                            clonetype_sample_sc,
                            clone_cat_sc,
                            cloneID_sc,
                            clonetype_sample_sc,
                            clone_cat_sc,
                            clone_cat_sc,
                            cloneID_sc
                        ]
                    )
        
                    clonecells.append(feature)
        
                if clonecells:
                    cells.append(pd.DataFrame(clonecells, columns=columns))
        
            df_sample = pd.concat(cells, ignore_index=True)
        
            os.makedirs(os.path.join(save_dir, "sample2data"), exist_ok=True)
        
            df_sample.to_csv(
                os.path.join(save_dir, "sample2data", f"sample_{sample}.csv"),
                index=False
            )

    def _save_FD_data(self, data_ref, save_dir="./"):
        Rt, D, N = data_ref['n0_r'].shape
        Rc, N = data_ref['n0_c'].shape
        
        columns = ['SampleID', 'Concentration', 'Drug', 'Number_tumor_cells', 'Number_all_cells', 'Well_position_1', 'Well_position_2']
        data = []
        for sample in range(N):
            for c in range(Rc):
                ntumor = data_ref['n_c'][c,sample].item()-data_ref['n0_c'][c,sample].item()
                well_x = np.random.randint(0,24)
                well_y = np.random.randint(0,7)
                data.append([f'sample_{sample}', '5', 'DMSO', ntumor, data_ref['n_c'][c,sample].item(), well_x, well_y])
            for r in range(Rt):
                for d in range(D):
                    ntumor = data_ref['n_r'][r,d,sample].item()-data_ref['n0_r'][r,d,sample].item()
                    well_x = np.random.randint(0,24)
                    well_y = np.random.randint(0,7)
                    data.append([f'sample_{sample}', '5', f'Drug_{d}', ntumor, data_ref['n_r'][r,d,sample].item(), well_x, well_y])
        FD_data = pd.DataFrame(data, columns=columns)
        FD_data.to_csv(os.path.join(save_dir, 'FD_data.csv'), index=False)

    def generate_and_save_misspecified_data(self, data_train: dict | None = None,
        neg_bin_n: float = 2, mode_nu: NuMode = NuMode.NOISE_CORRECTION, mode_theta: ThetaMode = ThetaMode.NOT_SHARED_DECOUPLED, save_dir="./"):
        data_ref, params = self.get_simulated_training_data(data_train=data_train, mode_nu=mode_nu, mode_theta=mode_theta, neg_bin_n=neg_bin_n)
        clone_infos = self._save_clone_infos(data_ref, params, save_dir=save_dir)
        self._save_RNA_sc(data_ref, params, clone_infos, save_dir=save_dir)
        self._save_FD_data(data_ref, save_dir=save_dir)
        return data_ref, params

        
    # ------------------------------------------------------------------
    # Train / test split  (delegated — identical to SimulatedData)
    # ------------------------------------------------------------------

    def get_data_split(
        self, data: dict, idxs_train: list[int], idxs_test: list[int]
    ) -> tuple[dict, dict]:
        Ntrain, Ntest = len(idxs_train), len(idxs_test)
        Ntot          = Ntrain + Ntest
        Kmax, R, C, D = data["Kmax"], data["R"], data["C"], data["D"]

        masks_train = self._build_split_masks(
            data, idxs_train, idxs_test, Kmax, R, C, D, Ntrain, Ntot
        )
        masks_test = self._build_subset_masks(data, idxs_test, Kmax, R, C, D)

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

        if data["n_rna"] is not None:
            n_rna_buf = torch.zeros((Kmax, Ntot))
            n_rna_buf[:, :Ntrain] = torch.as_tensor(data["n_rna"][:, tr])
            n_rna_buf[:, Ntrain:] = torch.as_tensor(data["n_rna"][:, idxs_test])
            data_train["n_rna"] = n_rna_buf
            data_train["ini_proportions"] = _ini_proportions(
                data_train["n_rna"], Kmax, Ntot
            )
        else:
            data_train["n_rna"] = None
            data_train["ini_proportions"] = data["ini_proportions"]

        data_train["n0_r"] = data["n0_r"][:, :, tr]
        data_train["n_r"]  = data["n_r"][:, :, tr]

        props_buf = torch.zeros((Ntot, Kmax))
        props_buf[:Ntrain] = data["proportions"][tr]
        props_buf[Ntrain:] = data["proportions"][idxs_test]
        data_train["proportions"] = props_buf
        data_train = _add_frac_stats(data_train, masks_train)

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
        if data["n_rna"] is not None:
            data_test["n_rna"]     = torch.as_tensor(data["n_rna"][:, te])
            data_test["ini_proportions"] = _ini_proportions(
                data_test["n_rna"], Kmax, Ntest
            )
        else:
            data_test["n_rna"] = None
            data_train["ini_proportions"] = data["ini_proportions"]

        data_test["n0_c"]        = data["n0_c"][:, te]
        data_test["n_c"]         = data["n_c"][:, te]
        data_test["n0_r"]        = data["n0_r"][:, :, te]
        data_test["n_r"]         = data["n_r"][:, :, te]
        data_test["proportions"] = data["proportions"][te]
        data_test = _add_frac_stats(data_test, masks_test)

        return data_train, data_test

    def get_params_split(
        self, params: dict, idxs_train: list[int], idxs_test: list[int]
    ) -> tuple[dict, dict]:
        params_train = copy.deepcopy(params)
        params_test  = copy.deepcopy(params)
        for key in ("proportions", "theta_fd"):
            params_train[key] = params[key][idxs_train, ...]
            params_test[key]  = params[key][idxs_test, ...]
        if "pi" in params:
            params_train["pi"] = params["pi"][:, :, idxs_train]
            params_test["pi"]  = params["pi"][:, :, idxs_test]
        if "nu_healthy_control" in params:
            params_train["nu_healthy_control"] = params["nu_healthy_control"][:, idxs_train]
            params_test["nu_healthy_control"] = params["nu_healthy_control"][:, idxs_test]
        if "nu_healthy_drug" in params:
            params_train["nu_healthy_drug"]  = params["nu_healthy_drug"][:, :, idxs_train]
            params_test["nu_healthy_drug"]  = params["nu_healthy_drug"][:, :, idxs_test]

        return params_train, params_test

    # ------------------------------------------------------------------
    # Data-representation transforms
    # ------------------------------------------------------------------

    def get_base_from_data(self, dic: dict) -> tuple[dict, "MisspecifiedSimulatedData"]:
        data = copy.deepcopy(dic)
        data["X"] = torch.zeros_like(data["X"])
        dataset = self._clone_with_topology(data["Kmax"])
        return data, dataset

    def get_bulk_from_data(self, dic: dict) -> tuple[dict, "MisspecifiedSimulatedData"]:
        data    = copy.deepcopy(dic)
        dataset = self._clone_with_topology(2)

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

        data["n_rna"]        = None
        data["masks"]["RNA"] = torch.zeros((2, data["N"]))

        return data, dataset

    def get_bimodal_from_data(self, dic: dict) -> tuple[dict, "MisspecifiedSimulatedData"]:
        data    = copy.deepcopy(dic)
        dataset = self._clone_with_topology(2)

        healthy = dataset.cat2clusters["healthy"]
        tumor   = dataset.cat2clusters["tumor"]
        Ndrug   = data["X"].shape[1]

        Z = torch.zeros((2, data["X"].shape[1], data["X"].shape[2]))
        for out_idx, clust_idxs in enumerate([healthy, tumor]):
            w = data["proportions"].T[clust_idxs, :Ndrug]
            w = w / w.sum(dim=0, keepdim=True)
            Z[out_idx] = (
                torch.nan_to_num(data["X"][clust_idxs]) * w[:, :, None]
            ).sum(dim=0)
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

    def _clone_with_topology(self, Kmax: int) -> "MisspecifiedSimulatedData":
        ds = MisspecifiedSimulatedData(
            response_misspec=self.response_misspec,
            noise_misspec=self.noise_misspec,
        )
        cluster2clonelabel, clonelabel2cat = _make_simple_topology(Kmax)
        ds.init_topology(cluster2clonelabel, clonelabel2cat)
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
