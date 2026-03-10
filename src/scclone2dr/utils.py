"""Utility functions for scClone2DR package."""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

def merge_data_params(data: dict, params: dict | None) -> dict:
    """Deep-copy *data* and overlay *params*, converting numpy arrays to tensors."""
    merged: dict = {}
    for key, val in data.items():
        merged[key] = val.clone().detach() if torch.is_tensor(val) else deepcopy(val)
    if params is not None:
        for key, val in params.items():
            merged[key] = torch.as_tensor(val) if isinstance(val, np.ndarray) else val
    return merged


def sigmoid(a: torch.Tensor) -> torch.Tensor:
    """Compute sigmoid function.
    
    Parameters
    ----------
    a : torch.Tensor
        Input tensor
        
    Returns
    -------
    torch.Tensor
        Sigmoid of input
    """
    return 1 / (1 + torch.exp(-a))


def get_robust(ls: np.ndarray) -> np.ndarray:
    """Get robust sorted array.
    
    Parameters
    ----------
    ls : np.ndarray
        Input array
        
    Returns
    -------
    np.ndarray
        Sorted array
    """
    ls = np.sort(ls)
    n = len(ls)
    return ls[:n]


def masked_softmax(vec, mask, dim=1):
    masked_vec = vec
    masked_vec[~mask] = -float('inf')
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec-max_vec)
    # we fill with 0 when a patient has no cell belonging to a given cloneID 
    exps = torch.nan_to_num(exps)

    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros=(masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps/masked_sums



def get_ini_proportions(data):
    N, Kmax = data['N'], data['Kmax']
    regu = 10 * torch.ones(data['n_rna'].shape)
    proportions = torch.tensor(np.copy((data['n_rna'] + regu) / torch.tile(torch.sum(data['n_rna']+regu, dim=0).reshape(1,N), (Kmax,1)))).T
    return proportions



def plot_CredInt(x, mean, bottom, top, color='#2187bb', horizontal_line_width=0.25, alpha=1, linestyle='-'):
    left = x - horizontal_line_width / 2
    right = x + horizontal_line_width / 2
    plt.plot([x, x], [top, bottom], color=color, alpha=alpha, linestyle=linestyle)
    plt.plot([left, right], [top, top], color=color, alpha=alpha, linestyle=linestyle)
    plt.plot([left, right], [bottom, bottom], color=color, alpha=alpha, linestyle=linestyle)
    
    
# Function to find distance
def shortest_distance(x1, y1, a, b, c): 
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    return d



def load_from_sampling(data, data_samp):
    #data['theta_fd'] = data_samp['theta_fd']
    data['n0_c'] = data_samp['n0_c']
    data['n0_r'] = data_samp['n0_r']
    try:
        data['n_c'] = data_samp['n_c']
        data['n_r'] = data_samp['n_r']
    except:
        pass
    data['n_rna'] = torch.tensor(data_samp['n_rna'])
    for i in range(data['N']):
        for k in range(data['Kmax']):
            if data['n_rna'][k,i]<0.5:
                data['n_rna'][k,i] = 1.

    N = data['n0_r'].shape[2]
    data['frac_r'] = 1. - data['n0_r']/data['n_r']
    data['frac_c'] = 1. - data['n0_c']/data['n_c']
    frac_r = 1. - torch.mean(data['n0_r']/data['n_r'], dim=0)
    frac_c = 1. - torch.mean(data['n0_c']/data['n_c'], dim=0)
    data['log_scores'] = torch.log(frac_c[:N] / frac_r)
    data['simulated_data'] = True
    return data

