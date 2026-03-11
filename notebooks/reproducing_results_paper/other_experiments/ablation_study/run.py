ls_mode_nu = ["fixed", "jack_mode"] 
ls_mode_theta = ['no overdispersion', 'not shared decoupled']

import sys
sys.path.append('../../')
import numpy as np
import pandas as pd
import math
import os
import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

import torch
import pyro
from torch.distributions import constraints
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.optim import Adam
import pyro.distributions as dist
import torch.distributions as tdist
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete, Predictive
pyro.enable_validation(True)

import generative_all_drugs_allincontrol
from generative_all_drugs_allincontrol.model_feat_control.sampling import *
from generative_all_drugs_allincontrol.model_feat_control.sampling_test import *
from generative_all_drugs_allincontrol.model_feat_control.prior import *

from generative_all_drugs_allincontrol.data_preprocessing.get_real_data import *
from generative_all_drugs_allincontrol.model_feat_control.svi_optim import *
from generative_all_drugs_allincontrol.visualization.vis_real_data import *
from generative_all_drugs_allincontrol.utils import *





import copy

ls_R = [5,10,15,20]
#ls_neg_bin_n = [1,10,100]
ls_neg_bin_n = [0.1, 0.01, 0.001]


for mode_nu in ls_mode_nu:
    for mode_theta in ls_mode_theta:
        if mode_theta=='no overdispersion':
            with_overdispersion = False
        else:
            with_overdispersion = True
        for ite in range(10):
            for neg_bin_n in ls_neg_bin_n:
                np.random.seed(ite)
                data_ref = get_simulated_training_data({'C':24,'R':1,'N':100,'Kmax':7, 'D':30, 'theta_rna':15}, neg_bin_n=neg_bin_n, mode_nu="jack_mode", mode_theta="not shared decoupled")
                idxs_train = [i for i in range(int(0.5*data_ref['N']))]
                idxs_test = [i for i in range(int(0.5*data_ref['N']), data_ref['N'])]
                for R in ls_R:
                    
                    print('R', R, 'neg_bin_n', neg_bin_n)
                    import os
                    directory = '/data/users/quentin/Paper_WP1/expes_paper/simu_ablation_study/save/{0}_{1}_R_{2}_neg_bin_{3}/'.format(mode_nu, mode_theta, str(R), str(neg_bin_n))
                    try:
                        os.makedirs(directory)
                    except:
                        pass

                    try:
                        DIC, DIC_sample = over_sample(data_ref, multi_C = 1, multi_R = R, mode_nu="jack_mode", mode_theta="not shared decoupled")
                        data_train, data_test = get_data_split_simu(DIC, idxs_train, idxs_test)


                        ls_params_obs = ['n0_c', 'n_c', 'n0_r', 'n_r', 'n_rna', 'masks', 'X', 'frac_r', 'frac_c']
                        data_train_svi = {param: data_train[param] for param in ls_params_obs}
                        ls_params_obs = ['n0_c', 'n0_r', 'n_rna', 'masks', 'X', 'X_nu_control', 'X_nu_drug']

                        pyro.clear_param_store()
                        trace = poutine.trace(lambda :guide_training(data_train)).get_trace()
                        trace.compute_log_prob()
                        print(trace.format_shapes())

                        penalty_l2 = 10
                        penalty_l1 = 10
                        if mode_theta=='no overdispersion':
                            train(lambda x: model_prior_without_overdispersion(x, fixed_proportions=False, mode_nu=mode_nu, include_score=False), guide_training, data_train, ls_params_obs, penalty_l1 = penalty_l1, penalty_l2 = penalty_l2, lr=0.005, n_steps=5000)
                        else:
                            train(lambda x: model_prior(x, fixed_proportions=False, mode_nu=mode_nu, theta_rna=None, mode_theta=mode_theta, include_score=False), guide_training, data_train, ls_params_obs, penalty_l1 = penalty_l1, penalty_l2 = penalty_l2, lr=0.01, n_steps=2000)

                        # Saving the learned parameters
                        params_svi = {}
                        for key,val in pyro.get_param_store().named_parameters():
                            print(key)
                            params_svi[key] = pyro.param(key).detach().numpy()

                        with open(directory+'data_train_ite_{0}.pkl'.format(str(ite)), 'wb') as f:
                            pickle.dump(data_train, f) 
                        with open(directory+'data_ite_{0}.pkl'.format(str(ite)), 'wb') as f:
                            pickle.dump(DIC, f) 
                        with open(directory+'data_sample_ite_{0}.pkl'.format(str(ite)), 'wb') as f:
                            pickle.dump(DIC_sample, f) 
                        with open(directory+'data_test_ite_{0}.pkl'.format(str(ite)), 'wb') as f:
                            pickle.dump(data_test, f) 
                        with open(directory+'params_svi_ite_{0}.pkl'.format(str(ite)), 'wb') as f:
                            pickle.dump(params_svi, f)
                            
                    except:
                        pass


