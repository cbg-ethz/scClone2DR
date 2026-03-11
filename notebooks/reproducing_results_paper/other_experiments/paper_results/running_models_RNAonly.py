import sys
sys.path.append('../../')
import scClone2DR as sccdr
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pickle
import torch
from tqdm import tqdm
import plotly.io as pio


import sys
sys.path.append('/data/users/quentin/final_package/')
import scClone2DR as sccdr
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pickle
import torch
from tqdm import tqdm
import os
import plotly.io as pio
rootpath = '/data/users/quentin/final_package/experiments/paper_results'

gene_set_collections = ['c6','hallmarks', 'c2_pid', 'c2_kegg_medicus']

for COHORT in ['melanoma']:
    for gene_set_collection in gene_set_collections:
        mode_features = 'metacells_{0}'.format(gene_set_collection)

        directory = os.path.join(rootpath,COHORT)
        if not os.path.exists(directory):
            os.makedirs(directory)
        pathsave = os.path.join(rootpath,COHORT,gene_set_collection)
        if not os.path.exists(pathsave):
            os.makedirs(pathsave)

        if COHORT=='melanoma':
            path_rna = '/data/users/04_share_reanalysis_results/02_melanoma/02_atypical_removed_preprocessing/{0}/'.format(mode_features)
            path_fastdrug = '/data/users/04_share_reanalysis_results/02_melanoma/MEL_CNN_abundances_no_plate_effect_correction.csv'
            concentration_DMSO = '100'
            concentration_drug = '5'
        elif COHORT=='aml':
            concentration_DMSO = '200'
            concentration_drug = '10'
            path_rna = '/data/users/04_share_reanalysis_results/01_aml/02_atypical_removed_preprocessing/{0}/'.format(mode_features)
            path_fastdrug = '/data/users/04_share_reanalysis_results/01_aml/AML_PCY_cell_numbers_no_plate_effect_correction.csv'
            path_info_cohort = '/data/users/04_share_reanalysis_results/01_aml/2024-08-15_aml_overview_scRNA.tsv'

        model = sccdr.models.scClone2DR(path_fastdrug=path_fastdrug, path_rna=path_rna)
        data_ref = model.get_real_data(concentration_DMSO=concentration_DMSO, concentration_drug=concentration_drug)

        idxs_train = [i for i in range(int(data_ref['N']))]
        idxs_test = []

        data_train, data_test, sample_names_train, sample_names_test = model.get_real_data_split(idxs_train, idxs_test)
        params_svi = model.train(data_train, penalty_l1=0.1, penalty_l2=0.1 , n_steps=2000)
        with open(os.path.join(rootpath,'{0}/{1}/params_svi.pkl'.format(COHORT, gene_set_collection)), 'wb') as handle:
            pickle.dump(params_svi, handle, protocol=pickle.HIGHEST_PROTOCOL)

        posterior_mean_params = model.sampling_from_posterior(data_ref, pathsave, params=params_svi, nb_ites=200, sample_names=model.sample_names)

        with open(os.path.join(rootpath,'{0}/{1}/posterior_mean_params.pkl'.format(COHORT, gene_set_collection)), 'wb') as handle:
            pickle.dump(posterior_mean_params, handle, protocol=pickle.HIGHEST_PROTOCOL)