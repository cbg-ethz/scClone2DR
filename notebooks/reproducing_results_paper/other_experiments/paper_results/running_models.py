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

gene_set_collections = ['gene'] #['c6','hallmarks','c2_pid','gene']
cohort2clonemodes = {'melanoma': ['scatrex','phenograph'], 'aml':['phenograph']} #{'melanoma': [ 'scatrex', 'phenograph'], 'aml':['phenograph']}
feat2penl1 = {'gene':51.2,'c6':3.2,'hallmarks':3.2,'c2_pid':12.8}
feat2penl2 = {'gene':3.2,'c6':1.6,'hallmarks':3.2,'c2_pid':3.2}

for gene_set_collection in gene_set_collections:
    for COHORT in ['aml']:
        for clonemode in cohort2clonemodes[COHORT]:
            pathsave = os.path.join(rootpath,'{0}/{1}_{2}'.format(COHORT, gene_set_collection, clonemode))
            if not os.path.exists(pathsave):
                os.makedirs(pathsave)
            
            mode_features = 'metacells_{0}_{1}'.format(gene_set_collection, clonemode)

            if COHORT=='melanoma':
                path_rna = '/data/users/04_share_reanalysis_results/melanoma_2025/02_atypical_removed_preprocessing/{0}/'.format(mode_features)
                path_fastdrug = '/data/users/04_share_reanalysis_results/melanoma_2025/MEL_CNN_abundances_no_plate_effect_correction.csv'
                concentration_DMSO = '100'
                concentration_drug = '5'
            elif COHORT=='aml':
                concentration_DMSO = '200'
                concentration_drug = '10'
                path_rna = '/data/users/04_share_reanalysis_results/aml_2025/02_atypical_removed_preprocessing/{0}/'.format(mode_features)
                path_fastdrug = '/data/users/04_share_reanalysis_results/01_aml/AML_PCY_cell_numbers_no_plate_effect_correction.csv'
                path_info_cohort = '/data/users/04_share_reanalysis_results/01_aml/2024-08-15_aml_overview_scRNA.tsv'

            model = sccdr.models.scClone2DR(path_fastdrug=path_fastdrug, path_rna=path_rna, type_guide="lowrank_MVN", rank=10)
            data_ref = model.get_real_data(concentration_DMSO=concentration_DMSO, concentration_drug=concentration_drug)

            idxs_train = [i for i in range(int(data_ref['N']))]
            idxs_test = []

            data_train, data_test, sample_names_train, sample_names_test = model.get_real_data_split(idxs_train, idxs_test)

            penalty_l1 = feat2penl1[gene_set_collection]
            penalty_l2 = feat2penl2[gene_set_collection]
            params_svi = model.train(data_train, penalty_l1=penalty_l1, penalty_l2=penalty_l2, n_steps=2000)

            with open(os.path.join(rootpath,'{0}/{1}_{2}/params_svi.pkl'.format(COHORT, gene_set_collection, clonemode)), 'wb') as handle:
                pickle.dump(params_svi, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if True:
                posterior_mean_params = model.sampling_from_posterior(data_ref, pathsave, params=params_svi, nb_ites=300, sample_names=model.sample_names)

                with open(os.path.join(rootpath,'{0}/{1}_{2}/posterior_mean_params.pkl'.format(COHORT, gene_set_collection, clonemode)), 'wb') as handle:
                    pickle.dump(posterior_mean_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            del data_ref
            del posterior_mean_params
            del params_svi
            del data_train
            del data_test