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
import random
import plotly.io as pio
rootpath = '/data/users/quentin/final_package/experiments/paper_results'

gene_set_collections = ['hallmarks'] #['c6','hallmarks', 'c2_pid', 'c2_kegg_medicus']
cohort2clonemodes = {'melanoma': ['scatrex','phenograph'], 'aml':['phenograph']}

penalties_l1 = [0.2*2**i for i in range(3,10)]
penalties_l2 = [0.05*2**i for i in range(3,10)]

penalties_l1 = [0.1]
penalties_l2 = [0.1]


stabilityname = 'stability_paper'

for COHORT in ['melanoma']:
    for clonemode in cohort2clonemodes[COHORT]:
        for gene_set_collection in gene_set_collections:
            for pl1, penalty_l1 in tqdm(enumerate(penalties_l1)):
                for penalty_l2 in penalties_l2:
            
                    tempdir = os.path.join(rootpath,'{0}/{1}_{2}/{3}/'.format(COHORT, gene_set_collection, clonemode, stabilityname))
                    if not os.path.exists(tempdir):
                        os.makedirs(tempdir)

                    mode_features = 'metacells_{0}_{1}'.format(gene_set_collection, clonemode)

                    directory = os.path.join(rootpath,COHORT)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    pathsave = os.path.join(rootpath,COHORT,gene_set_collection)
                    if not os.path.exists(pathsave):
                        os.makedirs(pathsave)

                    if COHORT=='melanoma':
                        path_rna = '/data/users/04_share_reanalysis_results/melanoma_2025/02_atypical_removed_preprocessing/{0}/'.format(mode_features)
                        path_fastdrug = '/data/users/04_share_reanalysis_results/melanoma_2025/MEL_CNN_abundances_no_plate_effect_correction.csv'
                        concentration_DMSO = '100'
                        concentration_drug = '5'
                    elif COHORT=='aml':
                        concentration_DMSO = '200'
                        concentration_drug = '10'
                        path_rna = '/data/users/04_share_reanalysis_results/01_aml/02_atypical_removed_preprocessing/{0}/'.format(mode_features)
                        path_fastdrug = '/data/users/04_share_reanalysis_results/01_aml/AML_PCY_cell_numbers_no_plate_effect_correction.csv'
                        path_info_cohort = '/data/users/04_share_reanalysis_results/01_aml/2024-08-15_aml_overview_scRNA.tsv'

                    model = sccdr.models.scClone2DR(path_fastdrug=path_fastdrug, path_rna=path_rna, type_guide="lowrank_MVN", rank=10)
                    data_ref = model.get_real_data(concentration_DMSO=concentration_DMSO, concentration_drug=concentration_drug)

                    for ite in range(0,50):
                        try:
                            np.random.seed(ite)
                            idxs = np.arange(int(data_ref['N']))
                            np.random.shuffle(idxs)  # NumPy's shuffle respects the seed
                            idxs_train = idxs[:int(0.63*data_ref['N'])]
                            idxs_test = idxs[int(0.63*data_ref['N']):]

                            data_train, data_test, sample_names_train, sample_names_test = model.get_real_data_split(idxs_train, idxs_test)
                            if gene_set_collection=='gene':
                                params_svi = model.train(data_train, penalty_l1=penalty_l1, penalty_l2=penalty_l2, n_steps=2000)
                            else:
                                params_svi = model.train(data_train, penalty_l1=penalty_l1, penalty_l2=penalty_l2 , n_steps=2000)
                            params_svi['sample_names_train'] = sample_names_train
                            params_svi['sample_names_test'] = sample_names_test
                            params_svi['idxs_train'] = idxs_train
                            params_svi['idxs_test'] = idxs_test

                            with open(os.path.join(rootpath,'{0}/{1}_{2}/{6}/params_svi_ite_{5}_l1_{3}_l2_{4}.pkl'.format(COHORT, gene_set_collection, clonemode, str(penalty_l1), str(penalty_l2), ite, stabilityname)), 'wb') as handle:
                                pickle.dump(params_svi, handle, protocol=pickle.HIGHEST_PROTOCOL)

                            if False:
                                posterior_mean_params = model.sampling_from_posterior(data_ref, pathsave, params=params_svi, nb_ites=300, sample_names=model.sample_names)

                                with open(os.path.join(rootpath,'{0}/{1}_{2}/posterior_mean_params.pkl'.format(COHORT, gene_set_collection, clonemode)), 'wb') as handle:
                                    pickle.dump(posterior_mean_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        except:
                            pass